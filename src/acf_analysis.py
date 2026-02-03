"""ACF/PACF 自相关分析模块

对BTC日线数据的多序列（对数收益率、平方收益率、绝对收益率、成交量）进行
自相关函数(ACF)、偏自相关函数(PACF)分析，自动检测显著滞后阶与周期性模式，
并执行 Ljung-Box 检验以验证序列依赖结构。
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.font_config import configure_chinese_font
configure_chinese_font()
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

from src.data_loader import load_klines
from src.preprocessing import add_derived_features


# ============================================================
# 常量配置
# ============================================================

# ACF/PACF 最大滞后阶数
ACF_MAX_LAGS = 100
PACF_MAX_LAGS = 40

# Ljung-Box 检验的滞后组
LJUNGBOX_LAG_GROUPS = [10, 20, 50, 100]

# 显著性水平对应的 z 值（双侧 5%）
Z_CRITICAL = 1.96

# 分析目标序列名称 -> 列名映射
SERIES_CONFIG = {
    "log_return": {
        "column": "log_return",
        "label": "对数收益率 (Log Return)",
        "purpose": "检测线性序列相关性",
    },
    "squared_return": {
        "column": "squared_return",
        "label": "平方收益率 (Squared Return)",
        "purpose": "检测波动聚集效应 / ARCH效应",
    },
    "abs_return": {
        "column": "abs_return",
        "label": "绝对收益率 (Absolute Return)",
        "purpose": "非线性依赖关系的稳健性检验",
    },
    "volume": {
        "column": "volume",
        "label": "成交量 (Volume)",
        "purpose": "检测成交量自相关性",
    },
}


# ============================================================
# 核心计算函数
# ============================================================

def compute_acf(series: pd.Series, nlags: int = ACF_MAX_LAGS) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算自相关函数及置信区间

    Parameters
    ----------
    series : pd.Series
        输入时间序列（已去除NaN）
    nlags : int
        最大滞后阶数

    Returns
    -------
    acf_values : np.ndarray
        ACF 值数组，shape=(nlags+1,)
    confint : np.ndarray
        置信区间数组，shape=(nlags+1, 2)
    """
    clean = series.dropna().values
    # alpha=0.05 对应 95% 置信区间
    acf_values, confint = acf(clean, nlags=nlags, alpha=0.05, fft=True)
    return acf_values, confint


def compute_pacf(series: pd.Series, nlags: int = PACF_MAX_LAGS) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算偏自相关函数及置信区间

    Parameters
    ----------
    series : pd.Series
        输入时间序列（已去除NaN）
    nlags : int
        最大滞后阶数

    Returns
    -------
    pacf_values : np.ndarray
        PACF 值数组
    confint : np.ndarray
        置信区间数组
    """
    clean = series.dropna().values
    # 确保 nlags 不超过样本量的一半
    max_allowed = len(clean) // 2 - 1
    nlags = min(nlags, max_allowed)
    pacf_values, confint = pacf(clean, nlags=nlags, alpha=0.05, method='ywm')
    return pacf_values, confint


def find_significant_lags(
    acf_values: np.ndarray,
    n_obs: int,
    start_lag: int = 1,
) -> List[int]:
    """
    识别超过 ±1.96/√N 置信带的显著滞后阶

    Parameters
    ----------
    acf_values : np.ndarray
        ACF 值数组（包含 lag 0）
    n_obs : int
        样本总数（用于计算 Bartlett 置信带宽度）
    start_lag : int
        从哪个滞后阶开始检测（默认跳过 lag 0）

    Returns
    -------
    significant : list of int
        显著的滞后阶列表
    """
    threshold = Z_CRITICAL / np.sqrt(n_obs)
    significant = []
    for lag in range(start_lag, len(acf_values)):
        if abs(acf_values[lag]) > threshold:
            significant.append(lag)
    return significant


def detect_periodic_pattern(
    significant_lags: List[int],
    min_period: int = 2,
    max_period: int = 50,
    min_occurrences: int = 3,
    tolerance: int = 1,
) -> List[Dict[str, Any]]:
    """
    检测显著滞后阶中的周期性模式

    算法：对每个候选周期 p，检查 p, 2p, 3p, ... 是否在显著滞后阶集合中
    （允许 ±tolerance 偏差），若命中次数 >= min_occurrences 则认为存在周期。

    Parameters
    ----------
    significant_lags : list of int
        显著滞后阶列表
    min_period : int
        最小候选周期
    max_period : int
        最大候选周期
    min_occurrences : int
        最少需要出现的周期倍数次数
    tolerance : int
        允许的滞后偏差（天数）

    Returns
    -------
    patterns : list of dict
        检测到的周期性模式列表，每个元素包含：
        - period: 周期长度
        - hits: 命中的滞后阶列表
        - count: 命中次数
        - fft_note: FFT 交叉验证说明
    """
    if not significant_lags:
        return []

    sig_set = set(significant_lags)
    max_lag = max(significant_lags)
    patterns = []

    for period in range(min_period, min(max_period + 1, max_lag + 1)):
        hits = []
        # 检查周期的整数倍是否出现在显著滞后阶中
        multiple = 1
        while period * multiple <= max_lag + tolerance:
            target = period * multiple
            # 在 ±tolerance 范围内查找匹配
            for offset in range(-tolerance, tolerance + 1):
                if (target + offset) in sig_set:
                    hits.append(target + offset)
                    break
            multiple += 1

        if len(hits) >= min_occurrences:
            # FFT 交叉验证说明：周期 p 天对应频率 1/p
            fft_freq = 1.0 / period
            patterns.append({
                "period": period,
                "hits": hits,
                "count": len(hits),
                "fft_note": (
                    f"若FFT频谱在 f={fft_freq:.4f} (1/{period}天) "
                    f"处存在峰值，则交叉验证通过"
                ),
            })

    # 按命中次数降序排列，去除被更短周期包含的冗余模式
    patterns.sort(key=lambda x: (-x["count"], x["period"]))
    filtered = _filter_harmonic_patterns(patterns)

    return filtered


def _filter_harmonic_patterns(
    patterns: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    过滤谐波冗余的周期模式

    如果周期 A 是周期 B 的整数倍且命中数不明显更多，则保留较短周期。
    """
    if len(patterns) <= 1:
        return patterns

    kept = []
    periods_kept = set()

    for pat in patterns:
        p = pat["period"]
        # 检查是否为已保留周期的整数倍
        is_harmonic = False
        for kp in periods_kept:
            if p % kp == 0 and p != kp:
                is_harmonic = True
                break
        if not is_harmonic:
            kept.append(pat)
            periods_kept.add(p)

    return kept


def run_ljungbox_test(
    series: pd.Series,
    lag_groups: List[int] = None,
) -> pd.DataFrame:
    """
    对序列执行 Ljung-Box 白噪声检验

    Parameters
    ----------
    series : pd.Series
        输入时间序列
    lag_groups : list of int
        检验的滞后阶组

    Returns
    -------
    results : pd.DataFrame
        包含 lag, lb_stat, lb_pvalue 的结果表
    """
    if lag_groups is None:
        lag_groups = LJUNGBOX_LAG_GROUPS

    clean = series.dropna()
    max_lag = max(lag_groups)

    # 确保最大滞后不超过样本量
    if max_lag >= len(clean):
        lag_groups = [lg for lg in lag_groups if lg < len(clean)]
        if not lag_groups:
            return pd.DataFrame(columns=["lag", "lb_stat", "lb_pvalue"])
        max_lag = max(lag_groups)

    lb_result = acorr_ljungbox(clean, lags=max_lag, return_df=True)

    rows = []
    for lg in lag_groups:
        if lg <= len(lb_result):
            rows.append({
                "lag": lg,
                "lb_stat": lb_result.loc[lg, "lb_stat"],
                "lb_pvalue": lb_result.loc[lg, "lb_pvalue"],
            })

    return pd.DataFrame(rows)


# ============================================================
# 可视化函数
# ============================================================

def _plot_acf_grid(
    acf_data: Dict[str, Tuple[np.ndarray, np.ndarray, int, List[int]]],
    output_path: Path,
) -> None:
    """
    绘制 2x2 ACF 图

    Parameters
    ----------
    acf_data : dict
        键为序列名称，值为 (acf_values, confint, n_obs, significant_lags) 元组
    output_path : Path
        输出文件路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("BTC 自相关函数 (ACF) 分析", fontsize=16, fontweight='bold', y=0.98)

    series_keys = list(SERIES_CONFIG.keys())

    for idx, key in enumerate(series_keys):
        ax = axes[idx // 2, idx % 2]

        if key not in acf_data:
            ax.set_visible(False)
            continue

        acf_vals, confint, n_obs, sig_lags = acf_data[key]
        config = SERIES_CONFIG[key]
        lags = np.arange(len(acf_vals))
        threshold = Z_CRITICAL / np.sqrt(n_obs)

        # 绘制 ACF 柱状图
        colors = []
        for lag in lags:
            if lag == 0:
                colors.append('#2196F3')  # lag 0 用蓝色
            elif lag in sig_lags:
                colors.append('#F44336')  # 显著滞后用红色
            else:
                colors.append('#90CAF9')  # 非显著用浅蓝

        ax.bar(lags, acf_vals, color=colors, width=0.8, alpha=0.85)

        # 绘制置信带
        ax.axhline(y=threshold, color='#E91E63', linestyle='--',
                    linewidth=1.2, alpha=0.7, label=f'±{Z_CRITICAL}/√N = ±{threshold:.4f}')
        ax.axhline(y=-threshold, color='#E91E63', linestyle='--',
                    linewidth=1.2, alpha=0.7)
        ax.axhline(y=0, color='black', linewidth=0.5)

        # 标注显著滞后阶（仅标注前10个避免拥挤）
        sig_lags_sorted = sorted(sig_lags)[:10]
        for lag in sig_lags_sorted:
            if lag < len(acf_vals):
                ax.annotate(
                    f'{lag}',
                    xy=(lag, acf_vals[lag]),
                    xytext=(0, 8 if acf_vals[lag] > 0 else -12),
                    textcoords='offset points',
                    fontsize=7,
                    color='#D32F2F',
                    ha='center',
                    fontweight='bold',
                )

        ax.set_title(f'{config["label"]}\n({config["purpose"]})', fontsize=11)
        ax.set_xlabel('滞后阶 (Lag)', fontsize=10)
        ax.set_ylabel('ACF', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_xlim(-1, len(acf_vals))
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(labelsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[ACF图] 已保存: {output_path}")


def _plot_pacf_grid(
    pacf_data: Dict[str, Tuple[np.ndarray, np.ndarray, int, List[int]]],
    output_path: Path,
) -> None:
    """
    绘制 2x2 PACF 图

    Parameters
    ----------
    pacf_data : dict
        键为序列名称，值为 (pacf_values, confint, n_obs, significant_lags) 元组
    output_path : Path
        输出文件路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("BTC 偏自相关函数 (PACF) 分析", fontsize=16, fontweight='bold', y=0.98)

    series_keys = list(SERIES_CONFIG.keys())

    for idx, key in enumerate(series_keys):
        ax = axes[idx // 2, idx % 2]

        if key not in pacf_data:
            ax.set_visible(False)
            continue

        pacf_vals, confint, n_obs, sig_lags = pacf_data[key]
        config = SERIES_CONFIG[key]
        lags = np.arange(len(pacf_vals))
        threshold = Z_CRITICAL / np.sqrt(n_obs)

        # 绘制 PACF 柱状图
        colors = []
        for lag in lags:
            if lag == 0:
                colors.append('#4CAF50')
            elif lag in sig_lags:
                colors.append('#FF5722')
            else:
                colors.append('#A5D6A7')

        ax.bar(lags, pacf_vals, color=colors, width=0.6, alpha=0.85)

        # 置信带
        ax.axhline(y=threshold, color='#E91E63', linestyle='--',
                    linewidth=1.2, alpha=0.7, label=f'±{Z_CRITICAL}/√N = ±{threshold:.4f}')
        ax.axhline(y=-threshold, color='#E91E63', linestyle='--',
                    linewidth=1.2, alpha=0.7)
        ax.axhline(y=0, color='black', linewidth=0.5)

        # 标注显著滞后阶
        sig_lags_sorted = sorted(sig_lags)[:10]
        for lag in sig_lags_sorted:
            if lag < len(pacf_vals):
                ax.annotate(
                    f'{lag}',
                    xy=(lag, pacf_vals[lag]),
                    xytext=(0, 8 if pacf_vals[lag] > 0 else -12),
                    textcoords='offset points',
                    fontsize=7,
                    color='#BF360C',
                    ha='center',
                    fontweight='bold',
                )

        ax.set_title(f'{config["label"]}\n(PACF - 偏自相关)', fontsize=11)
        ax.set_xlabel('滞后阶 (Lag)', fontsize=10)
        ax.set_ylabel('PACF', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_xlim(-1, len(pacf_vals))
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(labelsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[PACF图] 已保存: {output_path}")


def _plot_significant_lags_summary(
    all_sig_lags: Dict[str, List[int]],
    n_obs: int,
    output_path: Path,
) -> None:
    """
    绘制所有序列的显著滞后阶汇总热力图

    Parameters
    ----------
    all_sig_lags : dict
        键为序列名称，值为显著滞后阶列表
    n_obs : int
        样本总数
    output_path : Path
        输出文件路径
    """
    max_lag = ACF_MAX_LAGS
    series_names = list(SERIES_CONFIG.keys())
    labels = [SERIES_CONFIG[k]["label"].split(" (")[0] for k in series_names]

    # 构建二值矩阵：行=序列，列=滞后阶
    matrix = np.zeros((len(series_names), max_lag + 1))
    for i, key in enumerate(series_names):
        for lag in all_sig_lags.get(key, []):
            if lag <= max_lag:
                matrix[i, lag] = 1

    fig, ax = plt.subplots(figsize=(20, 4))
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='none')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('滞后阶 (Lag)', fontsize=11)
    ax.set_title('显著自相关滞后阶汇总 (ACF > 置信带)', fontsize=13, fontweight='bold')

    # 每隔 5 个标注 x 轴
    ax.set_xticks(range(0, max_lag + 1, 5))
    ax.tick_params(labelsize=8)

    plt.colorbar(im, ax=ax, label='显著 (1) / 不显著 (0)', shrink=0.8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[显著滞后汇总图] 已保存: {output_path}")


# ============================================================
# 多尺度 ACF 分析
# ============================================================

def multi_scale_acf_analysis(intervals: list = None) -> Dict:
    """多尺度 ACF 对比分析"""
    if intervals is None:
        intervals = ['1h', '4h', '1d', '1w']

    results = {}
    for interval in intervals:
        try:
            df_tf = load_klines(interval)
            prices = df_tf['close'].dropna()
            returns = np.log(prices / prices.shift(1)).dropna()
            abs_returns = returns.abs()

            if len(returns) < 100:
                continue

            # 计算 ACF（对数收益率和绝对收益率）
            acf_ret, _ = acf(returns.values, nlags=min(50, len(returns)//4), alpha=0.05, fft=True)
            acf_abs, _ = acf(abs_returns.values, nlags=min(50, len(abs_returns)//4), alpha=0.05, fft=True)

            # 计算自相关衰减速度（对 |r| 的 ACF 做指数衰减拟合）
            lags = np.arange(1, len(acf_abs))
            acf_vals = acf_abs[1:]
            positive_mask = acf_vals > 0
            if positive_mask.sum() > 5:
                log_lags = np.log(lags[positive_mask])
                log_acf = np.log(acf_vals[positive_mask])
                slope, _, r_value, _, _ = stats.linregress(log_lags, log_acf)
                decay_rate = -slope
            else:
                decay_rate = np.nan

            results[interval] = {
                'acf_returns': acf_ret,
                'acf_abs_returns': acf_abs,
                'decay_rate': decay_rate,
                'n_samples': len(returns),
            }
        except Exception as e:
            print(f"  {interval} 分析失败: {e}")

    return results


def plot_multi_scale_acf(ms_results: Dict, output_path: Path) -> None:
    """
    绘制多尺度 ACF 对比图

    Parameters
    ----------
    ms_results : dict
        multi_scale_acf_analysis 返回的结果字典
    output_path : Path
        输出文件路径
    """
    if not ms_results:
        print("[多尺度ACF] 无数据，跳过绘图")
        return

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle("多时间尺度 ACF 对比分析", fontsize=16, fontweight='bold', y=0.98)

    colors = {'1h': '#1E88E5', '4h': '#43A047', '1d': '#E53935', '1w': '#8E24AA'}

    # 上图：对数收益率 ACF
    ax1 = axes[0]
    for interval, data in ms_results.items():
        acf_ret = data['acf_returns']
        lags = np.arange(len(acf_ret))
        color = colors.get(interval, '#000000')
        ax1.plot(lags, acf_ret, label=f'{interval}', color=color, linewidth=1.5, alpha=0.8)

    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_xlabel('滞后阶 (Lag)', fontsize=11)
    ax1.set_ylabel('ACF', fontsize=11)
    ax1.set_title('对数收益率 ACF 多尺度对比', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(alpha=0.3)
    ax1.tick_params(labelsize=9)

    # 下图：绝对收益率 ACF
    ax2 = axes[1]
    for interval, data in ms_results.items():
        acf_abs = data['acf_abs_returns']
        lags = np.arange(len(acf_abs))
        color = colors.get(interval, '#000000')
        decay = data['decay_rate']
        label_text = f"{interval} (衰减率={decay:.3f})" if not np.isnan(decay) else f"{interval}"
        ax2.plot(lags, acf_abs, label=label_text, color=color, linewidth=1.5, alpha=0.8)

    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_xlabel('滞后阶 (Lag)', fontsize=11)
    ax2.set_ylabel('ACF', fontsize=11)
    ax2.set_title('绝对收益率 ACF 多尺度对比（长记忆性检测）', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(alpha=0.3)
    ax2.tick_params(labelsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[多尺度ACF图] 已保存: {output_path}")


def plot_acf_decay_vs_scale(ms_results: Dict, output_path: Path) -> None:
    """
    绘制自相关衰减速度 vs 时间尺度

    Parameters
    ----------
    ms_results : dict
        multi_scale_acf_analysis 返回的结果字典
    output_path : Path
        输出文件路径
    """
    if not ms_results:
        print("[ACF衰减vs尺度] 无数据，跳过绘图")
        return

    # 提取时间尺度和衰减率
    interval_mapping = {'1h': 1/24, '4h': 4/24, '1d': 1, '1w': 7}
    scales = []
    decay_rates = []
    labels = []

    for interval, data in ms_results.items():
        if interval in interval_mapping and not np.isnan(data['decay_rate']):
            scales.append(interval_mapping[interval])
            decay_rates.append(data['decay_rate'])
            labels.append(interval)

    if len(scales) < 2:
        print("[ACF衰减vs尺度] 有效数据点不足，跳过绘图")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # 对数坐标绘图
    ax.scatter(scales, decay_rates, s=150, c=['#1E88E5', '#43A047', '#E53935', '#8E24AA'][:len(scales)],
               alpha=0.8, edgecolors='black', linewidth=1.5, zorder=3)

    # 标注点
    for i, label in enumerate(labels):
        ax.annotate(label, xy=(scales[i], decay_rates[i]),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=10, fontweight='bold', color='#333333')

    # 拟合趋势线（如果有足够数据点）
    if len(scales) >= 3:
        log_scales = np.log(scales)
        slope, intercept, r_value, _, _ = stats.linregress(log_scales, decay_rates)
        x_fit = np.logspace(np.log10(min(scales)), np.log10(max(scales)), 100)
        y_fit = slope * np.log(x_fit) + intercept
        ax.plot(x_fit, y_fit, '--', color='#FF6F00', linewidth=2, alpha=0.6,
                label=f'拟合趋势 (R²={r_value**2:.3f})')
        ax.legend(fontsize=10)

    ax.set_xscale('log')
    ax.set_xlabel('时间尺度 (天, 对数)', fontsize=12, fontweight='bold')
    ax.set_ylabel('ACF 幂律衰减指数 d', fontsize=12, fontweight='bold')
    ax.set_title('自相关衰减速度 vs 时间尺度\n（检测跨尺度长记忆性）', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, which='both')
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[ACF衰减vs尺度图] 已保存: {output_path}")


# ============================================================
# 主入口函数
# ============================================================

def run_acf_analysis(
    df: pd.DataFrame,
    output_dir: Union[str, Path] = "output/acf",
) -> Dict[str, Any]:
    """
    ACF/PACF 自相关分析主入口

    对对数收益率、平方收益率、绝对收益率、成交量四个序列执行完整的
    自相关分析流程，包括：ACF计算、PACF计算、显著滞后检测、周期性
    模式识别、Ljung-Box检验以及可视化。

    Parameters
    ----------
    df : pd.DataFrame
        日线DataFrame，需包含 log_return, squared_return, abs_return, volume 列
        （通常由 preprocessing.add_derived_features 生成）
    output_dir : str or Path
        图表输出目录

    Returns
    -------
    results : dict
        分析结果字典，结构如下：
        {
            "acf": {series_name: {"values": ndarray, "significant_lags": list, ...}},
            "pacf": {series_name: {"values": ndarray, "significant_lags": list, ...}},
            "ljungbox": {series_name: DataFrame},
            "periodic_patterns": {series_name: list of dict},
            "summary": {...}
        }
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 验证必要列存在
    required_cols = [cfg["column"] for cfg in SERIES_CONFIG.values()]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame 缺少必要列: {missing}。请先调用 add_derived_features()。")

    print("=" * 70)
    print("ACF / PACF 自相关分析")
    print("=" * 70)
    print(f"样本量: {len(df)}")
    print(f"时间范围: {df.index.min()} ~ {df.index.max()}")
    print(f"ACF最大滞后: {ACF_MAX_LAGS} | PACF最大滞后: {PACF_MAX_LAGS}")
    print(f"置信水平: 95% (z={Z_CRITICAL})")
    print()

    # 存储结果
    results = {
        "acf": {},
        "pacf": {},
        "ljungbox": {},
        "periodic_patterns": {},
        "summary": {},
    }

    # 用于绘图的中间数据
    acf_plot_data = {}   # {key: (acf_vals, confint, n_obs, sig_lags_set)}
    pacf_plot_data = {}
    all_sig_lags = {}    # {key: list of significant lag indices}

    # --------------------------------------------------------
    # 逐序列分析
    # --------------------------------------------------------
    for key, config in SERIES_CONFIG.items():
        col = config["column"]
        label = config["label"]
        purpose = config["purpose"]
        series = df[col].dropna()
        n_obs = len(series)

        print(f"{'─' * 60}")
        print(f"序列: {label}")
        print(f"  目的: {purpose}")
        print(f"  有效样本: {n_obs}")

        # ---------- ACF ----------
        acf_vals, acf_confint = compute_acf(series, nlags=ACF_MAX_LAGS)
        sig_lags_acf = find_significant_lags(acf_vals, n_obs)
        sig_lags_set = set(sig_lags_acf)

        results["acf"][key] = {
            "values": acf_vals,
            "confint": acf_confint,
            "significant_lags": sig_lags_acf,
            "n_obs": n_obs,
            "threshold": Z_CRITICAL / np.sqrt(n_obs),
        }
        acf_plot_data[key] = (acf_vals, acf_confint, n_obs, sig_lags_set)
        all_sig_lags[key] = sig_lags_acf

        print(f"  [ACF] 显著滞后阶数: {len(sig_lags_acf)}")
        if sig_lags_acf:
            # 打印前 20 个显著滞后
            display_lags = sig_lags_acf[:20]
            lag_str = ", ".join(str(l) for l in display_lags)
            if len(sig_lags_acf) > 20:
                lag_str += f" ... (共{len(sig_lags_acf)}个)"
            print(f"        滞后阶: {lag_str}")
            # 打印最大 ACF 值的滞后阶（排除 lag 0）
            max_idx = max(range(1, len(acf_vals)), key=lambda i: abs(acf_vals[i]))
            print(f"        最大|ACF|: lag={max_idx}, ACF={acf_vals[max_idx]:.6f}")

        # ---------- PACF ----------
        pacf_vals, pacf_confint = compute_pacf(series, nlags=PACF_MAX_LAGS)
        sig_lags_pacf = find_significant_lags(pacf_vals, n_obs)
        sig_lags_pacf_set = set(sig_lags_pacf)

        results["pacf"][key] = {
            "values": pacf_vals,
            "confint": pacf_confint,
            "significant_lags": sig_lags_pacf,
            "n_obs": n_obs,
        }
        pacf_plot_data[key] = (pacf_vals, pacf_confint, n_obs, sig_lags_pacf_set)

        print(f"  [PACF] 显著滞后阶数: {len(sig_lags_pacf)}")
        if sig_lags_pacf:
            display_lags_p = sig_lags_pacf[:15]
            lag_str_p = ", ".join(str(l) for l in display_lags_p)
            if len(sig_lags_pacf) > 15:
                lag_str_p += f" ... (共{len(sig_lags_pacf)}个)"
            print(f"        滞后阶: {lag_str_p}")

        # ---------- 周期性模式检测 ----------
        periodic = detect_periodic_pattern(sig_lags_acf)
        results["periodic_patterns"][key] = periodic

        if periodic:
            print(f"  [周期性] 检测到 {len(periodic)} 个周期模式:")
            for pat in periodic:
                hit_str = ", ".join(str(h) for h in pat["hits"][:8])
                print(f"    - 周期 {pat['period']}天 (命中{pat['count']}次): "
                      f"lags=[{hit_str}]")
                print(f"      FFT验证: {pat['fft_note']}")
        else:
            print(f"  [周期性] 未检测到明显周期模式")

        # ---------- Ljung-Box 检验 ----------
        lb_df = run_ljungbox_test(series, LJUNGBOX_LAG_GROUPS)
        results["ljungbox"][key] = lb_df

        print(f"  [Ljung-Box检验]")
        if not lb_df.empty:
            for _, row in lb_df.iterrows():
                lag_val = int(row["lag"])
                stat = row["lb_stat"]
                pval = row["lb_pvalue"]
                # 判断显著性
                sig_mark = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                reject_str = "拒绝H0(存在自相关)" if pval < 0.05 else "不拒绝H0(无显著自相关)"
                print(f"    lag={lag_val:3d}: Q={stat:12.2f}, p={pval:.6f} {sig_mark} → {reject_str}")
        print()

    # --------------------------------------------------------
    # 汇总
    # --------------------------------------------------------
    print("=" * 70)
    print("分析汇总")
    print("=" * 70)

    summary = {}
    for key, config in SERIES_CONFIG.items():
        label_short = config["label"].split(" (")[0]
        acf_sig = results["acf"][key]["significant_lags"]
        pacf_sig = results["pacf"][key]["significant_lags"]
        lb = results["ljungbox"][key]
        periodic = results["periodic_patterns"][key]

        # Ljung-Box 在最大 lag 下是否显著
        lb_significant = False
        if not lb.empty:
            max_lag_row = lb.iloc[-1]
            lb_significant = max_lag_row["lb_pvalue"] < 0.05

        summary[key] = {
            "label": label_short,
            "acf_significant_count": len(acf_sig),
            "pacf_significant_count": len(pacf_sig),
            "ljungbox_rejects_white_noise": lb_significant,
            "periodic_patterns_count": len(periodic),
            "periodic_periods": [p["period"] for p in periodic],
        }

        lb_verdict = "存在自相关" if lb_significant else "无显著自相关"
        period_str = (
            ", ".join(f"{p}天" for p in summary[key]["periodic_periods"])
            if periodic else "无"
        )

        print(f"  {label_short}:")
        print(f"    ACF显著滞后: {len(acf_sig)}个 | PACF显著滞后: {len(pacf_sig)}个")
        print(f"    Ljung-Box: {lb_verdict} | 周期性模式: {period_str}")

    results["summary"] = summary

    # --------------------------------------------------------
    # 可视化
    # --------------------------------------------------------
    print()
    print("生成可视化图表...")

    # 1) ACF 2x2 网格图
    _plot_acf_grid(acf_plot_data, output_dir / "acf_grid.png")

    # 2) PACF 2x2 网格图
    _plot_pacf_grid(pacf_plot_data, output_dir / "pacf_grid.png")

    # 3) 显著滞后汇总热力图
    _plot_significant_lags_summary(
        all_sig_lags,
        n_obs=len(df.dropna(subset=["log_return"])),
        output_path=output_dir / "significant_lags_heatmap.png",
    )

    # 4) 多尺度 ACF 分析
    print("\n多尺度 ACF 对比分析...")
    ms_results = multi_scale_acf_analysis(['1h', '4h', '1d', '1w'])
    if ms_results:
        plot_multi_scale_acf(ms_results, output_dir / "acf_multi_scale.png")
        plot_acf_decay_vs_scale(ms_results, output_dir / "acf_decay_vs_scale.png")
    results["multi_scale"] = ms_results

    print()
    print("=" * 70)
    print("ACF/PACF 分析完成")
    print(f"图表输出目录: {output_dir.resolve()}")
    print("=" * 70)

    return results


# ============================================================
# 独立运行入口
# ============================================================

if __name__ == "__main__":
    from data_loader import load_daily
    from preprocessing import add_derived_features

    # 加载并预处理数据
    print("加载日线数据...")
    df = load_daily()
    print(f"原始数据: {len(df)} 行")

    print("添加衍生特征...")
    df = add_derived_features(df)
    print(f"预处理后: {len(df)} 行, 列={list(df.columns)}")
    print()

    # 执行 ACF/PACF 分析
    results = run_acf_analysis(df, output_dir="output/acf")

    # 打印结果概要
    print()
    print("返回结果键:")
    for k, v in results.items():
        if isinstance(v, dict):
            print(f"  results['{k}']: {list(v.keys())}")
        else:
            print(f"  results['{k}']: {type(v).__name__}")
