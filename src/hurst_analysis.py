"""
Hurst指数分析模块
================
通过R/S分析和DFA（去趋势波动分析）计算Hurst指数，
评估BTC价格序列的长程依赖性和市场状态（趋势/均值回归/随机游走）。

核心功能：
- R/S (Rescaled Range) 分析
- DFA (Detrended Fluctuation Analysis) via nolds
- R/S 与 DFA 交叉验证
- 滚动窗口Hurst指数追踪市场状态变化
- 多时间框架Hurst对比分析
"""

import matplotlib
matplotlib.use('Agg')

from src.font_config import configure_chinese_font
configure_chinese_font()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
try:
    import nolds
    HAS_NOLDS = True
except Exception:
    HAS_NOLDS = False
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loader import load_klines
from src.preprocessing import log_returns


# ============================================================
# Hurst指数判定标准
# ============================================================
TREND_THRESHOLD = 0.55       # H > 0.55 → 趋势性（持续性）
MEAN_REV_THRESHOLD = 0.45    # H < 0.45 → 均值回归（反持续性）
# 0.45 <= H <= 0.55 → 近似随机游走


def interpret_hurst(h: float) -> str:
    """根据Hurst指数值给出市场状态解读"""
    if h > TREND_THRESHOLD:
        return f"趋势性 (H={h:.4f} > {TREND_THRESHOLD})：序列具有长程正相关，价格趋势倾向于持续"
    elif h < MEAN_REV_THRESHOLD:
        return f"均值回归 (H={h:.4f} < {MEAN_REV_THRESHOLD})：序列具有长程负相关，价格倾向于反转"
    else:
        return f"随机游走 (H={h:.4f} ≈ 0.5)：序列近似无记忆，价格变动近似独立"


# ============================================================
# R/S (Rescaled Range) 分析
# ============================================================
def _rs_for_segment(segment: np.ndarray) -> float:
    """计算单个分段的R/S统计量"""
    n = len(segment)
    if n < 2:
        return np.nan

    # 计算均值偏差的累积和
    mean_val = np.mean(segment)
    deviations = segment - mean_val
    cumulative = np.cumsum(deviations)

    # 极差 R = max(累积偏差) - min(累积偏差)
    R = np.max(cumulative) - np.min(cumulative)

    # 标准差 S
    S = np.std(segment, ddof=1)
    if S == 0:
        return np.nan

    return R / S


def rs_hurst(series: np.ndarray, min_window: int = 10, max_window: Optional[int] = None,
             num_scales: int = 30) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    R/S重标极差分析计算Hurst指数

    Parameters
    ----------
    series : np.ndarray
        时间序列数据（通常为对数收益率）
    min_window : int
        最小窗口大小
    max_window : int, optional
        最大窗口大小，默认为序列长度的1/4
    num_scales : int
        尺度数量

    Returns
    -------
    H : float
        Hurst指数
    log_ns : np.ndarray
        log(窗口大小)
    log_rs : np.ndarray
        log(平均R/S值)
    """
    n = len(series)
    if max_window is None:
        max_window = n // 4

    # 生成对数均匀分布的窗口大小
    window_sizes = np.unique(
        np.logspace(np.log10(min_window), np.log10(max_window), num=num_scales).astype(int)
    )

    log_ns = []
    log_rs = []

    for w in window_sizes:
        if w < 10 or w > n // 2:
            continue

        # 将序列分成不重叠的分段
        num_segments = n // w
        if num_segments < 1:
            continue

        rs_values = []
        for i in range(num_segments):
            segment = series[i * w: (i + 1) * w]
            rs_val = _rs_for_segment(segment)
            if not np.isnan(rs_val):
                rs_values.append(rs_val)

        if len(rs_values) > 0:
            mean_rs = np.mean(rs_values)
            if mean_rs > 0:
                log_ns.append(np.log(w))
                log_rs.append(np.log(mean_rs))

    log_ns = np.array(log_ns)
    log_rs = np.array(log_rs)

    # 线性回归：log(R/S) = H * log(n) + c
    if len(log_ns) < 3:
        return 0.5, log_ns, log_rs

    coeffs = np.polyfit(log_ns, log_rs, 1)
    H = coeffs[0]

    return H, log_ns, log_rs


# ============================================================
# DFA (Detrended Fluctuation Analysis) - 使用nolds库
# ============================================================
def dfa_hurst(series: np.ndarray) -> float:
    """
    使用nolds库进行DFA分析，返回Hurst指数

    Parameters
    ----------
    series : np.ndarray
        时间序列数据

    Returns
    -------
    float
        DFA估计的Hurst指数（DFA指数α，对于分数布朗运动 α = H + 0.5 - 0.5 = H）
    """
    if HAS_NOLDS:
        # nolds.dfa 返回的是DFA scaling exponent α
        # 对于对数收益率序列（增量过程），α ≈ H
        # 对于累积序列（如价格），α ≈ H + 0.5
        alpha = nolds.dfa(series)
        return alpha
    else:
        # 自实现的简化DFA
        N = len(series)
        y = np.cumsum(series - np.mean(series))
        scales = np.unique(np.logspace(np.log10(4), np.log10(N // 4), 20).astype(int))
        flucts = []
        for s in scales:
            n_seg = N // s
            if n_seg < 1:
                continue
            rms_list = []
            for i in range(n_seg):
                seg = y[i*s:(i+1)*s]
                x = np.arange(s)
                coeffs = np.polyfit(x, seg, 1)
                trend = np.polyval(coeffs, x)
                rms_list.append(np.sqrt(np.mean((seg - trend)**2)))
            flucts.append(np.mean(rms_list))
        if len(flucts) < 2:
            return 0.5
        log_s = np.log(scales[:len(flucts)])
        log_f = np.log(flucts)
        alpha = np.polyfit(log_s, log_f, 1)[0]
        return alpha


# ============================================================
# 交叉验证：比较R/S和DFA结果
# ============================================================
def cross_validate_hurst(series: np.ndarray) -> Dict[str, float]:
    """
    使用R/S和DFA两种方法计算Hurst指数并交叉验证

    Returns
    -------
    dict
        包含两种方法的Hurst值及其差异
    """
    h_rs, _, _ = rs_hurst(series)
    h_dfa = dfa_hurst(series)

    result = {
        'R/S Hurst': h_rs,
        'DFA Hurst': h_dfa,
        '两种方法差异': abs(h_rs - h_dfa),
        '平均值': (h_rs + h_dfa) / 2,
    }
    return result


# ============================================================
# 滚动窗口Hurst指数
# ============================================================
def rolling_hurst(series: np.ndarray, dates: pd.DatetimeIndex,
                  window: int = 500, step: int = 30,
                  method: str = 'rs') -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """
    滚动窗口计算Hurst指数，追踪市场状态随时间的演变

    Parameters
    ----------
    series : np.ndarray
        时间序列（对数收益率）
    dates : pd.DatetimeIndex
        对应的日期索引
    window : int
        滚动窗口大小（默认500天）
    step : int
        滚动步长（默认30天）
    method : str
        'rs' 使用R/S分析，'dfa' 使用DFA分析

    Returns
    -------
    roll_dates : pd.DatetimeIndex
        每个窗口对应的日期（窗口末尾日期）
    roll_hurst : np.ndarray
        对应的Hurst指数值
    """
    n = len(series)
    roll_dates = []
    roll_hurst = []

    for start_idx in range(0, n - window + 1, step):
        end_idx = start_idx + window
        segment = series[start_idx:end_idx]

        if method == 'rs':
            h, _, _ = rs_hurst(segment)
        elif method == 'dfa':
            h = dfa_hurst(segment)
        else:
            raise ValueError(f"未知方法: {method}")

        roll_dates.append(dates[end_idx - 1])
        roll_hurst.append(h)

    return pd.DatetimeIndex(roll_dates), np.array(roll_hurst)


# ============================================================
# 多时间框架Hurst分析
# ============================================================
def multi_timeframe_hurst(intervals: List[str] = None) -> Dict[str, Dict[str, float]]:
    """
    在多个时间框架下计算Hurst指数

    Parameters
    ----------
    intervals : list of str
        时间框架列表，默认 ['1h', '4h', '1d', '1w']

    Returns
    -------
    dict
        每个时间框架的Hurst分析结果
    """
    if intervals is None:
        intervals = ['1h', '4h', '1d', '1w']

    results = {}
    for interval in intervals:
        try:
            print(f"\n正在加载 {interval} 数据...")
            df = load_klines(interval)
            prices = df['close'].dropna()

            if len(prices) < 100:
                print(f"  {interval} 数据量不足（{len(prices)}条），跳过")
                continue

            returns = log_returns(prices).values

            # R/S分析
            h_rs, _, _ = rs_hurst(returns)
            # DFA分析
            h_dfa = dfa_hurst(returns)

            results[interval] = {
                'R/S Hurst': h_rs,
                'DFA Hurst': h_dfa,
                '平均Hurst': (h_rs + h_dfa) / 2,
                '数据量': len(returns),
                '解读': interpret_hurst((h_rs + h_dfa) / 2),
            }

            print(f"  {interval}: R/S={h_rs:.4f}, DFA={h_dfa:.4f}, "
                  f"平均={results[interval]['平均Hurst']:.4f}")

        except FileNotFoundError:
            print(f"  {interval} 数据文件不存在，跳过")
        except Exception as e:
            print(f"  {interval} 分析失败: {e}")

    return results


# ============================================================
# 可视化函数
# ============================================================
def plot_rs_loglog(log_ns: np.ndarray, log_rs: np.ndarray, H: float,
                   output_dir: Path, filename: str = "hurst_rs_loglog.png"):
    """绘制R/S分析的log-log图"""
    fig, ax = plt.subplots(figsize=(10, 7))

    # 散点
    ax.scatter(log_ns, log_rs, color='steelblue', s=40, zorder=3, label='R/S 数据点')

    # 拟合线
    coeffs = np.polyfit(log_ns, log_rs, 1)
    fit_line = np.polyval(coeffs, log_ns)
    ax.plot(log_ns, fit_line, 'r-', linewidth=2, label=f'拟合线 (H = {H:.4f})')

    # 参考线：H=0.5（随机游走）
    ref_line = 0.5 * log_ns + (log_rs[0] - 0.5 * log_ns[0])
    ax.plot(log_ns, ref_line, 'k--', alpha=0.5, linewidth=1, label='H=0.5 (随机游走)')

    ax.set_xlabel('log(n) - 窗口大小的对数', fontsize=12)
    ax.set_ylabel('log(R/S) - 重标极差的对数', fontsize=12)
    ax.set_title(f'BTC R/S 分析 (Hurst指数 = {H:.4f})\n{interpret_hurst(H)}', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {filepath}")


def plot_rolling_hurst(roll_dates: pd.DatetimeIndex, roll_hurst: np.ndarray,
                       output_dir: Path, filename: str = "hurst_rolling.png"):
    """绘制滚动Hurst指数时间序列，带有市场状态色带"""
    fig, ax = plt.subplots(figsize=(14, 7))

    # 绘制Hurst指数曲线
    ax.plot(roll_dates, roll_hurst, color='steelblue', linewidth=1.5, label='滚动Hurst指数')

    # 状态色带
    ax.axhspan(TREND_THRESHOLD, max(roll_hurst.max() + 0.05, 0.8),
               alpha=0.1, color='green', label=f'趋势区 (H>{TREND_THRESHOLD})')
    ax.axhspan(MEAN_REV_THRESHOLD, TREND_THRESHOLD,
               alpha=0.1, color='yellow', label=f'随机游走区 ({MEAN_REV_THRESHOLD}<H<{TREND_THRESHOLD})')
    ax.axhspan(min(roll_hurst.min() - 0.05, 0.2), MEAN_REV_THRESHOLD,
               alpha=0.1, color='red', label=f'均值回归区 (H<{MEAN_REV_THRESHOLD})')

    # 参考线
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=TREND_THRESHOLD, color='green', linestyle=':', alpha=0.5)
    ax.axhline(y=MEAN_REV_THRESHOLD, color='red', linestyle=':', alpha=0.5)

    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('Hurst指数', fontsize=12)
    ax.set_title('BTC 滚动Hurst指数 (窗口=500天, 步长=30天)\n市场状态随时间演变', fontsize=13)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # 格式化日期轴
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate()

    fig.tight_layout()
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {filepath}")


def plot_multi_timeframe(results: Dict[str, Dict[str, float]],
                         output_dir: Path, filename: str = "hurst_multi_timeframe.png"):
    """绘制多时间框架Hurst指数对比图"""
    if not results:
        print("  没有可绘制的多时间框架结果")
        return

    intervals = list(results.keys())
    h_rs = [results[k]['R/S Hurst'] for k in intervals]
    h_dfa = [results[k]['DFA Hurst'] for k in intervals]
    h_avg = [results[k]['平均Hurst'] for k in intervals]

    x = np.arange(len(intervals))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))

    bars1 = ax.bar(x - width, h_rs, width, label='R/S Hurst', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x, h_dfa, width, label='DFA Hurst', color='coral', alpha=0.8)
    bars3 = ax.bar(x + width, h_avg, width, label='平均', color='seagreen', alpha=0.8)

    # 参考线
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1, label='H=0.5')
    ax.axhline(y=TREND_THRESHOLD, color='green', linestyle=':', alpha=0.4)
    ax.axhline(y=MEAN_REV_THRESHOLD, color='red', linestyle=':', alpha=0.4)

    # 在柱状图上标注数值
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('时间框架', fontsize=12)
    ax.set_ylabel('Hurst指数', fontsize=12)
    ax.set_title('BTC 多时间框架 Hurst指数对比', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(intervals)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {filepath}")


# ============================================================
# 主入口函数
# ============================================================
def run_hurst_analysis(df: pd.DataFrame, output_dir: str = "output/hurst") -> Dict:
    """
    Hurst指数综合分析主入口

    Parameters
    ----------
    df : pd.DataFrame
        K线数据（需包含 'close' 列和DatetimeIndex索引）
    output_dir : str
        图表输出目录

    Returns
    -------
    dict
        包含所有分析结果的字典
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    print("=" * 70)
    print("Hurst指数综合分析")
    print("=" * 70)

    # ----------------------------------------------------------
    # 1. 准备数据
    # ----------------------------------------------------------
    prices = df['close'].dropna()
    returns = log_returns(prices)
    returns_arr = returns.values

    print(f"\n数据概况:")
    print(f"  时间范围: {df.index.min()} ~ {df.index.max()}")
    print(f"  收益率序列长度: {len(returns_arr)}")

    # ----------------------------------------------------------
    # 2. R/S分析
    # ----------------------------------------------------------
    print("\n" + "-" * 50)
    print("【1】R/S (Rescaled Range) 分析")
    print("-" * 50)

    h_rs, log_ns, log_rs = rs_hurst(returns_arr)
    results['R/S Hurst'] = h_rs

    print(f"  R/S Hurst指数: {h_rs:.4f}")
    print(f"  解读: {interpret_hurst(h_rs)}")

    # 绘制R/S log-log图
    plot_rs_loglog(log_ns, log_rs, h_rs, output_dir)

    # ----------------------------------------------------------
    # 3. DFA分析（使用nolds库）
    # ----------------------------------------------------------
    print("\n" + "-" * 50)
    print("【2】DFA (Detrended Fluctuation Analysis) 分析")
    print("-" * 50)

    h_dfa = dfa_hurst(returns_arr)
    results['DFA Hurst'] = h_dfa

    print(f"  DFA Hurst指数: {h_dfa:.4f}")
    print(f"  解读: {interpret_hurst(h_dfa)}")

    # ----------------------------------------------------------
    # 4. 交叉验证
    # ----------------------------------------------------------
    print("\n" + "-" * 50)
    print("【3】交叉验证：R/S vs DFA")
    print("-" * 50)

    cv_results = cross_validate_hurst(returns_arr)
    results['交叉验证'] = cv_results

    print(f"  R/S Hurst:  {cv_results['R/S Hurst']:.4f}")
    print(f"  DFA Hurst:  {cv_results['DFA Hurst']:.4f}")
    print(f"  两种方法差异: {cv_results['两种方法差异']:.4f}")
    print(f"  平均值:     {cv_results['平均值']:.4f}")

    avg_h = cv_results['平均值']
    if cv_results['两种方法差异'] < 0.05:
        print("  ✓ 两种方法结果一致性较好（差异<0.05）")
    else:
        print("  ⚠ 两种方法结果存在一定差异（差异≥0.05），建议结合其他方法验证")

    print(f"\n  综合解读: {interpret_hurst(avg_h)}")
    results['综合Hurst'] = avg_h
    results['综合解读'] = interpret_hurst(avg_h)

    # ----------------------------------------------------------
    # 5. 滚动窗口Hurst（窗口500天，步长30天）
    # ----------------------------------------------------------
    print("\n" + "-" * 50)
    print("【4】滚动窗口Hurst指数 (窗口=500天, 步长=30天)")
    print("-" * 50)

    if len(returns_arr) >= 500:
        roll_dates, roll_h = rolling_hurst(
            returns_arr, returns.index, window=500, step=30, method='rs'
        )

        # 统计各状态占比
        n_trend = np.sum(roll_h > TREND_THRESHOLD)
        n_mean_rev = np.sum(roll_h < MEAN_REV_THRESHOLD)
        n_random = np.sum((roll_h >= MEAN_REV_THRESHOLD) & (roll_h <= TREND_THRESHOLD))
        total = len(roll_h)

        print(f"  滚动窗口数: {total}")
        print(f"  趋势状态占比:   {n_trend / total * 100:.1f}% ({n_trend}/{total})")
        print(f"  随机游走占比:   {n_random / total * 100:.1f}% ({n_random}/{total})")
        print(f"  均值回归占比:   {n_mean_rev / total * 100:.1f}% ({n_mean_rev}/{total})")
        print(f"  Hurst范围: [{roll_h.min():.4f}, {roll_h.max():.4f}]")
        print(f"  Hurst均值: {roll_h.mean():.4f}")

        results['滚动Hurst'] = {
            '窗口数': total,
            '趋势占比': n_trend / total,
            '随机游走占比': n_random / total,
            '均值回归占比': n_mean_rev / total,
            'Hurst范围': (roll_h.min(), roll_h.max()),
            'Hurst均值': roll_h.mean(),
        }

        # 绘制滚动Hurst图
        plot_rolling_hurst(roll_dates, roll_h, output_dir)
    else:
        print(f"  数据量不足（{len(returns_arr)}<500），跳过滚动窗口分析")

    # ----------------------------------------------------------
    # 6. 多时间框架Hurst分析
    # ----------------------------------------------------------
    print("\n" + "-" * 50)
    print("【5】多时间框架Hurst指数")
    print("-" * 50)

    mt_results = multi_timeframe_hurst(['1h', '4h', '1d', '1w'])
    results['多时间框架'] = mt_results

    # 绘制多时间框架对比图
    plot_multi_timeframe(mt_results, output_dir)

    # ----------------------------------------------------------
    # 7. 总结
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("分析总结")
    print("=" * 70)
    print(f"  日线综合Hurst指数: {avg_h:.4f}")
    print(f"  市场状态判断: {interpret_hurst(avg_h)}")

    if mt_results:
        print("\n  各时间框架Hurst指数:")
        for interval, data in mt_results.items():
            print(f"    {interval}: 平均H={data['平均Hurst']:.4f} - {data['解读']}")

    print(f"\n  判定标准:")
    print(f"    H > {TREND_THRESHOLD}: 趋势性（持续性，适合趋势跟随策略）")
    print(f"    H < {MEAN_REV_THRESHOLD}: 均值回归（反持续性，适合均值回归策略）")
    print(f"    {MEAN_REV_THRESHOLD} ≤ H ≤ {TREND_THRESHOLD}: 随机游走（无显著可预测性）")

    print(f"\n  图表已保存至: {output_dir.resolve()}")
    print("=" * 70)

    return results


# ============================================================
# 独立运行入口
# ============================================================
if __name__ == "__main__":
    from data_loader import load_daily

    print("加载BTC日线数据...")
    df = load_daily()
    print(f"数据加载完成: {len(df)} 条记录")

    results = run_hurst_analysis(df, output_dir="output/hurst")
