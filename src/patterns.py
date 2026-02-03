"""
K线形态识别与统计验证模块

手动实现常见蜡烛图形态（Doji、Hammer、Engulfing、Morning/Evening Star 等），
使用前向收益分析 + Wilson 置信区间 + FDR 校正进行统计验证。
"""

import matplotlib
matplotlib.use('Agg')

from src.font_config import configure_chinese_font
configure_chinese_font()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from src.data_loader import split_data, load_klines


# ============================================================
# 1. 辅助函数
# ============================================================

def _body(df: pd.DataFrame) -> pd.Series:
    """实体大小（绝对值）"""
    return (df['close'] - df['open']).abs()


def _body_signed(df: pd.DataFrame) -> pd.Series:
    """带符号的实体（正=阳线，负=阴线）"""
    return df['close'] - df['open']


def _upper_shadow(df: pd.DataFrame) -> pd.Series:
    """上影线长度"""
    return df['high'] - df[['open', 'close']].max(axis=1)


def _lower_shadow(df: pd.DataFrame) -> pd.Series:
    """下影线长度"""
    return df[['open', 'close']].min(axis=1) - df['low']


def _total_range(df: pd.DataFrame) -> pd.Series:
    """总振幅（high - low），避免零值"""
    return (df['high'] - df['low']).replace(0, np.nan)


def _is_bullish(df: pd.DataFrame) -> pd.Series:
    """是否阳线"""
    return df['close'] > df['open']


def _is_bearish(df: pd.DataFrame) -> pd.Series:
    """是否阴线"""
    return df['close'] < df['open']


# ============================================================
# 2. 形态识别函数（手动实现）
# ============================================================

def detect_doji(df: pd.DataFrame) -> pd.Series:
    """
    十字星 (Doji)
    条件: 实体 < 总振幅的 10%
    方向: 中性 (0)
    """
    body = _body(df)
    total = _total_range(df)
    return (body / total < 0.10).astype(int)


def detect_hammer(df: pd.DataFrame) -> pd.Series:
    """
    锤子线 (Hammer) — 底部反转看涨信号
    条件:
      - 下影线 > 实体的 2 倍
      - 上影线 < 实体的 0.5 倍（或 < 总振幅的 15%）
      - 实体在上半部分
    """
    body = _body(df)
    lower = _lower_shadow(df)
    upper = _upper_shadow(df)
    total = _total_range(df)

    cond = (
        (lower > 2 * body) &
        (upper < 0.5 * body + 1e-10) &  # 加小值避免零实体问题
        (body > 0)  # 排除doji
    )
    return cond.astype(int)


def detect_inverted_hammer(df: pd.DataFrame) -> pd.Series:
    """
    倒锤子线 (Inverted Hammer) — 底部反转看涨信号
    条件:
      - 上影线 > 实体的 2 倍
      - 下影线 < 实体的 0.5 倍
    """
    body = _body(df)
    lower = _lower_shadow(df)
    upper = _upper_shadow(df)

    cond = (
        (upper > 2 * body) &
        (lower < 0.5 * body + 1e-10) &
        (body > 0)
    )
    return cond.astype(int)


def detect_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """
    看涨吞没 (Bullish Engulfing)
    条件:
      - 前一根阴线，当前阳线
      - 当前实体完全包裹前一根实体
    """
    prev_bearish = _is_bearish(df).shift(1)
    curr_bullish = _is_bullish(df)

    # 当前开盘 < 前一根收盘 (前一根阴线收盘较低)
    # 当前收盘 > 前一根开盘
    cond = (
        prev_bearish &
        curr_bullish &
        (df['open'] <= df['close'].shift(1)) &
        (df['close'] >= df['open'].shift(1))
    )
    return cond.fillna(False).astype(int)


def detect_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """
    看跌吞没 (Bearish Engulfing)
    条件:
      - 前一根阳线，当前阴线
      - 当前实体完全包裹前一根实体
    """
    prev_bullish = _is_bullish(df).shift(1)
    curr_bearish = _is_bearish(df)

    cond = (
        prev_bullish &
        curr_bearish &
        (df['open'] >= df['close'].shift(1)) &
        (df['close'] <= df['open'].shift(1))
    )
    return cond.fillna(False).astype(int)


def detect_morning_star(df: pd.DataFrame) -> pd.Series:
    """
    晨星 (Morning Star) — 3根K线底部反转
    条件:
      - 第1根: 大阴线（实体 > 中位数实体）
      - 第2根: 小实体（实体 < 中位数实体 * 0.5），跳空低开或接近
      - 第3根: 大阳线，收盘超过第1根实体中点
    """
    body = _body(df)
    body_signed = _body_signed(df)
    median_body = body.rolling(window=20, min_periods=10).median()

    # 第1根大阴线
    bar1_big_bear = (body_signed.shift(2) < 0) & (body.shift(2) > median_body.shift(2))
    # 第2根小实体
    bar2_small = body.shift(1) < median_body.shift(1) * 0.5
    # 第3根大阳线，收盘超过第1根实体中点
    bar1_mid = (df['open'].shift(2) + df['close'].shift(2)) / 2
    bar3_big_bull = (body_signed > 0) & (body > median_body) & (df['close'] > bar1_mid)

    cond = bar1_big_bear & bar2_small & bar3_big_bull
    return cond.fillna(False).astype(int)


def detect_evening_star(df: pd.DataFrame) -> pd.Series:
    """
    暮星 (Evening Star) — 3根K线顶部反转
    条件:
      - 第1根: 大阳线
      - 第2根: 小实体
      - 第3根: 大阴线，收盘低于第1根实体中点
    """
    body = _body(df)
    body_signed = _body_signed(df)
    median_body = body.rolling(window=20, min_periods=10).median()

    bar1_big_bull = (body_signed.shift(2) > 0) & (body.shift(2) > median_body.shift(2))
    bar2_small = body.shift(1) < median_body.shift(1) * 0.5
    bar1_mid = (df['open'].shift(2) + df['close'].shift(2)) / 2
    bar3_big_bear = (body_signed < 0) & (body > median_body) & (df['close'] < bar1_mid)

    cond = bar1_big_bull & bar2_small & bar3_big_bear
    return cond.fillna(False).astype(int)


def detect_three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    """
    三阳开泰 (Three White Soldiers)
    条件:
      - 连续3根阳线
      - 每根开盘在前一根实体范围内
      - 每根收盘创新高
      - 上影线较小
    """
    bullish = _is_bullish(df)
    body = _body(df)
    upper = _upper_shadow(df)

    cond = (
        bullish & bullish.shift(1) & bullish.shift(2) &
        # 每根收盘逐步升高
        (df['close'] > df['close'].shift(1)) &
        (df['close'].shift(1) > df['close'].shift(2)) &
        # 每根开盘在前一根实体内
        (df['open'] >= df['open'].shift(1)) &
        (df['open'] <= df['close'].shift(1)) &
        (df['open'].shift(1) >= df['open'].shift(2)) &
        (df['open'].shift(1) <= df['close'].shift(2)) &
        # 上影线不超过实体的30%
        (upper < body * 0.3 + 1e-10) &
        (upper.shift(1) < body.shift(1) * 0.3 + 1e-10)
    )
    return cond.fillna(False).astype(int)


def detect_three_black_crows(df: pd.DataFrame) -> pd.Series:
    """
    三阴断头 (Three Black Crows)
    条件:
      - 连续3根阴线
      - 每根开盘在前一根实体范围内
      - 每根收盘创新低
      - 下影线较小
    """
    bearish = _is_bearish(df)
    body = _body(df)
    lower = _lower_shadow(df)

    cond = (
        bearish & bearish.shift(1) & bearish.shift(2) &
        # 每根收盘逐步降低
        (df['close'] < df['close'].shift(1)) &
        (df['close'].shift(1) < df['close'].shift(2)) &
        # 每根开盘在前一根实体内
        (df['open'] <= df['open'].shift(1)) &
        (df['open'] >= df['close'].shift(1)) &
        (df['open'].shift(1) <= df['open'].shift(2)) &
        (df['open'].shift(1) >= df['close'].shift(2)) &
        # 下影线不超过实体的30%
        (lower < body * 0.3 + 1e-10) &
        (lower.shift(1) < body.shift(1) * 0.3 + 1e-10)
    )
    return cond.fillna(False).astype(int)


def detect_pin_bar(df: pd.DataFrame) -> pd.Series:
    """
    Pin Bar (影线 > 总振幅的 2/3)
    分为上Pin Bar（看跌）和下Pin Bar（看涨），此处合并检测
    返回:
      +1 = 下Pin Bar (长下影，看涨)
      -1 = 上Pin Bar (长上影，看跌)
       0 = 无信号
    """
    total = _total_range(df)
    upper = _upper_shadow(df)
    lower = _lower_shadow(df)
    threshold = 2.0 / 3.0

    long_lower = (lower / total > threshold)  # 长下影 -> 看涨
    long_upper = (upper / total > threshold)  # 长上影 -> 看跌

    signal = pd.Series(0, index=df.index)
    signal[long_lower] = 1   # 看涨Pin Bar
    signal[long_upper] = -1  # 看跌Pin Bar
    # 如果同时满足（极端情况），取消信号
    signal[long_lower & long_upper] = 0
    return signal


def detect_shooting_star(df: pd.DataFrame) -> pd.Series:
    """
    流星线 (Shooting Star) — 顶部反转看跌信号
    条件:
      - 上影线 > 实体的 2 倍
      - 下影线 < 实体的 0.5 倍
      - 在上涨趋势末端（前2根收盘低于当前收盘）
    """
    body = _body(df)
    upper = _upper_shadow(df)
    lower = _lower_shadow(df)

    cond = (
        (upper > 2 * body) &
        (lower < 0.5 * body + 1e-10) &
        (body > 0) &
        (df['close'].shift(1) < df['high']) &
        (df['close'].shift(2) < df['close'].shift(1))
    )
    return cond.fillna(False).astype(int)


def detect_all_patterns(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    检测所有K线形态
    返回字典: {形态名称: 信号序列}

    对于方向性形态：
      - 看涨形态的值 > 0 表示检测到
      - 看跌形态的值 > 0 表示检测到
      - Pin Bar 特殊: +1=看涨, -1=看跌
    """
    patterns = {}

    # --- 单根K线形态 ---
    patterns['Doji'] = detect_doji(df)
    patterns['Hammer'] = detect_hammer(df)
    patterns['Inverted_Hammer'] = detect_inverted_hammer(df)
    patterns['Shooting_Star'] = detect_shooting_star(df)
    patterns['Pin_Bar_Bull'] = (detect_pin_bar(df) == 1).astype(int)
    patterns['Pin_Bar_Bear'] = (detect_pin_bar(df) == -1).astype(int)

    # --- 两根K线形态 ---
    patterns['Bullish_Engulfing'] = detect_bullish_engulfing(df)
    patterns['Bearish_Engulfing'] = detect_bearish_engulfing(df)

    # --- 三根K线形态 ---
    patterns['Morning_Star'] = detect_morning_star(df)
    patterns['Evening_Star'] = detect_evening_star(df)
    patterns['Three_White_Soldiers'] = detect_three_white_soldiers(df)
    patterns['Three_Black_Crows'] = detect_three_black_crows(df)

    return patterns


# 形态的预期方向映射（+1=看涨, -1=看跌, 0=中性）
PATTERN_EXPECTED_DIRECTION = {
    'Doji': 0,
    'Hammer': 1,
    'Inverted_Hammer': 1,
    'Shooting_Star': -1,
    'Pin_Bar_Bull': 1,
    'Pin_Bar_Bear': -1,
    'Bullish_Engulfing': 1,
    'Bearish_Engulfing': -1,
    'Morning_Star': 1,
    'Evening_Star': -1,
    'Three_White_Soldiers': 1,
    'Three_Black_Crows': -1,
}


# ============================================================
# 3. 前向收益分析
# ============================================================

def calc_forward_returns_multi(close: pd.Series, horizons: List[int] = None) -> pd.DataFrame:
    """计算多个前向周期的对数收益率"""
    if horizons is None:
        horizons = [1, 3, 5, 10, 20]
    fwd = pd.DataFrame(index=close.index)
    for h in horizons:
        fwd[f'fwd_{h}d'] = np.log(close.shift(-h) / close)
    return fwd


def analyze_pattern_returns(pattern_signal: pd.Series, fwd_returns: pd.DataFrame,
                            expected_dir: int = 0) -> Dict:
    """
    对单个形态进行前向收益分析

    参数:
        pattern_signal: 形态检测信号 (1=出现, 0=未出现)
        fwd_returns: 前向收益 DataFrame
        expected_dir: 预期方向 (+1=看涨, -1=看跌, 0=中性)

    返回:
        统计结果字典
    """
    mask = pattern_signal > 0  # Pin_Bar_Bear 已经处理为单独信号
    n_occurrences = mask.sum()

    result = {'n_occurrences': int(n_occurrences), 'expected_direction': expected_dir}

    if n_occurrences < 3:
        # 样本太少，跳过
        for col in fwd_returns.columns:
            result[f'{col}_mean'] = np.nan
            result[f'{col}_median'] = np.nan
            result[f'{col}_pct_positive'] = np.nan
            result[f'{col}_ttest_pval'] = np.nan
        result['hit_rate'] = np.nan
        result['wilson_ci_lower'] = np.nan
        result['wilson_ci_upper'] = np.nan
        return result

    for col in fwd_returns.columns:
        returns = fwd_returns.loc[mask, col].dropna()
        if len(returns) == 0:
            result[f'{col}_mean'] = np.nan
            result[f'{col}_median'] = np.nan
            result[f'{col}_pct_positive'] = np.nan
            result[f'{col}_ttest_pval'] = np.nan
            continue

        result[f'{col}_mean'] = returns.mean()
        result[f'{col}_median'] = returns.median()
        result[f'{col}_pct_positive'] = (returns > 0).mean()

        # 单样本 t-test: 均值是否显著不等于 0
        if len(returns) >= 5:
            t_stat, t_pval = stats.ttest_1samp(returns, 0)
            result[f'{col}_ttest_pval'] = t_pval
        else:
            result[f'{col}_ttest_pval'] = np.nan

    # --- 命中率 (hit rate) ---
    # 使用 fwd_1d 作为判断依据
    if 'fwd_1d' in fwd_returns.columns:
        ret_1d = fwd_returns.loc[mask, 'fwd_1d'].dropna()
        if len(ret_1d) > 0:
            if expected_dir == 1:
                # 看涨：收益>0 为命中
                hits = (ret_1d > 0).sum()
            elif expected_dir == -1:
                # 看跌：收益<0 为命中
                hits = (ret_1d < 0).sum()
            else:
                # 中性形态不做方向性预测，报告平均绝对收益幅度
                hit_rate = np.nan  # 不适用方向性命中率
                result['hit_rate'] = hit_rate
                result['hit_count'] = 0
                result['hit_n'] = int(len(ret_1d))
                result['avg_abs_return'] = ret_1d.abs().mean()
                result['wilson_ci_lower'] = np.nan
                result['wilson_ci_upper'] = np.nan
                result['binom_pval'] = np.nan
                return result

            n = len(ret_1d)
            hit_rate = hits / n
            result['hit_rate'] = hit_rate
            result['hit_count'] = int(hits)
            result['hit_n'] = int(n)

            # Wilson 置信区间
            ci_lower, ci_upper = wilson_confidence_interval(hits, n, alpha=0.05)
            result['wilson_ci_lower'] = ci_lower
            result['wilson_ci_upper'] = ci_upper

            # 二项检验: 命中率是否显著高于 50%
            binom_pval = stats.binomtest(hits, n, 0.5, alternative='greater').pvalue
            result['binom_pval'] = binom_pval
        else:
            result['hit_rate'] = np.nan
            result['wilson_ci_lower'] = np.nan
            result['wilson_ci_upper'] = np.nan
            result['binom_pval'] = np.nan
    else:
        result['hit_rate'] = np.nan
        result['wilson_ci_lower'] = np.nan
        result['wilson_ci_upper'] = np.nan

    return result


# ============================================================
# 4. Wilson 置信区间 + FDR 校正
# ============================================================

def wilson_confidence_interval(successes: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Wilson 置信区间计算

    比 Wald 区间更适合小样本和极端比例的情况

    参数:
        successes: 成功次数
        n: 总次数
        alpha: 显著性水平

    返回:
        (lower, upper) 置信区间
    """
    if n == 0:
        return (0.0, 1.0)

    p_hat = successes / n
    z = stats.norm.ppf(1 - alpha / 2)

    denominator = 1 + z ** 2 / n
    center = (p_hat + z ** 2 / (2 * n)) / denominator
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n)) / n) / denominator

    lower = max(0, center - margin)
    upper = min(1, center + margin)
    return (lower, upper)


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg FDR 校正

    参数:
        p_values: 原始 p 值数组
        alpha: 显著性水平

    返回:
        (rejected, adjusted_p): 是否拒绝原假设, 校正后p值
    """
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=bool), np.array([])

    valid_mask = ~np.isnan(p_values)
    adjusted = np.full(n, np.nan)
    rejected = np.full(n, False)

    valid_pvals = p_values[valid_mask]
    n_valid = len(valid_pvals)
    if n_valid == 0:
        return rejected, adjusted

    sorted_idx = np.argsort(valid_pvals)
    sorted_pvals = valid_pvals[sorted_idx]

    rank = np.arange(1, n_valid + 1)
    adjusted_sorted = sorted_pvals * n_valid / rank
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0, 1)

    valid_indices = np.where(valid_mask)[0]
    for i, idx in enumerate(sorted_idx):
        adjusted[valid_indices[idx]] = adjusted_sorted[i]
        rejected[valid_indices[idx]] = adjusted_sorted[i] <= alpha

    return rejected, adjusted


# ============================================================
# 5. 可视化
# ============================================================

def plot_pattern_counts(pattern_counts: Dict[str, int], output_dir: Path, prefix: str = "train"):
    """绘制形态出现次数的柱状图"""
    fig, ax = plt.subplots(figsize=(12, 6))

    names = list(pattern_counts.keys())
    counts = list(pattern_counts.values())
    colors = ['#2ecc71' if PATTERN_EXPECTED_DIRECTION.get(n, 0) >= 0 else '#e74c3c' for n in names]

    bars = ax.barh(range(len(names)), counts, color=colors, edgecolor='gray', linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Occurrence Count')
    ax.set_title(f'Pattern Occurrence Counts - {prefix.upper()} Set')

    # 在柱形上标注数值
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                str(count), va='center', fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / f"pattern_counts_{prefix}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [saved] pattern_counts_{prefix}.png")


def plot_forward_return_boxplots(patterns: Dict[str, pd.Series], fwd_returns: pd.DataFrame,
                                  output_dir: Path, prefix: str = "train"):
    """绘制各形态前向收益的箱线图"""
    horizons = [c for c in fwd_returns.columns if c.startswith('fwd_')]
    n_horizons = len(horizons)
    if n_horizons == 0:
        return

    # 筛选有足够样本的形态
    valid_patterns = {name: sig for name, sig in patterns.items() if sig.sum() >= 3}
    if not valid_patterns:
        return

    n_patterns = len(valid_patterns)
    fig, axes = plt.subplots(1, n_horizons, figsize=(4 * n_horizons, max(6, n_patterns * 0.4)))
    if n_horizons == 1:
        axes = [axes]

    for ax_idx, horizon in enumerate(horizons):
        data_list = []
        labels = []
        for name, sig in valid_patterns.items():
            mask = sig > 0
            ret = fwd_returns.loc[mask, horizon].dropna()
            if len(ret) > 0:
                data_list.append(ret.values)
                labels.append(f"{name} (n={len(ret)})")

        if data_list:
            bp = axes[ax_idx].boxplot(data_list, vert=False, patch_artist=True, widths=0.6)
            for patch, name in zip(bp['boxes'], valid_patterns.keys()):
                direction = PATTERN_EXPECTED_DIRECTION.get(name, 0)
                patch.set_facecolor('#a8e6cf' if direction >= 0 else '#ffb3b3')
                patch.set_alpha(0.7)
            axes[ax_idx].set_yticklabels(labels, fontsize=7)
            axes[ax_idx].axvline(x=0, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
            axes[ax_idx].set_xlabel('Log Return')
            horizon_label = horizon.replace('fwd_', '').replace('d', '-day')
            axes[ax_idx].set_title(f'{horizon_label} Forward Return')

    plt.suptitle(f'Pattern Forward Returns - {prefix.upper()} Set', fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / f"pattern_forward_returns_{prefix}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [saved] pattern_forward_returns_{prefix}.png")


def plot_hit_rate_with_ci(results_df: pd.DataFrame, output_dir: Path, prefix: str = "train"):
    """绘制命中率 + Wilson 置信区间"""
    # 筛选有效数据
    valid = results_df.dropna(subset=['hit_rate', 'wilson_ci_lower', 'wilson_ci_upper'])
    if len(valid) == 0:
        return

    fig, ax = plt.subplots(figsize=(12, max(6, len(valid) * 0.5)))

    names = valid.index.tolist()
    hit_rates = valid['hit_rate'].values
    ci_lower = valid['wilson_ci_lower'].values
    ci_upper = valid['wilson_ci_upper'].values

    y_pos = range(len(names))
    # 置信区间误差条
    xerr_lower = hit_rates - ci_lower
    xerr_upper = ci_upper - hit_rates
    xerr = np.array([xerr_lower, xerr_upper])

    colors = ['#2ecc71' if hr > 0.5 else '#e74c3c' for hr in hit_rates]
    ax.barh(y_pos, hit_rates, xerr=xerr, color=colors, edgecolor='gray',
            linewidth=0.5, alpha=0.8, capsize=3, ecolor='black')
    ax.axvline(x=0.5, color='blue', linestyle='--', linewidth=1.0, label='50% baseline')

    # 标注 FDR 校正结果
    if 'binom_adj_pval' in valid.columns:
        for i, name in enumerate(names):
            adj_p = valid.loc[name, 'binom_adj_pval']
            marker = ''
            if not np.isnan(adj_p):
                if adj_p < 0.01:
                    marker = ' ***'
                elif adj_p < 0.05:
                    marker = ' **'
                elif adj_p < 0.10:
                    marker = ' *'
            ax.text(ci_upper[i] + 0.01, i, f"{hit_rates[i]:.1%}{marker}", va='center', fontsize=8)
    else:
        for i in range(len(names)):
            ax.text(ci_upper[i] + 0.01, i, f"{hit_rates[i]:.1%}", va='center', fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Hit Rate')
    ax.set_title(f'Pattern Hit Rate with Wilson CI - {prefix.upper()} Set\n(* p<0.10, ** p<0.05, *** p<0.01 after FDR)')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    fig.savefig(output_dir / f"pattern_hit_rate_{prefix}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [saved] pattern_hit_rate_{prefix}.png")


# ============================================================
# 6. 多时间尺度形态分析
# ============================================================

def multi_timeframe_pattern_analysis(intervals=None) -> Dict:
    """多时间尺度形态识别与对比"""
    if intervals is None:
        intervals = ['1h', '4h', '1d']

    results = {}
    for interval in intervals:
        try:
            print(f"\n  加载 {interval} 数据进行形态识别...")
            df_tf = load_klines(interval)

            if len(df_tf) < 100:
                print(f"    {interval} 数据不足，跳过")
                continue

            # 检测所有形态
            patterns = detect_all_patterns(df_tf)

            # 计算前向收益
            close = df_tf['close']
            fwd_returns = calc_forward_returns_multi(close, horizons=[1, 3, 5])

            # 评估每个形态
            pattern_stats = {}
            for name, signal in patterns.items():
                n_occ = signal.sum() if hasattr(signal, 'sum') else (signal > 0).sum()
                expected_dir = PATTERN_EXPECTED_DIRECTION.get(name, 0)

                if n_occ >= 5:
                    result = analyze_pattern_returns(signal, fwd_returns, expected_dir)
                    pattern_stats[name] = {
                        'n_occurrences': int(n_occ),
                        'hit_rate': result.get('hit_rate', np.nan),
                    }
                else:
                    pattern_stats[name] = {
                        'n_occurrences': int(n_occ),
                        'hit_rate': np.nan,
                    }

            results[interval] = pattern_stats
            print(f"    {interval}: {sum(1 for v in pattern_stats.values() if v['n_occurrences'] > 0)} 种形态检测到")

        except FileNotFoundError:
            print(f"    {interval} 数据文件不存在，跳过")
        except Exception as e:
            print(f"    {interval} 分析失败: {e}")

    return results


def cross_scale_pattern_consistency(intervals=None) -> Dict:
    """
    跨尺度形态一致性分析
    检查同一日期多个尺度是否同时出现相同方向的形态

    返回:
        包含一致性统计的字典
    """
    if intervals is None:
        intervals = ['1h', '4h', '1d']

    # 加载所有时间尺度数据
    dfs = {}
    for interval in intervals:
        try:
            df = load_klines(interval)
            if len(df) >= 100:
                dfs[interval] = df
        except:
            continue

    if len(dfs) < 2:
        print("  跨尺度分析需要至少2个时间尺度的数据")
        return {}

    # 检测每个尺度的形态
    patterns_by_tf = {}
    for interval, df in dfs.items():
        patterns_by_tf[interval] = detect_all_patterns(df)

    # 统计跨尺度一致性
    consistency_stats = {}

    # 对每种形态，检查在同一日期的不同尺度上是否同时出现
    all_pattern_names = set()
    for patterns in patterns_by_tf.values():
        all_pattern_names.update(patterns.keys())

    for pattern_name in all_pattern_names:
        expected_dir = PATTERN_EXPECTED_DIRECTION.get(pattern_name, 0)
        if expected_dir == 0:  # 跳过中性形态
            continue

        # 找出所有尺度上该形态出现的日期
        occurrences_by_tf = {}
        for interval, patterns in patterns_by_tf.items():
            if pattern_name in patterns:
                signal = patterns[pattern_name]
                # 转换为日期（忽略时间）
                dates = signal[signal > 0].index.date if hasattr(signal.index, 'date') else signal[signal > 0].index
                occurrences_by_tf[interval] = set(dates)

        if len(occurrences_by_tf) < 2:
            continue

        # 计算交集（同时出现在多个尺度的日期数）
        all_dates = set()
        for dates in occurrences_by_tf.values():
            all_dates.update(dates)

        # 统计每个日期在多少个尺度上出现
        date_counts = {}
        for date in all_dates:
            count = sum(1 for dates in occurrences_by_tf.values() if date in dates)
            date_counts[date] = count

        # 计算一致性指标
        total_occurrences = sum(len(dates) for dates in occurrences_by_tf.values())
        multi_scale_occurrences = sum(1 for count in date_counts.values() if count >= 2)

        consistency_stats[pattern_name] = {
            'total_occurrences': total_occurrences,
            'multi_scale_occurrences': multi_scale_occurrences,
            'consistency_rate': multi_scale_occurrences / total_occurrences if total_occurrences > 0 else 0,
            'scales_available': len(occurrences_by_tf),
        }

    return consistency_stats


def plot_multi_timeframe_hit_rates(mt_results: Dict, output_dir: Path):
    """多尺度形态命中率对比图"""
    if not mt_results:
        return

    # 收集所有形态名称
    all_patterns = set()
    for tf_stats in mt_results.values():
        all_patterns.update(tf_stats.keys())

    # 筛选至少在一个尺度上有足够样本的形态
    valid_patterns = []
    for pattern in all_patterns:
        has_valid_data = False
        for tf_stats in mt_results.values():
            if pattern in tf_stats and tf_stats[pattern]['n_occurrences'] >= 5:
                if not np.isnan(tf_stats[pattern].get('hit_rate', np.nan)):
                    has_valid_data = True
                    break
        if has_valid_data:
            valid_patterns.append(pattern)

    if not valid_patterns:
        print("  没有足够的数据绘制多尺度命中率对比图")
        return

    # 准备绘图数据
    intervals = sorted(mt_results.keys())
    n_intervals = len(intervals)
    n_patterns = len(valid_patterns)

    fig, ax = plt.subplots(figsize=(max(12, n_patterns * 0.8), 8))

    x = np.arange(n_patterns)
    width = 0.8 / n_intervals

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    for i, interval in enumerate(intervals):
        hit_rates = []
        for pattern in valid_patterns:
            if pattern in mt_results[interval]:
                hr = mt_results[interval][pattern].get('hit_rate', np.nan)
            else:
                hr = np.nan
            hit_rates.append(hr)

        offset = (i - n_intervals / 2 + 0.5) * width
        bars = ax.bar(x + offset, hit_rates, width, label=interval,
                     color=colors[i % len(colors)], alpha=0.8, edgecolor='gray', linewidth=0.5)

        # 标注数值
        for j, (bar, hr) in enumerate(zip(bars, hit_rates)):
            if not np.isnan(hr) and bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                       f'{hr:.1%}', ha='center', va='bottom', fontsize=6, rotation=0)

    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.0, alpha=0.7, label='50% baseline')
    ax.set_xlabel('形态名称', fontsize=11)
    ax.set_ylabel('命中率', fontsize=11)
    ax.set_title('多时间尺度形态命中率对比', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_patterns, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9, loc='best')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    fig.savefig(output_dir / "pattern_multi_timeframe_hitrate.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [saved] pattern_multi_timeframe_hitrate.png")


def plot_cross_scale_consistency(consistency_stats: Dict, output_dir: Path):
    """展示跨尺度形态一致性统计"""
    if not consistency_stats:
        print("  没有跨尺度一致性数据可绘制")
        return

    # 筛选有效数据
    valid_stats = {k: v for k, v in consistency_stats.items() if v['total_occurrences'] >= 10}
    if not valid_stats:
        print("  没有足够的数据绘制跨尺度一致性图")
        return

    # 按一致性率排序
    sorted_patterns = sorted(valid_stats.items(), key=lambda x: x[1]['consistency_rate'], reverse=True)

    names = [name for name, _ in sorted_patterns]
    consistency_rates = [stats['consistency_rate'] for _, stats in sorted_patterns]
    multi_scale_counts = [stats['multi_scale_occurrences'] for _, stats in sorted_patterns]
    total_counts = [stats['total_occurrences'] for _, stats in sorted_patterns]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, len(names) * 0.4)))

    # 左图：一致性率
    y_pos = range(len(names))
    colors = ['#2ecc71' if rate > 0.3 else '#e74c3c' for rate in consistency_rates]
    bars1 = ax1.barh(y_pos, consistency_rates, color=colors, edgecolor='gray', linewidth=0.5, alpha=0.8)

    for i, (bar, rate, multi, total) in enumerate(zip(bars1, consistency_rates, multi_scale_counts, total_counts)):
        ax1.text(bar.get_width() + 0.01, i, f'{rate:.1%}\n({multi}/{total})',
                va='center', fontsize=7)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel('跨尺度一致性率', fontsize=11)
    ax1.set_title('形态跨尺度一致性率\n(同一日期出现在多个时间尺度的比例)', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.axvline(x=0.3, color='blue', linestyle='--', linewidth=0.8, alpha=0.5, label='30% threshold')
    ax1.legend(fontsize=8)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')

    # 右图：出现次数对比
    width = 0.35
    x_pos = np.arange(len(names))

    bars2 = ax2.barh(x_pos, total_counts, width, label='总出现次数', color='#3498db', alpha=0.7)
    bars3 = ax2.barh(x_pos + width, multi_scale_counts, width, label='多尺度出现次数', color='#e67e22', alpha=0.7)

    ax2.set_yticks(x_pos + width / 2)
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel('出现次数', fontsize=11)
    ax2.set_title('形态出现次数统计', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    fig.savefig(output_dir / "pattern_cross_scale_consistency.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [saved] pattern_cross_scale_consistency.png")


# ============================================================
# 7. 主流程
# ============================================================

def evaluate_patterns_on_set(df: pd.DataFrame, patterns: Dict[str, pd.Series],
                              set_name: str) -> pd.DataFrame:
    """
    在给定数据集上评估所有形态

    参数:
        df: 数据集 DataFrame (含 OHLCV)
        patterns: 形态信号字典
        set_name: 数据集名称（用于打印）

    返回:
        包含统计结果的 DataFrame
    """
    close = df['close']
    fwd_returns = calc_forward_returns_multi(close, horizons=[1, 3, 5, 10, 20])

    results = {}
    for name, signal in patterns.items():
        sig = signal.reindex(df.index).fillna(0)
        expected_dir = PATTERN_EXPECTED_DIRECTION.get(name, 0)
        results[name] = analyze_pattern_returns(sig, fwd_returns, expected_dir)

    results_df = pd.DataFrame(results).T
    results_df.index.name = 'pattern'

    print(f"\n{'='*60}")
    print(f"  {set_name} 数据集形态评估结果")
    print(f"{'='*60}")

    # 打印形态出现次数
    print(f"\n  形态出现次数:")
    for name in results_df.index:
        n = int(results_df.loc[name, 'n_occurrences'])
        print(f"    {name}: {n} 次")

    return results_df


def apply_fdr_to_patterns(results_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    对形态检验的多个 p 值进行 FDR 校正

    校正的 p 值列:
      - 各前向周期的 t-test p 值
      - 二项检验 p 值
    """
    # t-test p 值列
    ttest_cols = [c for c in results_df.columns if c.endswith('_ttest_pval')]
    all_pval_cols = ttest_cols.copy()

    if 'binom_pval' in results_df.columns:
        all_pval_cols.append('binom_pval')

    for col in all_pval_cols:
        pvals = results_df[col].values.astype(float)
        rejected, adjusted = benjamini_hochberg(pvals, alpha)
        adj_col = col.replace('_pval', '_adj_pval')
        rej_col = col.replace('_pval', '_rejected')
        results_df[adj_col] = adjusted
        results_df[rej_col] = rejected

    return results_df


def run_patterns_analysis(df: pd.DataFrame, output_dir: str) -> Dict:
    """
    K线形态识别与统计验证主入口

    参数:
        df: 完整的日线 DataFrame（含 open/high/low/close/volume 等列，DatetimeIndex）
        output_dir: 图表输出目录

    返回:
        包含训练集和验证集结果的字典
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  K线形态识别与统计验证")
    print("=" * 60)

    # --- 数据切分 ---
    train, val, test = split_data(df)
    print(f"\n训练集: {train.index.min()} ~ {train.index.max()}  ({len(train)} bars)")
    print(f"验证集: {val.index.min()} ~ {val.index.max()}  ({len(val)} bars)")

    # --- 检测所有形态（在全量数据上计算） ---
    all_patterns = detect_all_patterns(df)
    print(f"\n共检测 {len(all_patterns)} 种K线形态")

    # ============ 训练集评估 ============
    train_results = evaluate_patterns_on_set(train, all_patterns, "训练集 (TRAIN)")

    # FDR 校正
    train_results = apply_fdr_to_patterns(train_results, alpha=0.05)

    # 找出显著形态
    reject_cols = [c for c in train_results.columns if c.endswith('_rejected')]
    if reject_cols:
        train_results['any_fdr_pass'] = train_results[reject_cols].any(axis=1)
        fdr_passed_train = train_results[train_results['any_fdr_pass']].index.tolist()
    else:
        fdr_passed_train = []

    print(f"\n--- FDR 校正结果 (训练集) ---")
    if fdr_passed_train:
        print(f"  通过 FDR 校正的形态 ({len(fdr_passed_train)} 个):")
        for name in fdr_passed_train:
            row = train_results.loc[name]
            hr = row.get('hit_rate', np.nan)
            n = int(row.get('n_occurrences', 0))
            hr_str = f", hit_rate={hr:.1%}" if not np.isnan(hr) else ""
            print(f"    - {name}: n={n}{hr_str}")
    else:
        print("  没有形态通过 FDR 校正（alpha=0.05）")

    # --- 训练集可视化 ---
    print("\n--- 训练集可视化 ---")
    train_counts = {name: int(train_results.loc[name, 'n_occurrences']) for name in train_results.index}
    plot_pattern_counts(train_counts, output_dir, prefix="train")

    train_patterns_in_set = {name: sig.reindex(train.index).fillna(0) for name, sig in all_patterns.items()}
    train_fwd = calc_forward_returns_multi(train['close'], horizons=[1, 3, 5, 10, 20])
    plot_forward_return_boxplots(train_patterns_in_set, train_fwd, output_dir, prefix="train")
    plot_hit_rate_with_ci(train_results, output_dir, prefix="train")

    # ============ 验证集评估 ============
    val_results = evaluate_patterns_on_set(val, all_patterns, "验证集 (VAL)")
    val_results = apply_fdr_to_patterns(val_results, alpha=0.05)

    reject_cols_val = [c for c in val_results.columns if c.endswith('_rejected')]
    if reject_cols_val:
        val_results['any_fdr_pass'] = val_results[reject_cols_val].any(axis=1)
        fdr_passed_val = val_results[val_results['any_fdr_pass']].index.tolist()
    else:
        fdr_passed_val = []

    print(f"\n--- FDR 校正结果 (验证集) ---")
    if fdr_passed_val:
        print(f"  通过 FDR 校正的形态 ({len(fdr_passed_val)} 个):")
        for name in fdr_passed_val:
            row = val_results.loc[name]
            hr = row.get('hit_rate', np.nan)
            n = int(row.get('n_occurrences', 0))
            hr_str = f", hit_rate={hr:.1%}" if not np.isnan(hr) else ""
            print(f"    - {name}: n={n}{hr_str}")
    else:
        print("  没有形态通过 FDR 校正（alpha=0.05）")

    # --- 训练集 vs 验证集对比 ---
    if 'hit_rate' in train_results.columns and 'hit_rate' in val_results.columns:
        print(f"\n--- 训练集 vs 验证集命中率对比 ---")
        for name in train_results.index:
            tr_hr = train_results.loc[name, 'hit_rate'] if name in train_results.index else np.nan
            va_hr = val_results.loc[name, 'hit_rate'] if name in val_results.index else np.nan
            if np.isnan(tr_hr) or np.isnan(va_hr):
                continue
            diff = va_hr - tr_hr
            label = "STABLE" if abs(diff) < 0.05 else ("IMPROVE" if diff > 0 else "DECAY")
            print(f"  {name}: train={tr_hr:.1%}, val={va_hr:.1%}, diff={diff:+.1%} [{label}]")

    # --- 验证集可视化 ---
    print("\n--- 验证集可视化 ---")
    val_counts = {name: int(val_results.loc[name, 'n_occurrences']) for name in val_results.index}
    plot_pattern_counts(val_counts, output_dir, prefix="val")

    val_patterns_in_set = {name: sig.reindex(val.index).fillna(0) for name, sig in all_patterns.items()}
    val_fwd = calc_forward_returns_multi(val['close'], horizons=[1, 3, 5, 10, 20])
    plot_forward_return_boxplots(val_patterns_in_set, val_fwd, output_dir, prefix="val")
    plot_hit_rate_with_ci(val_results, output_dir, prefix="val")

    # ============ 多时间尺度形态分析 ============
    print("\n--- 多时间尺度形态分析 ---")
    mt_results = multi_timeframe_pattern_analysis(['1h', '4h', '1d'])
    if mt_results:
        plot_multi_timeframe_hit_rates(mt_results, output_dir)

    # ============ 跨尺度形态一致性分析 ============
    print("\n--- 跨尺度形态一致性分析 ---")
    consistency_stats = cross_scale_pattern_consistency(['1h', '4h', '1d'])
    if consistency_stats:
        plot_cross_scale_consistency(consistency_stats, output_dir)
        print(f"\n  检测到 {len(consistency_stats)} 种形态的跨尺度一致性")
        # 打印前5个一致性最高的形态
        sorted_patterns = sorted(consistency_stats.items(), key=lambda x: x[1]['consistency_rate'], reverse=True)
        print("\n  一致性率最高的形态:")
        for name, stats in sorted_patterns[:5]:
            rate = stats['consistency_rate']
            multi = stats['multi_scale_occurrences']
            total = stats['total_occurrences']
            print(f"    {name}: {rate:.1%} ({multi}/{total})")

    print(f"\n{'='*60}")
    print("  K线形态识别与统计验证完成")
    print(f"{'='*60}")

    return {
        'train_results': train_results,
        'val_results': val_results,
        'fdr_passed_train': fdr_passed_train,
        'fdr_passed_val': fdr_passed_val,
        'all_patterns': all_patterns,
        'mt_results': mt_results,
        'consistency_stats': consistency_stats,
    }
