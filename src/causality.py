"""Granger 因果检验模块

分析内容：
- 双向 Granger 因果检验（5 对变量，各 5 个滞后阶数）
- 跨时间尺度因果检验（小时级聚合特征 → 日级收益率）
- Bonferroni 多重检验校正
- 可视化：p 值热力图、显著因果关系网络图
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from typing import Optional, List, Tuple, Dict

from statsmodels.tsa.stattools import grangercausalitytests, adfuller

from src.data_loader import load_hourly
from src.preprocessing import log_returns, add_derived_features


# ============================================================
# 1. 因果检验对定义
# ============================================================

# 5 对双向因果关系，每对 (cause, effect)
CAUSALITY_PAIRS = [
    ('volume', 'log_return'),
    ('log_return', 'volume'),
    ('abs_return', 'volume'),
    ('volume', 'abs_return'),
    ('taker_buy_ratio', 'log_return'),
    ('log_return', 'taker_buy_ratio'),
    ('squared_return', 'volume'),
    ('volume', 'squared_return'),
    ('range_pct', 'log_return'),
    ('log_return', 'range_pct'),
]

# 测试的滞后阶数
TEST_LAGS = [1, 2, 3, 5, 10]


# ============================================================
# 2. ADF 平稳性检验辅助函数
# ============================================================

def _check_stationarity(series, name, alpha=0.05):
    """ADF 平稳性检验，非平稳则取差分"""
    result = adfuller(series.dropna(), autolag='AIC')
    if result[1] > alpha:
        print(f"  [注意] {name} 非平稳 (ADF p={result[1]:.4f})，使用差分序列")
        return series.diff().dropna(), True
    return series, False


# ============================================================
# 3. 单对 Granger 因果检验
# ============================================================

def granger_test_pair(
    df: pd.DataFrame,
    cause: str,
    effect: str,
    max_lag: int = 10,
    test_lags: Optional[List[int]] = None,
) -> List[Dict]:
    """
    对指定的 (cause → effect) 方向执行 Granger 因果检验

    Parameters
    ----------
    df : pd.DataFrame
        包含 cause 和 effect 列的数据
    cause : str
        原因变量列名
    effect : str
        结果变量列名
    max_lag : int
        最大滞后阶数
    test_lags : list of int, optional
        需要测试的滞后阶数列表

    Returns
    -------
    list of dict
        每个滞后阶数的检验结果
    """
    if test_lags is None:
        test_lags = TEST_LAGS

    # grangercausalitytests 要求: 第一列是 effect，第二列是 cause
    data = df[[effect, cause]].dropna()

    if len(data) < max_lag + 20:
        print(f"  [警告] {cause} → {effect}: 样本量不足 ({len(data)})，跳过")
        return []

    # ADF 平稳性检验，非平稳则取差分
    effect_series, effect_diffed = _check_stationarity(data[effect], effect)
    cause_series, cause_diffed = _check_stationarity(data[cause], cause)
    if effect_diffed or cause_diffed:
        data = pd.concat([effect_series, cause_series], axis=1).dropna()
        if len(data) < max_lag + 20:
            print(f"  [警告] {cause} → {effect}: 差分后样本量不足 ({len(data)})，跳过")
            return []

    results = []
    try:
        # 执行检验，maxlag 取最大值，一次获取所有滞后
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gc_results = grangercausalitytests(data, maxlag=max_lag, verbose=False)

        # 提取指定滞后阶数的结果
        for lag in test_lags:
            if lag > max_lag:
                continue
            test_result = gc_results[lag]
            # 取 ssr_ftest 的 F 统计量和 p 值
            f_stat = test_result[0]['ssr_ftest'][0]
            p_value = test_result[0]['ssr_ftest'][1]

            results.append({
                'cause': cause,
                'effect': effect,
                'lag': lag,
                'f_stat': f_stat,
                'p_value': p_value,
            })
    except Exception as e:
        print(f"  [错误] {cause} → {effect}: {e}")

    return results


# ============================================================
# 3. 批量因果检验
# ============================================================

def run_all_granger_tests(
    df: pd.DataFrame,
    pairs: Optional[List[Tuple[str, str]]] = None,
    test_lags: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    对所有变量对执行双向 Granger 因果检验

    Parameters
    ----------
    df : pd.DataFrame
        包含衍生特征的日线数据
    pairs : list of tuple, optional
        变量对列表 [(cause, effect), ...]
    test_lags : list of tuple, optional
        滞后阶数列表

    Returns
    -------
    pd.DataFrame
        所有检验结果汇总表
    """
    if pairs is None:
        pairs = CAUSALITY_PAIRS
    if test_lags is None:
        test_lags = TEST_LAGS

    max_lag = max(test_lags)
    all_results = []

    for cause, effect in pairs:
        if cause not in df.columns or effect not in df.columns:
            print(f"  [警告] 列 {cause} 或 {effect} 不存在，跳过")
            continue
        pair_results = granger_test_pair(df, cause, effect, max_lag=max_lag, test_lags=test_lags)
        all_results.extend(pair_results)

    results_df = pd.DataFrame(all_results)
    return results_df


# ============================================================
# 4. Bonferroni 校正
# ============================================================

def apply_bonferroni(results_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    对 Granger 检验结果应用 Bonferroni 多重检验校正

    Parameters
    ----------
    results_df : pd.DataFrame
        包含 p_value 列的检验结果
    alpha : float
        原始显著性水平

    Returns
    -------
    pd.DataFrame
        添加了校正后显著性判断的结果
    """
    n_tests = len(results_df)
    if n_tests == 0:
        return results_df

    out = results_df.copy()
    # Bonferroni 校正阈值
    corrected_alpha = alpha / n_tests
    out['bonferroni_alpha'] = corrected_alpha
    out['significant_raw'] = out['p_value'] < alpha
    out['significant_corrected'] = out['p_value'] < corrected_alpha

    return out


# ============================================================
# 5. 跨时间尺度因果检验
# ============================================================

def cross_timeframe_causality(
    daily_df: pd.DataFrame,
    test_lags: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    检验小时级聚合特征是否 Granger 因果于日级收益率

    具体步骤：
    1. 加载小时级数据
    2. 计算小时级波动率和成交量的日内聚合指标
    3. 与日线收益率合并
    4. 执行 Granger 因果检验

    Parameters
    ----------
    daily_df : pd.DataFrame
        日线数据（含 log_return）
    test_lags : list of int, optional
        滞后阶数列表

    Returns
    -------
    pd.DataFrame
        跨时间尺度因果检验结果
    """
    if test_lags is None:
        test_lags = TEST_LAGS

    # 加载小时数据
    try:
        hourly_raw = load_hourly()
    except (FileNotFoundError, Exception) as e:
        print(f"  [警告] 无法加载小时级数据，跳过跨时间尺度因果检验: {e}")
        return pd.DataFrame()

    # 计算小时级衍生特征
    hourly = add_derived_features(hourly_raw)

    # 日内聚合：按日期聚合小时数据
    hourly['date'] = hourly.index.date
    agg_dict = {}

    # 小时级日内波动率（对数收益率标准差）
    if 'log_return' in hourly.columns:
        hourly_vol = hourly.groupby('date')['log_return'].std()
        hourly_vol.name = 'hourly_intraday_vol'
        agg_dict['hourly_intraday_vol'] = hourly_vol

    # 小时级日内成交量总和
    if 'volume' in hourly.columns:
        hourly_volume = hourly.groupby('date')['volume'].sum()
        hourly_volume.name = 'hourly_volume_sum'
        agg_dict['hourly_volume_sum'] = hourly_volume

    # 小时级日内最大绝对收益率
    if 'abs_return' in hourly.columns:
        hourly_max_abs = hourly.groupby('date')['abs_return'].max()
        hourly_max_abs.name = 'hourly_max_abs_return'
        agg_dict['hourly_max_abs_return'] = hourly_max_abs

    if not agg_dict:
        print("  [警告] 小时级聚合特征为空，跳过")
        return pd.DataFrame()

    # 合并聚合结果
    hourly_agg = pd.DataFrame(agg_dict)
    hourly_agg.index = pd.to_datetime(hourly_agg.index)

    # 与日线数据合并
    daily_for_merge = daily_df[['log_return']].copy()
    merged = daily_for_merge.join(hourly_agg, how='inner')

    print(f"  [跨时间尺度] 合并后样本数: {len(merged)}")

    # 对每个小时级聚合特征检验 → 日级收益率
    cross_pairs = []
    for col in agg_dict.keys():
        cross_pairs.append((col, 'log_return'))

    max_lag = max(test_lags)
    all_results = []
    for cause, effect in cross_pairs:
        pair_results = granger_test_pair(merged, cause, effect, max_lag=max_lag, test_lags=test_lags)
        all_results.extend(pair_results)

    results_df = pd.DataFrame(all_results)
    return results_df


# ============================================================
# 6. 可视化：p 值热力图
# ============================================================

def plot_pvalue_heatmap(results_df: pd.DataFrame, output_dir: Path):
    """
    绘制 p 值热力图（变量对 x 滞后阶数）

    Parameters
    ----------
    results_df : pd.DataFrame
        因果检验结果
    output_dir : Path
        输出目录
    """
    if results_df.empty:
        print("  [警告] 无检验结果，跳过热力图绘制")
        return

    # 构建标签
    results_df = results_df.copy()
    results_df['pair'] = results_df['cause'] + ' → ' + results_df['effect']

    # 构建 pivot table: 行=pair, 列=lag
    pivot = results_df.pivot_table(index='pair', columns='lag', values='p_value')

    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.5)))

    # 绘制热力图
    im = ax.imshow(-np.log10(pivot.values + 1e-300), cmap='RdYlGn_r', aspect='auto')

    # 设置坐标轴
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'Lag {c}' for c in pivot.columns], fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # 在每个格子中标注 p 值
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if np.isnan(val):
                text = 'N/A'
            else:
                text = f'{val:.4f}'
            color = 'white' if -np.log10(val + 1e-300) > 2 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=8, color=color)

    # Bonferroni 校正线
    n_tests = len(results_df)
    if n_tests > 0:
        bonf_alpha = 0.05 / n_tests
        ax.set_title(
            f'Granger 因果检验 p 值热力图 (-log10)\n'
            f'Bonferroni 校正阈值: {bonf_alpha:.6f} (共 {n_tests} 次检验)',
            fontsize=13
        )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('-log10(p-value)', fontsize=11)

    fig.savefig(output_dir / 'granger_pvalue_heatmap.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] {output_dir / 'granger_pvalue_heatmap.png'}")


# ============================================================
# 7. 可视化：因果关系网络图
# ============================================================

def plot_causal_network(results_df: pd.DataFrame, output_dir: Path, alpha: float = 0.05):
    """
    绘制显著因果关系网络图（matplotlib 箭头实现）

    仅显示 Bonferroni 校正后仍显著的因果对（取最优滞后的结果）

    Parameters
    ----------
    results_df : pd.DataFrame
        含 significant_corrected 列的检验结果
    output_dir : Path
        输出目录
    alpha : float
        显著性水平
    """
    if results_df.empty or 'significant_corrected' not in results_df.columns:
        print("  [警告] 无校正后结果，跳过网络图绘制")
        return

    # 筛选显著因果对（取每对中 p 值最小的滞后）
    sig = results_df[results_df['significant_corrected']].copy()
    if sig.empty:
        print("  [信息] Bonferroni 校正后无显著因果关系，绘制空网络图")

    # 对每对取最小 p 值
    if not sig.empty:
        sig_best = sig.loc[sig.groupby(['cause', 'effect'])['p_value'].idxmin()]
    else:
        sig_best = pd.DataFrame(columns=results_df.columns)

    # 收集所有变量节点
    all_vars = set()
    for _, row in results_df.iterrows():
        all_vars.add(row['cause'])
        all_vars.add(row['effect'])
    all_vars = sorted(all_vars)
    n_vars = len(all_vars)

    if n_vars == 0:
        return

    # 布局：圆形排列
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False)
    positions = {v: (np.cos(a), np.sin(a)) for v, a in zip(all_vars, angles)}

    fig, ax = plt.subplots(figsize=(10, 10))

    # 绘制节点
    for var, (x, y) in positions.items():
        circle = plt.Circle((x, y), 0.12, color='steelblue', alpha=0.8)
        ax.add_patch(circle)
        ax.text(x, y, var, ha='center', va='center', fontsize=8,
                fontweight='bold', color='white')

    # 绘制显著因果箭头
    for _, row in sig_best.iterrows():
        cause_pos = positions[row['cause']]
        effect_pos = positions[row['effect']]

        # 计算起点和终点（缩短到节点边缘）
        dx = effect_pos[0] - cause_pos[0]
        dy = effect_pos[1] - cause_pos[1]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        if dist < 0.01:
            continue

        # 缩短箭头到节点圆的边缘
        shrink = 0.14
        start_x = cause_pos[0] + shrink * dx / dist
        start_y = cause_pos[1] + shrink * dy / dist
        end_x = effect_pos[0] - shrink * dx / dist
        end_y = effect_pos[1] - shrink * dy / dist

        # 箭头粗细与 -log10(p) 相关
        width = min(3.0, -np.log10(row['p_value'] + 1e-300) * 0.5)

        ax.annotate(
            '',
            xy=(end_x, end_y),
            xytext=(start_x, start_y),
            arrowprops=dict(
                arrowstyle='->', color='red', lw=width,
                connectionstyle='arc3,rad=0.1',
                mutation_scale=15,
            ),
        )
        # 标注滞后阶数和 p 值
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        ax.text(mid_x, mid_y, f'lag={int(row["lag"])}\np={row["p_value"]:.2e}',
                fontsize=7, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))

    n_sig = len(sig_best)
    n_total = len(results_df)
    ax.set_title(
        f'Granger 因果关系网络 (Bonferroni 校正后)\n'
        f'显著链接: {n_sig}/{n_total}',
        fontsize=14
    )
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect('equal')
    ax.axis('off')

    fig.savefig(output_dir / 'granger_causal_network.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] {output_dir / 'granger_causal_network.png'}")


# ============================================================
# 8. 结果打印
# ============================================================

def print_causality_results(results_df: pd.DataFrame):
    """打印所有因果检验结果"""
    if results_df.empty:
        print("  [信息] 无检验结果")
        return

    print("\n" + "=" * 90)
    print("Granger 因果检验结果明细")
    print("=" * 90)
    print(f"  {'因果方向':<40} {'滞后':>4} {'F统计量':>12} {'p值':>12} {'原始显著':>8} {'校正显著':>8}")
    print("  " + "-" * 88)

    for _, row in results_df.iterrows():
        pair_label = f"{row['cause']} → {row['effect']}"
        sig_raw = '***' if row.get('significant_raw', False) else ''
        sig_corr = '***' if row.get('significant_corrected', False) else ''
        print(f"  {pair_label:<40} {int(row['lag']):>4} "
              f"{row['f_stat']:>12.4f} {row['p_value']:>12.6f} "
              f"{sig_raw:>8} {sig_corr:>8}")

    # 汇总统计
    n_total = len(results_df)
    n_sig_raw = results_df.get('significant_raw', pd.Series(dtype=bool)).sum()
    n_sig_corr = results_df.get('significant_corrected', pd.Series(dtype=bool)).sum()

    print(f"\n  汇总: 共 {n_total} 次检验")
    print(f"    原始显著 (p < 0.05):     {n_sig_raw} ({n_sig_raw / n_total * 100:.1f}%)")
    print(f"    Bonferroni 校正后显著:   {n_sig_corr} ({n_sig_corr / n_total * 100:.1f}%)")

    if n_total > 0:
        bonf_alpha = 0.05 / n_total
        print(f"    Bonferroni 校正阈值:     {bonf_alpha:.6f}")


# ============================================================
# 9. 主入口
# ============================================================

def run_causality_analysis(
    df: pd.DataFrame,
    output_dir: str = "output/causality",
) -> Dict:
    """
    Granger 因果检验主函数

    Parameters
    ----------
    df : pd.DataFrame
        日线数据（已通过 add_derived_features 添加衍生特征）
    output_dir : str
        图表输出目录

    Returns
    -------
    dict
        包含所有检验结果的字典
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BTC Granger 因果检验分析")
    print("=" * 70)
    print(f"数据范围: {df.index.min()} ~ {df.index.max()}")
    print(f"样本数量: {len(df)}")
    print(f"测试滞后阶数: {TEST_LAGS}")
    print(f"因果变量对数: {len(CAUSALITY_PAIRS)}")
    print(f"总检验次数（含所有滞后）: {len(CAUSALITY_PAIRS) * len(TEST_LAGS)}")

    from src.font_config import configure_chinese_font
    configure_chinese_font()

    # --- 日线级 Granger 因果检验 ---
    print("\n>>> [1/4] 执行日线级 Granger 因果检验...")
    daily_results = run_all_granger_tests(df, pairs=CAUSALITY_PAIRS, test_lags=TEST_LAGS)

    if not daily_results.empty:
        daily_results = apply_bonferroni(daily_results, alpha=0.05)
        print_causality_results(daily_results)
    else:
        print("  [警告] 日线级因果检验未产生结果")

    # --- 跨时间尺度因果检验 ---
    print("\n>>> [2/4] 执行跨时间尺度因果检验（小时 → 日线）...")
    cross_results = cross_timeframe_causality(df, test_lags=TEST_LAGS)

    if not cross_results.empty:
        cross_results = apply_bonferroni(cross_results, alpha=0.05)
        print("\n跨时间尺度因果检验结果:")
        print_causality_results(cross_results)
    else:
        print("  [信息] 跨时间尺度因果检验无结果（可能小时数据不可用）")

    # --- 合并所有结果用于可视化 ---
    all_results = pd.concat([daily_results, cross_results], ignore_index=True)
    if not all_results.empty and 'significant_corrected' not in all_results.columns:
        all_results = apply_bonferroni(all_results, alpha=0.05)

    # --- p 值热力图（仅日线级结果，避免混淆） ---
    print("\n>>> [3/4] 绘制 p 值热力图...")
    plot_pvalue_heatmap(daily_results, output_dir)

    # --- 因果关系网络图 ---
    print("\n>>> [4/4] 绘制因果关系网络图...")
    # 使用所有结果（含跨时间尺度），直接使用各组已做的 Bonferroni 校正结果，
    # 不再重复校正（各组检验已独立校正，合并后再校正会导致双重惩罚）
    if not all_results.empty:
        plot_causal_network(all_results, output_dir)
    else:
        print("  [警告] 无可用结果，跳过网络图")

    print("\n" + "=" * 70)
    print("Granger 因果检验分析完成！")
    print(f"图表已保存至: {output_dir.resolve()}")
    print("=" * 70)

    return {
        'daily_results': daily_results,
        'cross_timeframe_results': cross_results,
        'all_results': all_results,
    }


# ============================================================
# 独立运行入口
# ============================================================

if __name__ == '__main__':
    from src.data_loader import load_daily
    from src.preprocessing import add_derived_features

    df = load_daily()
    df = add_derived_features(df)
    run_causality_analysis(df)
