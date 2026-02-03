"""日历效应分析模块 - 星期、月份、小时、季度、月初月末效应"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from itertools import combinations
from scipy import stats

from src.font_config import configure_chinese_font
configure_chinese_font()

# 星期名称映射（中英文）
WEEKDAY_NAMES_CN = {0: '周一', 1: '周二', 2: '周三', 3: '周四',
                    4: '周五', 5: '周六', 6: '周日'}
WEEKDAY_NAMES_EN = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu',
                    4: 'Fri', 5: 'Sat', 6: 'Sun'}

# 月份名称映射
MONTH_NAMES_CN = {1: '1月', 2: '2月', 3: '3月', 4: '4月',
                  5: '5月', 6: '6月', 7: '7月', 8: '8月',
                  9: '9月', 10: '10月', 11: '11月', 12: '12月'}
MONTH_NAMES_EN = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
                  5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
                  9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}


def _bonferroni_pairwise_mannwhitney(groups: dict, alpha: float = 0.05):
    """
    对多组数据进行 Mann-Whitney U 两两检验，并做 Bonferroni 校正。

    Parameters
    ----------
    groups : dict
        {组标签: 收益率序列}
    alpha : float
        显著性水平（校正前）

    Returns
    -------
    list[dict]
        每对检验的结果列表
    """
    keys = sorted(groups.keys())
    pairs = list(combinations(keys, 2))
    n_tests = len(pairs)
    corrected_alpha = alpha / n_tests if n_tests > 0 else alpha

    results = []
    for k1, k2 in pairs:
        g1, g2 = groups[k1].dropna(), groups[k2].dropna()
        if len(g1) < 3 or len(g2) < 3:
            continue
        stat, pval = stats.mannwhitneyu(g1, g2, alternative='two-sided')
        results.append({
            'group1': k1,
            'group2': k2,
            'U_stat': stat,
            'p_value': pval,
            'p_corrected': min(pval * n_tests, 1.0),  # Bonferroni 校正
            'significant': pval * n_tests < alpha,
            'corrected_alpha': corrected_alpha,
        })
    return results


def _kruskal_wallis_test(groups: dict):
    """
    Kruskal-Wallis H 检验（非参数单因素检验）。

    Parameters
    ----------
    groups : dict
        {组标签: 收益率序列}

    Returns
    -------
    dict
        包含 H 统计量、p 值等
    """
    valid_groups = [g.dropna().values for g in groups.values() if len(g.dropna()) >= 3]
    if len(valid_groups) < 2:
        return {'H_stat': np.nan, 'p_value': np.nan, 'n_groups': len(valid_groups)}

    h_stat, p_val = stats.kruskal(*valid_groups)
    return {'H_stat': h_stat, 'p_value': p_val, 'n_groups': len(valid_groups)}


# --------------------------------------------------------------------------
# 1. 星期效应分析
# --------------------------------------------------------------------------
def analyze_day_of_week(df: pd.DataFrame, output_dir: Path):
    """
    分析日收益率的星期效应。

    Parameters
    ----------
    df : pd.DataFrame
        日线数据（需含 log_return 列，DatetimeIndex 索引）
    output_dir : Path
        图片保存目录
    """
    print("\n" + "=" * 70)
    print("【星期效应分析】Day-of-Week Effect")
    print("=" * 70)

    df = df.dropna(subset=['log_return']).copy()
    df['weekday'] = df.index.dayofweek  # 0=周一, 6=周日

    # --- 描述性统计 ---
    groups = {wd: df.loc[df['weekday'] == wd, 'log_return'] for wd in range(7)}

    print("\n--- 各星期对数收益率统计 ---")
    stats_rows = []
    for wd in range(7):
        g = groups[wd]
        row = {
            '星期': WEEKDAY_NAMES_CN[wd],
            '样本量': len(g),
            '均值': g.mean(),
            '中位数': g.median(),
            '标准差': g.std(),
            '偏度': g.skew(),
            '峰度': g.kurtosis(),
        }
        stats_rows.append(row)
    stats_df = pd.DataFrame(stats_rows)
    print(stats_df.to_string(index=False, float_format='{:.6f}'.format))

    # --- Kruskal-Wallis 检验 ---
    kw_result = _kruskal_wallis_test(groups)
    print(f"\nKruskal-Wallis H 检验: H={kw_result['H_stat']:.4f}, "
          f"p={kw_result['p_value']:.6f}")
    if kw_result['p_value'] < 0.05:
        print("  => 在 5% 显著性水平下，各星期收益率存在显著差异")
    else:
        print("  => 在 5% 显著性水平下，各星期收益率无显著差异")

    # --- Mann-Whitney U 两两检验 (Bonferroni 校正) ---
    pairwise = _bonferroni_pairwise_mannwhitney(groups)
    sig_pairs = [p for p in pairwise if p['significant']]
    print(f"\nMann-Whitney U 两两检验 (Bonferroni 校正, {len(pairwise)} 对比较):")
    if sig_pairs:
        for p in sig_pairs:
            print(f"  {WEEKDAY_NAMES_CN[p['group1']]} vs {WEEKDAY_NAMES_CN[p['group2']]}: "
                  f"U={p['U_stat']:.1f}, p_raw={p['p_value']:.6f}, "
                  f"p_corrected={p['p_corrected']:.6f} *")
    else:
        print("  无显著差异的配对（校正后）")

    # --- 可视化: 箱线图 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 箱线图
    box_data = [groups[wd].values for wd in range(7)]
    bp = axes[0].boxplot(box_data, labels=[WEEKDAY_NAMES_CN[i] for i in range(7)],
                         patch_artist=True, showfliers=False, showmeans=True,
                         meanprops=dict(marker='D', markerfacecolor='red', markersize=5))
    colors = plt.cm.Set3(np.linspace(0, 1, 7))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    axes[0].axhline(y=0, color='grey', linestyle='--', alpha=0.5)
    axes[0].set_title('BTC 日收益率 - 星期效应（箱线图）', fontsize=13)
    axes[0].set_ylabel('对数收益率')
    axes[0].set_xlabel('星期')

    # 均值柱状图
    means = [groups[wd].mean() for wd in range(7)]
    sems = [groups[wd].sem() for wd in range(7)]
    bar_colors = ['#2ecc71' if m > 0 else '#e74c3c' for m in means]
    axes[1].bar(range(7), means, yerr=sems, color=bar_colors,
                alpha=0.8, capsize=3, edgecolor='black', linewidth=0.5)
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels([WEEKDAY_NAMES_CN[i] for i in range(7)])
    axes[1].axhline(y=0, color='grey', linestyle='--', alpha=0.5)
    axes[1].set_title('BTC 日均收益率 - 星期效应（均值±SE）', fontsize=13)
    axes[1].set_ylabel('平均对数收益率')
    axes[1].set_xlabel('星期')

    plt.tight_layout()
    fig_path = output_dir / 'calendar_weekday_effect.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n图表已保存: {fig_path}")


# --------------------------------------------------------------------------
# 2. 月份效应分析
# --------------------------------------------------------------------------
def analyze_month_of_year(df: pd.DataFrame, output_dir: Path):
    """
    分析日收益率的月份效应，并绘制年×月热力图。

    Parameters
    ----------
    df : pd.DataFrame
        日线数据（需含 log_return 列）
    output_dir : Path
        图片保存目录
    """
    print("\n" + "=" * 70)
    print("【月份效应分析】Month-of-Year Effect")
    print("=" * 70)

    df = df.dropna(subset=['log_return']).copy()
    df['month'] = df.index.month
    df['year'] = df.index.year

    # --- 描述性统计 ---
    groups = {m: df.loc[df['month'] == m, 'log_return'] for m in range(1, 13)}

    print("\n--- 各月份对数收益率统计 ---")
    stats_rows = []
    for m in range(1, 13):
        g = groups[m]
        row = {
            '月份': MONTH_NAMES_CN[m],
            '样本量': len(g),
            '均值': g.mean(),
            '中位数': g.median(),
            '标准差': g.std(),
        }
        stats_rows.append(row)
    stats_df = pd.DataFrame(stats_rows)
    print(stats_df.to_string(index=False, float_format='{:.6f}'.format))

    # --- Kruskal-Wallis 检验 ---
    kw_result = _kruskal_wallis_test(groups)
    print(f"\nKruskal-Wallis H 检验: H={kw_result['H_stat']:.4f}, "
          f"p={kw_result['p_value']:.6f}")
    if kw_result['p_value'] < 0.05:
        print("  => 在 5% 显著性水平下，各月份收益率存在显著差异")
    else:
        print("  => 在 5% 显著性水平下，各月份收益率无显著差异")

    # --- Mann-Whitney U 两两检验 (Bonferroni 校正) ---
    pairwise = _bonferroni_pairwise_mannwhitney(groups)
    sig_pairs = [p for p in pairwise if p['significant']]
    print(f"\nMann-Whitney U 两两检验 (Bonferroni 校正, {len(pairwise)} 对比较):")
    if sig_pairs:
        for p in sig_pairs:
            print(f"  {MONTH_NAMES_CN[p['group1']]} vs {MONTH_NAMES_CN[p['group2']]}: "
                  f"U={p['U_stat']:.1f}, p_raw={p['p_value']:.6f}, "
                  f"p_corrected={p['p_corrected']:.6f} *")
    else:
        print("  无显著差异的配对（校正后）")

    # --- 可视化 ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 均值柱状图
    means = [groups[m].mean() for m in range(1, 13)]
    sems = [groups[m].sem() for m in range(1, 13)]
    bar_colors = ['#2ecc71' if m > 0 else '#e74c3c' for m in means]
    axes[0].bar(range(1, 13), means, yerr=sems, color=bar_colors,
                alpha=0.8, capsize=3, edgecolor='black', linewidth=0.5)
    axes[0].set_xticks(range(1, 13))
    axes[0].set_xticklabels([MONTH_NAMES_EN[i] for i in range(1, 13)])
    axes[0].axhline(y=0, color='grey', linestyle='--', alpha=0.5)
    axes[0].set_title('BTC 月均收益率（均值±SE）', fontsize=13)
    axes[0].set_ylabel('平均对数收益率')
    axes[0].set_xlabel('月份')

    # 年×月 热力图：每月累计收益率
    monthly_returns = df.groupby(['year', 'month'])['log_return'].sum().unstack(fill_value=np.nan)
    monthly_returns.columns = [MONTH_NAMES_EN[c] for c in monthly_returns.columns]
    sns.heatmap(monthly_returns, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                linewidths=0.5, ax=axes[1], cbar_kws={'label': '累计对数收益率'})
    axes[1].set_title('BTC 年×月 累计对数收益率热力图', fontsize=13)
    axes[1].set_ylabel('年份')
    axes[1].set_xlabel('月份')

    plt.tight_layout()
    fig_path = output_dir / 'calendar_month_effect.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n图表已保存: {fig_path}")


# --------------------------------------------------------------------------
# 3. 小时效应分析（1h 数据）
# --------------------------------------------------------------------------
def analyze_hour_of_day(df_hourly: pd.DataFrame, output_dir: Path):
    """
    分析小时级别收益率与成交量的日内效应。

    Parameters
    ----------
    df_hourly : pd.DataFrame
        小时线数据（需含 close、volume 列，DatetimeIndex 索引）
    output_dir : Path
        图片保存目录
    """
    print("\n" + "=" * 70)
    print("【小时效应分析】Hour-of-Day Effect")
    print("=" * 70)

    df = df_hourly.copy()
    # 计算小时收益率
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df = df.dropna(subset=['log_return'])
    df['hour'] = df.index.hour

    # --- 描述性统计 ---
    groups_ret = {h: df.loc[df['hour'] == h, 'log_return'] for h in range(24)}
    groups_vol = {h: df.loc[df['hour'] == h, 'volume'] for h in range(24)}

    print("\n--- 各小时对数收益率与成交量统计 ---")
    stats_rows = []
    for h in range(24):
        gr = groups_ret[h]
        gv = groups_vol[h]
        row = {
            '小时(UTC)': f'{h:02d}:00',
            '样本量': len(gr),
            '收益率均值': gr.mean(),
            '收益率中位数': gr.median(),
            '收益率标准差': gr.std(),
            '成交量均值': gv.mean(),
        }
        stats_rows.append(row)
    stats_df = pd.DataFrame(stats_rows)
    print(stats_df.to_string(index=False, float_format='{:.6f}'.format))

    # --- Kruskal-Wallis 检验 (收益率) ---
    kw_ret = _kruskal_wallis_test(groups_ret)
    print(f"\n收益率 Kruskal-Wallis H 检验: H={kw_ret['H_stat']:.4f}, "
          f"p={kw_ret['p_value']:.6f}")
    if kw_ret['p_value'] < 0.05:
        print("  => 在 5% 显著性水平下，各小时收益率存在显著差异")
    else:
        print("  => 在 5% 显著性水平下，各小时收益率无显著差异")

    # --- Kruskal-Wallis 检验 (成交量) ---
    kw_vol = _kruskal_wallis_test(groups_vol)
    print(f"\n成交量 Kruskal-Wallis H 检验: H={kw_vol['H_stat']:.4f}, "
          f"p={kw_vol['p_value']:.6f}")
    if kw_vol['p_value'] < 0.05:
        print("  => 在 5% 显著性水平下，各小时成交量存在显著差异")
    else:
        print("  => 在 5% 显著性水平下，各小时成交量无显著差异")

    # --- 可视化 ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    hours = list(range(24))
    hour_labels = [f'{h:02d}' for h in hours]

    # 收益率
    ret_means = [groups_ret[h].mean() for h in hours]
    ret_sems = [groups_ret[h].sem() for h in hours]
    bar_colors_ret = ['#2ecc71' if m > 0 else '#e74c3c' for m in ret_means]
    axes[0].bar(hours, ret_means, yerr=ret_sems, color=bar_colors_ret,
                alpha=0.8, capsize=2, edgecolor='black', linewidth=0.3)
    axes[0].set_xticks(hours)
    axes[0].set_xticklabels(hour_labels)
    axes[0].axhline(y=0, color='grey', linestyle='--', alpha=0.5)
    axes[0].set_title('BTC 小时均收益率 (UTC, 均值±SE)', fontsize=13)
    axes[0].set_ylabel('平均对数收益率')
    axes[0].set_xlabel('小时 (UTC)')

    # 成交量
    vol_means = [groups_vol[h].mean() for h in hours]
    axes[1].bar(hours, vol_means, color='steelblue', alpha=0.8,
                edgecolor='black', linewidth=0.3)
    axes[1].set_xticks(hours)
    axes[1].set_xticklabels(hour_labels)
    axes[1].set_title('BTC 小时均成交量 (UTC)', fontsize=13)
    axes[1].set_ylabel('平均成交量 (BTC)')
    axes[1].set_xlabel('小时 (UTC)')
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    plt.tight_layout()
    fig_path = output_dir / 'calendar_hour_effect.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n图表已保存: {fig_path}")


# --------------------------------------------------------------------------
# 4. 季度效应 & 月初月末效应
# --------------------------------------------------------------------------
def analyze_quarter_and_month_boundary(df: pd.DataFrame, output_dir: Path):
    """
    分析季度效应，以及每月前5日/后5日的收益率差异。

    Parameters
    ----------
    df : pd.DataFrame
        日线数据（需含 log_return 列）
    output_dir : Path
        图片保存目录
    """
    print("\n" + "=" * 70)
    print("【季度效应 & 月初/月末效应分析】")
    print("=" * 70)

    df = df.dropna(subset=['log_return']).copy()
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['day'] = df.index.day

    # ========== 季度效应 ==========
    groups_q = {q: df.loc[df['quarter'] == q, 'log_return'] for q in range(1, 5)}

    print("\n--- 各季度对数收益率统计 ---")
    quarter_names = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'}
    for q in range(1, 5):
        g = groups_q[q]
        print(f"  {quarter_names[q]}: 均值={g.mean():.6f}, 中位数={g.median():.6f}, "
              f"标准差={g.std():.6f}, 样本量={len(g)}")

    kw_q = _kruskal_wallis_test(groups_q)
    print(f"\n季度 Kruskal-Wallis H 检验: H={kw_q['H_stat']:.4f}, p={kw_q['p_value']:.6f}")
    if kw_q['p_value'] < 0.05:
        print("  => 在 5% 显著性水平下，各季度收益率存在显著差异")
    else:
        print("  => 在 5% 显著性水平下，各季度收益率无显著差异")

    # 季度两两比较
    pairwise_q = _bonferroni_pairwise_mannwhitney(groups_q)
    sig_q = [p for p in pairwise_q if p['significant']]
    if sig_q:
        print(f"\n季度两两检验 (Bonferroni 校正, {len(pairwise_q)} 对):")
        for p in sig_q:
            print(f"  {quarter_names[p['group1']]} vs {quarter_names[p['group2']]}: "
                  f"U={p['U_stat']:.1f}, p_corrected={p['p_corrected']:.6f} *")

    # ========== 月初/月末效应 ==========
    # 判断每月最后5天：通过计算每个日期距当月末的天数
    from pandas.tseries.offsets import MonthEnd
    df['month_end'] = df.index + MonthEnd(0)  # 当月最后一天
    df['days_to_end'] = (df['month_end'] - df.index).dt.days

    # 月初前5天 vs 月末后5天
    mask_start = df['day'] <= 5
    mask_end = df['days_to_end'] < 5  # 距离月末不到5天（即最后5天）

    ret_start = df.loc[mask_start, 'log_return']
    ret_end = df.loc[mask_end, 'log_return']
    ret_mid = df.loc[~mask_start & ~mask_end, 'log_return']

    print("\n--- 月初 / 月中 / 月末 收益率统计 ---")
    for label, data in [('月初(前5日)', ret_start), ('月中', ret_mid), ('月末(后5日)', ret_end)]:
        print(f"  {label}: 均值={data.mean():.6f}, 中位数={data.median():.6f}, "
              f"标准差={data.std():.6f}, 样本量={len(data)}")

    # Mann-Whitney U 检验：月初 vs 月末
    if len(ret_start) >= 3 and len(ret_end) >= 3:
        u_stat, p_val = stats.mannwhitneyu(ret_start, ret_end, alternative='two-sided')
        print(f"\n月初 vs 月末 Mann-Whitney U 检验: U={u_stat:.1f}, p={p_val:.6f}")
        if p_val < 0.05:
            print("  => 在 5% 显著性水平下，月初与月末收益率存在显著差异")
        else:
            print("  => 在 5% 显著性水平下，月初与月末收益率无显著差异")

    # --- 可视化 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 季度柱状图
    q_means = [groups_q[q].mean() for q in range(1, 5)]
    q_sems = [groups_q[q].sem() for q in range(1, 5)]
    q_colors = ['#2ecc71' if m > 0 else '#e74c3c' for m in q_means]
    axes[0].bar(range(1, 5), q_means, yerr=q_sems, color=q_colors,
                alpha=0.8, capsize=4, edgecolor='black', linewidth=0.5)
    axes[0].set_xticks(range(1, 5))
    axes[0].set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
    axes[0].axhline(y=0, color='grey', linestyle='--', alpha=0.5)
    axes[0].set_title('BTC 季度均收益率（均值±SE）', fontsize=13)
    axes[0].set_ylabel('平均对数收益率')
    axes[0].set_xlabel('季度')

    # 月初/月中/月末 柱状图
    boundary_means = [ret_start.mean(), ret_mid.mean(), ret_end.mean()]
    boundary_sems = [ret_start.sem(), ret_mid.sem(), ret_end.sem()]
    boundary_colors = ['#3498db', '#95a5a6', '#e67e22']
    axes[1].bar(range(3), boundary_means, yerr=boundary_sems, color=boundary_colors,
                alpha=0.8, capsize=4, edgecolor='black', linewidth=0.5)
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels(['月初(前5日)', '月中', '月末(后5日)'])
    axes[1].axhline(y=0, color='grey', linestyle='--', alpha=0.5)
    axes[1].set_title('BTC 月初/月中/月末 均收益率（均值±SE）', fontsize=13)
    axes[1].set_ylabel('平均对数收益率')

    plt.tight_layout()
    fig_path = output_dir / 'calendar_quarter_boundary_effect.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n图表已保存: {fig_path}")

    # 清理临时列
    df.drop(columns=['month_end', 'days_to_end'], inplace=True, errors='ignore')


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------
def run_calendar_analysis(
    df: pd.DataFrame,
    df_hourly: pd.DataFrame = None,
    output_dir: str = 'output/calendar',
):
    """
    日历效应分析主入口。

    Parameters
    ----------
    df : pd.DataFrame
        日线数据，已通过 add_derived_features 添加衍生特征（含 log_return 列）
    df_hourly : pd.DataFrame, optional
        小时线原始数据（含 close、volume 列）。若为 None 则跳过小时效应分析。
    output_dir : str or Path
        输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 70)
    print("#  BTC 日历效应分析 (Calendar Effects Analysis)")
    print("#" * 70)

    # 1. 星期效应
    analyze_day_of_week(df, output_dir)

    # 2. 月份效应
    analyze_month_of_year(df, output_dir)

    # 3. 小时效应（若有小时数据）
    if df_hourly is not None and len(df_hourly) > 0:
        analyze_hour_of_day(df_hourly, output_dir)
    else:
        print("\n[跳过] 小时效应分析：未提供小时数据 (df_hourly is None)")

    # 4. 季度 & 月初月末效应
    analyze_quarter_and_month_boundary(df, output_dir)

    print("\n" + "#" * 70)
    print("#  日历效应分析完成")
    print("#" * 70)


# --------------------------------------------------------------------------
# 可独立运行
# --------------------------------------------------------------------------
if __name__ == '__main__':
    from data_loader import load_daily, load_hourly
    from preprocessing import add_derived_features

    # 加载数据
    df_daily = load_daily()
    df_daily = add_derived_features(df_daily)

    try:
        df_hourly = load_hourly()
    except Exception as e:
        print(f"[警告] 加载小时数据失败: {e}")
        df_hourly = None

    run_calendar_analysis(df_daily, df_hourly, output_dir='output/calendar')
