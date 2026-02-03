"""BTC 减半周期分析模块 - 减半前后价格行为、波动率、累计收益对比"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from scipy import stats

from src.font_config import configure_chinese_font
configure_chinese_font()

# BTC 减半日期（数据范围 2017-2026 内的两次减半）
HALVING_DATES = [
    pd.Timestamp('2020-05-11'),
    pd.Timestamp('2024-04-20'),
]
HALVING_LABELS = ['第三次减半 (2020-05-11)', '第四次减半 (2024-04-20)']

# 分析窗口：减半前后各 500 天
WINDOW_DAYS = 500


def _extract_halving_window(df: pd.DataFrame, halving_date: pd.Timestamp,
                            window: int = WINDOW_DAYS):
    """
    提取减半日期前后的数据窗口。

    Parameters
    ----------
    df : pd.DataFrame
        日线数据（DatetimeIndex 索引，含 close 和 log_return 列）
    halving_date : pd.Timestamp
        减半日期
    window : int
        前后各取的天数

    Returns
    -------
    pd.DataFrame
        窗口数据，附加 'days_from_halving' 列（减半日=0）
    """
    start = halving_date - pd.Timedelta(days=window)
    end = halving_date + pd.Timedelta(days=window)
    mask = (df.index >= start) & (df.index <= end)
    window_df = df.loc[mask].copy()

    # 计算距减半日的天数差
    window_df['days_from_halving'] = (window_df.index - halving_date).days
    return window_df


def _normalize_price(window_df: pd.DataFrame, halving_date: pd.Timestamp):
    """
    以减半日价格为基准（=100）归一化价格。

    Parameters
    ----------
    window_df : pd.DataFrame
        窗口数据（含 close 列）
    halving_date : pd.Timestamp
        减半日期

    Returns
    -------
    pd.Series
        归一化后的价格序列（减半日=100）
    """
    # 找到距减半日最近的交易日
    idx = window_df.index.get_indexer([halving_date], method='nearest')[0]
    base_price = window_df['close'].iloc[idx]
    return (window_df['close'] / base_price) * 100


def analyze_normalized_trajectories(windows: list, output_dir: Path):
    """
    绘制归一化价格轨迹叠加图。

    Parameters
    ----------
    windows : list[dict]
        每个元素包含 'df', 'normalized', 'label', 'halving_date'
    output_dir : Path
        图片保存目录
    """
    print("\n" + "-" * 60)
    print("【归一化价格轨迹叠加】")
    print("-" * 60)

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ['#2980b9', '#e74c3c']
    linestyles = ['-', '--']

    for i, w in enumerate(windows):
        days = w['df']['days_from_halving']
        normalized = w['normalized']
        ax.plot(days, normalized, color=colors[i], linestyle=linestyles[i],
                linewidth=1.5, label=w['label'], alpha=0.85)

    ax.axvline(x=0, color='gold', linestyle='-', linewidth=2,
               alpha=0.8, label='减半日')
    ax.axhline(y=100, color='grey', linestyle=':', alpha=0.4)

    ax.set_title('BTC 减半周期 - 归一化价格轨迹叠加（减半日=100）', fontsize=14)
    ax.set_xlabel(f'距减半日天数（前后各 {WINDOW_DAYS} 天）')
    ax.set_ylabel('归一化价格')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig_path = output_dir / 'halving_normalized_trajectories.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"图表已保存: {fig_path}")


def analyze_pre_post_returns(windows: list, output_dir: Path):
    """
    对比减半前后平均收益率，进行 Welch's t 检验。

    Parameters
    ----------
    windows : list[dict]
        窗口数据列表
    output_dir : Path
        图片保存目录
    """
    print("\n" + "-" * 60)
    print("【减半前后收益率对比 & Welch's t 检验】")
    print("-" * 60)

    all_pre_returns = []
    all_post_returns = []

    for w in windows:
        df_w = w['df']
        pre = df_w.loc[df_w['days_from_halving'] < 0, 'log_return'].dropna()
        post = df_w.loc[df_w['days_from_halving'] > 0, 'log_return'].dropna()
        all_pre_returns.append(pre)
        all_post_returns.append(post)

        print(f"\n{w['label']}:")
        print(f"  减半前 {WINDOW_DAYS}天: 均值={pre.mean():.6f}, 标准差={pre.std():.6f}, "
              f"中位数={pre.median():.6f}, N={len(pre)}")
        print(f"  减半后 {WINDOW_DAYS}天: 均值={post.mean():.6f}, 标准差={post.std():.6f}, "
              f"中位数={post.median():.6f}, N={len(post)}")

        # 单周期 Welch's t-test
        if len(pre) >= 3 and len(post) >= 3:
            t_stat, p_val = stats.ttest_ind(pre, post, equal_var=False)
            print(f"  Welch's t 检验: t={t_stat:.4f}, p={p_val:.6f}")
            if p_val < 0.05:
                print("    => 减半前后收益率在 5% 水平下存在显著差异")
            else:
                print("    => 减半前后收益率在 5% 水平下无显著差异")

    # 合并所有周期的前后收益率进行总体检验
    combined_pre = pd.concat(all_pre_returns)
    combined_post = pd.concat(all_post_returns)
    print(f"\n--- 合并所有减半周期 ---")
    print(f"  合并减半前: 均值={combined_pre.mean():.6f}, N={len(combined_pre)}")
    print(f"  合并减半后: 均值={combined_post.mean():.6f}, N={len(combined_post)}")
    t_stat_all, p_val_all = stats.ttest_ind(combined_pre, combined_post, equal_var=False)
    print(f"  合并 Welch's t 检验: t={t_stat_all:.4f}, p={p_val_all:.6f}")

    # --- 可视化: 减半前后收益率对比柱状图（含置信区间） ---
    fig, axes = plt.subplots(1, len(windows), figsize=(7 * len(windows), 6))
    if len(windows) == 1:
        axes = [axes]

    for i, w in enumerate(windows):
        df_w = w['df']
        pre = df_w.loc[df_w['days_from_halving'] < 0, 'log_return'].dropna()
        post = df_w.loc[df_w['days_from_halving'] > 0, 'log_return'].dropna()

        means = [pre.mean(), post.mean()]
        # 95% 置信区间
        ci_pre = stats.t.interval(0.95, len(pre) - 1, loc=pre.mean(), scale=pre.sem())
        ci_post = stats.t.interval(0.95, len(post) - 1, loc=post.mean(), scale=post.sem())
        errors = [
            [means[0] - ci_pre[0], means[1] - ci_post[0]],
            [ci_pre[1] - means[0], ci_post[1] - means[1]],
        ]

        colors_bar = ['#3498db', '#e67e22']
        axes[i].bar(['减半前', '减半后'], means, yerr=errors, color=colors_bar,
                     alpha=0.8, capsize=5, edgecolor='black', linewidth=0.5)
        axes[i].axhline(y=0, color='grey', linestyle='--', alpha=0.5)
        axes[i].set_title(w['label'] + '\n日均对数收益率（95% CI）', fontsize=12)
        axes[i].set_ylabel('平均对数收益率')

    plt.tight_layout()
    fig_path = output_dir / 'halving_pre_post_returns.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n图表已保存: {fig_path}")


def analyze_cumulative_returns(windows: list, output_dir: Path):
    """
    绘制减半后累计收益率对比。

    Parameters
    ----------
    windows : list[dict]
        窗口数据列表
    output_dir : Path
        图片保存目录
    """
    print("\n" + "-" * 60)
    print("【减半后累计收益率对比】")
    print("-" * 60)

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ['#2980b9', '#e74c3c']

    for i, w in enumerate(windows):
        df_w = w['df']
        post = df_w.loc[df_w['days_from_halving'] >= 0].copy()
        if len(post) == 0:
            print(f"  {w['label']}: 无减半后数据")
            continue

        # 累计对数收益率
        post_returns = post['log_return'].fillna(0)
        cum_return = post_returns.cumsum()
        # 转为百分比形式
        cum_return_pct = (np.exp(cum_return) - 1) * 100

        days = post['days_from_halving']
        ax.plot(days, cum_return_pct, color=colors[i], linewidth=1.5,
                label=w['label'], alpha=0.85)

        # 输出关键节点
        final_cum = cum_return_pct.iloc[-1] if len(cum_return_pct) > 0 else 0
        print(f"  {w['label']}: 减半后 {len(post)} 天累计收益率 = {final_cum:.2f}%")

        # 输出一些关键时间节点的累计收益
        for target_day in [30, 90, 180, 365, WINDOW_DAYS]:
            mask_day = days <= target_day
            if mask_day.any():
                val = cum_return_pct.loc[mask_day].iloc[-1]
                actual_day = days.loc[mask_day].iloc[-1]
                print(f"    第 {actual_day} 天: {val:.2f}%")

    ax.axhline(y=0, color='grey', linestyle=':', alpha=0.4)
    ax.set_title('BTC 减半后累计收益率对比', fontsize=14)
    ax.set_xlabel('距减半日天数')
    ax.set_ylabel('累计收益率 (%)')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}%'))

    fig_path = output_dir / 'halving_cumulative_returns.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n图表已保存: {fig_path}")


def analyze_volatility_change(windows: list, output_dir: Path):
    """
    Levene 检验：减半前后波动率变化。

    Parameters
    ----------
    windows : list[dict]
        窗口数据列表
    output_dir : Path
        图片保存目录
    """
    print("\n" + "-" * 60)
    print("【减半前后波动率变化 - Levene 检验】")
    print("-" * 60)

    for w in windows:
        df_w = w['df']
        pre = df_w.loc[df_w['days_from_halving'] < 0, 'log_return'].dropna()
        post = df_w.loc[df_w['days_from_halving'] > 0, 'log_return'].dropna()

        print(f"\n{w['label']}:")
        print(f"  减半前波动率（日标准差）: {pre.std():.6f} "
              f"(年化: {pre.std() * np.sqrt(365):.4f})")
        print(f"  减半后波动率（日标准差）: {post.std():.6f} "
              f"(年化: {post.std() * np.sqrt(365):.4f})")

        if len(pre) >= 3 and len(post) >= 3:
            lev_stat, lev_p = stats.levene(pre, post, center='median')
            print(f"  Levene 检验: W={lev_stat:.4f}, p={lev_p:.6f}")
            if lev_p < 0.05:
                print("    => 在 5% 水平下，减半前后波动率存在显著变化")
            else:
                print("    => 在 5% 水平下，减半前后波动率无显著变化")


def analyze_inter_cycle_correlation(windows: list):
    """
    两个减半周期归一化轨迹的 Pearson 相关系数。

    Parameters
    ----------
    windows : list[dict]
        窗口数据列表（需要至少2个周期）
    """
    print("\n" + "-" * 60)
    print("【周期间轨迹相关性 - Pearson 相关】")
    print("-" * 60)

    if len(windows) < 2:
        print("  仅有1个周期，无法计算周期间相关性。")
        return

    # 按照 days_from_halving 对齐两个周期
    w1, w2 = windows[0], windows[1]
    df1 = w1['df'][['days_from_halving']].copy()
    df1['norm_price_1'] = w1['normalized'].values

    df2 = w2['df'][['days_from_halving']].copy()
    df2['norm_price_2'] = w2['normalized'].values

    # 以 days_from_halving 为键进行内连接
    merged = pd.merge(df1, df2, on='days_from_halving', how='inner')

    if len(merged) < 10:
        print(f"  重叠天数过少（{len(merged)}天），无法可靠计算相关性。")
        return

    r, p_val = stats.pearsonr(merged['norm_price_1'], merged['norm_price_2'])
    print(f"  重叠天数: {len(merged)}")
    print(f"  Pearson 相关系数: r={r:.4f}, p={p_val:.6f}")

    if abs(r) > 0.7:
        print("  => 两个减半周期的价格轨迹呈强相关")
    elif abs(r) > 0.4:
        print("  => 两个减半周期的价格轨迹呈中等相关")
    else:
        print("  => 两个减半周期的价格轨迹相关性较弱")

    # 分别看减半前和减半后的相关性
    pre_merged = merged[merged['days_from_halving'] < 0]
    post_merged = merged[merged['days_from_halving'] > 0]

    if len(pre_merged) >= 10:
        r_pre, p_pre = stats.pearsonr(pre_merged['norm_price_1'], pre_merged['norm_price_2'])
        print(f"  减半前轨迹相关性: r={r_pre:.4f}, p={p_pre:.6f} (N={len(pre_merged)})")

    if len(post_merged) >= 10:
        r_post, p_post = stats.pearsonr(post_merged['norm_price_1'], post_merged['norm_price_2'])
        print(f"  减半后轨迹相关性: r={r_post:.4f}, p={p_post:.6f} (N={len(post_merged)})")


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------
def run_halving_analysis(
    df: pd.DataFrame,
    output_dir: str = 'output/halving',
):
    """
    BTC 减半周期分析主入口。

    Parameters
    ----------
    df : pd.DataFrame
        日线数据，已通过 add_derived_features 添加衍生特征（含 close、log_return 列）
    output_dir : str or Path
        输出目录

    Notes
    -----
    重要局限性: 数据范围内仅含2次减半事件（2020、2024），样本量极少，
    统计检验的功效（power）很低，结论仅供参考，不能作为因果推断依据。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 70)
    print("#  BTC 减半周期分析 (Halving Cycle Analysis)")
    print("#" * 70)

    # ===== 重要局限性说明 =====
    print("\n⚠️  重要局限性说明:")
    print(f"  本分析仅覆盖 {len(HALVING_DATES)} 次减半事件（样本量极少）。")
    print("  统计检验的功效（statistical power）很低，")
    print("  任何「显著性」结论都应谨慎解读，不能作为因果推断依据。")
    print("  结果主要用于描述性分析和模式探索。\n")

    # 提取每次减半的窗口数据
    windows = []
    for i, (hdate, hlabel) in enumerate(zip(HALVING_DATES, HALVING_LABELS)):
        w_df = _extract_halving_window(df, hdate, WINDOW_DAYS)
        if len(w_df) == 0:
            print(f"[警告] {hlabel} 窗口内无数据，跳过。")
            continue

        normalized = _normalize_price(w_df, hdate)

        print(f"周期 {i + 1}: {hlabel}")
        print(f"  数据范围: {w_df.index.min().date()} ~ {w_df.index.max().date()}")
        print(f"  数据量: {len(w_df)} 天")
        print(f"  减半日价格: {w_df['close'].iloc[w_df.index.get_indexer([hdate], method='nearest')[0]]:.2f} USDT")

        windows.append({
            'df': w_df,
            'normalized': normalized,
            'label': hlabel,
            'halving_date': hdate,
        })

    if len(windows) == 0:
        print("[错误] 无有效减半窗口数据，分析中止。")
        return

    # 1. 归一化价格轨迹叠加
    analyze_normalized_trajectories(windows, output_dir)

    # 2. 减半前后收益率对比
    analyze_pre_post_returns(windows, output_dir)

    # 3. 减半后累计收益率
    analyze_cumulative_returns(windows, output_dir)

    # 4. 波动率变化 (Levene 检验)
    analyze_volatility_change(windows, output_dir)

    # 5. 周期间轨迹相关性
    analyze_inter_cycle_correlation(windows)

    # ===== 综合可视化: 三合一图 =====
    _plot_combined_summary(windows, output_dir)

    print("\n" + "#" * 70)
    print("#  减半周期分析完成")
    print(f"#  注意: 仅 {len(windows)} 个周期，结论统计功效有限")
    print("#" * 70)


def _plot_combined_summary(windows: list, output_dir: Path):
    """
    综合图: 归一化轨迹 + 减半前后收益率柱状图 + 累计收益率对比。

    Parameters
    ----------
    windows : list[dict]
        窗口数据列表
    output_dir : Path
        图片保存目录
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = ['#2980b9', '#e74c3c']
    linestyles = ['-', '--']

    # (0,0) 归一化轨迹
    ax = axes[0, 0]
    for i, w in enumerate(windows):
        days = w['df']['days_from_halving']
        ax.plot(days, w['normalized'], color=colors[i], linestyle=linestyles[i],
                linewidth=1.5, label=w['label'], alpha=0.85)
    ax.axvline(x=0, color='gold', linewidth=2, alpha=0.8, label='减半日')
    ax.axhline(y=100, color='grey', linestyle=':', alpha=0.4)
    ax.set_title('归一化价格轨迹（减半日=100）', fontsize=12)
    ax.set_xlabel('距减半日天数')
    ax.set_ylabel('归一化价格')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (0,1) 减半前后日均收益率
    ax = axes[0, 1]
    x_pos = np.arange(len(windows))
    width = 0.35
    pre_means, post_means, pre_errs, post_errs = [], [], [], []
    for w in windows:
        df_w = w['df']
        pre = df_w.loc[df_w['days_from_halving'] < 0, 'log_return'].dropna()
        post = df_w.loc[df_w['days_from_halving'] > 0, 'log_return'].dropna()
        pre_means.append(pre.mean())
        post_means.append(post.mean())
        pre_errs.append(pre.sem() * 1.96)  # 95% CI
        post_errs.append(post.sem() * 1.96)

    ax.bar(x_pos - width / 2, pre_means, width, yerr=pre_errs, label='减半前',
           color='#3498db', alpha=0.8, capsize=4, edgecolor='black', linewidth=0.5)
    ax.bar(x_pos + width / 2, post_means, width, yerr=post_errs, label='减半后',
           color='#e67e22', alpha=0.8, capsize=4, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([w['label'].split('(')[0].strip() for w in windows], fontsize=9)
    ax.axhline(y=0, color='grey', linestyle='--', alpha=0.5)
    ax.set_title('减半前后日均对数收益率（95% CI）', fontsize=12)
    ax.set_ylabel('平均对数收益率')
    ax.legend(fontsize=9)

    # (1,0) 累计收益率
    ax = axes[1, 0]
    for i, w in enumerate(windows):
        df_w = w['df']
        post = df_w.loc[df_w['days_from_halving'] >= 0].copy()
        if len(post) == 0:
            continue
        cum_ret = post['log_return'].fillna(0).cumsum()
        cum_ret_pct = (np.exp(cum_ret) - 1) * 100
        ax.plot(post['days_from_halving'], cum_ret_pct, color=colors[i],
                linewidth=1.5, label=w['label'], alpha=0.85)
    ax.axhline(y=0, color='grey', linestyle=':', alpha=0.4)
    ax.set_title('减半后累计收益率对比', fontsize=12)
    ax.set_xlabel('距减半日天数')
    ax.set_ylabel('累计收益率 (%)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}%'))

    # (1,1) 波动率对比（滚动30天）
    ax = axes[1, 1]
    for i, w in enumerate(windows):
        df_w = w['df']
        rolling_vol = df_w['log_return'].rolling(30).std() * np.sqrt(365)
        ax.plot(df_w['days_from_halving'], rolling_vol, color=colors[i],
                linewidth=1.2, label=w['label'], alpha=0.8)
    ax.axvline(x=0, color='gold', linewidth=2, alpha=0.8, label='减半日')
    ax.set_title('滚动30天年化波动率', fontsize=12)
    ax.set_xlabel('距减半日天数')
    ax.set_ylabel('年化波动率')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle('BTC 减半周期综合分析', fontsize=15, y=1.01)
    plt.tight_layout()
    fig_path = output_dir / 'halving_combined_summary.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n综合图表已保存: {fig_path}")


# --------------------------------------------------------------------------
# 可独立运行
# --------------------------------------------------------------------------
if __name__ == '__main__':
    from data_loader import load_daily
    from preprocessing import add_derived_features

    # 加载数据
    df_daily = load_daily()
    df_daily = add_derived_features(df_daily)

    run_halving_analysis(df_daily, output_dir='output/halving')
