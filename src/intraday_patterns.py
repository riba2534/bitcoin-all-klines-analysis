"""
日内模式分析模块
分析不同时间粒度下的日内交易模式，包括成交量/波动率U型曲线、时段差异等
"""

import matplotlib
matplotlib.use("Agg")
from src.font_config import configure_chinese_font
configure_chinese_font()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
from scipy.stats import f_oneway, kruskal
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import load_klines
from src.preprocessing import log_returns


def compute_intraday_volume_pattern(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    计算日内成交量U型曲线

    Args:
        df: 包含 volume 列的 DataFrame，索引为 DatetimeIndex

    Returns:
        hourly_stats: 按小时聚合的统计数据
        test_result: 统计检验结果
    """
    print("  - 计算日内成交量模式...")

    # 按小时聚合
    df_copy = df.copy()
    df_copy['hour'] = df_copy.index.hour

    hourly_stats = df_copy.groupby('hour').agg({
        'volume': ['mean', 'median', 'std'],
        'close': 'count'
    })
    hourly_stats.columns = ['volume_mean', 'volume_median', 'volume_std', 'count']

    # 检验U型曲线：开盘和收盘时段（0-2h, 22-23h）成交量是否显著高于中间时段（11-13h）
    early_hours = df_copy[df_copy['hour'].isin([0, 1, 2, 22, 23])]['volume']
    middle_hours = df_copy[df_copy['hour'].isin([11, 12, 13])]['volume']

    # Welch's t-test (不假设方差相等)
    t_stat, p_value = stats.ttest_ind(early_hours, middle_hours, equal_var=False)

    # 计算效应量 (Cohen's d)
    pooled_std = np.sqrt((early_hours.std()**2 + middle_hours.std()**2) / 2)
    effect_size = (early_hours.mean() - middle_hours.mean()) / pooled_std

    test_result = {
        'name': '日内成交量U型检验',
        'p_value': p_value,
        'effect_size': effect_size,
        'significant': p_value < 0.05,
        'early_mean': early_hours.mean(),
        'middle_mean': middle_hours.mean(),
        'description': f"开盘收盘时段成交量均值 vs 中间时段: {early_hours.mean():.2f} vs {middle_hours.mean():.2f}"
    }

    return hourly_stats, test_result


def compute_intraday_volatility_pattern(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    计算日内波动率微笑模式

    Args:
        df: 包含价格数据的 DataFrame

    Returns:
        hourly_vol: 按小时的波动率统计
        test_result: 统计检验结果
    """
    print("  - 计算日内波动率模式...")

    # 计算对数收益率
    df_copy = df.copy()
    df_copy['log_return'] = log_returns(df_copy['close'])
    df_copy['abs_return'] = df_copy['log_return'].abs()
    df_copy['hour'] = df_copy.index.hour

    # 按小时聚合波动率
    hourly_vol = df_copy.groupby('hour').agg({
        'abs_return': ['mean', 'std'],
        'log_return': lambda x: x.std()
    })
    hourly_vol.columns = ['abs_return_mean', 'abs_return_std', 'return_std']

    # 检验波动率微笑：早晚时段波动率是否高于中间时段
    early_vol = df_copy[df_copy['hour'].isin([0, 1, 2, 22, 23])]['abs_return']
    middle_vol = df_copy[df_copy['hour'].isin([11, 12, 13])]['abs_return']

    t_stat, p_value = stats.ttest_ind(early_vol, middle_vol, equal_var=False)

    pooled_std = np.sqrt((early_vol.std()**2 + middle_vol.std()**2) / 2)
    effect_size = (early_vol.mean() - middle_vol.mean()) / pooled_std

    test_result = {
        'name': '日内波动率微笑检验',
        'p_value': p_value,
        'effect_size': effect_size,
        'significant': p_value < 0.05,
        'early_mean': early_vol.mean(),
        'middle_mean': middle_vol.mean(),
        'description': f"开盘收盘时段波动率 vs 中间时段: {early_vol.mean():.6f} vs {middle_vol.mean():.6f}"
    }

    return hourly_vol, test_result


def compute_session_analysis(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    分析亚洲/欧洲/美洲时段的PnL和波动率差异

    时段定义 (UTC):
    - 亚洲: 00-08
    - 欧洲: 08-16
    - 美洲: 16-24

    Args:
        df: 价格数据

    Returns:
        session_stats: 各时段统计数据
        test_result: ANOVA/Kruskal-Wallis检验结果
    """
    print("  - 分析三大时区交易模式...")

    df_copy = df.copy()
    df_copy['log_return'] = log_returns(df_copy['close'])
    df_copy['hour'] = df_copy.index.hour

    # 定义时段
    def assign_session(hour):
        if 0 <= hour < 8:
            return 'Asia'
        elif 8 <= hour < 16:
            return 'Europe'
        else:
            return 'America'

    df_copy['session'] = df_copy['hour'].apply(assign_session)

    # 按时段聚合
    session_stats = df_copy.groupby('session').agg({
        'log_return': ['mean', 'std', 'count'],
        'volume': ['mean', 'sum']
    })
    session_stats.columns = ['return_mean', 'return_std', 'count', 'volume_mean', 'volume_sum']

    # ANOVA检验收益率差异
    asia_returns = df_copy[df_copy['session'] == 'Asia']['log_return'].dropna()
    europe_returns = df_copy[df_copy['session'] == 'Europe']['log_return'].dropna()
    america_returns = df_copy[df_copy['session'] == 'America']['log_return'].dropna()

    # 正态性检验（需要至少8个样本）
    def safe_normaltest(data):
        if len(data) >= 8:
            try:
                _, p = stats.normaltest(data)
                return p
            except:
                return 0.0  # 假设非正态
        return 0.0  # 样本不足，假设非正态

    p_asia = safe_normaltest(asia_returns)
    p_europe = safe_normaltest(europe_returns)
    p_america = safe_normaltest(america_returns)

    # 如果数据不符合正态分布，使用Kruskal-Wallis；否则使用ANOVA
    if min(p_asia, p_europe, p_america) < 0.05:
        stat, p_value = kruskal(asia_returns, europe_returns, america_returns)
        test_name = 'Kruskal-Wallis'
    else:
        stat, p_value = f_oneway(asia_returns, europe_returns, america_returns)
        test_name = 'ANOVA'

    # 计算效应量 (eta-squared)
    grand_mean = df_copy['log_return'].mean()
    ss_between = sum([
        len(asia_returns) * (asia_returns.mean() - grand_mean)**2,
        len(europe_returns) * (europe_returns.mean() - grand_mean)**2,
        len(america_returns) * (america_returns.mean() - grand_mean)**2
    ])
    ss_total = ((df_copy['log_return'] - grand_mean)**2).sum()
    eta_squared = ss_between / ss_total

    test_result = {
        'name': f'时段收益率差异检验 ({test_name})',
        'p_value': p_value,
        'effect_size': eta_squared,
        'significant': p_value < 0.05,
        'test_statistic': stat,
        'description': f"亚洲/欧洲/美洲时段收益率: {asia_returns.mean():.6f}/{europe_returns.mean():.6f}/{america_returns.mean():.6f}"
    }

    # 波动率差异检验
    asia_vol = df_copy[df_copy['session'] == 'Asia']['log_return'].abs()
    europe_vol = df_copy[df_copy['session'] == 'Europe']['log_return'].abs()
    america_vol = df_copy[df_copy['session'] == 'America']['log_return'].abs()

    stat_vol, p_value_vol = kruskal(asia_vol, europe_vol, america_vol)

    test_result_vol = {
        'name': '时段波动率差异检验 (Kruskal-Wallis)',
        'p_value': p_value_vol,
        'effect_size': None,
        'significant': p_value_vol < 0.05,
        'description': f"亚洲/欧洲/美洲时段波动率: {asia_vol.mean():.6f}/{europe_vol.mean():.6f}/{america_vol.mean():.6f}"
    }

    return session_stats, [test_result, test_result_vol]


def compute_hourly_day_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算小时 x 星期几的成交量/波动率热力图数据

    Args:
        df: 价格数据

    Returns:
        heatmap_data: 热力图数据 (hour x day_of_week)
    """
    print("  - 计算小时-星期热力图...")

    df_copy = df.copy()
    df_copy['log_return'] = log_returns(df_copy['close'])
    df_copy['abs_return'] = df_copy['log_return'].abs()
    df_copy['hour'] = df_copy.index.hour
    df_copy['day_of_week'] = df_copy.index.dayofweek

    # 按小时和星期聚合
    heatmap_volume = df_copy.pivot_table(
        values='volume',
        index='hour',
        columns='day_of_week',
        aggfunc='mean'
    )

    heatmap_volatility = df_copy.pivot_table(
        values='abs_return',
        index='hour',
        columns='day_of_week',
        aggfunc='mean'
    )

    return heatmap_volume, heatmap_volatility


def compute_intraday_autocorr(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    计算日内收益率自相关结构

    Args:
        df: 价格数据

    Returns:
        autocorr_stats: 各时段的自相关系数
        test_result: 统计检验结果
    """
    print("  - 计算日内收益率自相关...")

    df_copy = df.copy()
    df_copy['log_return'] = log_returns(df_copy['close'])
    df_copy['hour'] = df_copy.index.hour

    # 按时段计算lag-1自相关
    sessions = {
        'Asia': range(0, 8),
        'Europe': range(8, 16),
        'America': range(16, 24)
    }

    autocorr_results = []

    for session_name, hours in sessions.items():
        session_data = df_copy[df_copy['hour'].isin(hours)]['log_return'].dropna()

        if len(session_data) > 1:
            # 计算lag-1自相关
            autocorr = session_data.autocorr(lag=1)

            # Ljung-Box检验
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(session_data, lags=[1], return_df=True)

            autocorr_results.append({
                'session': session_name,
                'autocorr_lag1': autocorr,
                'lb_statistic': lb_result['lb_stat'].iloc[0],
                'lb_pvalue': lb_result['lb_pvalue'].iloc[0]
            })

    autocorr_df = pd.DataFrame(autocorr_results)

    # 检验三个时段的自相关是否显著不同
    test_result = {
        'name': '日内收益率自相关分析',
        'p_value': None,
        'effect_size': None,
        'significant': any(autocorr_df['lb_pvalue'] < 0.05),
        'description': f"各时段lag-1自相关: " + ", ".join([
            f"{row['session']}={row['autocorr_lag1']:.4f}"
            for _, row in autocorr_df.iterrows()
        ])
    }

    return autocorr_df, test_result


def compute_multi_granularity_stability(intervals: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """
    比较不同粒度下日内模式的稳定性

    Args:
        intervals: 时间粒度列表，如 ['1m', '5m', '15m', '1h']

    Returns:
        correlation_matrix: 不同粒度日内模式的相关系数矩阵
        test_result: 统计检验结果
    """
    print("  - 分析多粒度日内模式稳定性...")

    hourly_patterns = {}

    for interval in intervals:
        print(f"    加载 {interval} 数据...")
        try:
            df = load_klines(interval)
            if df is None or len(df) == 0:
                print(f"    {interval} 数据为空，跳过")
                continue

            # 计算日内成交量模式
            df_copy = df.copy()
            df_copy['hour'] = df_copy.index.hour
            hourly_volume = df_copy.groupby('hour')['volume'].mean()

            # 标准化
            hourly_volume_norm = (hourly_volume - hourly_volume.mean()) / hourly_volume.std()
            hourly_patterns[interval] = hourly_volume_norm

        except Exception as e:
            print(f"    处理 {interval} 数据时出错: {e}")
            continue

    if len(hourly_patterns) < 2:
        return pd.DataFrame(), {
            'name': '多粒度稳定性分析',
            'p_value': None,
            'effect_size': None,
            'significant': False,
            'description': '数据不足，无法进行多粒度对比'
        }

    # 计算相关系数矩阵
    pattern_df = pd.DataFrame(hourly_patterns)
    corr_matrix = pattern_df.corr()

    # 计算平均相关系数（作为稳定性指标）
    avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()

    test_result = {
        'name': '多粒度日内模式稳定性',
        'p_value': None,
        'effect_size': avg_corr,
        'significant': avg_corr > 0.7,
        'description': f"不同粒度日内模式平均相关系数: {avg_corr:.4f}"
    }

    return corr_matrix, test_result


def bootstrap_test(data1: np.ndarray, data2: np.ndarray, n_bootstrap: int = 1000) -> float:
    """
    Bootstrap检验两组数据均值差异的稳健性

    Returns:
        p_value: Bootstrap p值
    """
    observed_diff = data1.mean() - data2.mean()

    # 合并数据
    combined = np.concatenate([data1, data2])
    n1, n2 = len(data1), len(data2)

    # Bootstrap重采样
    diffs = []
    for _ in range(n_bootstrap):
        np.random.shuffle(combined)
        boot_diff = combined[:n1].mean() - combined[n1:n1+n2].mean()
        diffs.append(boot_diff)

    # 计算p值
    p_value = np.mean(np.abs(diffs) >= np.abs(observed_diff))
    return p_value


def train_test_split_temporal(df: pd.DataFrame, train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    按时间顺序分割训练集和测试集

    Args:
        df: 数据
        train_ratio: 训练集比例

    Returns:
        train_df, test_df
    """
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def validate_finding(finding: Dict, df: pd.DataFrame) -> Dict:
    """
    在测试集上验证发现的稳健性

    Args:
        finding: 包含统计检验结果的字典
        df: 完整数据

    Returns:
        更新后的finding，添加test_set_consistent和bootstrap_robust字段
    """
    train_df, test_df = train_test_split_temporal(df)

    # 根据finding的name类型进行不同的验证
    if '成交量U型' in finding['name']:
        # 在测试集上重新计算
        train_df['hour'] = train_df.index.hour
        test_df['hour'] = test_df.index.hour

        train_early = train_df[train_df['hour'].isin([0, 1, 2, 22, 23])]['volume'].values
        train_middle = train_df[train_df['hour'].isin([11, 12, 13])]['volume'].values

        test_early = test_df[test_df['hour'].isin([0, 1, 2, 22, 23])]['volume'].values
        test_middle = test_df[test_df['hour'].isin([11, 12, 13])]['volume'].values

        # 测试集检验
        _, test_p = stats.ttest_ind(test_early, test_middle, equal_var=False)
        test_set_consistent = (test_p < 0.05) == finding['significant']

        # Bootstrap检验
        bootstrap_p = bootstrap_test(train_early, train_middle, n_bootstrap=1000)
        bootstrap_robust = bootstrap_p < 0.05

    elif '波动率微笑' in finding['name']:
        train_df['log_return'] = log_returns(train_df['close'])
        train_df['abs_return'] = train_df['log_return'].abs()
        train_df['hour'] = train_df.index.hour

        test_df['log_return'] = log_returns(test_df['close'])
        test_df['abs_return'] = test_df['log_return'].abs()
        test_df['hour'] = test_df.index.hour

        train_early = train_df[train_df['hour'].isin([0, 1, 2, 22, 23])]['abs_return'].values
        train_middle = train_df[train_df['hour'].isin([11, 12, 13])]['abs_return'].values

        test_early = test_df[test_df['hour'].isin([0, 1, 2, 22, 23])]['abs_return'].values
        test_middle = test_df[test_df['hour'].isin([11, 12, 13])]['abs_return'].values

        _, test_p = stats.ttest_ind(test_early, test_middle, equal_var=False)
        test_set_consistent = (test_p < 0.05) == finding['significant']

        bootstrap_p = bootstrap_test(train_early, train_middle, n_bootstrap=1000)
        bootstrap_robust = bootstrap_p < 0.05

    else:
        # 其他类型的finding暂不验证
        test_set_consistent = None
        bootstrap_robust = None

    finding['test_set_consistent'] = test_set_consistent
    finding['bootstrap_robust'] = bootstrap_robust

    return finding


def plot_intraday_patterns(hourly_stats: pd.DataFrame, hourly_vol: pd.DataFrame,
                          output_dir: str):
    """
    绘制日内成交量和波动率U型曲线
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 成交量曲线
    ax1 = axes[0]
    hours = hourly_stats.index
    ax1.plot(hours, hourly_stats['volume_mean'], 'o-', linewidth=2, markersize=8,
             color='#2E86AB', label='平均成交量')
    ax1.fill_between(hours,
                     hourly_stats['volume_mean'] - hourly_stats['volume_std'],
                     hourly_stats['volume_mean'] + hourly_stats['volume_std'],
                     alpha=0.3, color='#2E86AB')
    ax1.set_xlabel('UTC小时', fontsize=12)
    ax1.set_ylabel('成交量', fontsize=12)
    ax1.set_title('日内成交量模式 (U型曲线)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(0, 24, 2))

    # 波动率曲线
    ax2 = axes[1]
    ax2.plot(hourly_vol.index, hourly_vol['abs_return_mean'], 's-', linewidth=2,
             markersize=8, color='#A23B72', label='平均绝对收益率')
    ax2.fill_between(hourly_vol.index,
                     hourly_vol['abs_return_mean'] - hourly_vol['abs_return_std'],
                     hourly_vol['abs_return_mean'] + hourly_vol['abs_return_std'],
                     alpha=0.3, color='#A23B72')
    ax2.set_xlabel('UTC小时', fontsize=12)
    ax2.set_ylabel('绝对收益率', fontsize=12)
    ax2.set_title('日内波动率模式 (微笑曲线)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(0, 24, 2))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/intraday_volume_pattern.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  - 已保存: intraday_volume_pattern.png")


def plot_session_heatmap(heatmap_volume: pd.DataFrame, heatmap_volatility: pd.DataFrame,
                        output_dir: str):
    """
    绘制小时 x 星期热力图
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # 成交量热力图
    ax1 = axes[0]
    sns.heatmap(heatmap_volume, cmap='YlOrRd', annot=False, fmt='.0f',
                cbar_kws={'label': '平均成交量'}, ax=ax1)
    ax1.set_xlabel('星期 (0=周一, 6=周日)', fontsize=12)
    ax1.set_ylabel('UTC小时', fontsize=12)
    ax1.set_title('日内成交量热力图 (小时 x 星期)', fontsize=14, fontweight='bold')

    # 波动率热力图
    ax2 = axes[1]
    sns.heatmap(heatmap_volatility, cmap='Purples', annot=False, fmt='.6f',
                cbar_kws={'label': '平均绝对收益率'}, ax=ax2)
    ax2.set_xlabel('星期 (0=周一, 6=周日)', fontsize=12)
    ax2.set_ylabel('UTC小时', fontsize=12)
    ax2.set_title('日内波动率热力图 (小时 x 星期)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/intraday_session_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  - 已保存: intraday_session_heatmap.png")


def plot_session_pnl(df: pd.DataFrame, output_dir: str):
    """
    绘制三大时区PnL对比箱线图
    """
    df_copy = df.copy()
    df_copy['log_return'] = log_returns(df_copy['close'])
    df_copy['hour'] = df_copy.index.hour

    def assign_session(hour):
        if 0 <= hour < 8:
            return '亚洲 (00-08 UTC)'
        elif 8 <= hour < 16:
            return '欧洲 (08-16 UTC)'
        else:
            return '美洲 (16-24 UTC)'

    df_copy['session'] = df_copy['hour'].apply(assign_session)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 收益率箱线图
    ax1 = axes[0]
    session_order = ['亚洲 (00-08 UTC)', '欧洲 (08-16 UTC)', '美洲 (16-24 UTC)']
    df_plot = df_copy[df_copy['log_return'].notna()]

    bp1 = ax1.boxplot([df_plot[df_plot['session'] == s]['log_return'] for s in session_order],
                       labels=session_order,
                       patch_artist=True,
                       showfliers=False)

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.set_ylabel('对数收益率', fontsize=12)
    ax1.set_title('三大时区收益率分布对比', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # 波动率箱线图
    ax2 = axes[1]
    df_plot['abs_return'] = df_plot['log_return'].abs()

    bp2 = ax2.boxplot([df_plot[df_plot['session'] == s]['abs_return'] for s in session_order],
                       labels=session_order,
                       patch_artist=True,
                       showfliers=False)

    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_ylabel('绝对收益率', fontsize=12)
    ax2.set_title('三大时区波动率分布对比', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/intraday_session_pnl.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  - 已保存: intraday_session_pnl.png")


def plot_stability_comparison(corr_matrix: pd.DataFrame, output_dir: str):
    """
    绘制不同粒度日内模式稳定性对比
    """
    if corr_matrix.empty:
        print("  - 跳过稳定性对比图表（数据不足）")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                center=0.5, vmin=0, vmax=1,
                square=True, linewidths=1, cbar_kws={'label': '相关系数'},
                ax=ax)

    ax.set_title('不同粒度日内成交量模式相关性', fontsize=14, fontweight='bold')
    ax.set_xlabel('时间粒度', fontsize=12)
    ax.set_ylabel('时间粒度', fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/intraday_stability.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  - 已保存: intraday_stability.png")


def run_intraday_analysis(df: pd.DataFrame = None, output_dir: str = "output/intraday") -> Dict:
    """
    执行完整的日内模式分析

    Args:
        df: 可选，如果提供则使用该数据；否则从load_klines加载
        output_dir: 输出目录

    Returns:
        结果字典，包含findings和summary
    """
    print("\n" + "="*80)
    print("开始日内模式分析")
    print("="*80)

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    findings = []

    # 1. 加载主要分析数据（使用1h数据以平衡性能和细节）
    print("\n[1/6] 加载1小时粒度数据进行主要分析...")
    if df is None:
        df_1h = load_klines('1h')
        if df_1h is None or len(df_1h) == 0:
            print("错误: 无法加载1h数据")
            return {"findings": [], "summary": {"error": "数据加载失败"}}
    else:
        df_1h = df

    print(f"  - 数据范围: {df_1h.index[0]} 到 {df_1h.index[-1]}")
    print(f"  - 数据点数: {len(df_1h):,}")

    # 2. 日内成交量U型曲线
    print("\n[2/6] 分析日内成交量U型曲线...")
    hourly_stats, volume_test = compute_intraday_volume_pattern(df_1h)
    volume_test = validate_finding(volume_test, df_1h)
    findings.append(volume_test)

    # 3. 日内波动率微笑
    print("\n[3/6] 分析日内波动率微笑模式...")
    hourly_vol, vol_test = compute_intraday_volatility_pattern(df_1h)
    vol_test = validate_finding(vol_test, df_1h)
    findings.append(vol_test)

    # 4. 时段分析
    print("\n[4/6] 分析三大时区交易特征...")
    session_stats, session_tests = compute_session_analysis(df_1h)
    findings.extend(session_tests)

    # 5. 日内自相关
    print("\n[5/6] 分析日内收益率自相关...")
    autocorr_df, autocorr_test = compute_intraday_autocorr(df_1h)
    findings.append(autocorr_test)

    # 6. 多粒度稳定性对比
    print("\n[6/6] 对比多粒度日内模式稳定性...")
    intervals = ['1m', '5m', '15m', '1h']
    corr_matrix, stability_test = compute_multi_granularity_stability(intervals)
    findings.append(stability_test)

    # 生成热力图数据
    print("\n生成热力图数据...")
    heatmap_volume, heatmap_volatility = compute_hourly_day_heatmap(df_1h)

    # 绘制图表
    print("\n生成图表...")
    plot_intraday_patterns(hourly_stats, hourly_vol, output_dir)
    plot_session_heatmap(heatmap_volume, heatmap_volatility, output_dir)
    plot_session_pnl(df_1h, output_dir)
    plot_stability_comparison(corr_matrix, output_dir)

    # 生成总结
    summary = {
        'total_findings': len(findings),
        'significant_findings': sum(1 for f in findings if f.get('significant', False)),
        'data_points': len(df_1h),
        'date_range': f"{df_1h.index[0]} 到 {df_1h.index[-1]}",
        'hourly_volume_pattern': {
            'u_shape_confirmed': volume_test['significant'],
            'early_vs_middle_ratio': volume_test.get('early_mean', 0) / volume_test.get('middle_mean', 1)
        },
        'session_analysis': {
            'best_session': session_stats['return_mean'].idxmax(),
            'most_volatile_session': session_stats['return_std'].idxmax(),
            'highest_volume_session': session_stats['volume_mean'].idxmax()
        },
        'multi_granularity_stability': {
            'average_correlation': stability_test.get('effect_size', 0),
            'stable': stability_test.get('significant', False)
        }
    }

    print("\n" + "="*80)
    print("日内模式分析完成")
    print("="*80)
    print(f"\n总发现数: {summary['total_findings']}")
    print(f"显著发现数: {summary['significant_findings']}")
    print(f"最佳交易时段: {summary['session_analysis']['best_session']}")
    print(f"最高波动时段: {summary['session_analysis']['most_volatile_session']}")
    print(f"多粒度稳定性: {'稳定' if summary['multi_granularity_stability']['stable'] else '不稳定'} "
          f"(平均相关: {summary['multi_granularity_stability']['average_correlation']:.3f})")

    return {
        'findings': findings,
        'summary': summary
    }


if __name__ == "__main__":
    # 测试运行
    result = run_intraday_analysis()

    print("\n" + "="*80)
    print("详细发现:")
    print("="*80)
    for i, finding in enumerate(result['findings'], 1):
        print(f"\n{i}. {finding['name']}")
        print(f"   显著性: {'是' if finding.get('significant') else '否'} (p={finding.get('p_value', 'N/A')})")
        if finding.get('effect_size') is not None:
            print(f"   效应量: {finding['effect_size']:.4f}")
        print(f"   描述: {finding['description']}")
        if finding.get('test_set_consistent') is not None:
            print(f"   测试集一致性: {'是' if finding['test_set_consistent'] else '否'}")
        if finding.get('bootstrap_robust') is not None:
            print(f"   Bootstrap稳健性: {'是' if finding['bootstrap_robust'] else '否'}")
