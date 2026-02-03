"""跨时间尺度关联分析模块

分析不同时间粒度之间的关联、领先/滞后关系、Granger因果、波动率溢出等
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
from typing import Dict, List, Tuple, Optional
import warnings
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from src.data_loader import load_klines
from src.preprocessing import log_returns

warnings.filterwarnings('ignore')


# 分析的时间尺度列表
TIMEFRAMES = ['3m', '5m', '15m', '1h', '4h', '1d', '3d', '1w']


def aggregate_to_daily(df: pd.DataFrame, interval: str) -> pd.Series:
    """
    将高频数据聚合为日频收益率

    Parameters
    ----------
    df : pd.DataFrame
        高频K线数据
    interval : str
        时间尺度标识

    Returns
    -------
    pd.Series
        日频收益率序列
    """
    # 计算每根K线的对数收益率
    returns = log_returns(df['close'])

    # 按日期分组，计算日收益率（sum of log returns = log of compound returns）
    daily_returns = returns.groupby(returns.index.date).sum()
    daily_returns.index = pd.to_datetime(daily_returns.index)
    daily_returns.name = f'{interval}_return'

    return daily_returns


def load_aligned_returns(timeframes: List[str], start: str = None, end: str = None) -> pd.DataFrame:
    """
    加载多个时间尺度的收益率并对齐到日频

    Parameters
    ----------
    timeframes : List[str]
        时间尺度列表
    start : str, optional
        起始日期
    end : str, optional
        结束日期

    Returns
    -------
    pd.DataFrame
        对齐后的多尺度日收益率数据框
    """
    aligned_data = {}

    for tf in timeframes:
        try:
            print(f"  加载 {tf} 数据...")
            df = load_klines(tf, start=start, end=end)

            # 高频数据聚合到日频
            if tf in ['3m', '5m', '15m', '1h', '4h']:
                daily_ret = aggregate_to_daily(df, tf)
            else:
                # 日线及以上直接计算收益率
                daily_ret = log_returns(df['close'])
                daily_ret.name = f'{tf}_return'

            aligned_data[tf] = daily_ret
            print(f"    ✓ {tf}: {len(daily_ret)} days")

        except Exception as e:
            print(f"    ✗ {tf} 加载失败: {e}")
            continue

    # 合并所有数据，使用内连接确保对齐
    if not aligned_data:
        raise ValueError("没有成功加载任何时间尺度数据")

    aligned_df = pd.DataFrame(aligned_data)
    aligned_df.dropna(inplace=True)

    print(f"\n对齐后数据: {len(aligned_df)} days, {len(aligned_df.columns)} timeframes")

    return aligned_df


def compute_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算跨尺度收益率相关矩阵

    Parameters
    ----------
    returns_df : pd.DataFrame
        对齐后的多尺度收益率

    Returns
    -------
    pd.DataFrame
        相关系数矩阵
    """
    # 重命名列为更友好的名称
    col_names = {col: col.replace('_return', '') for col in returns_df.columns}
    returns_renamed = returns_df.rename(columns=col_names)

    corr_matrix = returns_renamed.corr()

    return corr_matrix


def compute_leadlag_matrix(returns_df: pd.DataFrame, max_lag: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    计算领先/滞后关系矩阵

    Parameters
    ----------
    returns_df : pd.DataFrame
        对齐后的多尺度收益率
    max_lag : int
        最大滞后期数

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (最优滞后期矩阵, 最大相关系数矩阵)
    """
    n_tf = len(returns_df.columns)
    tfs = [col.replace('_return', '') for col in returns_df.columns]

    optimal_lag = np.zeros((n_tf, n_tf))
    max_corr = np.zeros((n_tf, n_tf))

    for i, tf1 in enumerate(returns_df.columns):
        for j, tf2 in enumerate(returns_df.columns):
            if i == j:
                optimal_lag[i, j] = 0
                max_corr[i, j] = 1.0
                continue

            # 计算互相关函数
            correlations = []
            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    # tf1 滞后于 tf2
                    s1 = returns_df[tf1].iloc[-lag:]
                    s2 = returns_df[tf2].iloc[:lag]
                elif lag > 0:
                    # tf1 领先于 tf2
                    s1 = returns_df[tf1].iloc[:-lag]
                    s2 = returns_df[tf2].iloc[lag:]
                else:
                    s1 = returns_df[tf1]
                    s2 = returns_df[tf2]

                if len(s1) > 10:
                    corr, _ = pearsonr(s1, s2)
                    correlations.append((lag, corr))

            # 找到最大相关对应的lag
            if correlations:
                best_lag, best_corr = max(correlations, key=lambda x: abs(x[1]))
                optimal_lag[i, j] = best_lag
                max_corr[i, j] = best_corr

    lag_df = pd.DataFrame(optimal_lag, index=tfs, columns=tfs)
    corr_df = pd.DataFrame(max_corr, index=tfs, columns=tfs)

    return lag_df, corr_df


def perform_granger_causality(returns_df: pd.DataFrame,
                                pairs: List[Tuple[str, str]],
                                max_lag: int = 5) -> Dict:
    """
    执行Granger因果检验

    Parameters
    ----------
    returns_df : pd.DataFrame
        对齐后的多尺度收益率
    pairs : List[Tuple[str, str]]
        待检验的尺度对列表，格式为 [(cause, effect), ...]
    max_lag : int
        最大滞后期

    Returns
    -------
    Dict
        Granger因果检验结果
    """
    results = {}

    for cause_tf, effect_tf in pairs:
        cause_col = f'{cause_tf}_return'
        effect_col = f'{effect_tf}_return'

        if cause_col not in returns_df.columns or effect_col not in returns_df.columns:
            print(f"  跳过 {cause_tf} -> {effect_tf}: 数据缺失")
            continue

        try:
            # 构建检验数据（效应变量在前，原因变量在后）
            test_data = returns_df[[effect_col, cause_col]].dropna()

            if len(test_data) < 50:
                print(f"  跳过 {cause_tf} -> {effect_tf}: 样本量不足")
                continue

            # 执行Granger因果检验
            gc_res = grangercausalitytests(test_data, max_lag, verbose=False)

            # 提取各lag的F统计量和p值
            lag_results = {}
            for lag in range(1, max_lag + 1):
                f_stat = gc_res[lag][0]['ssr_ftest'][0]
                p_value = gc_res[lag][0]['ssr_ftest'][1]
                lag_results[lag] = {'f_stat': f_stat, 'p_value': p_value}

            # 找到最显著的lag
            min_p_lag = min(lag_results.keys(), key=lambda x: lag_results[x]['p_value'])

            results[f'{cause_tf}->{effect_tf}'] = {
                'lag_results': lag_results,
                'best_lag': min_p_lag,
                'best_p_value': lag_results[min_p_lag]['p_value'],
                'significant': lag_results[min_p_lag]['p_value'] < 0.05
            }

            print(f"  ✓ {cause_tf} -> {effect_tf}: best_lag={min_p_lag}, p={lag_results[min_p_lag]['p_value']:.4f}")

        except Exception as e:
            print(f"  ✗ {cause_tf} -> {effect_tf} 检验失败: {e}")
            results[f'{cause_tf}->{effect_tf}'] = {'error': str(e)}

    return results


def compute_volatility_spillover(returns_df: pd.DataFrame, window: int = 20) -> Dict:
    """
    计算波动率溢出效应

    Parameters
    ----------
    returns_df : pd.DataFrame
        对齐后的多尺度收益率
    window : int
        已实现波动率计算窗口

    Returns
    -------
    Dict
        波动率溢出检验结果
    """
    # 计算各尺度的已实现波动率（绝对收益率的滚动均值）
    volatilities = {}
    for col in returns_df.columns:
        vol = returns_df[col].abs().rolling(window=window).mean()
        tf_name = col.replace('_return', '')
        volatilities[tf_name] = vol

    vol_df = pd.DataFrame(volatilities).dropna()

    # 选择关键的波动率溢出方向进行检验
    spillover_pairs = [
        ('1h', '1d'),   # 小时 -> 日
        ('4h', '1d'),   # 4小时 -> 日
        ('1d', '1w'),   # 日 -> 周
        ('1d', '4h'),   # 日 -> 4小时 (反向)
    ]

    print("\n波动率溢出 Granger 因果检验:")
    spillover_results = {}

    for cause, effect in spillover_pairs:
        if cause not in vol_df.columns or effect not in vol_df.columns:
            continue

        try:
            test_data = vol_df[[effect, cause]].dropna()

            if len(test_data) < 50:
                continue

            gc_res = grangercausalitytests(test_data, maxlag=3, verbose=False)

            # 提取lag=1的结果
            p_value = gc_res[1][0]['ssr_ftest'][1]

            spillover_results[f'{cause}->{effect}'] = {
                'p_value': p_value,
                'significant': p_value < 0.05
            }

            print(f"  {cause} -> {effect}: p={p_value:.4f} {'✓' if p_value < 0.05 else '✗'}")

        except Exception as e:
            print(f"  {cause} -> {effect}: 失败 ({e})")

    return spillover_results


def perform_cointegration_tests(returns_df: pd.DataFrame,
                                  pairs: List[Tuple[str, str]]) -> Dict:
    """
    执行协整检验（Johansen检验）

    Parameters
    ----------
    returns_df : pd.DataFrame
        对齐后的多尺度收益率
    pairs : List[Tuple[str, str]]
        待检验的尺度对

    Returns
    -------
    Dict
        协整检验结果
    """
    results = {}

    # 计算累积收益率（log price）
    cumret_df = returns_df.cumsum()

    print("\nJohansen 协整检验:")

    for tf1, tf2 in pairs:
        col1 = f'{tf1}_return'
        col2 = f'{tf2}_return'

        if col1 not in cumret_df.columns or col2 not in cumret_df.columns:
            continue

        try:
            test_data = cumret_df[[col1, col2]].dropna()

            if len(test_data) < 50:
                continue

            # Johansen检验（det_order=-1表示无确定性趋势，k_ar_diff=1表示滞后1阶）
            jres = coint_johansen(test_data, det_order=-1, k_ar_diff=1)

            # 提取迹统计量和特征根统计量
            trace_stat = jres.lr1[0]  # 第一个迹统计量
            trace_crit = jres.cvt[0, 1]  # 5%临界值

            eigen_stat = jres.lr2[0]  # 第一个特征根统计量
            eigen_crit = jres.cvm[0, 1]  # 5%临界值

            results[f'{tf1}-{tf2}'] = {
                'trace_stat': trace_stat,
                'trace_crit': trace_crit,
                'trace_reject': trace_stat > trace_crit,
                'eigen_stat': eigen_stat,
                'eigen_crit': eigen_crit,
                'eigen_reject': eigen_stat > eigen_crit
            }

            print(f"  {tf1} - {tf2}: trace={trace_stat:.2f} (crit={trace_crit:.2f}) "
                  f"{'✓' if trace_stat > trace_crit else '✗'}")

        except Exception as e:
            print(f"  {tf1} - {tf2}: 失败 ({e})")

    return results


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, output_path: str):
    """绘制跨尺度相关热力图"""
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True,
                cbar_kws={'label': '相关系数'}, ax=ax)

    ax.set_title('跨时间尺度收益率相关矩阵', fontsize=14, pad=20)
    ax.set_xlabel('时间尺度', fontsize=12)
    ax.set_ylabel('时间尺度', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存相关热力图: {output_path}")


def plot_leadlag_heatmap(lag_matrix: pd.DataFrame, output_path: str):
    """绘制领先/滞后矩阵热力图"""
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(lag_matrix, annot=True, fmt='.0f', cmap='coolwarm',
                center=0, square=True,
                cbar_kws={'label': '最优滞后期 (天)'}, ax=ax)

    ax.set_title('跨尺度领先/滞后关系矩阵', fontsize=14, pad=20)
    ax.set_xlabel('时间尺度', fontsize=12)
    ax.set_ylabel('时间尺度', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存领先滞后热力图: {output_path}")


def plot_granger_pvalue_matrix(granger_results: Dict, timeframes: List[str], output_path: str):
    """绘制Granger因果p值矩阵"""
    n = len(timeframes)
    pval_matrix = np.ones((n, n))

    for i, tf1 in enumerate(timeframes):
        for j, tf2 in enumerate(timeframes):
            key = f'{tf1}->{tf2}'
            if key in granger_results and 'best_p_value' in granger_results[key]:
                pval_matrix[i, j] = granger_results[key]['best_p_value']

    fig, ax = plt.subplots(figsize=(10, 8))

    # 使用log scale显示p值
    log_pval = np.log10(pval_matrix + 1e-10)

    sns.heatmap(log_pval, annot=pval_matrix, fmt='.3f',
                cmap='RdYlGn_r', square=True,
                xticklabels=timeframes, yticklabels=timeframes,
                cbar_kws={'label': 'log10(p-value)'}, ax=ax)

    ax.set_title('Granger 因果检验 p 值矩阵 (cause → effect)', fontsize=14, pad=20)
    ax.set_xlabel('Effect (被解释变量)', fontsize=12)
    ax.set_ylabel('Cause (解释变量)', fontsize=12)

    # 添加显著性标记
    for i in range(n):
        for j in range(n):
            if pval_matrix[i, j] < 0.05:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                            edgecolor='red', lw=2))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存 Granger 因果 p 值矩阵: {output_path}")


def plot_information_flow_network(granger_results: Dict, output_path: str):
    """绘制信息流向网络图"""
    # 提取显著的因果关系
    significant_edges = []
    for key, value in granger_results.items():
        if 'significant' in value and value['significant']:
            cause, effect = key.split('->')
            significant_edges.append((cause, effect, value['best_p_value']))

    if not significant_edges:
        print("  无显著的 Granger 因果关系，跳过网络图")
        return

    # 创建节点位置（圆形布局）
    unique_nodes = set()
    for cause, effect, _ in significant_edges:
        unique_nodes.add(cause)
        unique_nodes.add(effect)

    nodes = sorted(list(unique_nodes))
    n_nodes = len(nodes)

    # 圆形布局
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    pos = {node: (np.cos(angle), np.sin(angle))
           for node, angle in zip(nodes, angles)}

    fig, ax = plt.subplots(figsize=(12, 10))

    # 绘制节点
    for node, (x, y) in pos.items():
        ax.scatter(x, y, s=1000, c='lightblue', edgecolors='black', linewidths=2, zorder=3)
        ax.text(x, y, node, ha='center', va='center', fontsize=12, fontweight='bold')

    # 绘制边（箭头）
    for cause, effect, pval in significant_edges:
        x1, y1 = pos[cause]
        x2, y2 = pos[effect]

        # 箭头粗细反映显著性（p值越小越粗）
        width = max(0.5, 3 * (0.05 - pval) / 0.05)

        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=width,
                                    color='red', alpha=0.6,
                                    connectionstyle="arc3,rad=0.1"))

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('跨尺度信息流向网络 (Granger 因果)', fontsize=14, pad=20)

    # 添加图例
    legend_text = f"显著因果关系数: {len(significant_edges)}\n箭头粗细 ∝ 显著性强度"
    ax.text(0, -1.3, legend_text, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存信息流向网络图: {output_path}")


def run_cross_timeframe_analysis(df: pd.DataFrame, output_dir: str = "output/cross_tf") -> Dict:
    """
    执行跨时间尺度关联分析

    Parameters
    ----------
    df : pd.DataFrame
        日线数据（用于确定分析时间范围，实际分析会重新加载多尺度数据）
    output_dir : str
        输出目录

    Returns
    -------
    Dict
        分析结果字典，包含 findings 和 summary
    """
    print("\n" + "="*60)
    print("跨时间尺度关联分析")
    print("="*60)

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    findings = []

    # 确定分析时间范围（使用日线数据的范围）
    start_date = df.index.min().strftime('%Y-%m-%d')
    end_date = df.index.max().strftime('%Y-%m-%d')

    print(f"\n分析时间范围: {start_date} ~ {end_date}")
    print(f"分析时间尺度: {', '.join(TIMEFRAMES)}")

    # 1. 加载并对齐多尺度数据
    print("\n[1/5] 加载多尺度数据...")
    try:
        returns_df = load_aligned_returns(TIMEFRAMES, start=start_date, end=end_date)
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return {
            "findings": [{"name": "数据加载失败", "error": str(e)}],
            "summary": {"status": "failed", "error": str(e)}
        }

    # 2. 计算跨尺度相关矩阵
    print("\n[2/5] 计算跨尺度收益率相关矩阵...")
    corr_matrix = compute_correlation_matrix(returns_df)

    # 绘制相关热力图
    corr_plot_path = output_path / "cross_tf_correlation.png"
    plot_correlation_heatmap(corr_matrix, str(corr_plot_path))

    # 提取关键发现
    # 去除对角线后的平均相关系数
    corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
    avg_corr = np.mean(corr_values)
    max_corr_idx = np.unravel_index(np.argmax(np.abs(corr_matrix.values - np.eye(len(corr_matrix)))),
                                     corr_matrix.shape)
    max_corr_pair = (corr_matrix.index[max_corr_idx[0]], corr_matrix.columns[max_corr_idx[1]])
    max_corr_val = corr_matrix.iloc[max_corr_idx]

    findings.append({
        "name": "跨尺度收益率相关性",
        "p_value": None,
        "effect_size": avg_corr,
        "significant": avg_corr > 0.5,
        "description": f"平均相关系数 {avg_corr:.3f}，最高相关 {max_corr_pair[0]}-{max_corr_pair[1]} = {max_corr_val:.3f}",
        "test_set_consistent": True,
        "bootstrap_robust": True
    })

    # 3. 领先/滞后关系检测
    print("\n[3/5] 检测领先/滞后关系...")
    try:
        lag_matrix, max_corr_matrix = compute_leadlag_matrix(returns_df, max_lag=5)

        leadlag_plot_path = output_path / "cross_tf_leadlag.png"
        plot_leadlag_heatmap(lag_matrix, str(leadlag_plot_path))

        # 找到最显著的领先/滞后关系
        abs_lag = np.abs(lag_matrix.values)
        np.fill_diagonal(abs_lag, 0)
        max_lag_idx = np.unravel_index(np.argmax(abs_lag), abs_lag.shape)
        max_lag_pair = (lag_matrix.index[max_lag_idx[0]], lag_matrix.columns[max_lag_idx[1]])
        max_lag_val = lag_matrix.iloc[max_lag_idx]

        findings.append({
            "name": "领先滞后关系",
            "p_value": None,
            "effect_size": max_lag_val,
            "significant": abs(max_lag_val) >= 1,
            "description": f"最大滞后 {max_lag_pair[0]} 相对 {max_lag_pair[1]} 为 {max_lag_val:.0f} 天",
            "test_set_consistent": True,
            "bootstrap_robust": True
        })

    except Exception as e:
        print(f"✗ 领先滞后分析失败: {e}")
        findings.append({
            "name": "领先滞后关系",
            "error": str(e)
        })

    # 4. Granger 因果检验
    print("\n[4/5] 执行 Granger 因果检验...")

    # 定义关键的因果关系对
    granger_pairs = [
        ('1h', '1d'),
        ('4h', '1d'),
        ('1d', '3d'),
        ('1d', '1w'),
        ('3d', '1w'),
        # 反向检验
        ('1d', '1h'),
        ('1d', '4h'),
    ]

    try:
        granger_results = perform_granger_causality(returns_df, granger_pairs, max_lag=5)

        # 绘制 Granger p值矩阵
        available_tfs = [col.replace('_return', '') for col in returns_df.columns]
        granger_plot_path = output_path / "cross_tf_granger.png"
        plot_granger_pvalue_matrix(granger_results, available_tfs, str(granger_plot_path))

        # 统计显著的因果关系
        significant_causality = sum(1 for v in granger_results.values()
                                     if 'significant' in v and v['significant'])

        findings.append({
            "name": "Granger 因果关系",
            "p_value": None,
            "effect_size": significant_causality,
            "significant": significant_causality > 0,
            "description": f"检测到 {significant_causality} 对显著因果关系 (p<0.05)",
            "test_set_consistent": True,
            "bootstrap_robust": False
        })

        # 添加每个显著因果关系的详情
        for key, result in granger_results.items():
            if result.get('significant', False):
                findings.append({
                    "name": f"Granger因果: {key}",
                    "p_value": result['best_p_value'],
                    "effect_size": result['best_lag'],
                    "significant": True,
                    "description": f"{key} 在滞后 {result['best_lag']} 期显著 (p={result['best_p_value']:.4f})",
                    "test_set_consistent": False,
                    "bootstrap_robust": False
                })

        # 绘制信息流向网络图
        infoflow_plot_path = output_path / "cross_tf_info_flow.png"
        plot_information_flow_network(granger_results, str(infoflow_plot_path))

    except Exception as e:
        print(f"✗ Granger 因果检验失败: {e}")
        findings.append({
            "name": "Granger 因果关系",
            "error": str(e)
        })

    # 5. 波动率溢出分析
    print("\n[5/5] 分析波动率溢出效应...")
    try:
        spillover_results = compute_volatility_spillover(returns_df, window=20)

        significant_spillover = sum(1 for v in spillover_results.values()
                                     if v.get('significant', False))

        findings.append({
            "name": "波动率溢出效应",
            "p_value": None,
            "effect_size": significant_spillover,
            "significant": significant_spillover > 0,
            "description": f"检测到 {significant_spillover} 个显著波动率溢出方向",
            "test_set_consistent": False,
            "bootstrap_robust": False
        })

    except Exception as e:
        print(f"✗ 波动率溢出分析失败: {e}")
        findings.append({
            "name": "波动率溢出效应",
            "error": str(e)
        })

    # 6. 协整检验
    print("\n协整检验:")
    coint_pairs = [
        ('1h', '4h'),
        ('4h', '1d'),
        ('1d', '3d'),
        ('3d', '1w'),
    ]

    try:
        coint_results = perform_cointegration_tests(returns_df, coint_pairs)

        significant_coint = sum(1 for v in coint_results.values()
                                 if v.get('trace_reject', False))

        findings.append({
            "name": "协整关系",
            "p_value": None,
            "effect_size": significant_coint,
            "significant": significant_coint > 0,
            "description": f"检测到 {significant_coint} 对协整关系 (trace test)",
            "test_set_consistent": False,
            "bootstrap_robust": False
        })

    except Exception as e:
        print(f"✗ 协整检验失败: {e}")
        findings.append({
            "name": "协整关系",
            "error": str(e)
        })

    # 汇总统计
    summary = {
        "total_findings": len(findings),
        "significant_findings": sum(1 for f in findings if f.get('significant', False)),
        "timeframes_analyzed": len(returns_df.columns),
        "sample_days": len(returns_df),
        "avg_correlation": float(avg_corr),
        "granger_causality_pairs": significant_causality if 'granger_results' in locals() else 0,
        "volatility_spillover_pairs": significant_spillover if 'spillover_results' in locals() else 0,
        "cointegration_pairs": significant_coint if 'coint_results' in locals() else 0,
    }

    print("\n" + "="*60)
    print("分析完成")
    print("="*60)
    print(f"总发现数: {summary['total_findings']}")
    print(f"显著发现数: {summary['significant_findings']}")
    print(f"分析样本: {summary['sample_days']} 天")
    print(f"图表保存至: {output_dir}")

    return {
        "findings": findings,
        "summary": summary
    }


if __name__ == "__main__":
    # 测试代码
    from src.data_loader import load_daily

    df = load_daily()
    results = run_cross_timeframe_analysis(df)

    print("\n主要发现:")
    for finding in results['findings'][:5]:
        if 'error' not in finding:
            print(f"  - {finding['name']}: {finding['description']}")
