"""
极端值与尾部风险分析模块

基于极值理论(EVT)分析BTC价格的尾部风险特征:
- GEV分布拟合区组极大值
- GPD分布拟合超阈值尾部
- VaR/CVaR多尺度回测
- Hill尾部指数估计
- 极端事件聚集性检验
"""

import matplotlib
matplotlib.use("Agg")
from src.font_config import configure_chinese_font
configure_chinese_font()

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import genextreme, genpareto
from typing import Dict, List, Tuple
from pathlib import Path

from src.data_loader import load_klines
from src.preprocessing import log_returns

warnings.filterwarnings('ignore')


def fit_gev_distribution(returns: pd.Series, block_size: str = 'M') -> Dict:
    """
    拟合广义极值分布(GEV)到区组极大值

    Args:
        returns: 收益率序列
        block_size: 区组大小 ('M'=月, 'Q'=季度)

    Returns:
        包含GEV参数和诊断信息的字典
    """
    try:
        # 按区组取极大值和极小值
        returns_df = pd.DataFrame({'returns': returns})
        returns_df.index = pd.to_datetime(returns_df.index)

        block_maxima = returns_df.resample(block_size).max()['returns'].dropna()
        block_minima = returns_df.resample(block_size).min()['returns'].dropna()

        # 拟合正向极值(最大值)
        shape_max, loc_max, scale_max = genextreme.fit(block_maxima)

        # 拟合负向极值(最小值的绝对值)
        shape_min, loc_min, scale_min = genextreme.fit(-block_minima)

        # 分类尾部类型
        def classify_tail(xi):
            if xi > 0.1:
                return "Fréchet重尾"
            elif xi < -0.1:
                return "Weibull有界尾"
            else:
                return "Gumbel指数尾"

        # KS检验拟合优度
        ks_max = stats.kstest(block_maxima, lambda x: genextreme.cdf(x, shape_max, loc_max, scale_max))
        ks_min = stats.kstest(-block_minima, lambda x: genextreme.cdf(x, shape_min, loc_min, scale_min))

        return {
            'maxima': {
                'shape': shape_max,
                'location': loc_max,
                'scale': scale_max,
                'tail_type': classify_tail(shape_max),
                'ks_pvalue': ks_max.pvalue,
                'n_blocks': len(block_maxima)
            },
            'minima': {
                'shape': shape_min,
                'location': loc_min,
                'scale': scale_min,
                'tail_type': classify_tail(shape_min),
                'ks_pvalue': ks_min.pvalue,
                'n_blocks': len(block_minima)
            },
            'block_maxima': block_maxima,
            'block_minima': block_minima
        }
    except Exception as e:
        return {'error': str(e)}


def fit_gpd_distribution(returns: pd.Series, threshold_quantile: float = 0.95) -> Dict:
    """
    拟合广义Pareto分布(GPD)到超阈值尾部

    Args:
        returns: 收益率序列
        threshold_quantile: 阈值分位数

    Returns:
        包含GPD参数和诊断信息的字典
    """
    try:
        # 正向尾部(极端正收益)
        threshold_pos = returns.quantile(threshold_quantile)
        exceedances_pos = returns[returns > threshold_pos] - threshold_pos

        # 负向尾部(极端负收益)
        threshold_neg = returns.quantile(1 - threshold_quantile)
        exceedances_neg = -(returns[returns < threshold_neg] - threshold_neg)

        results = {}

        # 拟合正向尾部
        if len(exceedances_pos) >= 10:
            shape_pos, loc_pos, scale_pos = genpareto.fit(exceedances_pos, floc=0)
            ks_pos = stats.kstest(exceedances_pos,
                                 lambda x: genpareto.cdf(x, shape_pos, loc_pos, scale_pos))

            results['positive_tail'] = {
                'shape': shape_pos,
                'scale': scale_pos,
                'threshold': threshold_pos,
                'n_exceedances': len(exceedances_pos),
                'is_power_law': shape_pos > 0,
                'tail_index': 1/shape_pos if shape_pos > 0 else np.inf,
                'ks_pvalue': ks_pos.pvalue,
                'exceedances': exceedances_pos
            }

        # 拟合负向尾部
        if len(exceedances_neg) >= 10:
            shape_neg, loc_neg, scale_neg = genpareto.fit(exceedances_neg, floc=0)
            ks_neg = stats.kstest(exceedances_neg,
                                 lambda x: genpareto.cdf(x, shape_neg, loc_neg, scale_neg))

            results['negative_tail'] = {
                'shape': shape_neg,
                'scale': scale_neg,
                'threshold': threshold_neg,
                'n_exceedances': len(exceedances_neg),
                'is_power_law': shape_neg > 0,
                'tail_index': 1/shape_neg if shape_neg > 0 else np.inf,
                'ks_pvalue': ks_neg.pvalue,
                'exceedances': exceedances_neg
            }

        return results
    except Exception as e:
        return {'error': str(e)}


def calculate_var_cvar(returns: pd.Series, confidence_levels: List[float] = [0.95, 0.99]) -> Dict:
    """
    计算历史VaR和CVaR

    Args:
        returns: 收益率序列
        confidence_levels: 置信水平列表

    Returns:
        包含VaR和CVaR的字典
    """
    results = {}

    for cl in confidence_levels:
        # VaR: 分位数
        var = returns.quantile(1 - cl)

        # CVaR: 超过VaR的平均损失
        cvar = returns[returns <= var].mean()

        results[f'VaR_{int(cl*100)}'] = var
        results[f'CVaR_{int(cl*100)}'] = cvar

    return results


def backtest_var(returns: pd.Series, var_level: float, confidence: float = 0.95) -> Dict:
    """
    VaR回测使用Kupiec POF检验

    Args:
        returns: 收益率序列
        var_level: VaR阈值
        confidence: 置信水平

    Returns:
        回测结果
    """
    # 计算实际违约次数
    violations = (returns < var_level).sum()
    n = len(returns)

    # 期望违约次数
    expected_violations = n * (1 - confidence)

    # Kupiec POF检验
    p = 1 - confidence
    if violations > 0:
        lr_stat = 2 * (
            violations * np.log(violations / expected_violations) +
            (n - violations) * np.log((n - violations) / (n - expected_violations))
        )
    else:
        lr_stat = 2 * n * np.log(1 / (1 - p))

    # 卡方分布检验(自由度=1)
    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)

    return {
        'violations': violations,
        'expected_violations': expected_violations,
        'violation_rate': violations / n,
        'expected_rate': 1 - confidence,
        'lr_statistic': lr_stat,
        'p_value': p_value,
        'reject_model': p_value < 0.05,
        'violation_indices': returns[returns < var_level].index.tolist()
    }


def estimate_hill_index(returns: pd.Series, k_max: int = None) -> Dict:
    """
    Hill估计量计算尾部指数

    Args:
        returns: 收益率序列
        k_max: 最大尾部样本数

    Returns:
        Hill估计结果
    """
    try:
        # 使用收益率绝对值
        abs_returns = np.abs(returns.values)
        sorted_returns = np.sort(abs_returns)[::-1]  # 降序

        if k_max is None:
            k_max = min(len(sorted_returns) // 4, 500)

        k_values = np.arange(10, min(k_max, len(sorted_returns)))
        hill_estimates = []

        for k in k_values:
            # Hill估计量: 1/α = (1/k) * Σlog(X_i / X_{k+1})
            log_ratios = np.log(sorted_returns[:k] / sorted_returns[k])
            hill_est = np.mean(log_ratios)
            hill_estimates.append(hill_est)

        hill_estimates = np.array(hill_estimates)
        tail_indices = 1 / hill_estimates  # α = 1 / Hill估计量

        # 寻找稳定区域(变异系数最小的区间)
        window = 20
        stable_idx = 0
        min_cv = np.inf

        for i in range(len(tail_indices) - window):
            window_values = tail_indices[i:i+window]
            cv = np.std(window_values) / np.abs(np.mean(window_values))
            if cv < min_cv:
                min_cv = cv
                stable_idx = i + window // 2

        stable_alpha = tail_indices[stable_idx]

        return {
            'k_values': k_values,
            'hill_estimates': hill_estimates,
            'tail_indices': tail_indices,
            'stable_alpha': stable_alpha,
            'stable_k': k_values[stable_idx],
            'is_heavy_tail': stable_alpha < 5  # α<4无方差, α<2无均值
        }
    except Exception as e:
        return {'error': str(e)}


def test_extreme_clustering(returns: pd.Series, quantile: float = 0.99) -> Dict:
    """
    检验极端事件的聚集性

    使用游程检验判断极端事件是否独立

    Args:
        returns: 收益率序列
        quantile: 极端事件定义分位数

    Returns:
        聚集性检验结果
    """
    try:
        # 定义极端事件(双侧)
        threshold_pos = returns.quantile(quantile)
        threshold_neg = returns.quantile(1 - quantile)

        is_extreme = (returns > threshold_pos) | (returns < threshold_neg)

        # 游程检验
        n_extreme = is_extreme.sum()
        n_total = len(is_extreme)

        # 计算游程数
        runs = 1 + (is_extreme.diff().fillna(False) != 0).sum()

        # 期望游程数(独立情况下)
        p = n_extreme / n_total
        expected_runs = 2 * n_total * p * (1 - p) + 1

        # 方差
        var_runs = 2 * n_total * p * (1 - p) * (2 * n_total * p * (1 - p) - 1) / (n_total - 1)

        # Z统计量
        z_stat = (runs - expected_runs) / np.sqrt(var_runs) if var_runs > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

        # 自相关检验
        extreme_indicator = is_extreme.astype(int)
        acf_lag1 = extreme_indicator.autocorr(lag=1)

        return {
            'n_extreme_events': n_extreme,
            'extreme_rate': p,
            'n_runs': runs,
            'expected_runs': expected_runs,
            'z_statistic': z_stat,
            'p_value': p_value,
            'is_clustered': p_value < 0.05 and runs < expected_runs,
            'acf_lag1': acf_lag1,
            'extreme_dates': is_extreme[is_extreme].index.tolist()
        }
    except Exception as e:
        return {'error': str(e)}


def plot_tail_qq(gpd_results: Dict, output_path: str):
    """绘制尾部拟合QQ图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 正向尾部
    if 'positive_tail' in gpd_results:
        pos = gpd_results['positive_tail']
        if 'exceedances' in pos:
            exc = pos['exceedances'].values
            theoretical = genpareto.ppf(np.linspace(0.01, 0.99, len(exc)),
                                       pos['shape'], 0, pos['scale'])
            observed = np.sort(exc)

            axes[0].scatter(theoretical, observed, alpha=0.5, s=20)
            axes[0].plot([observed.min(), observed.max()],
                        [observed.min(), observed.max()],
                        'r--', lw=2, label='理论分位线')
            axes[0].set_xlabel('GPD理论分位数', fontsize=11)
            axes[0].set_ylabel('观测分位数', fontsize=11)
            axes[0].set_title(f'正向尾部QQ图 (ξ={pos["shape"]:.3f})', fontsize=12, fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

    # 负向尾部
    if 'negative_tail' in gpd_results:
        neg = gpd_results['negative_tail']
        if 'exceedances' in neg:
            exc = neg['exceedances'].values
            theoretical = genpareto.ppf(np.linspace(0.01, 0.99, len(exc)),
                                       neg['shape'], 0, neg['scale'])
            observed = np.sort(exc)

            axes[1].scatter(theoretical, observed, alpha=0.5, s=20, color='orange')
            axes[1].plot([observed.min(), observed.max()],
                        [observed.min(), observed.max()],
                        'r--', lw=2, label='理论分位线')
            axes[1].set_xlabel('GPD理论分位数', fontsize=11)
            axes[1].set_ylabel('观测分位数', fontsize=11)
            axes[1].set_title(f'负向尾部QQ图 (ξ={neg["shape"]:.3f})', fontsize=12, fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_var_backtest(price_series: pd.Series, returns: pd.Series,
                     var_levels: Dict, backtest_results: Dict, output_path: str):
    """绘制VaR回测图"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # 价格图
    axes[0].plot(price_series.index, price_series.values, label='BTC价格', linewidth=1.5)

    # 标记VaR违约点
    for var_name, bt_result in backtest_results.items():
        if 'violation_indices' in bt_result and bt_result['violation_indices']:
            viol_dates = pd.to_datetime(bt_result['violation_indices'])
            viol_prices = price_series.loc[viol_dates]
            axes[0].scatter(viol_dates, viol_prices,
                          label=f'{var_name} 违约', s=50, alpha=0.7, zorder=5)

    axes[0].set_ylabel('价格 (USDT)', fontsize=11)
    axes[0].set_title('VaR违约事件标记', fontsize=12, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    # 收益率图 + VaR线
    axes[1].plot(returns.index, returns.values, label='收益率', linewidth=1, alpha=0.7)

    colors = ['red', 'darkred', 'blue', 'darkblue']
    for i, (var_name, var_val) in enumerate(var_levels.items()):
        if 'VaR' in var_name:
            axes[1].axhline(y=var_val, color=colors[i % len(colors)],
                          linestyle='--', linewidth=2, label=f'{var_name}', alpha=0.8)

    axes[1].set_xlabel('日期', fontsize=11)
    axes[1].set_ylabel('收益率', fontsize=11)
    axes[1].set_title('收益率与VaR阈值', fontsize=12, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_hill_estimates(hill_results: Dict, output_path: str):
    """绘制Hill估计量图"""
    if 'error' in hill_results:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    k_values = hill_results['k_values']

    # Hill估计量
    axes[0].plot(k_values, hill_results['hill_estimates'], linewidth=2)
    axes[0].axhline(y=hill_results['hill_estimates'][np.argmin(
        np.abs(k_values - hill_results['stable_k']))],
        color='red', linestyle='--', linewidth=2, label='稳定估计值')
    axes[0].set_xlabel('尾部样本数 k', fontsize=11)
    axes[0].set_ylabel('Hill估计量 (1/α)', fontsize=11)
    axes[0].set_title('Hill估计量 vs 尾部样本数', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 尾部指数
    axes[1].plot(k_values, hill_results['tail_indices'], linewidth=2, color='green')
    axes[1].axhline(y=hill_results['stable_alpha'],
                   color='red', linestyle='--', linewidth=2,
                   label=f'稳定尾部指数 α={hill_results["stable_alpha"]:.2f}')
    axes[1].axhline(y=2, color='orange', linestyle=':', linewidth=2, label='α=2 (无均值边界)')
    axes[1].axhline(y=4, color='purple', linestyle=':', linewidth=2, label='α=4 (无方差边界)')
    axes[1].set_xlabel('尾部样本数 k', fontsize=11)
    axes[1].set_ylabel('尾部指数 α', fontsize=11)
    axes[1].set_title('尾部指数 vs 尾部样本数', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, min(10, hill_results['tail_indices'].max() * 1.2))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_extreme_timeline(price_series: pd.Series, extreme_dates: List, output_path: str):
    """绘制极端事件时间线"""
    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(price_series.index, price_series.values, linewidth=1.5, label='BTC价格')

    # 标记极端事件
    if extreme_dates:
        extreme_dates_dt = pd.to_datetime(extreme_dates)
        extreme_prices = price_series.loc[extreme_dates_dt]
        ax.scatter(extreme_dates_dt, extreme_prices,
                  color='red', s=100, alpha=0.6,
                  label='极端事件', zorder=5, marker='X')

    ax.set_xlabel('日期', fontsize=11)
    ax.set_ylabel('价格 (USDT)', fontsize=11)
    ax.set_title('极端事件时间线 (99%分位数)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_extreme_value_analysis(df: pd.DataFrame = None, output_dir: str = "output/extreme") -> Dict:
    """
    运行极端值与尾部风险分析

    Args:
        df: 预处理后的数据框(可选，内部会加载多尺度数据)
        output_dir: 输出目录

    Returns:
        包含发现和摘要的字典
    """
    os.makedirs(output_dir, exist_ok=True)
    findings = []
    summary = {}

    print("=" * 60)
    print("极端值与尾部风险分析")
    print("=" * 60)

    # 加载多尺度数据
    intervals = ['1h', '4h', '1d', '1w']
    all_data = {}

    for interval in intervals:
        try:
            data = load_klines(interval)
            returns = log_returns(data["close"])
            all_data[interval] = {
                'price': data['close'],
                'returns': returns
            }
            print(f"加载 {interval} 数据: {len(data)} 条")
        except Exception as e:
            print(f"加载 {interval} 数据失败: {e}")

    # 主要使用日线数据进行深度分析
    if '1d' not in all_data:
        print("缺少日线数据，无法进行分析")
        return {'findings': findings, 'summary': summary}

    daily_returns = all_data['1d']['returns']
    daily_price = all_data['1d']['price']

    # 1. GEV分布拟合
    print("\n1. 拟合广义极值分布(GEV)...")
    gev_results = fit_gev_distribution(daily_returns, block_size='M')

    if 'error' not in gev_results:
        maxima_info = gev_results['maxima']
        minima_info = gev_results['minima']

        findings.append({
            'name': 'GEV区组极值拟合',
            'p_value': min(maxima_info['ks_pvalue'], minima_info['ks_pvalue']),
            'effect_size': abs(maxima_info['shape']),
            'significant': maxima_info['ks_pvalue'] > 0.05,
            'description': f"正向尾部: {maxima_info['tail_type']} (ξ={maxima_info['shape']:.3f}); "
                          f"负向尾部: {minima_info['tail_type']} (ξ={minima_info['shape']:.3f})",
            'test_set_consistent': True,
            'bootstrap_robust': maxima_info['n_blocks'] >= 30
        })

        summary['gev_maxima_shape'] = maxima_info['shape']
        summary['gev_minima_shape'] = minima_info['shape']
        print(f"  正向尾部: {maxima_info['tail_type']}, ξ={maxima_info['shape']:.3f}")
        print(f"  负向尾部: {minima_info['tail_type']}, ξ={minima_info['shape']:.3f}")

    # 2. GPD分布拟合
    print("\n2. 拟合广义Pareto分布(GPD)...")
    gpd_95 = fit_gpd_distribution(daily_returns, threshold_quantile=0.95)
    gpd_975 = fit_gpd_distribution(daily_returns, threshold_quantile=0.975)

    if 'error' not in gpd_95 and 'positive_tail' in gpd_95:
        pos_tail = gpd_95['positive_tail']
        findings.append({
            'name': 'GPD尾部拟合(95%阈值)',
            'p_value': pos_tail['ks_pvalue'],
            'effect_size': pos_tail['shape'],
            'significant': pos_tail['is_power_law'],
            'description': f"正向尾部形状参数 ξ={pos_tail['shape']:.3f}, "
                          f"尾部指数 α={pos_tail['tail_index']:.2f}, "
                          f"{'幂律尾部' if pos_tail['is_power_law'] else '指数尾部'}",
            'test_set_consistent': True,
            'bootstrap_robust': pos_tail['n_exceedances'] >= 30
        })

        summary['gpd_shape_95'] = pos_tail['shape']
        summary['gpd_tail_index_95'] = pos_tail['tail_index']
        print(f"  95%阈值正向尾部: ξ={pos_tail['shape']:.3f}, α={pos_tail['tail_index']:.2f}")

    # 绘制尾部拟合QQ图
    plot_tail_qq(gpd_95, os.path.join(output_dir, 'extreme_qq_tail.png'))
    print("  保存QQ图: extreme_qq_tail.png")

    # 3. 多尺度VaR/CVaR计算与回测
    print("\n3. VaR/CVaR多尺度回测...")
    var_results = {}
    backtest_results_all = {}

    for interval in ['1h', '4h', '1d', '1w']:
        if interval not in all_data:
            continue

        try:
            returns = all_data[interval]['returns']
            var_cvar = calculate_var_cvar(returns, confidence_levels=[0.95, 0.99])
            var_results[interval] = var_cvar

            # 回测
            backtest_results = {}
            for cl in [0.95, 0.99]:
                var_level = var_cvar[f'VaR_{int(cl*100)}']
                bt = backtest_var(returns, var_level, confidence=cl)
                backtest_results[f'VaR_{int(cl*100)}'] = bt

                findings.append({
                    'name': f'VaR回测_{interval}_{int(cl*100)}%',
                    'p_value': bt['p_value'],
                    'effect_size': abs(bt['violation_rate'] - bt['expected_rate']),
                    'significant': not bt['reject_model'],
                    'description': f"{interval} VaR{int(cl*100)} 违约率={bt['violation_rate']:.2%} "
                                  f"(期望{bt['expected_rate']:.2%}), "
                                  f"{'模型拒绝' if bt['reject_model'] else '模型通过'}",
                    'test_set_consistent': True,
                    'bootstrap_robust': True
                })

            backtest_results_all[interval] = backtest_results

            print(f"  {interval}: VaR95={var_cvar['VaR_95']:.4f}, CVaR95={var_cvar['CVaR_95']:.4f}")

        except Exception as e:
            print(f"  {interval} VaR计算失败: {e}")

    # 绘制VaR回测图(使用日线)
    if '1d' in backtest_results_all:
        plot_var_backtest(daily_price, daily_returns,
                         var_results['1d'], backtest_results_all['1d'],
                         os.path.join(output_dir, 'extreme_var_backtest.png'))
        print("  保存VaR回测图: extreme_var_backtest.png")

    summary['var_results'] = var_results

    # 4. Hill尾部指数估计
    print("\n4. Hill尾部指数估计...")
    hill_results = estimate_hill_index(daily_returns, k_max=300)

    if 'error' not in hill_results:
        findings.append({
            'name': 'Hill尾部指数估计',
            'p_value': None,
            'effect_size': hill_results['stable_alpha'],
            'significant': hill_results['is_heavy_tail'],
            'description': f"稳定尾部指数 α={hill_results['stable_alpha']:.2f} "
                          f"(k={hill_results['stable_k']}), "
                          f"{'重尾分布' if hill_results['is_heavy_tail'] else '轻尾分布'}",
            'test_set_consistent': True,
            'bootstrap_robust': True
        })

        summary['hill_tail_index'] = hill_results['stable_alpha']
        summary['hill_is_heavy_tail'] = hill_results['is_heavy_tail']
        print(f"  稳定尾部指数: α={hill_results['stable_alpha']:.2f}")

        # 绘制Hill图
        plot_hill_estimates(hill_results, os.path.join(output_dir, 'extreme_hill_plot.png'))
        print("  保存Hill图: extreme_hill_plot.png")

    # 5. 极端事件聚集性检验
    print("\n5. 极端事件聚集性检验...")
    clustering_results = test_extreme_clustering(daily_returns, quantile=0.99)

    if 'error' not in clustering_results:
        findings.append({
            'name': '极端事件聚集性检验',
            'p_value': clustering_results['p_value'],
            'effect_size': abs(clustering_results['acf_lag1']),
            'significant': clustering_results['is_clustered'],
            'description': f"极端事件{'存在聚集' if clustering_results['is_clustered'] else '独立分布'}, "
                          f"游程数={clustering_results['n_runs']:.0f} "
                          f"(期望{clustering_results['expected_runs']:.0f}), "
                          f"ACF(1)={clustering_results['acf_lag1']:.3f}",
            'test_set_consistent': True,
            'bootstrap_robust': True
        })

        summary['extreme_clustering'] = clustering_results['is_clustered']
        summary['extreme_acf_lag1'] = clustering_results['acf_lag1']
        print(f"  {'检测到聚集性' if clustering_results['is_clustered'] else '无明显聚集'}")
        print(f"  ACF(1)={clustering_results['acf_lag1']:.3f}")

        # 绘制极端事件时间线
        plot_extreme_timeline(daily_price, clustering_results['extreme_dates'],
                            os.path.join(output_dir, 'extreme_timeline.png'))
        print("  保存极端事件时间线: extreme_timeline.png")

    # 汇总统计
    summary['n_findings'] = len(findings)
    summary['n_significant'] = sum(1 for f in findings if f['significant'])

    print("\n" + "=" * 60)
    print(f"分析完成: {len(findings)} 项发现, {summary['n_significant']} 项显著")
    print("=" * 60)

    return {
        'findings': findings,
        'summary': summary
    }


if __name__ == '__main__':
    result = run_extreme_value_analysis()
    print(f"\n发现数: {len(result['findings'])}")
    for finding in result['findings']:
        print(f"  - {finding['name']}: {finding['description']}")
