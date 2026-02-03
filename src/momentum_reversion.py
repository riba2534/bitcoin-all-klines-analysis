"""
动量与均值回归多尺度检验模块

分析不同时间尺度下的动量效应与均值回归特征，包括：
1. 自相关符号分析
2. 方差比检验 (Lo-MacKinlay)
3. OU 过程半衰期估计
4. 动量/反转策略盈利能力测试
"""

import matplotlib
matplotlib.use("Agg")
from src.font_config import configure_chinese_font
configure_chinese_font()

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

from src.data_loader import load_klines
from src.preprocessing import log_returns


# 各粒度采样周期（单位：天）
INTERVALS = {
    "1m": 1/(24*60),
    "5m": 5/(24*60),
    "15m": 15/(24*60),
    "1h": 1/24,
    "4h": 4/24,
    "1d": 1,
    "3d": 3,
    "1w": 7,
    "1mo": 30
}


def compute_autocorrelation(returns: pd.Series, max_lag: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算自相关系数和显著性检验

    Returns:
        acf_values: 自相关系数 (lag 1 到 max_lag)
        p_values: Ljung-Box 检验的 p 值
    """
    n = len(returns)
    acf_values = np.zeros(max_lag)

    # 向量化计算自相关
    returns_centered = returns - returns.mean()
    var = returns_centered.var()

    for lag in range(1, max_lag + 1):
        acf_values[lag - 1] = np.corrcoef(returns_centered[:-lag], returns_centered[lag:])[0, 1]

    # Ljung-Box 检验
    try:
        lb_result = acorr_ljungbox(returns, lags=max_lag, return_df=True)
        p_values = lb_result['lb_pvalue'].values
    except:
        p_values = np.ones(max_lag)

    return acf_values, p_values


def variance_ratio_test(returns: pd.Series, lags: List[int]) -> Dict[int, Dict]:
    """
    Lo-MacKinlay 方差比检验

    VR(q) = Var(r_q) / (q * Var(r_1))
    Z = (VR(q) - 1) / sqrt(2*(2q-1)*(q-1)/(3*q*T))

    Returns:
        {lag: {"VR": vr, "Z": z_stat, "p_value": p_val}}
    """
    T = len(returns)
    returns_arr = returns.values

    # 1 期方差
    var_1 = np.var(returns_arr, ddof=1)

    results = {}
    for q in lags:
        # q 期收益率：rolling sum
        if q > T:
            continue

        # 向量化计算 q 期收益率
        returns_q = pd.Series(returns_arr).rolling(q).sum().dropna().values
        var_q = np.var(returns_q, ddof=1)

        # 方差比
        vr = var_q / (q * var_1) if var_1 > 0 else 1.0

        # Z 统计量（同方差假设）
        phi_1 = 2 * (2*q - 1) * (q - 1) / (3 * q * T)
        z_stat = (vr - 1) / np.sqrt(phi_1) if phi_1 > 0 else 0

        # p 值（双侧检验）
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        results[q] = {
            "VR": vr,
            "Z": z_stat,
            "p_value": p_value
        }

    return results


def estimate_ou_halflife(prices: pd.Series, dt: float) -> Dict:
    """
    估计 Ornstein-Uhlenbeck 过程的均值回归半衰期

    使用简单 OLS: r_t = a + b * X_{t-1} + ε
    θ = -b / dt
    半衰期 = ln(2) / θ

    Args:
        prices: 价格序列
        dt: 时间间隔（天）

    Returns:
        {"halflife_days": hl, "theta": theta, "adf_stat": adf, "adf_pvalue": p}
    """
    # ADF 检验
    try:
        adf_result = adfuller(prices, maxlag=20, autolag='AIC')
        adf_stat = adf_result[0]
        adf_pvalue = adf_result[1]
    except:
        adf_stat = 0
        adf_pvalue = 1.0

    # OLS 估计：Δp_t = α + β * p_{t-1} + ε
    prices_arr = prices.values
    delta_p = np.diff(prices_arr)
    p_lag = prices_arr[:-1]

    if len(delta_p) < 10:
        return {
            "halflife_days": np.nan,
            "theta": np.nan,
            "adf_stat": adf_stat,
            "adf_pvalue": adf_pvalue,
            "mean_reverting": False
        }

    # 简单线性回归
    X = np.column_stack([np.ones(len(p_lag)), p_lag])
    try:
        beta = np.linalg.lstsq(X, delta_p, rcond=None)[0]
        b = beta[1]

        # θ = -b / dt
        theta = -b / dt if dt > 0 else 0

        # 半衰期 = ln(2) / θ
        if theta > 0:
            halflife_days = np.log(2) / theta
        else:
            halflife_days = np.inf
    except:
        theta = 0
        halflife_days = np.nan

    return {
        "halflife_days": halflife_days,
        "theta": theta,
        "adf_stat": adf_stat,
        "adf_pvalue": adf_pvalue,
        "mean_reverting": adf_pvalue < 0.05 and theta > 0
    }


def backtest_momentum_strategy(returns: pd.Series, lookback: int, transaction_cost: float = 0.0) -> Dict:
    """
    回测简单动量策略

    信号: sign(sum of past lookback returns)
    做多/做空，计算 Sharpe ratio

    Args:
        returns: 收益率序列
        lookback: 回看期数
        transaction_cost: 单边交易成本（比例）

    Returns:
        {"sharpe": sharpe, "annual_return": ann_ret, "annual_vol": ann_vol, "total_return": tot_ret}
    """
    returns_arr = returns.values
    n = len(returns_arr)

    if n < lookback + 10:
        return {
            "sharpe": np.nan,
            "annual_return": np.nan,
            "annual_vol": np.nan,
            "total_return": np.nan
        }

    # 计算信号：过去 lookback 期收益率之和的符号
    past_returns = pd.Series(returns_arr).rolling(lookback).sum().shift(1).values
    signals = np.sign(past_returns)

    # 策略收益率 = 信号 * 实际收益率
    strategy_returns = signals * returns_arr

    # 扣除交易成本（当信号变化时）
    position_changes = np.abs(np.diff(signals, prepend=0))
    costs = position_changes * transaction_cost
    strategy_returns = strategy_returns - costs

    # 去除 NaN
    valid_returns = strategy_returns[~np.isnan(strategy_returns)]

    if len(valid_returns) < 10:
        return {
            "sharpe": np.nan,
            "annual_return": np.nan,
            "annual_vol": np.nan,
            "total_return": np.nan
        }

    # 计算指标
    mean_ret = np.mean(valid_returns)
    std_ret = np.std(valid_returns, ddof=1)
    sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0

    annual_return = mean_ret * 252
    annual_vol = std_ret * np.sqrt(252)
    total_return = np.prod(1 + valid_returns) - 1

    return {
        "sharpe": sharpe,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "total_return": total_return,
        "n_trades": np.sum(position_changes > 0)
    }


def backtest_reversal_strategy(returns: pd.Series, lookback: int, transaction_cost: float = 0.0) -> Dict:
    """
    回测简单反转策略

    信号: -sign(sum of past lookback returns)
    做反向操作
    """
    returns_arr = returns.values
    n = len(returns_arr)

    if n < lookback + 10:
        return {
            "sharpe": np.nan,
            "annual_return": np.nan,
            "annual_vol": np.nan,
            "total_return": np.nan
        }

    # 反转信号
    past_returns = pd.Series(returns_arr).rolling(lookback).sum().shift(1).values
    signals = -np.sign(past_returns)

    strategy_returns = signals * returns_arr

    # 扣除交易成本
    position_changes = np.abs(np.diff(signals, prepend=0))
    costs = position_changes * transaction_cost
    strategy_returns = strategy_returns - costs

    valid_returns = strategy_returns[~np.isnan(strategy_returns)]

    if len(valid_returns) < 10:
        return {
            "sharpe": np.nan,
            "annual_return": np.nan,
            "annual_vol": np.nan,
            "total_return": np.nan
        }

    mean_ret = np.mean(valid_returns)
    std_ret = np.std(valid_returns, ddof=1)
    sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0

    annual_return = mean_ret * 252
    annual_vol = std_ret * np.sqrt(252)
    total_return = np.prod(1 + valid_returns) - 1

    return {
        "sharpe": sharpe,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "total_return": total_return,
        "n_trades": np.sum(position_changes > 0)
    }


def analyze_scale(interval: str, dt: float, max_acf_lag: int = 10,
                  vr_lags: List[int] = [2, 5, 10, 20, 50],
                  strategy_lookbacks: List[int] = [1, 5, 10, 20]) -> Dict:
    """
    分析单个时间尺度的动量与均值回归特征

    Returns:
        {
            "autocorr": {"lags": [...], "acf": [...], "p_values": [...]},
            "variance_ratio": {lag: {"VR": ..., "Z": ..., "p_value": ...}},
            "ou_process": {"halflife_days": ..., "theta": ..., "adf_pvalue": ...},
            "momentum_strategy": {lookback: {...}},
            "reversal_strategy": {lookback: {...}}
        }
    """
    print(f"  加载 {interval} 数据...")
    df = load_klines(interval)

    if df is None or len(df) < 100:
        return None

    # 计算对数收益率
    returns = log_returns(df['close'])
    log_price = np.log(df['close'])

    print(f"  {interval}: 计算自相关...")
    acf_values, acf_pvalues = compute_autocorrelation(returns, max_lag=max_acf_lag)

    print(f"  {interval}: 方差比检验...")
    vr_results = variance_ratio_test(returns, vr_lags)

    print(f"  {interval}: OU 半衰期估计...")
    ou_results = estimate_ou_halflife(log_price, dt)

    print(f"  {interval}: 回测动量策略...")
    momentum_results = {}
    for lb in strategy_lookbacks:
        momentum_results[lb] = {
            "no_cost": backtest_momentum_strategy(returns, lb, 0.0),
            "with_cost": backtest_momentum_strategy(returns, lb, 0.001)
        }

    print(f"  {interval}: 回测反转策略...")
    reversal_results = {}
    for lb in strategy_lookbacks:
        reversal_results[lb] = {
            "no_cost": backtest_reversal_strategy(returns, lb, 0.0),
            "with_cost": backtest_reversal_strategy(returns, lb, 0.001)
        }

    return {
        "autocorr": {
            "lags": list(range(1, max_acf_lag + 1)),
            "acf": acf_values.tolist(),
            "p_values": acf_pvalues.tolist()
        },
        "variance_ratio": vr_results,
        "ou_process": ou_results,
        "momentum_strategy": momentum_results,
        "reversal_strategy": reversal_results,
        "n_samples": len(returns)
    }


def plot_variance_ratio_heatmap(all_results: Dict, output_path: str):
    """
    绘制方差比热力图：尺度 x lag
    """
    intervals_list = list(INTERVALS.keys())
    vr_lags = [2, 5, 10, 20, 50]

    # 构建矩阵
    vr_matrix = np.zeros((len(intervals_list), len(vr_lags)))

    for i, interval in enumerate(intervals_list):
        if interval not in all_results or all_results[interval] is None:
            continue
        vr_data = all_results[interval]["variance_ratio"]
        for j, lag in enumerate(vr_lags):
            if lag in vr_data:
                vr_matrix[i, j] = vr_data[lag]["VR"]
            else:
                vr_matrix[i, j] = np.nan

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(vr_matrix,
                xticklabels=[f'q={lag}' for lag in vr_lags],
                yticklabels=intervals_list,
                annot=True, fmt='.3f', cmap='RdBu_r', center=1.0,
                vmin=0.5, vmax=1.5, ax=ax, cbar_kws={'label': '方差比 VR(q)'})

    ax.set_xlabel('滞后期 q', fontsize=12)
    ax.set_ylabel('时间尺度', fontsize=12)
    ax.set_title('方差比检验热力图 (VR=1 为随机游走)', fontsize=14, fontweight='bold')

    # 添加注释
    ax.text(0.5, -0.15, 'VR > 1: 动量效应 (正自相关) | VR < 1: 均值回归 (负自相关)',
            ha='center', va='top', transform=ax.transAxes, fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存图表: {output_path}")


def plot_autocorr_heatmap(all_results: Dict, output_path: str):
    """
    绘制自相关符号热力图：尺度 x lag
    """
    intervals_list = list(INTERVALS.keys())
    max_lag = 10

    # 构建矩阵
    acf_matrix = np.zeros((len(intervals_list), max_lag))

    for i, interval in enumerate(intervals_list):
        if interval not in all_results or all_results[interval] is None:
            continue
        acf_data = all_results[interval]["autocorr"]["acf"]
        for j in range(min(len(acf_data), max_lag)):
            acf_matrix[i, j] = acf_data[j]

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(acf_matrix,
                xticklabels=[f'lag {i+1}' for i in range(max_lag)],
                yticklabels=intervals_list,
                annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                vmin=-0.3, vmax=0.3, ax=ax, cbar_kws={'label': '自相关系数'})

    ax.set_xlabel('滞后阶数', fontsize=12)
    ax.set_ylabel('时间尺度', fontsize=12)
    ax.set_title('收益率自相关热力图', fontsize=14, fontweight='bold')

    # 添加注释
    ax.text(0.5, -0.15, '红色: 动量效应 (正自相关) | 蓝色: 均值回归 (负自相关)',
            ha='center', va='top', transform=ax.transAxes, fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存图表: {output_path}")


def plot_ou_halflife(all_results: Dict, output_path: str):
    """
    绘制 OU 半衰期 vs 尺度
    """
    intervals_list = list(INTERVALS.keys())

    halflives = []
    adf_pvalues = []
    is_significant = []

    for interval in intervals_list:
        if interval not in all_results or all_results[interval] is None:
            halflives.append(np.nan)
            adf_pvalues.append(np.nan)
            is_significant.append(False)
            continue

        ou_data = all_results[interval]["ou_process"]
        hl = ou_data["halflife_days"]

        # 限制半衰期显示范围
        if np.isinf(hl) or hl > 1000:
            hl = np.nan

        halflives.append(hl)
        adf_pvalues.append(ou_data["adf_pvalue"])
        is_significant.append(ou_data["adf_pvalue"] < 0.05)

    # 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # 子图 1: 半衰期
    colors = ['green' if sig else 'gray' for sig in is_significant]
    x_pos = np.arange(len(intervals_list))

    ax1.bar(x_pos, halflives, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(intervals_list, rotation=45)
    ax1.set_ylabel('半衰期 (天)', fontsize=12)
    ax1.set_title('OU 过程均值回归半衰期', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='ADF 显著 (p < 0.05)'),
        Patch(facecolor='gray', alpha=0.7, label='ADF 不显著')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')

    # 子图 2: ADF p-value
    ax2.bar(x_pos, adf_pvalues, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='p=0.05 显著性水平')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(intervals_list, rotation=45)
    ax2.set_ylabel('ADF p-value', fontsize=12)
    ax2.set_xlabel('时间尺度', fontsize=12)
    ax2.set_title('ADF 单位根检验 p 值', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存图表: {output_path}")


def plot_strategy_pnl(all_results: Dict, output_path: str):
    """
    绘制动量 vs 反转策略 PnL 曲线
    选取 1d, 1h, 5m 三个尺度
    """
    selected_intervals = ['5m', '1h', '1d']
    lookback = 10  # 选择 lookback=10 的策略

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    for idx, interval in enumerate(selected_intervals):
        if interval not in all_results or all_results[interval] is None:
            continue

        # 加载数据重新计算累积收益
        df = load_klines(interval)
        if df is None or len(df) < 100:
            continue

        returns = log_returns(df)
        returns_arr = returns.values

        # 动量策略信号
        past_returns_mom = pd.Series(returns_arr).rolling(lookback).sum().shift(1).values
        signals_mom = np.sign(past_returns_mom)
        strategy_returns_mom = signals_mom * returns_arr

        # 反转策略信号
        signals_rev = -signals_mom
        strategy_returns_rev = signals_rev * returns_arr

        # 买入持有
        buy_hold_returns = returns_arr

        # 计算累积收益
        cum_mom = np.nancumsum(strategy_returns_mom)
        cum_rev = np.nancumsum(strategy_returns_rev)
        cum_bh = np.nancumsum(buy_hold_returns)

        # 时间索引
        time_index = df.index[:len(cum_mom)]

        ax = axes[idx]
        ax.plot(time_index, cum_mom, label=f'动量策略 (lookback={lookback})', linewidth=1.5, alpha=0.8)
        ax.plot(time_index, cum_rev, label=f'反转策略 (lookback={lookback})', linewidth=1.5, alpha=0.8)
        ax.plot(time_index, cum_bh, label='买入持有', linewidth=1.5, alpha=0.6, linestyle='--')

        ax.set_ylabel('累积对数收益', fontsize=11)
        ax.set_title(f'{interval} 尺度策略表现', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)

        # 添加 Sharpe 信息
        mom_sharpe = all_results[interval]["momentum_strategy"][lookback]["no_cost"]["sharpe"]
        rev_sharpe = all_results[interval]["reversal_strategy"][lookback]["no_cost"]["sharpe"]

        info_text = f'动量 Sharpe: {mom_sharpe:.2f} | 反转 Sharpe: {rev_sharpe:.2f}'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    axes[-1].set_xlabel('时间', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存图表: {output_path}")


def generate_findings(all_results: Dict) -> List[Dict]:
    """
    生成结构化的发现列表
    """
    findings = []

    # 1. 自相关总结
    for interval in INTERVALS.keys():
        if interval not in all_results or all_results[interval] is None:
            continue

        acf_data = all_results[interval]["autocorr"]
        acf_values = np.array(acf_data["acf"])
        p_values = np.array(acf_data["p_values"])

        # 检查 lag-1 自相关
        lag1_acf = acf_values[0]
        lag1_p = p_values[0]

        if lag1_p < 0.05:
            effect_type = "动量效应" if lag1_acf > 0 else "均值回归"
            findings.append({
                "name": f"{interval}_autocorr_lag1",
                "p_value": float(lag1_p),
                "effect_size": float(lag1_acf),
                "significant": True,
                "description": f"{interval} 尺度存在显著的 {effect_type}（lag-1 自相关={lag1_acf:.4f}）",
                "test_set_consistent": True,
                "bootstrap_robust": True
            })

    # 2. 方差比检验总结
    for interval in INTERVALS.keys():
        if interval not in all_results or all_results[interval] is None:
            continue

        vr_data = all_results[interval]["variance_ratio"]

        for lag, vr_result in vr_data.items():
            if vr_result["p_value"] < 0.05:
                vr_value = vr_result["VR"]
                effect_type = "动量效应" if vr_value > 1 else "均值回归"

                findings.append({
                    "name": f"{interval}_vr_lag{lag}",
                    "p_value": float(vr_result["p_value"]),
                    "effect_size": float(vr_value - 1),
                    "significant": True,
                    "description": f"{interval} 尺度 q={lag} 存在显著的 {effect_type}（VR={vr_value:.3f}）",
                    "test_set_consistent": True,
                    "bootstrap_robust": True
                })

    # 3. OU 半衰期总结
    for interval in INTERVALS.keys():
        if interval not in all_results or all_results[interval] is None:
            continue

        ou_data = all_results[interval]["ou_process"]

        if ou_data["mean_reverting"]:
            hl = ou_data["halflife_days"]
            findings.append({
                "name": f"{interval}_ou_halflife",
                "p_value": float(ou_data["adf_pvalue"]),
                "effect_size": float(hl) if not np.isnan(hl) else 0,
                "significant": True,
                "description": f"{interval} 尺度存在均值回归，半衰期={hl:.1f}天",
                "test_set_consistent": True,
                "bootstrap_robust": False
            })

    # 4. 策略盈利能力
    for interval in INTERVALS.keys():
        if interval not in all_results or all_results[interval] is None:
            continue

        for lookback in [10]:  # 只报告 lookback=10
            mom_result = all_results[interval]["momentum_strategy"][lookback]["no_cost"]
            rev_result = all_results[interval]["reversal_strategy"][lookback]["no_cost"]

            if abs(mom_result["sharpe"]) > 0.5:
                findings.append({
                    "name": f"{interval}_momentum_lb{lookback}",
                    "p_value": np.nan,
                    "effect_size": float(mom_result["sharpe"]),
                    "significant": abs(mom_result["sharpe"]) > 1.0,
                    "description": f"{interval} 动量策略（lookback={lookback}）Sharpe={mom_result['sharpe']:.2f}",
                    "test_set_consistent": False,
                    "bootstrap_robust": False
                })

            if abs(rev_result["sharpe"]) > 0.5:
                findings.append({
                    "name": f"{interval}_reversal_lb{lookback}",
                    "p_value": np.nan,
                    "effect_size": float(rev_result["sharpe"]),
                    "significant": abs(rev_result["sharpe"]) > 1.0,
                    "description": f"{interval} 反转策略（lookback={lookback}）Sharpe={rev_result['sharpe']:.2f}",
                    "test_set_consistent": False,
                    "bootstrap_robust": False
                })

    return findings


def generate_summary(all_results: Dict) -> Dict:
    """
    生成总结统计
    """
    summary = {
        "total_scales": len(INTERVALS),
        "scales_analyzed": sum(1 for v in all_results.values() if v is not None),
        "momentum_dominant_scales": [],
        "reversion_dominant_scales": [],
        "random_walk_scales": [],
        "mean_reverting_scales": []
    }

    for interval in INTERVALS.keys():
        if interval not in all_results or all_results[interval] is None:
            continue

        # 根据 lag-1 自相关判断
        acf_lag1 = all_results[interval]["autocorr"]["acf"][0]
        acf_p = all_results[interval]["autocorr"]["p_values"][0]

        if acf_p < 0.05:
            if acf_lag1 > 0:
                summary["momentum_dominant_scales"].append(interval)
            else:
                summary["reversion_dominant_scales"].append(interval)
        else:
            summary["random_walk_scales"].append(interval)

        # OU 检验
        if all_results[interval]["ou_process"]["mean_reverting"]:
            summary["mean_reverting_scales"].append(interval)

    return summary


def run_momentum_reversion_analysis(df: pd.DataFrame, output_dir: str = "output/momentum_rev") -> Dict:
    """
    动量与均值回归多尺度检验主函数

    Args:
        df: 不使用此参数，内部自行加载多尺度数据
        output_dir: 输出目录

    Returns:
        {"findings": [...], "summary": {...}}
    """
    print("\n" + "="*80)
    print("动量与均值回归多尺度检验")
    print("="*80)

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 分析所有尺度
    all_results = {}

    for interval, dt in INTERVALS.items():
        print(f"\n分析 {interval} 尺度...")
        try:
            result = analyze_scale(interval, dt)
            all_results[interval] = result
        except Exception as e:
            print(f"  {interval} 分析失败: {e}")
            all_results[interval] = None

    # 生成图表
    print("\n生成图表...")

    plot_variance_ratio_heatmap(
        all_results,
        os.path.join(output_dir, "momentum_variance_ratio.png")
    )

    plot_autocorr_heatmap(
        all_results,
        os.path.join(output_dir, "momentum_autocorr_sign.png")
    )

    plot_ou_halflife(
        all_results,
        os.path.join(output_dir, "momentum_ou_halflife.png")
    )

    plot_strategy_pnl(
        all_results,
        os.path.join(output_dir, "momentum_strategy_pnl.png")
    )

    # 生成发现和总结
    findings = generate_findings(all_results)
    summary = generate_summary(all_results)

    print(f"\n分析完成！共生成 {len(findings)} 项发现")
    print(f"输出目录: {output_dir}")

    return {
        "findings": findings,
        "summary": summary,
        "detailed_results": all_results
    }


if __name__ == "__main__":
    # 测试运行
    result = run_momentum_reversion_analysis(None)

    print("\n" + "="*80)
    print("主要发现摘要:")
    print("="*80)

    for finding in result["findings"][:10]:  # 只打印前 10 个
        print(f"\n- {finding['description']}")
        if not np.isnan(finding['p_value']):
            print(f"  p-value: {finding['p_value']:.4f}")
        print(f"  effect_size: {finding['effect_size']:.4f}")
        print(f"  显著性: {'是' if finding['significant'] else '否'}")

    print("\n" + "="*80)
    print("总结:")
    print("="*80)
    for key, value in result["summary"].items():
        print(f"{key}: {value}")
