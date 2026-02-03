"""幂律增长拟合与走廊模型分析

通过幂律模型拟合BTC价格的长期增长趋势，构建价格走廊，
并与指数增长模型进行比较，评估当前价格在历史分布中的位置。
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from pathlib import Path
from typing import Tuple, Dict

# 中文显示支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def _compute_days_since_start(df: pd.DataFrame) -> np.ndarray:
    """计算距离起始日的天数（从1开始，避免log(0)）"""
    days = (df.index - df.index[0]).days.astype(float) + 1.0
    return days


def _fit_power_law(log_days: np.ndarray, log_prices: np.ndarray) -> Dict:
    """对数-对数线性回归拟合幂律模型

    模型: log(price) = slope * log(days) + intercept
    等价于: price = exp(intercept) * days^slope

    Returns
    -------
    dict
        包含 slope, intercept, r_squared, residuals, fitted_values
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_days, log_prices)
    fitted = slope * log_days + intercept
    residuals = log_prices - fitted

    return {
        'slope': slope,           # 幂律指数 α
        'intercept': intercept,   # log(c)
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'std_err': std_err,
        'residuals': residuals,
        'fitted_values': fitted,
    }


def _build_corridor(
    log_days: np.ndarray,
    fit_result: Dict,
    quantiles: Tuple[float, ...] = (0.05, 0.50, 0.95),
) -> Dict[float, np.ndarray]:
    """基于残差分位数构建幂律走廊

    Parameters
    ----------
    log_days : array
        log(天数) 序列
    fit_result : dict
        幂律拟合结果
    quantiles : tuple
        走廊分位数

    Returns
    -------
    dict
        分位数 -> 走廊价格（原始尺度）
    """
    residuals = fit_result['residuals']
    corridor = {}
    for q in quantiles:
        q_val = np.quantile(residuals, q)
        # log_price = slope * log_days + intercept + quantile_offset
        log_price_band = fit_result['slope'] * log_days + fit_result['intercept'] + q_val
        corridor[q] = np.exp(log_price_band)
    return corridor


def _power_law_func(days: np.ndarray, c: float, alpha: float) -> np.ndarray:
    """幂律函数: price = c * days^alpha"""
    return c * np.power(days, alpha)


def _exponential_func(days: np.ndarray, c: float, beta: float) -> np.ndarray:
    """指数函数: price = c * exp(beta * days)"""
    return c * np.exp(beta * days)


def _compute_aic_bic(n: int, k: int, rss: float) -> Tuple[float, float]:
    """计算AIC和BIC

    Parameters
    ----------
    n : int
        样本量
    k : int
        模型参数个数
    rss : float
        残差平方和

    Returns
    -------
    tuple
        (AIC, BIC)
    """
    # 对数似然 (假设正态分布残差)
    log_likelihood = -n / 2 * (np.log(2 * np.pi * rss / n) + 1)
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood
    return aic, bic


def _fit_and_compare_models(
    days: np.ndarray, prices: np.ndarray
) -> Dict:
    """拟合幂律和指数增长模型并比较AIC/BIC

    Returns
    -------
    dict
        包含两个模型的参数、AIC、BIC及比较结论
    """
    n = len(prices)
    k = 2  # 两个模型都有2个参数

    # --- 幂律拟合: price = c * days^alpha ---
    try:
        popt_pl, _ = curve_fit(
            _power_law_func, days, prices,
            p0=[1.0, 1.5], maxfev=10000
        )
        prices_pred_pl = _power_law_func(days, *popt_pl)
        rss_pl = np.sum((prices - prices_pred_pl) ** 2)
        aic_pl, bic_pl = _compute_aic_bic(n, k, rss_pl)
    except RuntimeError:
        # curve_fit 失败时回退到对数空间OLS估计
        log_d = np.log(days)
        log_p = np.log(prices)
        slope, intercept, _, _, _ = stats.linregress(log_d, log_p)
        popt_pl = [np.exp(intercept), slope]
        prices_pred_pl = _power_law_func(days, *popt_pl)
        rss_pl = np.sum((prices - prices_pred_pl) ** 2)
        aic_pl, bic_pl = _compute_aic_bic(n, k, rss_pl)

    # --- 指数拟合: price = c * exp(beta * days) ---
    # 初始值通过log空间OLS估计
    log_p = np.log(prices)
    beta_init, log_c_init, _, _, _ = stats.linregress(days, log_p)
    try:
        popt_exp, _ = curve_fit(
            _exponential_func, days, prices,
            p0=[np.exp(log_c_init), beta_init], maxfev=10000
        )
        prices_pred_exp = _exponential_func(days, *popt_exp)
        rss_exp = np.sum((prices - prices_pred_exp) ** 2)
        aic_exp, bic_exp = _compute_aic_bic(n, k, rss_exp)
    except (RuntimeError, OverflowError):
        # 指数拟合容易溢出，使用log空间线性回归作替代
        popt_exp = [np.exp(log_c_init), beta_init]
        prices_pred_exp = _exponential_func(days, *popt_exp)
        # 裁剪防止溢出
        prices_pred_exp = np.clip(prices_pred_exp, 0, prices.max() * 100)
        rss_exp = np.sum((prices - prices_pred_exp) ** 2)
        aic_exp, bic_exp = _compute_aic_bic(n, k, rss_exp)

    return {
        'power_law': {
            'params': {'c': popt_pl[0], 'alpha': popt_pl[1]},
            'aic': aic_pl,
            'bic': bic_pl,
            'rss': rss_pl,
            'predicted': prices_pred_pl,
        },
        'exponential': {
            'params': {'c': popt_exp[0], 'beta': popt_exp[1]},
            'aic': aic_exp,
            'bic': bic_exp,
            'rss': rss_exp,
            'predicted': prices_pred_exp,
        },
        'preferred': 'power_law' if aic_pl < aic_exp else 'exponential',
    }


def _compute_current_percentile(residuals: np.ndarray) -> float:
    """计算当前价格（最后一个残差）在历史残差分布中的百分位

    Returns
    -------
    float
        百分位数 (0-100)
    """
    current_residual = residuals[-1]
    percentile = stats.percentileofscore(residuals, current_residual)
    return percentile


# =============================================================================
#  可视化函数
# =============================================================================

def _plot_loglog_regression(
    log_days: np.ndarray,
    log_prices: np.ndarray,
    fit_result: Dict,
    dates: pd.DatetimeIndex,
    output_dir: Path,
):
    """图1: 对数-对数散点图 + 回归线"""
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.scatter(log_days, log_prices, s=3, alpha=0.5, color='steelblue', label='实际价格')
    ax.plot(log_days, fit_result['fitted_values'], color='red', linewidth=2,
            label=f"回归线: slope={fit_result['slope']:.4f}, R²={fit_result['r_squared']:.4f}")

    ax.set_xlabel('log(天数)', fontsize=12)
    ax.set_ylabel('log(价格)', fontsize=12)
    ax.set_title('BTC 幂律拟合 — 对数-对数回归', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / 'power_law_loglog_regression.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [图] 对数-对数回归已保存: {output_dir / 'power_law_loglog_regression.png'}")


def _plot_corridor(
    df: pd.DataFrame,
    days: np.ndarray,
    corridor: Dict[float, np.ndarray],
    fit_result: Dict,
    output_dir: Path,
):
    """图2: 幂律走廊模型（价格 + 5%/50%/95% 通道）"""
    fig, ax = plt.subplots(figsize=(14, 7))

    # 实际价格
    ax.semilogy(df.index, df['close'], color='black', linewidth=0.8, label='BTC 收盘价')

    # 走廊带
    colors = {0.05: 'green', 0.50: 'orange', 0.95: 'red'}
    labels = {0.05: '5% 下界', 0.50: '50% 中位线', 0.95: '95% 上界'}
    for q, band in corridor.items():
        ax.semilogy(df.index, band, color=colors[q], linewidth=1.5,
                     linestyle='--', label=labels[q])

    # 填充走廊区间
    ax.fill_between(df.index, corridor[0.05], corridor[0.95],
                     alpha=0.1, color='blue', label='90% 走廊区间')

    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('价格 (USDT, 对数尺度)', fontsize=12)
    ax.set_title('BTC 幂律走廊模型', fontsize=14)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')

    fig.savefig(output_dir / 'power_law_corridor.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [图] 幂律走廊已保存: {output_dir / 'power_law_corridor.png'}")


def _plot_model_comparison(
    df: pd.DataFrame,
    days: np.ndarray,
    comparison: Dict,
    output_dir: Path,
):
    """图3: 幂律 vs 指数增长模型对比"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 左图: 价格对比
    ax1 = axes[0]
    ax1.semilogy(df.index, df['close'], color='black', linewidth=0.8, label='实际价格')
    ax1.semilogy(df.index, comparison['power_law']['predicted'],
                  color='blue', linewidth=1.5, linestyle='--', label='幂律拟合')
    ax1.semilogy(df.index, np.clip(comparison['exponential']['predicted'], 1e-1, None),
                  color='red', linewidth=1.5, linestyle='--', label='指数拟合')
    ax1.set_xlabel('日期', fontsize=11)
    ax1.set_ylabel('价格 (USDT, 对数尺度)', fontsize=11)
    ax1.set_title('模型拟合对比', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')

    # 右图: AIC/BIC 柱状图
    ax2 = axes[1]
    models = ['幂律模型', '指数模型']
    aic_vals = [comparison['power_law']['aic'], comparison['exponential']['aic']]
    bic_vals = [comparison['power_law']['bic'], comparison['exponential']['bic']]

    x = np.arange(len(models))
    width = 0.35
    bars1 = ax2.bar(x - width / 2, aic_vals, width, label='AIC', color='steelblue')
    bars2 = ax2.bar(x + width / 2, bic_vals, width, label='BIC', color='coral')

    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=11)
    ax2.set_ylabel('信息准则值', fontsize=11)
    ax2.set_title('AIC / BIC 模型比较', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar in bars1:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / 'power_law_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [图] 模型对比已保存: {output_dir / 'power_law_model_comparison.png'}")


def _plot_residual_distribution(
    residuals: np.ndarray,
    current_percentile: float,
    output_dir: Path,
):
    """图4: 残差分布 + 当前位置"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(residuals, bins=60, density=True, alpha=0.6, color='steelblue',
            edgecolor='white', label='残差分布')

    # 当前位置
    current_res = residuals[-1]
    ax.axvline(current_res, color='red', linewidth=2, linestyle='--',
               label=f'当前位置: {current_percentile:.1f}%')

    # 分位数线
    for q, color, label in [(0.05, 'green', '5%'), (0.50, 'orange', '50%'), (0.95, 'red', '95%')]:
        q_val = np.quantile(residuals, q)
        ax.axvline(q_val, color=color, linewidth=1, linestyle=':',
                   alpha=0.7, label=f'{label} 分位: {q_val:.3f}')

    ax.set_xlabel('残差 (log尺度)', fontsize=12)
    ax.set_ylabel('密度', fontsize=12)
    ax.set_title(f'幂律残差分布 — 当前价格位于 {current_percentile:.1f}% 分位', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / 'power_law_residual_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [图] 残差分布已保存: {output_dir / 'power_law_residual_distribution.png'}")


# =============================================================================
#  主入口
# =============================================================================

def run_power_law_analysis(df: pd.DataFrame, output_dir: str = "output") -> Dict:
    """幂律增长拟合与走廊模型 — 主入口函数

    Parameters
    ----------
    df : pd.DataFrame
        由 data_loader.load_daily() 返回的日线数据，含 DatetimeIndex 和 close 列
    output_dir : str
        图表输出目录

    Returns
    -------
    dict
        分析结果摘要
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  BTC 幂律增长分析")
    print("=" * 60)

    prices = df['close'].dropna()

    # ---- 步骤1: 准备数据 ----
    days = _compute_days_since_start(df.loc[prices.index])
    log_days = np.log(days)
    log_prices = np.log(prices.values)

    print(f"\n数据范围: {prices.index[0].date()} ~ {prices.index[-1].date()}")
    print(f"样本数量: {len(prices)}")

    # ---- 步骤2: 对数-对数线性回归 ----
    print("\n--- 对数-对数线性回归 ---")
    fit_result = _fit_power_law(log_days, log_prices)
    print(f"  幂律指数 (slope/α): {fit_result['slope']:.6f}")
    print(f"  截距 log(c):        {fit_result['intercept']:.6f}")
    print(f"  等价系数 c:         {np.exp(fit_result['intercept']):.6f}")
    print(f"  R²:                 {fit_result['r_squared']:.6f}")
    print(f"  p-value:            {fit_result['p_value']:.2e}")
    print(f"  标准误差:            {fit_result['std_err']:.6f}")

    # ---- 步骤3: 幂律走廊模型 ----
    print("\n--- 幂律走廊模型 ---")
    quantiles = (0.05, 0.50, 0.95)
    corridor = _build_corridor(log_days, fit_result, quantiles)
    for q in quantiles:
        print(f"  {int(q * 100):>3d}% 分位当前走廊价格: ${corridor[q][-1]:,.0f}")

    # ---- 步骤4: 模型比较 (幂律 vs 指数) ----
    print("\n--- 模型比较: 幂律 vs 指数 ---")
    comparison = _fit_and_compare_models(days, prices.values)

    pl = comparison['power_law']
    exp = comparison['exponential']
    print(f"  幂律模型:  c={pl['params']['c']:.4f}, α={pl['params']['alpha']:.4f}")
    print(f"             AIC={pl['aic']:.0f}, BIC={pl['bic']:.0f}")
    print(f"  指数模型:  c={exp['params']['c']:.4f}, β={exp['params']['beta']:.6f}")
    print(f"             AIC={exp['aic']:.0f}, BIC={exp['bic']:.0f}")
    print(f"  AIC 差值 (幂律-指数): {pl['aic'] - exp['aic']:.0f}")
    print(f"  BIC 差值 (幂律-指数): {pl['bic'] - exp['bic']:.0f}")
    print(f"  >> 优选模型: {comparison['preferred']}")

    # ---- 步骤5: 当前价格位置 ----
    print("\n--- 当前价格位置 ---")
    current_percentile = _compute_current_percentile(fit_result['residuals'])
    current_price = prices.iloc[-1]
    print(f"  当前价格: ${current_price:,.2f}")
    print(f"  历史残差分位: {current_percentile:.1f}%")
    if current_percentile > 90:
        print("  >> 警告: 当前价格处于历史高估区域")
    elif current_percentile < 10:
        print("  >> 提示: 当前价格处于历史低估区域")
    else:
        print("  >> 当前价格处于历史正常波动范围内")

    # ---- 步骤6: 生成可视化 ----
    print("\n--- 生成可视化图表 ---")
    _plot_loglog_regression(log_days, log_prices, fit_result, prices.index, output_dir)
    _plot_corridor(df.loc[prices.index], days, corridor, fit_result, output_dir)
    _plot_model_comparison(df.loc[prices.index], days, comparison, output_dir)
    _plot_residual_distribution(fit_result['residuals'], current_percentile, output_dir)

    print("\n" + "=" * 60)
    print("  幂律分析完成")
    print("=" * 60)

    # 返回结果摘要
    return {
        'r_squared': fit_result['r_squared'],
        'power_exponent': fit_result['slope'],
        'intercept': fit_result['intercept'],
        'corridor_prices': {q: corridor[q][-1] for q in quantiles},
        'model_comparison': {
            'power_law_aic': pl['aic'],
            'power_law_bic': pl['bic'],
            'exponential_aic': exp['aic'],
            'exponential_bic': exp['bic'],
            'preferred': comparison['preferred'],
        },
        'current_price': current_price,
        'current_percentile': current_percentile,
    }


if __name__ == '__main__':
    from data_loader import load_daily
    df = load_daily()
    results = run_power_law_analysis(df, output_dir='../output/power_law')
