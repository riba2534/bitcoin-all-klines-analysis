"""收益率分布分析与GARCH建模模块

分析内容：
- 正态性检验（KS、JB、AD）
- 厚尾特征分析（峰度、偏度、超越比率）
- 多时间尺度收益率分布对比
- QQ图
- GARCH(1,1) 条件波动率建模
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from pathlib import Path
from typing import Optional

from src.data_loader import load_klines
from src.preprocessing import log_returns


# ============================================================
# 1. 正态性检验
# ============================================================

def normality_tests(returns: pd.Series) -> dict:
    """
    对收益率序列进行多种正态性检验

    Parameters
    ----------
    returns : pd.Series
        对数收益率序列（已去除NaN）

    Returns
    -------
    dict
        包含KS、JB、AD检验统计量和p值的字典
    """
    r = returns.dropna().values

    # Kolmogorov-Smirnov 检验（与标准正态比较）
    r_standardized = (r - r.mean()) / r.std()
    ks_stat, ks_p = stats.kstest(r_standardized, 'norm')

    # Jarque-Bera 检验
    jb_stat, jb_p = stats.jarque_bera(r)

    # Anderson-Darling 检验
    ad_result = stats.anderson(r, dist='norm')

    results = {
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_p,
        'jb_statistic': jb_stat,
        'jb_pvalue': jb_p,
        'ad_statistic': ad_result.statistic,
        'ad_critical_values': dict(zip(
            [f'{sl}%' for sl in ad_result.significance_level],
            ad_result.critical_values
        )),
    }
    return results


# ============================================================
# 2. 厚尾分析
# ============================================================

def fat_tail_analysis(returns: pd.Series) -> dict:
    """
    厚尾特征分析：峰度、偏度、σ超越比率

    Parameters
    ----------
    returns : pd.Series
        对数收益率序列

    Returns
    -------
    dict
        峰度、偏度、3σ/4σ超越比率及其与正态分布的对比
    """
    r = returns.dropna().values
    mu, sigma = r.mean(), r.std()

    # 基础统计
    excess_kurtosis = stats.kurtosis(r)  # scipy默认是excess kurtosis
    skewness = stats.skew(r)

    # 实际超越比率
    r_std = (r - mu) / sigma
    exceed_3sigma = np.mean(np.abs(r_std) > 3)
    exceed_4sigma = np.mean(np.abs(r_std) > 4)

    # 正态分布理论超越比率
    normal_3sigma = 2 * (1 - stats.norm.cdf(3))  # ≈ 0.0027
    normal_4sigma = 2 * (1 - stats.norm.cdf(4))  # ≈ 0.0001

    results = {
        'excess_kurtosis': excess_kurtosis,
        'skewness': skewness,
        'exceed_3sigma_actual': exceed_3sigma,
        'exceed_3sigma_normal': normal_3sigma,
        'exceed_3sigma_ratio': exceed_3sigma / normal_3sigma if normal_3sigma > 0 else np.inf,
        'exceed_4sigma_actual': exceed_4sigma,
        'exceed_4sigma_normal': normal_4sigma,
        'exceed_4sigma_ratio': exceed_4sigma / normal_4sigma if normal_4sigma > 0 else np.inf,
    }
    return results


# ============================================================
# 3. 多时间尺度分布对比
# ============================================================

def multi_timeframe_distributions() -> dict:
    """
    加载1h/4h/1d/1w数据，计算各时间尺度的对数收益率分布

    Returns
    -------
    dict
        {interval: pd.Series} 各时间尺度的对数收益率
    """
    intervals = ['1h', '4h', '1d', '1w']
    distributions = {}
    for interval in intervals:
        try:
            df = load_klines(interval)
            ret = log_returns(df['close'])
            distributions[interval] = ret
        except FileNotFoundError:
            print(f"[警告] {interval} 数据文件不存在，跳过")
    return distributions


# ============================================================
# 4. GARCH(1,1) 建模
# ============================================================

def fit_garch11(returns: pd.Series) -> dict:
    """
    拟合GARCH(1,1)模型

    Parameters
    ----------
    returns : pd.Series
        对数收益率序列（百分比化后传入arch库）

    Returns
    -------
    dict
        包含模型参数、持续性、条件波动率序列的字典
    """
    from arch import arch_model

    # arch库推荐使用百分比收益率以改善数值稳定性
    r_pct = returns.dropna() * 100

    # 拟合GARCH(1,1)，均值模型用常数均值
    model = arch_model(r_pct, vol='Garch', p=1, q=1, mean='Constant', dist='Normal')
    result = model.fit(disp='off')

    # 提取参数
    params = result.params
    omega = params.get('omega', np.nan)
    alpha = params.get('alpha[1]', np.nan)
    beta = params.get('beta[1]', np.nan)
    persistence = alpha + beta

    # 条件波动率（转回原始比例）
    cond_vol = result.conditional_volatility / 100

    results = {
        'model_summary': str(result.summary()),
        'omega': omega,
        'alpha': alpha,
        'beta': beta,
        'persistence': persistence,
        'log_likelihood': result.loglikelihood,
        'aic': result.aic,
        'bic': result.bic,
        'conditional_volatility': cond_vol,
        'result_obj': result,
    }
    return results


# ============================================================
# 5. 可视化
# ============================================================

def plot_histogram_vs_normal(returns: pd.Series, output_dir: Path):
    """绘制收益率直方图与正态分布对比"""
    r = returns.dropna().values
    mu, sigma = r.mean(), r.std()

    fig, ax = plt.subplots(figsize=(12, 6))

    # 直方图
    n_bins = 150
    ax.hist(r, bins=n_bins, density=True, alpha=0.65, color='steelblue',
            edgecolor='white', linewidth=0.3, label='BTC日对数收益率')

    # 正态分布拟合曲线
    x = np.linspace(r.min(), r.max(), 500)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
            label=f'正态分布 N({mu:.5f}, {sigma:.4f}²)')

    ax.set_xlabel('日对数收益率', fontsize=12)
    ax.set_ylabel('概率密度', fontsize=12)
    ax.set_title('BTC日对数收益率分布 vs 正态分布', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / 'returns_histogram_vs_normal.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[保存] {output_dir / 'returns_histogram_vs_normal.png'}")


def plot_qq(returns: pd.Series, output_dir: Path):
    """绘制QQ图"""
    fig, ax = plt.subplots(figsize=(8, 8))
    r = returns.dropna().values

    # QQ图
    (osm, osr), (slope, intercept, _) = stats.probplot(r, dist='norm')
    ax.scatter(osm, osr, s=5, alpha=0.5, color='steelblue', label='样本分位数')
    # 理论线
    x_line = np.array([osm.min(), osm.max()])
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2, label='理论正态线')

    ax.set_xlabel('理论分位数（正态）', fontsize=12)
    ax.set_ylabel('样本分位数', fontsize=12)
    ax.set_title('BTC日对数收益率 QQ图', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / 'returns_qq_plot.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[保存] {output_dir / 'returns_qq_plot.png'}")


def plot_multi_timeframe(distributions: dict, output_dir: Path):
    """绘制多时间尺度收益率分布对比"""
    n_plots = len(distributions)
    if n_plots == 0:
        print("[警告] 无可用的多时间尺度数据")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    interval_names = {
        '1h': '1小时', '4h': '4小时', '1d': '1天', '1w': '1周'
    }

    for idx, (interval, ret) in enumerate(distributions.items()):
        if idx >= 4:
            break
        ax = axes[idx]
        r = ret.dropna().values
        mu, sigma = r.mean(), r.std()

        ax.hist(r, bins=100, density=True, alpha=0.65, color='steelblue',
                edgecolor='white', linewidth=0.3)

        x = np.linspace(r.min(), r.max(), 500)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=1.5)

        # 统计信息
        kurt = stats.kurtosis(r)
        skew = stats.skew(r)
        label = interval_names.get(interval, interval)
        ax.set_title(f'{label}收益率 (峰度={kurt:.2f}, 偏度={skew:.3f})', fontsize=11)
        ax.set_xlabel('对数收益率', fontsize=10)
        ax.set_ylabel('概率密度', fontsize=10)
        ax.grid(True, alpha=0.3)

    # 隐藏多余子图
    for idx in range(len(distributions), 4):
        axes[idx].set_visible(False)

    fig.suptitle('多时间尺度BTC对数收益率分布', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / 'multi_timeframe_distributions.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[保存] {output_dir / 'multi_timeframe_distributions.png'}")


def plot_garch_conditional_vol(garch_results: dict, output_dir: Path):
    """绘制GARCH(1,1)条件波动率时序图"""
    cond_vol = garch_results['conditional_volatility']

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(cond_vol.index, cond_vol.values, linewidth=0.8, color='steelblue')
    ax.fill_between(cond_vol.index, 0, cond_vol.values, alpha=0.2, color='steelblue')

    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('条件波动率', fontsize=12)
    ax.set_title(
        f'GARCH(1,1) 条件波动率  '
        f'(α={garch_results["alpha"]:.4f}, β={garch_results["beta"]:.4f}, '
        f'持续性={garch_results["persistence"]:.4f})',
        fontsize=13
    )
    ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / 'garch_conditional_volatility.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[保存] {output_dir / 'garch_conditional_volatility.png'}")


# ============================================================
# 6. 结果打印
# ============================================================

def print_normality_results(results: dict):
    """打印正态性检验结果"""
    print("\n" + "=" * 60)
    print("正态性检验结果")
    print("=" * 60)

    print(f"\n[KS检验] Kolmogorov-Smirnov")
    print(f"  统计量: {results['ks_statistic']:.6f}")
    print(f"  p值:    {results['ks_pvalue']:.2e}")
    print(f"  结论:   {'拒绝正态假设' if results['ks_pvalue'] < 0.05 else '不能拒绝正态假设'}")

    print(f"\n[JB检验] Jarque-Bera")
    print(f"  统计量: {results['jb_statistic']:.4f}")
    print(f"  p值:    {results['jb_pvalue']:.2e}")
    print(f"  结论:   {'拒绝正态假设' if results['jb_pvalue'] < 0.05 else '不能拒绝正态假设'}")

    print(f"\n[AD检验] Anderson-Darling")
    print(f"  统计量: {results['ad_statistic']:.4f}")
    print("  临界值:")
    for level, cv in results['ad_critical_values'].items():
        reject = results['ad_statistic'] > cv
        print(f"    {level}: {cv:.4f} {'(拒绝)' if reject else '(不拒绝)'}")


def print_fat_tail_results(results: dict):
    """打印厚尾分析结果"""
    print("\n" + "=" * 60)
    print("厚尾特征分析")
    print("=" * 60)
    print(f"  超额峰度 (excess kurtosis): {results['excess_kurtosis']:.4f}")
    print(f"    (正态分布=0，值越大尾部越厚)")
    print(f"  偏度 (skewness):             {results['skewness']:.4f}")
    print(f"    (正态分布=0，负值表示左偏)")

    print(f"\n  3σ超越比率:")
    print(f"    实际: {results['exceed_3sigma_actual']:.6f} "
          f"({results['exceed_3sigma_actual'] * 100:.3f}%)")
    print(f"    正态: {results['exceed_3sigma_normal']:.6f} "
          f"({results['exceed_3sigma_normal'] * 100:.3f}%)")
    print(f"    倍数: {results['exceed_3sigma_ratio']:.2f}x")

    print(f"\n  4σ超越比率:")
    print(f"    实际: {results['exceed_4sigma_actual']:.6f} "
          f"({results['exceed_4sigma_actual'] * 100:.4f}%)")
    print(f"    正态: {results['exceed_4sigma_normal']:.6f} "
          f"({results['exceed_4sigma_normal'] * 100:.4f}%)")
    print(f"    倍数: {results['exceed_4sigma_ratio']:.2f}x")


def print_garch_results(results: dict):
    """打印GARCH(1,1)建模结果"""
    print("\n" + "=" * 60)
    print("GARCH(1,1) 建模结果")
    print("=" * 60)
    print(f"  ω (omega):    {results['omega']:.6f}")
    print(f"  α (alpha[1]): {results['alpha']:.6f}")
    print(f"  β (beta[1]):  {results['beta']:.6f}")
    print(f"  持续性 (α+β): {results['persistence']:.6f}")
    print(f"    {'高持续性（接近1）→波动率冲击衰减缓慢' if results['persistence'] > 0.9 else '中等持续性'}")
    print(f"  对数似然值:    {results['log_likelihood']:.4f}")
    print(f"  AIC:           {results['aic']:.4f}")
    print(f"  BIC:           {results['bic']:.4f}")


# ============================================================
# 7. 主入口
# ============================================================

def run_returns_analysis(df: pd.DataFrame, output_dir: str = "output/returns"):
    """
    收益率分布分析主函数

    Parameters
    ----------
    df : pd.DataFrame
        日线K线数据（含'close'列，DatetimeIndex索引）
    output_dir : str
        图表输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BTC 收益率分布分析与 GARCH 建模")
    print("=" * 60)
    print(f"数据范围: {df.index.min()} ~ {df.index.max()}")
    print(f"样本数量: {len(df)}")

    # 计算日对数收益率
    daily_returns = log_returns(df['close'])
    print(f"日对数收益率样本数: {len(daily_returns)}")

    # --- 正态性检验 ---
    print("\n>>> 执行正态性检验...")
    norm_results = normality_tests(daily_returns)
    print_normality_results(norm_results)

    # --- 厚尾分析 ---
    print("\n>>> 执行厚尾分析...")
    tail_results = fat_tail_analysis(daily_returns)
    print_fat_tail_results(tail_results)

    # --- 多时间尺度分布 ---
    print("\n>>> 加载多时间尺度数据...")
    distributions = multi_timeframe_distributions()
    # 打印各尺度统计
    print("\n多时间尺度对数收益率统计:")
    print(f"  {'尺度':<8} {'样本数':>8} {'均值':>12} {'标准差':>12} {'峰度':>10} {'偏度':>10}")
    print("  " + "-" * 62)
    for interval, ret in distributions.items():
        r = ret.dropna().values
        print(f"  {interval:<8} {len(r):>8d} {r.mean():>12.6f} {r.std():>12.6f} "
              f"{stats.kurtosis(r):>10.4f} {stats.skew(r):>10.4f}")

    # --- GARCH(1,1) 建模 ---
    print("\n>>> 拟合 GARCH(1,1) 模型...")
    garch_results = fit_garch11(daily_returns)
    print_garch_results(garch_results)

    # --- 生成可视化 ---
    print("\n>>> 生成可视化图表...")

    from src.font_config import configure_chinese_font
    configure_chinese_font()

    plot_histogram_vs_normal(daily_returns, output_dir)
    plot_qq(daily_returns, output_dir)
    plot_multi_timeframe(distributions, output_dir)
    plot_garch_conditional_vol(garch_results, output_dir)

    print("\n" + "=" * 60)
    print("收益率分布分析完成！")
    print(f"图表已保存至: {output_dir.resolve()}")
    print("=" * 60)

    # 返回所有结果供后续使用
    return {
        'normality': norm_results,
        'fat_tail': tail_results,
        'multi_timeframe': distributions,
        'garch': garch_results,
    }


# ============================================================
# 独立运行入口
# ============================================================

if __name__ == '__main__':
    from src.data_loader import load_daily
    df = load_daily()
    run_returns_analysis(df)
