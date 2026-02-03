"""波动率聚集与非对称GARCH建模模块

分析内容：
- 多窗口已实现波动率（7d, 30d, 90d）
- 波动率自相关幂律衰减检验（长记忆性）
- GARCH/EGARCH/GJR-GARCH 模型对比
- 杠杆效应分析：收益率与未来波动率的相关性
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import acf
from pathlib import Path
from typing import Optional

from src.data_loader import load_daily, load_klines
from src.preprocessing import log_returns

# 时间尺度（以天为单位）用于X轴
INTERVAL_DAYS = {"5m": 5/(24*60), "1h": 1/24, "4h": 4/24, "1d": 1.0}


# ============================================================
# 1. 多窗口已实现波动率
# ============================================================

def multi_window_realized_vol(returns: pd.Series,
                               windows: list = [7, 30, 90]) -> pd.DataFrame:
    """
    计算多窗口已实现波动率（年化）

    Parameters
    ----------
    returns : pd.Series
        日对数收益率
    windows : list
        滚动窗口列表（天数）

    Returns
    -------
    pd.DataFrame
        各窗口已实现波动率，列名为 'rv_7d', 'rv_30d', 'rv_90d' 等
    """
    vol_df = pd.DataFrame(index=returns.index)
    for w in windows:
        # 已实现波动率 = sqrt(sum(r^2)) * sqrt(365/window) 进行年化
        rv = np.sqrt((returns ** 2).rolling(window=w).sum()) * np.sqrt(365 / w)
        vol_df[f'rv_{w}d'] = rv
    return vol_df.dropna(how='all')


# ============================================================
# 2. 波动率自相关幂律衰减检验（长记忆性）
# ============================================================

def volatility_acf_power_law(returns: pd.Series,
                              max_lags: int = 200) -> dict:
    """
    检验|收益率|的自相关函数是否服从幂律衰减：ACF(k) ~ k^(-d)

    长记忆性判断：若 0 < d < 1，则存在长记忆

    Parameters
    ----------
    returns : pd.Series
        日对数收益率
    max_lags : int
        最大滞后阶数

    Returns
    -------
    dict
        包含幂律拟合参数d、拟合优度R²、ACF值等
    """
    abs_returns = returns.dropna().abs()

    # 计算ACF
    acf_values = acf(abs_returns, nlags=max_lags, fft=True)
    # 从lag=1开始（lag=0始终为1）
    lags = np.arange(1, max_lags + 1)
    acf_vals = acf_values[1:]

    # 只取正的ACF值来做对数拟合
    positive_mask = acf_vals > 0
    lags_pos = lags[positive_mask]
    acf_pos = acf_vals[positive_mask]

    if len(lags_pos) < 10:
        print("[警告] 正的ACF值过少，无法可靠拟合幂律")
        return {
            'd': np.nan, 'r_squared': np.nan,
            'lags': lags, 'acf_values': acf_vals,
            'is_long_memory': False,
        }

    # 对数-对数线性回归: log(ACF) = -d * log(k) + c
    log_lags = np.log(lags_pos)
    log_acf = np.log(acf_pos)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_lags, log_acf)

    d = -slope  # 幂律衰减指数
    r_squared = r_value ** 2

    # 非线性拟合作为对照（幂律函数直接拟合）
    def power_law(k, a, d_param):
        return a * k ** (-d_param)

    try:
        popt, pcov = curve_fit(power_law, lags_pos, acf_pos,
                               p0=[acf_pos[0], d], maxfev=5000)
        d_nonlinear = popt[1]
    except (RuntimeError, ValueError):
        d_nonlinear = np.nan

    results = {
        'd': d,
        'd_nonlinear': d_nonlinear,
        'r_squared': r_squared,
        'slope': slope,
        'intercept': intercept,
        'p_value': p_value,
        'std_err': std_err,
        'lags': lags,
        'acf_values': acf_vals,
        'lags_positive': lags_pos,
        'acf_positive': acf_pos,
        'is_long_memory': 0 < d < 1,
    }
    return results


def multi_scale_volatility_analysis(intervals=None):
    """多尺度波动率聚集分析"""
    if intervals is None:
        intervals = ['5m', '1h', '4h', '1d']

    results = {}
    for interval in intervals:
        try:
            print(f"\n  分析 {interval} 尺度波动率...")
            df_tf = load_klines(interval)
            prices = df_tf['close'].dropna()
            returns = np.log(prices / prices.shift(1)).dropna()

            # 对大数据截断
            if len(returns) > 200000:
                returns = returns.iloc[-200000:]

            if len(returns) < 200:
                print(f"    {interval} 数据不足，跳过")
                continue

            # ACF 幂律衰减（长记忆参数 d）
            acf_result = volatility_acf_power_law(returns, max_lags=min(200, len(returns)//5))

            results[interval] = {
                'd': acf_result['d'],
                'd_nonlinear': acf_result.get('d_nonlinear', np.nan),
                'r_squared': acf_result['r_squared'],
                'is_long_memory': acf_result['is_long_memory'],
                'n_samples': len(returns),
            }

            print(f"    d={acf_result['d']:.4f}, R²={acf_result['r_squared']:.4f}, long_memory={acf_result['is_long_memory']}")

        except FileNotFoundError:
            print(f"    {interval} 数据文件不存在，跳过")
        except Exception as e:
            print(f"    {interval} 分析失败: {e}")

    return results


# ============================================================
# 3. GARCH / EGARCH / GJR-GARCH 模型对比
# ============================================================

def compare_garch_models(returns: pd.Series) -> dict:
    """
    拟合GARCH(1,1)、EGARCH(1,1)、GJR-GARCH(1,1)并比较AIC/BIC

    Parameters
    ----------
    returns : pd.Series
        日对数收益率

    Returns
    -------
    dict
        各模型参数、AIC/BIC、杠杆效应参数
    """
    from arch import arch_model

    r_pct = returns.dropna() * 100  # 百分比收益率
    results = {}

    # --- GARCH(1,1) ---
    model_garch = arch_model(r_pct, vol='Garch', p=1, q=1,
                              mean='Constant', dist='Normal')
    res_garch = model_garch.fit(disp='off')
    results['GARCH'] = {
        'params': dict(res_garch.params),
        'aic': res_garch.aic,
        'bic': res_garch.bic,
        'log_likelihood': res_garch.loglikelihood,
        'conditional_volatility': res_garch.conditional_volatility / 100,
        'result_obj': res_garch,
    }

    # --- EGARCH(1,1) ---
    model_egarch = arch_model(r_pct, vol='EGARCH', p=1, q=1,
                               mean='Constant', dist='Normal')
    res_egarch = model_egarch.fit(disp='off')
    # EGARCH的gamma参数反映杠杆效应（负值表示负收益增大波动率）
    egarch_params = dict(res_egarch.params)
    results['EGARCH'] = {
        'params': egarch_params,
        'aic': res_egarch.aic,
        'bic': res_egarch.bic,
        'log_likelihood': res_egarch.loglikelihood,
        'conditional_volatility': res_egarch.conditional_volatility / 100,
        'leverage_param': egarch_params.get('gamma[1]', np.nan),
        'result_obj': res_egarch,
    }

    # --- GJR-GARCH(1,1) ---
    # GJR-GARCH 在 arch 库中通过 vol='Garch', o=1 实现
    model_gjr = arch_model(r_pct, vol='Garch', p=1, o=1, q=1,
                            mean='Constant', dist='Normal')
    res_gjr = model_gjr.fit(disp='off')
    gjr_params = dict(res_gjr.params)
    results['GJR-GARCH'] = {
        'params': gjr_params,
        'aic': res_gjr.aic,
        'bic': res_gjr.bic,
        'log_likelihood': res_gjr.loglikelihood,
        'conditional_volatility': res_gjr.conditional_volatility / 100,
        # gamma[1] > 0 表示负冲击产生更大波动
        'leverage_param': gjr_params.get('gamma[1]', np.nan),
        'result_obj': res_gjr,
    }

    return results


# ============================================================
# 4. 杠杆效应分析
# ============================================================

def leverage_effect_analysis(returns: pd.Series,
                              forward_windows: list = [5, 10, 20]) -> dict:
    """
    分析收益率与未来波动率的相关性（杠杆效应）

    杠杆效应：负收益倾向于增加未来波动率，正收益倾向于减少未来波动率
    表现为 corr(r_t, vol_{t+k}) < 0

    Parameters
    ----------
    returns : pd.Series
        日对数收益率
    forward_windows : list
        前瞻波动率窗口列表

    Returns
    -------
    dict
        各窗口下的相关系数及显著性
    """
    r = returns.dropna()
    results = {}

    for w in forward_windows:
        # 前瞻已实现波动率
        future_vol = r.abs().rolling(window=w).mean().shift(-w)
        # 对齐有效数据
        valid = pd.DataFrame({'return': r, 'future_vol': future_vol}).dropna()

        if len(valid) < 30:
            results[f'{w}d'] = {
                'correlation': np.nan,
                'p_value': np.nan,
                'n_samples': len(valid),
            }
            continue

        corr, p_val = stats.pearsonr(valid['return'], valid['future_vol'])
        # Spearman秩相关作为稳健性检查
        spearman_corr, spearman_p = stats.spearmanr(valid['return'], valid['future_vol'])

        results[f'{w}d'] = {
            'pearson_correlation': corr,
            'pearson_pvalue': p_val,
            'spearman_correlation': spearman_corr,
            'spearman_pvalue': spearman_p,
            'n_samples': len(valid),
            'return_series': valid['return'],
            'future_vol_series': valid['future_vol'],
        }

    return results


# ============================================================
# 5. 可视化
# ============================================================

def plot_realized_volatility(vol_df: pd.DataFrame, output_dir: Path):
    """绘制多窗口已实现波动率时序图"""
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    labels = {'rv_7d': '7天', 'rv_30d': '30天', 'rv_90d': '90天'}

    for idx, col in enumerate(vol_df.columns):
        label = labels.get(col, col)
        ax.plot(vol_df.index, vol_df[col], linewidth=0.8,
                color=colors[idx % len(colors)],
                label=f'{label}已实现波动率（年化）', alpha=0.85)

    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('年化波动率', fontsize=12)
    ax.set_title('BTC 多窗口已实现波动率', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / 'realized_volatility_multiwindow.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[保存] {output_dir / 'realized_volatility_multiwindow.png'}")


def plot_acf_power_law(acf_results: dict, output_dir: Path):
    """绘制ACF幂律衰减拟合图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    lags = acf_results['lags']
    acf_vals = acf_results['acf_values']

    # 左图：ACF原始值
    ax1 = axes[0]
    ax1.bar(lags, acf_vals, width=1, alpha=0.6, color='steelblue')
    ax1.set_xlabel('滞后阶数', fontsize=11)
    ax1.set_ylabel('ACF', fontsize=11)
    ax1.set_title('|收益率| 自相关函数', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linewidth=0.5)

    # 右图：对数-对数图 + 幂律拟合
    ax2 = axes[1]
    lags_pos = acf_results['lags_positive']
    acf_pos = acf_results['acf_positive']

    ax2.scatter(np.log(lags_pos), np.log(acf_pos), s=10, alpha=0.5,
                color='steelblue', label='实际ACF')

    # 拟合线
    d = acf_results['d']
    intercept = acf_results['intercept']
    x_fit = np.linspace(np.log(lags_pos.min()), np.log(lags_pos.max()), 100)
    y_fit = -d * x_fit + intercept
    ax2.plot(x_fit, y_fit, 'r-', linewidth=2,
             label=f'幂律拟合: d={d:.3f}, R²={acf_results["r_squared"]:.3f}')

    ax2.set_xlabel('log(滞后阶数)', fontsize=11)
    ax2.set_ylabel('log(ACF)', fontsize=11)
    ax2.set_title('幂律衰减拟合（双对数坐标）', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / 'acf_power_law_fit.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[保存] {output_dir / 'acf_power_law_fit.png'}")


def plot_model_comparison(model_results: dict, output_dir: Path):
    """绘制GARCH模型对比图（AIC/BIC + 条件波动率对比）"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    model_names = list(model_results.keys())
    aic_values = [model_results[m]['aic'] for m in model_names]
    bic_values = [model_results[m]['bic'] for m in model_names]

    # 上图：AIC/BIC 对比柱状图
    ax1 = axes[0]
    x = np.arange(len(model_names))
    width = 0.35
    bars1 = ax1.bar(x - width / 2, aic_values, width, label='AIC',
                     color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width / 2, bic_values, width, label='BIC',
                     color='coral', alpha=0.8)

    ax1.set_xlabel('模型', fontsize=12)
    ax1.set_ylabel('信息准则值', fontsize=12)
    ax1.set_title('GARCH 模型信息准则对比（越小越好）', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, fontsize=11)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')

    # 在柱状图上标注数值
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)

    # 下图：各模型条件波动率时序对比
    ax2 = axes[1]
    colors = {'GARCH': '#1f77b4', 'EGARCH': '#ff7f0e', 'GJR-GARCH': '#2ca02c'}
    for name in model_names:
        cv = model_results[name]['conditional_volatility']
        ax2.plot(cv.index, cv.values, linewidth=0.7,
                 color=colors.get(name, 'gray'),
                 label=name, alpha=0.8)

    ax2.set_xlabel('日期', fontsize=12)
    ax2.set_ylabel('条件波动率', fontsize=12)
    ax2.set_title('各GARCH模型条件波动率对比', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / 'garch_model_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[保存] {output_dir / 'garch_model_comparison.png'}")


def plot_leverage_effect(leverage_results: dict, output_dir: Path):
    """绘制杠杆效应散点图"""
    # 找到有数据的窗口
    valid_windows = [w for w, r in leverage_results.items()
                     if 'return_series' in r]
    n_plots = len(valid_windows)
    if n_plots == 0:
        print("[警告] 无有效杠杆效应数据可绘制")
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for idx, window_key in enumerate(valid_windows):
        ax = axes[idx]
        data = leverage_results[window_key]
        ret = data['return_series']
        fvol = data['future_vol_series']

        # 散点图（采样避免过多点）
        n_sample = min(len(ret), 2000)
        sample_idx = np.random.choice(len(ret), n_sample, replace=False)
        ax.scatter(ret.values[sample_idx], fvol.values[sample_idx],
                   s=5, alpha=0.3, color='steelblue')

        # 回归线
        z = np.polyfit(ret.values, fvol.values, 1)
        p = np.poly1d(z)
        x_line = np.linspace(ret.min(), ret.max(), 100)
        ax.plot(x_line, p(x_line), 'r-', linewidth=2)

        corr = data['pearson_correlation']
        p_val = data['pearson_pvalue']
        ax.set_xlabel('当日对数收益率', fontsize=11)
        ax.set_ylabel(f'未来{window_key}平均|收益率|', fontsize=11)
        ax.set_title(f'杠杆效应 ({window_key})\n'
                     f'Pearson r={corr:.4f}, p={p_val:.2e}', fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / 'leverage_effect_scatter.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[保存] {output_dir / 'leverage_effect_scatter.png'}")


def plot_long_memory_vs_scale(ms_results: dict, output_dir: Path):
    """绘制波动率长记忆参数 d vs 时间尺度"""
    if not ms_results:
        print("[警告] 无多尺度分析结果可绘制")
        return

    # 提取数据
    intervals = list(ms_results.keys())
    d_values = [ms_results[i]['d'] for i in intervals]
    time_scales = [INTERVAL_DAYS.get(i, np.nan) for i in intervals]

    # 过滤掉无效值
    valid_data = [(t, d, i) for t, d, i in zip(time_scales, d_values, intervals)
                  if not np.isnan(t) and not np.isnan(d)]

    if not valid_data:
        print("[警告] 无有效数据用于绘制长记忆参数图")
        return

    time_scales_valid, d_values_valid, intervals_valid = zip(*valid_data)

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))

    # 散点图（对数X轴）
    ax.scatter(time_scales_valid, d_values_valid, s=100, color='steelblue',
               edgecolors='black', linewidth=1.5, alpha=0.8, zorder=3)

    # 标注每个点的时间尺度
    for t, d, interval in zip(time_scales_valid, d_values_valid, intervals_valid):
        ax.annotate(interval, (t, d), xytext=(5, 5),
                   textcoords='offset points', fontsize=10, color='darkblue')

    # 参考线
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.6,
              label='d=0 (无长记忆)', zorder=1)
    ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.6,
              label='d=0.5 (临界值)', zorder=1)

    # 设置对数X轴
    ax.set_xscale('log')
    ax.set_xlabel('时间尺度（天，对数刻度）', fontsize=12)
    ax.set_ylabel('长记忆参数 d', fontsize=12)
    ax.set_title('波动率长记忆参数 vs 时间尺度', fontsize=14)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    fig.savefig(output_dir / 'volatility_long_memory_vs_scale.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[保存] {output_dir / 'volatility_long_memory_vs_scale.png'}")


# ============================================================
# 6. 结果打印
# ============================================================

def print_realized_vol_summary(vol_df: pd.DataFrame):
    """打印已实现波动率统计摘要"""
    print("\n" + "=" * 60)
    print("多窗口已实现波动率统计（年化）")
    print("=" * 60)
    summary = vol_df.describe().T
    for col in vol_df.columns:
        s = vol_df[col].dropna()
        print(f"\n  {col}:")
        print(f"    均值:   {s.mean():.4f} ({s.mean() * 100:.2f}%)")
        print(f"    中位数: {s.median():.4f} ({s.median() * 100:.2f}%)")
        print(f"    最大值: {s.max():.4f} ({s.max() * 100:.2f}%)")
        print(f"    最小值: {s.min():.4f} ({s.min() * 100:.2f}%)")
        print(f"    标准差: {s.std():.4f}")


def print_acf_power_law_results(results: dict):
    """打印ACF幂律衰减检验结果"""
    print("\n" + "=" * 60)
    print("波动率自相关幂律衰减检验（长记忆性）")
    print("=" * 60)
    print(f"  幂律衰减指数 d (线性拟合):   {results['d']:.4f}")
    print(f"  幂律衰减指数 d (非线性拟合): {results['d_nonlinear']:.4f}")
    print(f"  拟合优度 R²:                  {results['r_squared']:.4f}")
    print(f"  回归斜率:                     {results['slope']:.4f}")
    print(f"  回归截距:                     {results['intercept']:.4f}")
    print(f"  p值:                          {results['p_value']:.2e}")
    print(f"  标准误:                       {results['std_err']:.4f}")
    print(f"\n  长记忆性判断 (0 < d < 1):     "
          f"{'是 - 存在长记忆性' if results['is_long_memory'] else '否'}")
    if results['is_long_memory']:
        print(f"    → |收益率|的自相关以幂律速度缓慢衰减")
        print(f"    → 波动率聚集具有长记忆特征，GARCH模型的持续性可能不足以刻画")


def print_model_comparison(model_results: dict):
    """打印GARCH模型对比结果"""
    print("\n" + "=" * 60)
    print("GARCH / EGARCH / GJR-GARCH 模型对比")
    print("=" * 60)

    print(f"\n  {'模型':<14} {'AIC':>12} {'BIC':>12} {'对数似然':>12}")
    print("  " + "-" * 52)
    for name, res in model_results.items():
        print(f"  {name:<14} {res['aic']:>12.2f} {res['bic']:>12.2f} "
              f"{res['log_likelihood']:>12.2f}")

    # 找到最优模型
    best_aic = min(model_results.items(), key=lambda x: x[1]['aic'])
    best_bic = min(model_results.items(), key=lambda x: x[1]['bic'])
    print(f"\n  AIC最优模型: {best_aic[0]} (AIC={best_aic[1]['aic']:.2f})")
    print(f"  BIC最优模型: {best_bic[0]} (BIC={best_bic[1]['bic']:.2f})")

    # 杠杆效应参数
    print("\n  杠杆效应参数:")
    for name in ['EGARCH', 'GJR-GARCH']:
        if name in model_results and 'leverage_param' in model_results[name]:
            gamma = model_results[name]['leverage_param']
            print(f"    {name} gamma[1] = {gamma:.6f}")
            if name == 'EGARCH':
                # EGARCH中gamma<0表示负冲击增大波动
                if gamma < 0:
                    print(f"      → gamma < 0: 负收益（下跌）产生更大波动，存在杠杆效应")
                else:
                    print(f"      → gamma >= 0: 未观察到明显杠杆效应")
            elif name == 'GJR-GARCH':
                # GJR-GARCH中gamma>0表示负冲击的额外影响
                if gamma > 0:
                    print(f"      → gamma > 0: 负冲击产生额外波动增量，存在杠杆效应")
                else:
                    print(f"      → gamma <= 0: 未观察到明显杠杆效应")

    # 打印各模型详细参数
    print("\n  各模型详细参数:")
    for name, res in model_results.items():
        print(f"\n  [{name}]")
        for param_name, param_val in res['params'].items():
            print(f"    {param_name}: {param_val:.6f}")


def print_leverage_results(leverage_results: dict):
    """打印杠杆效应分析结果"""
    print("\n" + "=" * 60)
    print("杠杆效应分析：收益率与未来波动率的相关性")
    print("=" * 60)
    print(f"\n  {'窗口':<8} {'Pearson r':>12} {'p值':>12} "
          f"{'Spearman r':>12} {'p值':>12} {'样本数':>8}")
    print("  " + "-" * 66)
    for window, data in leverage_results.items():
        if 'pearson_correlation' in data:
            print(f"  {window:<8} "
                  f"{data['pearson_correlation']:>12.4f} "
                  f"{data['pearson_pvalue']:>12.2e} "
                  f"{data['spearman_correlation']:>12.4f} "
                  f"{data['spearman_pvalue']:>12.2e} "
                  f"{data['n_samples']:>8d}")
        else:
            print(f"  {window:<8} {'N/A':>12} {'N/A':>12} "
                  f"{'N/A':>12} {'N/A':>12} {data.get('n_samples', 0):>8d}")

    # 总结
    print("\n  解读:")
    print("    - 相关系数 < 0: 负收益（下跌）后波动率上升 → 存在杠杆效应")
    print("    - 相关系数 ≈ 0: 收益率方向与未来波动率无关")
    print("    - 相关系数 > 0: 正收益（上涨）后波动率上升（反向杠杆/波动率反馈效应）")
    print("    - 注意: BTC作为加密货币，杠杆效应可能与传统股票不同")


# ============================================================
# 7. 主入口
# ============================================================

def run_volatility_analysis(df: pd.DataFrame, output_dir: str = "output/volatility"):
    """
    波动率聚集与非对称GARCH分析主函数

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
    print("BTC 波动率聚集与非对称 GARCH 分析")
    print("=" * 60)
    print(f"数据范围: {df.index.min()} ~ {df.index.max()}")
    print(f"样本数量: {len(df)}")

    # 计算日对数收益率
    daily_returns = log_returns(df['close'])
    print(f"日对数收益率样本数: {len(daily_returns)}")

    from src.font_config import configure_chinese_font
    configure_chinese_font()

    # 固定随机种子以保证杠杆效应散点图采样可复现
    np.random.seed(42)

    # --- 多窗口已实现波动率 ---
    print("\n>>> 计算多窗口已实现波动率 (7d, 30d, 90d)...")
    vol_df = multi_window_realized_vol(daily_returns, windows=[7, 30, 90])
    print_realized_vol_summary(vol_df)
    plot_realized_volatility(vol_df, output_dir)

    # --- ACF幂律衰减检验 ---
    print("\n>>> 执行波动率自相关幂律衰减检验...")
    acf_results = volatility_acf_power_law(daily_returns, max_lags=200)
    print_acf_power_law_results(acf_results)
    plot_acf_power_law(acf_results, output_dir)

    # --- GARCH模型对比 ---
    print("\n>>> 拟合 GARCH / EGARCH / GJR-GARCH 模型...")
    model_results = compare_garch_models(daily_returns)
    print_model_comparison(model_results)
    plot_model_comparison(model_results, output_dir)

    # --- 杠杆效应分析 ---
    print("\n>>> 执行杠杆效应分析...")
    leverage_results = leverage_effect_analysis(daily_returns,
                                                forward_windows=[5, 10, 20])
    print_leverage_results(leverage_results)
    plot_leverage_effect(leverage_results, output_dir)

    # --- 多尺度波动率分析 ---
    print("\n>>> 多尺度波动率聚集分析 (5m, 1h, 4h, 1d)...")
    ms_vol_results = multi_scale_volatility_analysis(['5m', '1h', '4h', '1d'])
    if ms_vol_results:
        plot_long_memory_vs_scale(ms_vol_results, output_dir)

    print("\n" + "=" * 60)
    print("波动率分析完成！")
    print(f"图表已保存至: {output_dir.resolve()}")
    print("=" * 60)

    # 返回所有结果供后续使用
    return {
        'realized_vol': vol_df,
        'acf_power_law': acf_results,
        'model_comparison': model_results,
        'leverage_effect': leverage_results,
        'multi_scale_volatility': ms_vol_results,
    }


# ============================================================
# 独立运行入口
# ============================================================

if __name__ == '__main__':
    df = load_daily()
    run_volatility_analysis(df)
