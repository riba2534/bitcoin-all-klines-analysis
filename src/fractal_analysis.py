"""
分形维数与自相似性分析模块
========================
通过盒计数法（Box-Counting）计算BTC价格序列的分形维数，
并通过蒙特卡洛模拟与随机游走对比，检验BTC价格是否具有显著不同的分形特征。

核心功能：
- 盒计数法（Box-Counting Dimension）计算分形维数
- 蒙特卡洛模拟对比（Z检验）
- 多尺度自相似性分析
"""

import matplotlib
matplotlib.use('Agg')

from src.font_config import configure_chinese_font
configure_chinese_font()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loader import load_klines
from src.preprocessing import log_returns

import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 盒计数法（Box-Counting Dimension）
# ============================================================
def box_counting_dimension(prices: np.ndarray,
                           num_scales: int = 30,
                           min_boxes: int = 5,
                           max_boxes: int = None) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    盒计数法计算价格序列的分形维数

    方法：
    1. 将价格序列归一化到 [0,1] x [0,1] 空间
    2. 在不同尺度(box size)下计数覆盖曲线所需的盒子数
    3. 通过 log(count) vs log(1/scale) 的线性回归得到分形维数

    Parameters
    ----------
    prices : np.ndarray
        价格序列
    num_scales : int
        尺度数量
    min_boxes : int
        最小划分数量
    max_boxes : int, optional
        最大划分数量，默认为序列长度的1/4

    Returns
    -------
    D : float
        盒计数分形维数
    log_inv_scales : np.ndarray
        log(1/scale) 数组
    log_counts : np.ndarray
        log(count) 数组
    """
    n = len(prices)
    if max_boxes is None:
        max_boxes = n // 4

    # 步骤1：归一化到 [0,1] x [0,1]
    # x轴：时间归一化
    x = np.linspace(0, 1, n)
    # y轴：价格归一化
    y = (prices - prices.min()) / (prices.max() - prices.min())

    # 步骤2：在不同尺度下计数
    # 生成对数均匀分布的划分数量
    box_counts_list = np.unique(
        np.logspace(np.log10(min_boxes), np.log10(max_boxes), num=num_scales).astype(int)
    )

    log_inv_scales = []
    log_counts = []

    for num_boxes_per_side in box_counts_list:
        if num_boxes_per_side < 2:
            continue

        # 独立归一化 x 和 y 到盒子网格，避免纵横比失真
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        if x_range == 0:
            x_range = 1.0
        if y_range == 0:
            y_range = 1.0
        x_box = np.floor((x - x.min()) / x_range * (num_boxes_per_side - 1)).astype(int)
        y_box = np.floor((y - y.min()) / y_range * (num_boxes_per_side - 1)).astype(int)
        x_box = np.clip(x_box, 0, num_boxes_per_side - 1)
        y_box = np.clip(y_box, 0, num_boxes_per_side - 1)

        # 还需要考虑相邻点之间的连线经过的盒子
        occupied = set()
        for i in range(n):
            occupied.add((x_box[i], y_box[i]))

        # 对于相邻点，如果它们不在同一个盒子中，需要插值连接
        for i in range(n - 1):
            if x_box[i] == x_box[i + 1] and y_box[i] == y_box[i + 1]:
                continue

            # 线性插值找出经过的所有盒子
            steps = max(abs(x_box[i + 1] - x_box[i]), abs(y_box[i + 1] - y_box[i])) + 1
            if steps <= 1:
                continue

            for t in np.linspace(0, 1, steps + 1):
                xi = x[i] + t * (x[i + 1] - x[i])
                yi = y[i] + t * (y[i + 1] - y[i])
                bx = int(np.clip(np.floor((xi - x.min()) / x_range * (num_boxes_per_side - 1)), 0, num_boxes_per_side - 1))
                by = int(np.clip(np.floor((yi - y.min()) / y_range * (num_boxes_per_side - 1)), 0, num_boxes_per_side - 1))
                occupied.add((bx, by))

        count = len(occupied)
        box_size = 1.0 / num_boxes_per_side  # 等效盒子大小，用于缩放关系
        if count > 0:
            log_inv_scales.append(np.log(1.0 / box_size))
            log_counts.append(np.log(count))

    log_inv_scales = np.array(log_inv_scales)
    log_counts = np.array(log_counts)

    # 步骤3：线性回归
    if len(log_inv_scales) < 3:
        return 1.5, log_inv_scales, log_counts

    coeffs = np.polyfit(log_inv_scales, log_counts, 1)
    D = coeffs[0]  # 斜率即分形维数

    return D, log_inv_scales, log_counts


# ============================================================
# 蒙特卡洛模拟对比
# ============================================================
def generate_random_walk(n: int, seed: Optional[int] = None) -> np.ndarray:
    """
    生成一条与BTC价格序列等长的随机游走

    Parameters
    ----------
    n : int
        序列长度
    seed : int, optional
        随机种子

    Returns
    -------
    np.ndarray
        随机游走价格序列
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    # 生成标准正态分布的增量
    increments = rng.randn(n - 1)
    # 累积求和得到随机游走
    walk = np.cumsum(increments)
    # 加上一个正的起始值避免负数
    walk = walk - walk.min() + 1.0
    return walk


def monte_carlo_fractal_test(prices: np.ndarray, n_simulations: int = 100,
                              seed: int = 42) -> Dict:
    """
    蒙特卡洛模拟检验BTC分形维数是否显著偏离随机游走

    方法：
    1. 生成n_simulations条随机游走
    2. 计算每条的分形维数
    3. 与BTC分形维数做Z检验

    Parameters
    ----------
    prices : np.ndarray
        BTC价格序列
    n_simulations : int
        模拟次数（默认100）
    seed : int
        随机种子（可重复性）

    Returns
    -------
    dict
        包含BTC分形维数、随机游走分形维数分布、Z检验结果
    """
    n = len(prices)

    # 计算BTC分形维数
    print(f"  计算BTC分形维数...")
    d_btc, _, _ = box_counting_dimension(prices)
    print(f"  BTC分形维数: {d_btc:.4f}")

    # 蒙特卡洛模拟
    print(f"  运行{n_simulations}次随机游走模拟...")
    d_random = []
    for i in range(n_simulations):
        if (i + 1) % 20 == 0:
            print(f"    进度: {i + 1}/{n_simulations}")
        rw = generate_random_walk(n, seed=seed + i)
        d_rw, _, _ = box_counting_dimension(rw)
        d_random.append(d_rw)

    d_random = np.array(d_random)

    # Z检验：BTC分形维数 vs 随机游走分形维数分布
    mean_rw = np.mean(d_random)
    std_rw = np.std(d_random, ddof=1)

    if std_rw > 0:
        z_score = (d_btc - mean_rw) / std_rw
        # 双侧p值
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    else:
        z_score = np.nan
        p_value = np.nan

    result = {
        'BTC分形维数': d_btc,
        '随机游走均值': mean_rw,
        '随机游走标准差': std_rw,
        '随机游走范围': (d_random.min(), d_random.max()),
        'Z统计量': z_score,
        'p值': p_value,
        '显著性(α=0.05)': p_value < 0.05 if not np.isnan(p_value) else False,
        '随机游走分形维数': d_random,
    }

    return result


# ============================================================
# 多尺度自相似性分析
# ============================================================
def multi_scale_self_similarity(prices: np.ndarray,
                                 scales: List[int] = None) -> Dict:
    """
    多尺度自相似性分析：在不同聚合级别下比较统计特征

    方法：
    对价格序列按不同尺度聚合后，比较收益率分布的统计矩
    如果序列具有自相似性，其缩放后的统计特征应保持一致

    Parameters
    ----------
    prices : np.ndarray
        价格序列
    scales : list of int
        聚合尺度，默认 [1, 2, 5, 10, 20, 50]

    Returns
    -------
    dict
        各尺度下的统计特征
    """
    if scales is None:
        scales = [1, 2, 5, 10, 20, 50]

    results = {}

    for scale in scales:
        # 对价格序列按scale聚合（每scale个点取一个）
        aggregated = prices[::scale]
        if len(aggregated) < 30:
            continue

        # 计算对数收益率
        returns = np.diff(np.log(aggregated))
        if len(returns) < 10:
            continue

        results[scale] = {
            '样本量': len(returns),
            '均值': np.mean(returns),
            '标准差': np.std(returns),
            '偏度': float(stats.skew(returns)),
            '峰度': float(stats.kurtosis(returns)),
            # 标准差的缩放关系：如果H是Hurst指数，std(scale) ∝ scale^H
            '标准差(原始)': np.std(returns),
        }

    # 计算缩放指数：log(std) vs log(scale) 的斜率
    valid_scales = sorted(results.keys())
    if len(valid_scales) >= 3:
        log_scales = np.log(valid_scales)
        log_stds = np.log([results[s]['标准差'] for s in valid_scales])
        scaling_exponent = np.polyfit(log_scales, log_stds, 1)[0]
        scaling_result = {
            '缩放指数(H估计)': scaling_exponent,
            '各尺度统计': results,
        }
    else:
        scaling_result = {
            '缩放指数(H估计)': np.nan,
            '各尺度统计': results,
        }

    return scaling_result


# ============================================================
# 多重分形 DFA (MF-DFA)
# ============================================================
def mfdfa_analysis(series: np.ndarray, q_list=None, scales=None) -> Dict:
    """
    多重分形 DFA (MF-DFA)

    计算广义 Hurst 指数 h(q) 和多重分形谱 f(α)

    Parameters
    ----------
    series : np.ndarray
        时间序列（对数收益率）
    q_list : list
        q 值列表，默认 [-5, -4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 5]
    scales : list
        尺度列表，默认对数均匀分布

    Returns
    -------
    dict
        包含 hq, q_list, h_list, tau, alpha, f_alpha, multifractal_width
    """
    if q_list is None:
        q_list = [-5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 5]

    N = len(series)
    if scales is None:
        scales = np.unique(np.logspace(np.log10(10), np.log10(N//4), 20).astype(int))

    # 累积偏差序列
    Y = np.cumsum(series - np.mean(series))

    # 对每个尺度和 q 值计算波动函数
    Fq = {}
    for s in scales:
        n_seg = N // s
        if n_seg < 1:
            continue

        # 正向和反向分段
        var_list = []
        for v in range(n_seg):
            segment = Y[v*s:(v+1)*s]
            x = np.arange(s)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            var_list.append(np.mean((segment - trend)**2))

        for v in range(n_seg):
            segment = Y[N - (v+1)*s:N - v*s]
            x = np.arange(s)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            var_list.append(np.mean((segment - trend)**2))

        var_arr = np.array(var_list)
        var_arr = var_arr[var_arr > 0]  # 去除零方差

        if len(var_arr) == 0:
            continue

        for q in q_list:
            if q == 0:
                fq_val = np.exp(0.5 * np.mean(np.log(var_arr)))
            else:
                fq_val = (np.mean(var_arr ** (q/2))) ** (1/q)

            if q not in Fq:
                Fq[q] = {'scales': [], 'fq': []}
            Fq[q]['scales'].append(s)
            Fq[q]['fq'].append(fq_val)

    # 对每个 q 拟合 h(q)
    hq = {}
    for q in q_list:
        if q not in Fq or len(Fq[q]['scales']) < 3:
            continue
        log_s = np.log(Fq[q]['scales'])
        log_fq = np.log(Fq[q]['fq'])
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_s, log_fq)
        hq[q] = slope

    # 计算多重分形谱 f(α)
    q_vals = sorted(hq.keys())
    h_vals = [hq[q] for q in q_vals]

    # τ(q) = q*h(q) - 1
    tau = [q * hq[q] - 1 for q in q_vals]

    # α = dτ/dq (数值微分)
    alpha = np.gradient(tau, q_vals)

    # f(α) = q*α - τ
    f_alpha = [q_vals[i] * alpha[i] - tau[i] for i in range(len(q_vals))]

    return {
        'hq': hq,  # {q: h(q)}
        'q_list': q_vals,
        'h_list': h_vals,
        'tau': tau,
        'alpha': list(alpha),
        'f_alpha': f_alpha,
        'multifractal_width': max(alpha) - min(alpha) if len(alpha) > 0 else 0,
    }


# ============================================================
# 多时间尺度分形对比
# ============================================================
def multi_timeframe_fractal(df_1h: pd.DataFrame, df_4h: pd.DataFrame, df_1d: pd.DataFrame) -> Dict:
    """
    多时间尺度分形分析对比

    对 1h, 4h, 1d 数据分别做盒计数和 MF-DFA

    Parameters
    ----------
    df_1h : pd.DataFrame
        1小时K线数据
    df_4h : pd.DataFrame
        4小时K线数据
    df_1d : pd.DataFrame
        日线K线数据

    Returns
    -------
    dict
        各时间尺度的分形维数和多重分形宽度
    """
    results = {}

    for name, df in [('1h', df_1h), ('4h', df_4h), ('1d', df_1d)]:
        if df is None or len(df) == 0:
            continue

        prices = df['close'].dropna().values
        if len(prices) < 100:
            continue

        # 盒计数分形维数
        D, _, _ = box_counting_dimension(prices)

        # 计算对数收益率用于 MF-DFA
        returns = np.diff(np.log(prices))

        # 大数据截断（MF-DFA 计算开销较大）
        if len(returns) > 50000:
            returns = returns[-50000:]

        # MF-DFA 分析
        try:
            mfdfa_result = mfdfa_analysis(returns)
            multifractal_width = mfdfa_result['multifractal_width']
            h_q2 = mfdfa_result['hq'].get(2, np.nan)  # q=2 对应标准 Hurst 指数
        except Exception as e:
            print(f"  {name} MF-DFA 计算失败: {e}")
            multifractal_width = np.nan
            h_q2 = np.nan

        results[name] = {
            '样本量': len(prices),
            '分形维数': D,
            'Hurst(从D)': 2.0 - D,  # 仅对自仿射 fBm 严格成立，真实数据为近似值
            '多重分形宽度': multifractal_width,
            'Hurst(MF-DFA,q=2)': h_q2,
        }

    return results


# ============================================================
# 可视化函数
# ============================================================
def plot_box_counting(log_inv_scales: np.ndarray, log_counts: np.ndarray, D: float,
                      output_dir: Path, filename: str = "fractal_box_counting.png"):
    """绘制盒计数法的log-log图"""
    fig, ax = plt.subplots(figsize=(10, 7))

    # 散点
    ax.scatter(log_inv_scales, log_counts, color='steelblue', s=40, zorder=3,
               label='盒计数数据点')

    # 拟合线
    coeffs = np.polyfit(log_inv_scales, log_counts, 1)
    fit_line = np.polyval(coeffs, log_inv_scales)
    ax.plot(log_inv_scales, fit_line, 'r-', linewidth=2,
            label=f'拟合线 (D = {D:.4f})')

    # 参考线：D=1.5（纯随机游走理论值）
    ref_line = 1.5 * log_inv_scales + (log_counts[0] - 1.5 * log_inv_scales[0])
    ax.plot(log_inv_scales, ref_line, 'k--', alpha=0.5, linewidth=1,
            label='D=1.5 (随机游走理论值)')

    ax.set_xlabel('log(1/ε) - 尺度倒数的对数', fontsize=12)
    ax.set_ylabel('log(N(ε)) - 盒子数的对数', fontsize=12)
    ax.set_title(f'BTC 盒计数法分析 (分形维数 D = {D:.4f})', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {filepath}")


def plot_monte_carlo(mc_results: Dict, output_dir: Path,
                     filename: str = "fractal_monte_carlo.png"):
    """绘制蒙特卡洛模拟结果：随机游走分形维数直方图 vs BTC"""
    fig, ax = plt.subplots(figsize=(10, 7))

    d_random = mc_results['随机游走分形维数']
    d_btc = mc_results['BTC分形维数']

    # 直方图
    ax.hist(d_random, bins=20, density=True, alpha=0.7, color='steelblue',
            edgecolor='white', label=f'随机游走 (n={len(d_random)})')

    # BTC分形维数的竖线
    ax.axvline(x=d_btc, color='red', linewidth=2.5, linestyle='-',
               label=f'BTC (D={d_btc:.4f})')

    # 随机游走均值的竖线
    ax.axvline(x=mc_results['随机游走均值'], color='blue', linewidth=1.5, linestyle='--',
               label=f'随机游走均值 (D={mc_results["随机游走均值"]:.4f})')

    # 添加正态分布拟合曲线
    x_range = np.linspace(d_random.min() - 0.05, d_random.max() + 0.05, 200)
    pdf = stats.norm.pdf(x_range, mc_results['随机游走均值'], mc_results['随机游走标准差'])
    ax.plot(x_range, pdf, 'b-', alpha=0.5, linewidth=1)

    # 标注统计信息
    info_text = (
        f"Z统计量: {mc_results['Z统计量']:.2f}\n"
        f"p值: {mc_results['p值']:.4f}\n"
        f"显著性(α=0.05): {'是' if mc_results['显著性(α=0.05)'] else '否'}"
    )
    ax.text(0.02, 0.95, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.set_xlabel('分形维数 D', fontsize=12)
    ax.set_ylabel('概率密度', fontsize=12)
    ax.set_title('BTC分形维数 vs 随机游走蒙特卡洛模拟', fontsize=13)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {filepath}")


def plot_self_similarity(scaling_result: Dict, output_dir: Path,
                         filename: str = "fractal_self_similarity.png"):
    """绘制多尺度自相似性分析图"""
    scale_stats = scaling_result['各尺度统计']
    if not scale_stats:
        print("  没有可绘制的自相似性结果")
        return

    scales = sorted(scale_stats.keys())
    stds = [scale_stats[s]['标准差'] for s in scales]
    skews = [scale_stats[s]['偏度'] for s in scales]
    kurts = [scale_stats[s]['峰度'] for s in scales]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 图1：log(std) vs log(scale) — 缩放关系
    ax1 = axes[0]
    log_scales = np.log(scales)
    log_stds = np.log(stds)

    ax1.scatter(log_scales, log_stds, color='steelblue', s=60, zorder=3)

    if len(log_scales) >= 3:
        coeffs = np.polyfit(log_scales, log_stds, 1)
        fit_line = np.polyval(coeffs, log_scales)
        ax1.plot(log_scales, fit_line, 'r-', linewidth=2,
                 label=f'拟合斜率 H≈{coeffs[0]:.4f}')

    # 参考线 H=0.5
    ref_line = 0.5 * log_scales + (log_stds[0] - 0.5 * log_scales[0])
    ax1.plot(log_scales, ref_line, 'k--', alpha=0.5, label='H=0.5 参考线')

    ax1.set_xlabel('log(聚合尺度)', fontsize=11)
    ax1.set_ylabel('log(标准差)', fontsize=11)
    ax1.set_title('缩放关系 (标准差 vs 尺度)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 图2：偏度随尺度变化
    ax2 = axes[1]
    ax2.bar(range(len(scales)), skews, color='coral', alpha=0.8)
    ax2.set_xticks(range(len(scales)))
    ax2.set_xticklabels([str(s) for s in scales])
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('聚合尺度', fontsize=11)
    ax2.set_ylabel('偏度', fontsize=11)
    ax2.set_title('偏度随尺度变化', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    # 图3：峰度随尺度变化
    ax3 = axes[2]
    ax3.bar(range(len(scales)), kurts, color='seagreen', alpha=0.8)
    ax3.set_xticks(range(len(scales)))
    ax3.set_xticklabels([str(s) for s in scales])
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='正态分布峰度=0')
    ax3.set_xlabel('聚合尺度', fontsize=11)
    ax3.set_ylabel('超额峰度', fontsize=11)
    ax3.set_title('峰度随尺度变化', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'BTC 多尺度自相似性分析 (缩放指数 H≈{scaling_result["缩放指数(H估计)"]:.4f})',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {filepath}")


def plot_mfdfa(mfdfa_result: Dict, output_dir: Path,
               filename: str = "fractal_mfdfa.png"):
    """绘制 MF-DFA 分析结果：h(q) 和 f(α) 谱"""
    if not mfdfa_result or len(mfdfa_result.get('q_list', [])) == 0:
        print("  没有可绘制的 MF-DFA 结果")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 图1: h(q) vs q 曲线
    ax1 = axes[0]
    q_list = mfdfa_result['q_list']
    h_list = mfdfa_result['h_list']

    ax1.plot(q_list, h_list, 'o-', color='steelblue', linewidth=2, markersize=6)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='H=0.5 (随机游走)')
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    ax1.set_xlabel('矩阶 q', fontsize=12)
    ax1.set_ylabel('广义 Hurst 指数 h(q)', fontsize=12)
    ax1.set_title('MF-DFA 广义 Hurst 指数谱', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 图2: f(α) 多重分形谱
    ax2 = axes[1]
    alpha = mfdfa_result['alpha']
    f_alpha = mfdfa_result['f_alpha']

    ax2.plot(alpha, f_alpha, 'o-', color='seagreen', linewidth=2, markersize=6)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='f(α)=1 理论峰值')

    # 标注多重分形宽度
    width = mfdfa_result['multifractal_width']
    ax2.text(0.05, 0.95, f'多重分形宽度 Δα = {width:.4f}',
             transform=ax2.transAxes, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax2.set_xlabel('奇异指数 α', fontsize=12)
    ax2.set_ylabel('多重分形谱 f(α)', fontsize=12)
    ax2.set_title('多重分形谱 f(α)', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'BTC 多重分形 DFA 分析 (Δα = {width:.4f})',
                 fontsize=14, y=1.00)
    fig.tight_layout()
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {filepath}")


def plot_multi_timeframe_fractal(mtf_results: Dict, output_dir: Path,
                                   filename: str = "fractal_multi_timeframe.png"):
    """绘制多时间尺度分形对比图"""
    if not mtf_results:
        print("  没有可绘制的多时间尺度对比结果")
        return

    timeframes = sorted(mtf_results.keys(), key=lambda x: {'1h': 1, '4h': 4, '1d': 24}[x])
    fractal_dims = [mtf_results[tf]['分形维数'] for tf in timeframes]
    multifractal_widths = [mtf_results[tf]['多重分形宽度'] for tf in timeframes]
    hurst_from_d = [mtf_results[tf]['Hurst(从D)'] for tf in timeframes]
    hurst_mfdfa = [mtf_results[tf]['Hurst(MF-DFA,q=2)'] for tf in timeframes]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 图1: 分形维数对比
    ax1 = axes[0, 0]
    x_pos = np.arange(len(timeframes))
    bars1 = ax1.bar(x_pos, fractal_dims, color='steelblue', alpha=0.8)
    ax1.axhline(y=1.5, color='red', linestyle='--', alpha=0.7, label='D=1.5 (随机游走)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(timeframes)
    ax1.set_ylabel('分形维数 D', fontsize=11)
    ax1.set_title('不同时间尺度的分形维数', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # 在柱子上标注数值
    for i, (bar, val) in enumerate(zip(bars1, fractal_dims)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    # 图2: 多重分形宽度对比
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x_pos, multifractal_widths, color='seagreen', alpha=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(timeframes)
    ax2.set_ylabel('多重分形宽度 Δα', fontsize=11)
    ax2.set_title('不同时间尺度的多重分形宽度', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    # 在柱子上标注数值
    for i, (bar, val) in enumerate(zip(bars2, multifractal_widths)):
        if not np.isnan(val):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    # 图3: Hurst 指数对比（两种方法）
    ax3 = axes[1, 0]
    width = 0.35
    x_pos = np.arange(len(timeframes))
    bars3a = ax3.bar(x_pos - width/2, hurst_from_d, width, label='Hurst(从D推算)',
                     color='coral', alpha=0.8)
    bars3b = ax3.bar(x_pos + width/2, hurst_mfdfa, width, label='Hurst(MF-DFA,q=2)',
                     color='orchid', alpha=0.8)
    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='H=0.5 (随机游走)')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(timeframes)
    ax3.set_ylabel('Hurst 指数 H', fontsize=11)
    ax3.set_title('不同时间尺度的 Hurst 指数对比', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')

    # 图4: 样本量信息
    ax4 = axes[1, 1]
    samples = [mtf_results[tf]['样本量'] for tf in timeframes]
    bars4 = ax4.bar(x_pos, samples, color='skyblue', alpha=0.8)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(timeframes)
    ax4.set_ylabel('样本量', fontsize=11)
    ax4.set_title('不同时间尺度的数据量', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')

    # 在柱子上标注数值
    for i, (bar, val) in enumerate(zip(bars4, samples)):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(samples)*0.01,
                f'{val}', ha='center', va='bottom', fontsize=10)

    fig.suptitle('BTC 多时间尺度分形特征对比 (1h vs 4h vs 1d)',
                 fontsize=14, y=0.995)
    fig.tight_layout()
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {filepath}")


# ============================================================
# 主入口函数
# ============================================================
def run_fractal_analysis(df: pd.DataFrame, output_dir: str = "output/fractal") -> Dict:
    """
    分形维数与自相似性综合分析主入口

    Parameters
    ----------
    df : pd.DataFrame
        K线数据（需包含 'close' 列和DatetimeIndex索引）
    output_dir : str
        图表输出目录

    Returns
    -------
    dict
        包含所有分析结果的字典
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    print("=" * 70)
    print("分形维数与自相似性分析")
    print("=" * 70)

    # ----------------------------------------------------------
    # 1. 准备数据
    # ----------------------------------------------------------
    prices = df['close'].dropna().values

    print(f"\n数据概况:")
    print(f"  时间范围: {df.index.min()} ~ {df.index.max()}")
    print(f"  价格序列长度: {len(prices)}")
    print(f"  价格范围: {prices.min():.2f} ~ {prices.max():.2f}")

    # ----------------------------------------------------------
    # 2. 盒计数法分形维数
    # ----------------------------------------------------------
    print("\n" + "-" * 50)
    print("【1】盒计数法 (Box-Counting Dimension)")
    print("-" * 50)

    D, log_inv_scales, log_counts = box_counting_dimension(prices)
    results['盒计数分形维数'] = D

    print(f"  BTC分形维数: D = {D:.4f}")
    print(f"  理论参考值:")
    print(f"    D = 1.0: 光滑曲线（完全可预测）")
    print(f"    D = 1.5: 纯随机游走（布朗运动）")
    print(f"    D = 2.0: 完全填充平面（极端不规则）")

    if D < 1.3:
        interpretation = "序列非常光滑，可能存在强趋势特征"
    elif D < 1.45:
        interpretation = "序列较为光滑，具有一定趋势持续性"
    elif D < 1.55:
        interpretation = "序列接近随机游走特征"
    elif D < 1.7:
        interpretation = "序列较为粗糙，具有一定均值回归倾向"
    else:
        interpretation = "序列非常不规则，高度波动"

    print(f"  BTC解读: {interpretation}")
    results['维数解读'] = interpretation

    # 分形维数与Hurst指数的关系: D = 2 - H
    h_from_d = 2.0 - D
    print(f"\n  由分形维数推算Hurst指数 (D = 2 - H):")
    print(f"    H ≈ {h_from_d:.4f}")
    results['Hurst(从D推算)'] = h_from_d

    # 绘制盒计数log-log图
    plot_box_counting(log_inv_scales, log_counts, D, output_dir)

    # ----------------------------------------------------------
    # 3. 蒙特卡洛模拟对比
    # ----------------------------------------------------------
    print("\n" + "-" * 50)
    print("【2】蒙特卡洛模拟对比 (100次随机游走)")
    print("-" * 50)

    mc_results = monte_carlo_fractal_test(prices, n_simulations=100, seed=42)
    results['蒙特卡洛检验'] = {
        k: v for k, v in mc_results.items() if k != '随机游走分形维数'
    }

    print(f"\n  结果汇总:")
    print(f"    BTC分形维数:     D = {mc_results['BTC分形维数']:.4f}")
    print(f"    随机游走均值:    D = {mc_results['随机游走均值']:.4f} ± {mc_results['随机游走标准差']:.4f}")
    print(f"    随机游走范围:    [{mc_results['随机游走范围'][0]:.4f}, {mc_results['随机游走范围'][1]:.4f}]")
    print(f"    Z统计量:         {mc_results['Z统计量']:.4f}")
    print(f"    p值:             {mc_results['p值']:.6f}")
    print(f"    显著性(α=0.05):  {'是 - BTC与随机游走显著不同' if mc_results['显著性(α=0.05)'] else '否 - 无法拒绝随机游走假设'}")

    # 绘制蒙特卡洛结果图
    plot_monte_carlo(mc_results, output_dir)

    # ----------------------------------------------------------
    # 4. 多尺度自相似性分析
    # ----------------------------------------------------------
    print("\n" + "-" * 50)
    print("【3】多尺度自相似性分析")
    print("-" * 50)

    scaling_result = multi_scale_self_similarity(prices, scales=[1, 2, 5, 10, 20, 50])
    results['多尺度自相似性'] = {
        k: v for k, v in scaling_result.items() if k != '各尺度统计'
    }
    results['多尺度自相似性']['缩放指数(H估计)'] = scaling_result['缩放指数(H估计)']

    print(f"\n  缩放指数 (波动率缩放关系 H估计): {scaling_result['缩放指数(H估计)']:.4f}")
    print(f"  各尺度统计特征:")
    for scale, stat in sorted(scaling_result['各尺度统计'].items()):
        print(f"    尺度={scale:3d}: 样本={stat['样本量']:5d}, "
              f"std={stat['标准差']:.6f}, "
              f"偏度={stat['偏度']:.4f}, "
              f"峰度={stat['峰度']:.4f}")

    # 自相似性判定
    scale_stats = scaling_result['各尺度统计']
    if scale_stats:
        valid_scales = sorted(scale_stats.keys())
        if len(valid_scales) >= 2:
            kurts = [scale_stats[s]['峰度'] for s in valid_scales]
            # 如果峰度随尺度增大而趋向0（正态），说明大尺度下趋向正态
            if all(k > 1.0 for k in kurts):
                print("\n  自相似性判定: 所有尺度均呈现超额峰度（尖峰厚尾），")
                print("  表明BTC收益率分布在各尺度下均偏离正态分布，具有分形特征")
            elif kurts[-1] < kurts[0] * 0.5:
                print("\n  自相似性判定: 峰度随聚合尺度增大而显著下降，")
                print("  表明大尺度下收益率趋于正态，自相似性有限")
            else:
                print("\n  自相似性判定: 峰度随尺度变化不大，具有一定自相似性")

    # 绘制自相似性图
    plot_self_similarity(scaling_result, output_dir)

    # ----------------------------------------------------------
    # 4. 多重分形 DFA 分析
    # ----------------------------------------------------------
    print("\n" + "-" * 50)
    print("【4】多重分形 DFA (MF-DFA) 分析")
    print("-" * 50)

    # 计算对数收益率
    returns = np.diff(np.log(prices))

    # 大数据截断
    if len(returns) > 50000:
        print(f"  数据量较大 ({len(returns)}), 截断至最后 50000 个点进行 MF-DFA 分析")
        returns_for_mfdfa = returns[-50000:]
    else:
        returns_for_mfdfa = returns

    try:
        mfdfa_result = mfdfa_analysis(returns_for_mfdfa)
        results['MF-DFA'] = {
            '多重分形宽度': mfdfa_result['multifractal_width'],
            'Hurst(q=2)': mfdfa_result['hq'].get(2, np.nan),
            'Hurst(q=-2)': mfdfa_result['hq'].get(-2, np.nan),
        }

        print(f"\n  MF-DFA 分析结果:")
        print(f"    多重分形宽度 Δα = {mfdfa_result['multifractal_width']:.4f}")
        print(f"    Hurst 指数 (q=2): H = {mfdfa_result['hq'].get(2, np.nan):.4f}")
        print(f"    Hurst 指数 (q=-2): H = {mfdfa_result['hq'].get(-2, np.nan):.4f}")

        if mfdfa_result['multifractal_width'] > 0.3:
            mf_interpretation = "显著多重分形特征 - 价格波动具有复杂的标度行为"
        elif mfdfa_result['multifractal_width'] > 0.15:
            mf_interpretation = "中等多重分形特征 - 存在一定的多尺度结构"
        else:
            mf_interpretation = "弱多重分形特征 - 接近单一分形"

        print(f"    解读: {mf_interpretation}")
        results['MF-DFA']['解读'] = mf_interpretation

        # 绘制 MF-DFA 图
        plot_mfdfa(mfdfa_result, output_dir)

    except Exception as e:
        print(f"  MF-DFA 分析失败: {e}")
        results['MF-DFA'] = {'错误': str(e)}

    # ----------------------------------------------------------
    # 5. 多时间尺度分形对比
    # ----------------------------------------------------------
    print("\n" + "-" * 50)
    print("【5】多时间尺度分形对比 (1h vs 4h vs 1d)")
    print("-" * 50)

    try:
        # 加载不同时间尺度数据
        print("  加载 1h 数据...")
        df_1h = load_klines('1h')
        print(f"    1h 数据: {len(df_1h)} 条")

        print("  加载 4h 数据...")
        df_4h = load_klines('4h')
        print(f"    4h 数据: {len(df_4h)} 条")

        # df 是日线数据
        df_1d = df
        print(f"  日线数据: {len(df_1d)} 条")

        # 多时间尺度分析
        mtf_results = multi_timeframe_fractal(df_1h, df_4h, df_1d)
        results['多时间尺度对比'] = mtf_results

        print(f"\n  多时间尺度对比结果:")
        for tf in sorted(mtf_results.keys(), key=lambda x: {'1h': 1, '4h': 4, '1d': 24}[x]):
            res = mtf_results[tf]
            print(f"    {tf:3s}: 样本={res['样本量']:6d}, D={res['分形维数']:.4f}, "
                  f"H(从D)={res['Hurst(从D)']:.4f}, Δα={res['多重分形宽度']:.4f}")

        # 绘制多时间尺度对比图
        plot_multi_timeframe_fractal(mtf_results, output_dir)

    except Exception as e:
        print(f"  多时间尺度对比失败: {e}")
        results['多时间尺度对比'] = {'错误': str(e)}

    # ----------------------------------------------------------
    # 6. 总结
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("分析总结")
    print("=" * 70)
    print(f"  盒计数分形维数: D = {D:.4f}")
    print(f"  由D推算Hurst指数: H = {h_from_d:.4f}")
    print(f"  维数解读: {interpretation}")
    print(f"\n  蒙特卡洛检验:")
    if mc_results['显著性(α=0.05)']:
        print(f"    BTC价格序列的分形维数与纯随机游走存在显著差异 (p={mc_results['p值']:.6f})")
        if D < mc_results['随机游走均值']:
            print(f"    BTC的D({D:.4f}) < 随机游走的D({mc_results['随机游走均值']:.4f})，")
            print("    表明BTC价格比纯随机游走更「光滑」，即存在趋势持续性")
        else:
            print(f"    BTC的D({D:.4f}) > 随机游走的D({mc_results['随机游走均值']:.4f})，")
            print("    表明BTC价格比纯随机游走更「粗糙」，即存在均值回归特征")
    else:
        print(f"    无法在5%显著性水平下拒绝BTC为随机游走的假设 (p={mc_results['p值']:.6f})")

    print(f"\n  波动率缩放指数: H ≈ {scaling_result['缩放指数(H估计)']:.4f}")
    print(f"    H > 0.5: 波动率超线性增长 → 趋势持续性")
    print(f"    H < 0.5: 波动率亚线性增长 → 均值回归性")
    print(f"    H ≈ 0.5: 波动率线性增长 → 随机游走")

    print(f"\n  图表已保存至: {output_dir.resolve()}")
    print("=" * 70)

    return results


# ============================================================
# 独立运行入口
# ============================================================
if __name__ == "__main__":
    from data_loader import load_daily

    print("加载BTC日线数据...")
    df = load_daily()
    print(f"数据加载完成: {len(df)} 条记录")

    results = run_fractal_analysis(df, output_dir="output/fractal")
