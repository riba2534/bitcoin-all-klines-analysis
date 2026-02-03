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

        # 盒子大小（在归一化空间中）
        box_size = 1.0 / num_boxes_per_side

        # 计算每个数据点所在的盒子编号
        # x方向：时间划分
        x_box = np.floor(x / box_size).astype(int)
        x_box = np.clip(x_box, 0, num_boxes_per_side - 1)

        # y方向：价格划分
        y_box = np.floor(y / box_size).astype(int)
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
                bx = int(np.clip(np.floor(xi / box_size), 0, num_boxes_per_side - 1))
                by = int(np.clip(np.floor(yi / box_size), 0, num_boxes_per_side - 1))
                occupied.add((bx, by))

        count = len(occupied)
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
    # 5. 总结
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
