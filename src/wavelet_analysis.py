"""小波变换分析模块 - CWT时频分析、全局小波谱、显著性检验、周期强度追踪"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
from scipy.signal import detrend
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.preprocessing import log_returns, standardize


# ============================================================================
# 核心参数配置
# ============================================================================

WAVELET = 'cmor1.5-1.0'          # 复Morlet小波 (bandwidth=1.5, center_freq=1.0)
MIN_PERIOD = 7                     # 最小周期（天）
MAX_PERIOD = 1500                  # 最大周期（天）
NUM_SCALES = 256                   # 尺度数量
KEY_PERIODS = [30, 90, 365, 1400]  # 关键追踪周期（天）
N_SURROGATES = 1000                # Monte Carlo替代数据数量
SIGNIFICANCE_LEVEL = 0.95          # 显著性水平
DPI = 150                          # 图像分辨率


# ============================================================================
# 辅助函数：尺度与周期转换
# ============================================================================

def _periods_to_scales(periods: np.ndarray, wavelet: str, dt: float = 1.0) -> np.ndarray:
    """将周期（天）转换为CWT尺度参数

    Parameters
    ----------
    periods : np.ndarray
        目标周期数组（天）
    wavelet : str
        小波名称
    dt : float
        采样间隔（天）

    Returns
    -------
    np.ndarray
        对应的尺度数组
    """
    central_freq = pywt.central_frequency(wavelet)
    scales = central_freq * periods / dt
    return scales


def _scales_to_periods(scales: np.ndarray, wavelet: str, dt: float = 1.0) -> np.ndarray:
    """将CWT尺度参数转换为周期（天）"""
    central_freq = pywt.central_frequency(wavelet)
    periods = scales * dt / central_freq
    return periods


# ============================================================================
# 核心计算：连续小波变换
# ============================================================================

def compute_cwt(
    signal: np.ndarray,
    dt: float = 1.0,
    wavelet: str = WAVELET,
    min_period: float = MIN_PERIOD,
    max_period: float = MAX_PERIOD,
    num_scales: int = NUM_SCALES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算连续小波变换（CWT）

    Parameters
    ----------
    signal : np.ndarray
        输入时间序列（建议已标准化）
    dt : float
        采样间隔（天）
    wavelet : str
        小波函数名称
    min_period : float
        最小分析周期（天）
    max_period : float
        最大分析周期（天）
    num_scales : int
        尺度分辨率

    Returns
    -------
    coeffs : np.ndarray
        CWT系数矩阵 (n_scales, n_times)
    periods : np.ndarray
        对应周期数组（天）
    scales : np.ndarray
        尺度数组
    """
    # 生成对数等间隔的周期序列
    periods = np.logspace(np.log10(min_period), np.log10(max_period), num_scales)
    scales = _periods_to_scales(periods, wavelet, dt)

    # 执行CWT
    coeffs, _ = pywt.cwt(signal, scales, wavelet, sampling_period=dt)

    return coeffs, periods, scales


def compute_power_spectrum(coeffs: np.ndarray) -> np.ndarray:
    """计算小波功率谱 |W(s,t)|^2

    Parameters
    ----------
    coeffs : np.ndarray
        CWT系数矩阵

    Returns
    -------
    np.ndarray
        功率谱矩阵
    """
    return np.abs(coeffs) ** 2


# ============================================================================
# 影响锥（Cone of Influence）
# ============================================================================

def compute_coi(n: int, dt: float = 1.0, wavelet: str = WAVELET) -> np.ndarray:
    """计算影响锥（COI）边界

    影响锥标识边界效应显著的区域。对于Morlet小波，
    COI对应于e-folding时间 sqrt(2) * scale。

    Parameters
    ----------
    n : int
        时间序列长度
    dt : float
        采样间隔
    wavelet : str
        小波名称

    Returns
    -------
    coi_periods : np.ndarray
        每个时间点对应的COI周期边界（天）
    """
    # e-folding time for Morlet wavelet: sqrt(2) * s
    # COI period = sqrt(2) * s * dt / central_freq
    central_freq = pywt.central_frequency(wavelet)
    # 从两端递增到中间
    t = np.arange(n) * dt
    coi_time = np.minimum(t, (n - 1) * dt - t)
    # 转换为周期：COI_period = sqrt(2) * coi_time * central_freq (反推)
    # 实际上 COI boundary in period space: period = sqrt(2) * dt * index / central_freq * central_freq
    # 简化: coi_period = sqrt(2) * coi_time
    coi_periods = np.sqrt(2) * coi_time
    # 最小值截断到最小周期
    coi_periods = np.maximum(coi_periods, dt)
    return coi_periods


# ============================================================================
# AR(1) 红噪声显著性检验（Monte Carlo方法）
# ============================================================================

def _estimate_ar1(signal: np.ndarray) -> float:
    """估计信号的AR(1)自相关系数（lag-1 autocorrelation）

    Parameters
    ----------
    signal : np.ndarray
        输入时间序列

    Returns
    -------
    float
        lag-1自相关系数
    """
    n = len(signal)
    x = signal - np.mean(signal)
    c0 = np.sum(x ** 2) / n
    c1 = np.sum(x[:-1] * x[1:]) / n
    if c0 == 0:
        return 0.0
    alpha = c1 / c0
    return np.clip(alpha, -0.999, 0.999)


def _generate_ar1_surrogate(n: int, alpha: float, variance: float) -> np.ndarray:
    """生成AR(1)红噪声替代数据

    x(t) = alpha * x(t-1) + noise

    Parameters
    ----------
    n : int
        序列长度
    alpha : float
        AR(1)系数
    variance : float
        原始信号方差

    Returns
    -------
    np.ndarray
        AR(1)替代序列
    """
    noise_std = np.sqrt(variance * (1 - alpha ** 2))
    noise = np.random.normal(0, noise_std, n)
    surrogate = np.zeros(n)
    surrogate[0] = noise[0]
    for i in range(1, n):
        surrogate[i] = alpha * surrogate[i - 1] + noise[i]
    return surrogate


def significance_test_monte_carlo(
    signal: np.ndarray,
    periods: np.ndarray,
    dt: float = 1.0,
    wavelet: str = WAVELET,
    n_surrogates: int = N_SURROGATES,
    significance_level: float = SIGNIFICANCE_LEVEL,
) -> Tuple[np.ndarray, np.ndarray]:
    """AR(1)红噪声Monte Carlo显著性检验

    生成大量AR(1)替代数据，计算其全局小波谱分布，
    得到指定置信水平的阈值。

    Parameters
    ----------
    signal : np.ndarray
        原始时间序列
    periods : np.ndarray
        CWT分析的周期数组
    dt : float
        采样间隔
    wavelet : str
        小波名称
    n_surrogates : int
        替代数据数量
    significance_level : float
        显著性水平（如0.95对应95%置信度）

    Returns
    -------
    significance_threshold : np.ndarray
        各周期的显著性阈值
    surrogate_spectra : np.ndarray
        所有替代数据的全局谱 (n_surrogates, n_periods)
    """
    n = len(signal)
    alpha = _estimate_ar1(signal)
    variance = np.var(signal)
    scales = _periods_to_scales(periods, wavelet, dt)

    print(f"  AR(1) 系数 alpha = {alpha:.4f}")
    print(f"  生成 {n_surrogates} 个AR(1)替代数据进行Monte Carlo检验...")

    surrogate_global_spectra = np.zeros((n_surrogates, len(periods)))

    for i in range(n_surrogates):
        surrogate = _generate_ar1_surrogate(n, alpha, variance)
        coeffs_surr, _ = pywt.cwt(surrogate, scales, wavelet, sampling_period=dt)
        power_surr = np.abs(coeffs_surr) ** 2
        surrogate_global_spectra[i, :] = np.mean(power_surr, axis=1)

        if (i + 1) % 200 == 0:
            print(f"    Monte Carlo 进度: {i + 1}/{n_surrogates}")

    # 计算指定分位数作为显著性阈值
    percentile = significance_level * 100
    significance_threshold = np.percentile(surrogate_global_spectra, percentile, axis=0)

    return significance_threshold, surrogate_global_spectra


# ============================================================================
# 全局小波谱
# ============================================================================

def compute_global_wavelet_spectrum(power: np.ndarray) -> np.ndarray:
    """计算全局小波谱（时间平均功率）

    Parameters
    ----------
    power : np.ndarray
        功率谱矩阵 (n_scales, n_times)

    Returns
    -------
    np.ndarray
        全局小波谱 (n_scales,)
    """
    return np.mean(power, axis=1)


def find_significant_periods(
    global_spectrum: np.ndarray,
    significance_threshold: np.ndarray,
    periods: np.ndarray,
) -> List[Dict]:
    """找出超过显著性阈值的周期峰

    在全局谱中检测超过95%置信水平的局部极大值。

    Parameters
    ----------
    global_spectrum : np.ndarray
        全局小波谱
    significance_threshold : np.ndarray
        显著性阈值
    periods : np.ndarray
        周期数组

    Returns
    -------
    list of dict
        显著周期列表，每项包含 period, power, threshold, ratio
    """
    # 找出超过阈值的区域
    above_mask = global_spectrum > significance_threshold

    significant = []
    if not np.any(above_mask):
        return significant

    # 在超过阈值的连续区间内找峰值
    diff = np.diff(above_mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    # 处理边界情况
    if above_mask[0]:
        starts = np.insert(starts, 0, 0)
    if above_mask[-1]:
        ends = np.append(ends, len(above_mask))

    for s, e in zip(starts, ends):
        segment = global_spectrum[s:e]
        peak_idx = s + np.argmax(segment)
        significant.append({
            'period': float(periods[peak_idx]),
            'power': float(global_spectrum[peak_idx]),
            'threshold': float(significance_threshold[peak_idx]),
            'ratio': float(global_spectrum[peak_idx] / significance_threshold[peak_idx]),
        })

    # 按功率降序排列
    significant.sort(key=lambda x: x['power'], reverse=True)
    return significant


# ============================================================================
# 关键周期功率时间演化
# ============================================================================

def extract_power_at_periods(
    power: np.ndarray,
    periods: np.ndarray,
    key_periods: List[float] = None,
) -> Dict[float, np.ndarray]:
    """提取关键周期处的功率随时间变化

    Parameters
    ----------
    power : np.ndarray
        功率谱矩阵 (n_scales, n_times)
    periods : np.ndarray
        周期数组
    key_periods : list of float
        要追踪的关键周期（天）

    Returns
    -------
    dict
        {period: power_time_series} 映射
    """
    if key_periods is None:
        key_periods = KEY_PERIODS

    result = {}
    for target_period in key_periods:
        # 找到最接近目标周期的尺度索引
        idx = np.argmin(np.abs(periods - target_period))
        actual_period = periods[idx]
        result[target_period] = {
            'power': power[idx, :],
            'actual_period': float(actual_period),
        }

    return result


# ============================================================================
# 可视化模块
# ============================================================================

def plot_cwt_scalogram(
    power: np.ndarray,
    periods: np.ndarray,
    dates: pd.DatetimeIndex,
    coi_periods: np.ndarray,
    output_path: Path,
    title: str = 'BTC/USDT CWT 时频功率谱（Scalogram）',
) -> None:
    """绘制CWT scalogram（时间-周期-功率热力图）含影响锥

    Parameters
    ----------
    power : np.ndarray
        功率谱矩阵
    periods : np.ndarray
        周期数组（天）
    dates : pd.DatetimeIndex
        时间索引
    coi_periods : np.ndarray
        影响锥边界
    output_path : Path
        输出文件路径
    title : str
        图标题
    """
    fig, ax = plt.subplots(figsize=(16, 8))

    # 使用对数归一化的伪彩色图
    t = mdates.date2num(dates.to_pydatetime())
    T, P = np.meshgrid(t, periods)

    # 功率取对数以获得更好的视觉效果
    power_plot = power.copy()
    power_plot[power_plot <= 0] = np.min(power_plot[power_plot > 0]) * 0.1

    im = ax.pcolormesh(
        T, P, power_plot,
        cmap='jet',
        norm=LogNorm(vmin=np.percentile(power_plot, 5), vmax=np.percentile(power_plot, 99)),
        shading='auto',
    )

    # 绘制影响锥（COI）
    coi_t = mdates.date2num(dates.to_pydatetime())
    ax.fill_between(
        coi_t, coi_periods, periods[-1] * 1.1,
        alpha=0.3, facecolor='white', hatch='x',
        label='影响锥 (COI)',
    )

    # Y轴对数刻度
    ax.set_yscale('log')
    ax.set_ylim(periods[0], periods[-1])
    ax.invert_yaxis()

    # 标记关键周期
    for kp in KEY_PERIODS:
        if periods[0] <= kp <= periods[-1]:
            ax.axhline(y=kp, color='white', linestyle='--', alpha=0.6, linewidth=0.8)
            ax.text(t[-1] + (t[-1] - t[0]) * 0.01, kp, f'{kp}d',
                    color='white', fontsize=8, va='center')

    # 格式化
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('周期（天）', fontsize=12)
    ax.set_title(title, fontsize=14)

    cbar = fig.colorbar(im, ax=ax, pad=0.08, shrink=0.8)
    cbar.set_label('功率（对数尺度）', fontsize=10)

    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Scalogram 已保存: {output_path}")


def plot_global_spectrum(
    global_spectrum: np.ndarray,
    significance_threshold: np.ndarray,
    periods: np.ndarray,
    significant_periods: List[Dict],
    output_path: Path,
    title: str = 'BTC/USDT 全局小波谱 + 95%显著性',
) -> None:
    """绘制全局小波谱及95%红噪声显著性阈值

    Parameters
    ----------
    global_spectrum : np.ndarray
        全局小波谱
    significance_threshold : np.ndarray
        95%显著性阈值
    periods : np.ndarray
        周期数组
    significant_periods : list of dict
        显著周期信息
    output_path : Path
        输出路径
    title : str
        图标题
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(periods, global_spectrum, 'b-', linewidth=1.5, label='全局小波谱')
    ax.plot(periods, significance_threshold, 'r--', linewidth=1.2, label='95% 红噪声显著性')

    # 填充显著区域
    above = global_spectrum > significance_threshold
    ax.fill_between(
        periods, global_spectrum, significance_threshold,
        where=above, alpha=0.25, color='blue', label='显著区域',
    )

    # 标注显著周期峰值
    for sp in significant_periods:
        ax.annotate(
            f"{sp['period']:.0f}d\n({sp['ratio']:.1f}x)",
            xy=(sp['period'], sp['power']),
            xytext=(sp['period'] * 1.3, sp['power'] * 1.2),
            fontsize=9,
            arrowprops=dict(arrowstyle='->', color='darkblue', lw=1.0),
            color='darkblue',
            fontweight='bold',
        )

    # 标记关键周期
    for kp in KEY_PERIODS:
        if periods[0] <= kp <= periods[-1]:
            ax.axvline(x=kp, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
            ax.text(kp, ax.get_ylim()[1] * 0.95, f'{kp}d',
                    ha='center', va='top', fontsize=8, color='gray')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('周期（天）', fontsize=12)
    ax.set_ylabel('功率', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  全局小波谱 已保存: {output_path}")


def plot_key_period_power(
    key_power: Dict[float, Dict],
    dates: pd.DatetimeIndex,
    coi_periods: np.ndarray,
    output_path: Path,
    title: str = 'BTC/USDT 关键周期功率时间演化',
) -> None:
    """绘制关键周期处的功率随时间变化

    Parameters
    ----------
    key_power : dict
        extract_power_at_periods 的返回结果
    dates : pd.DatetimeIndex
        时间索引
    coi_periods : np.ndarray
        影响锥边界
    output_path : Path
        输出路径
    title : str
        图标题
    """
    n_periods = len(key_power)
    fig, axes = plt.subplots(n_periods, 1, figsize=(16, 3.5 * n_periods), sharex=True)
    if n_periods == 1:
        axes = [axes]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, (target_period, info) in enumerate(key_power.items()):
        ax = axes[i]
        power_ts = info['power']
        actual_period = info['actual_period']

        # 标记COI内外区域
        in_coi = coi_periods < actual_period  # COI内=不可靠
        reliable_power = power_ts.copy()
        reliable_power[in_coi] = np.nan
        unreliable_power = power_ts.copy()
        unreliable_power[~in_coi] = np.nan

        color = colors[i % len(colors)]
        ax.plot(dates, reliable_power, color=color, linewidth=1.0,
                label=f'{target_period}d (实际 {actual_period:.1f}d)')
        ax.plot(dates, unreliable_power, color=color, linewidth=0.8,
                alpha=0.3, linestyle='--', label='COI 内（不可靠）')

        # 对功率做平滑以显示趋势
        window = max(int(target_period / 5), 7)
        smoothed = pd.Series(power_ts).rolling(window=window, center=True, min_periods=1).mean()
        ax.plot(dates, smoothed, color='black', linewidth=1.5, alpha=0.6, label=f'平滑 ({window}d)')

        ax.set_ylabel('功率', fontsize=10)
        ax.set_title(f'周期 ~ {target_period} 天', fontsize=11)
        ax.legend(loc='upper right', fontsize=8, ncol=3)
        ax.grid(True, alpha=0.3)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[-1].set_xlabel('日期', fontsize=12)

    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  关键周期功率图 已保存: {output_path}")


# ============================================================================
# 主入口函数
# ============================================================================

def run_wavelet_analysis(
    df: pd.DataFrame,
    output_dir: str,
    wavelet: str = WAVELET,
    min_period: float = MIN_PERIOD,
    max_period: float = MAX_PERIOD,
    num_scales: int = NUM_SCALES,
    key_periods: List[float] = None,
    n_surrogates: int = N_SURROGATES,
) -> Dict:
    """执行完整的小波变换分析流程

    Parameters
    ----------
    df : pd.DataFrame
        日线 DataFrame，需包含 'close' 列和 DatetimeIndex
    output_dir : str
        输出目录路径
    wavelet : str
        小波函数名
    min_period : float
        最小分析周期（天）
    max_period : float
        最大分析周期（天）
    num_scales : int
        尺度分辨率
    key_periods : list of float
        要追踪的关键周期
    n_surrogates : int
        Monte Carlo替代数据数量

    Returns
    -------
    dict
        包含所有分析结果的字典:
        - coeffs: CWT系数矩阵
        - power: 功率谱矩阵
        - periods: 周期数组
        - global_spectrum: 全局小波谱
        - significance_threshold: 95%显著性阈值
        - significant_periods: 显著周期列表
        - key_period_power: 关键周期功率演化
        - ar1_alpha: AR(1)系数
        - dates: 时间索引
    """
    if key_periods is None:
        key_periods = KEY_PERIODS

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 数据准备 ----
    print("=" * 70)
    print("小波变换分析 (Continuous Wavelet Transform)")
    print("=" * 70)

    prices = df['close'].dropna()
    dates = prices.index
    n = len(prices)

    print(f"\n[数据概况]")
    print(f"  时间范围: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}")
    print(f"  样本数: {n}")
    print(f"  小波函数: {wavelet}")
    print(f"  分析周期范围: {min_period}d ~ {max_period}d")

    # 对数收益率 + 标准化，作为CWT输入信号
    log_ret = log_returns(prices)
    signal = standardize(log_ret).values
    signal_dates = log_ret.index

    # 处理可能的NaN/Inf
    valid_mask = np.isfinite(signal)
    if not np.all(valid_mask):
        print(f"  警告: 移除 {np.sum(~valid_mask)} 个非有限值")
        signal = signal[valid_mask]
        signal_dates = signal_dates[valid_mask]

    n_signal = len(signal)
    print(f"  CWT输入信号长度: {n_signal}")

    # ---- 2. 连续小波变换 ----
    print(f"\n[CWT 计算]")
    print(f"  尺度数量: {num_scales}")

    coeffs, periods, scales = compute_cwt(
        signal, dt=1.0, wavelet=wavelet,
        min_period=min_period, max_period=max_period, num_scales=num_scales,
    )
    power = compute_power_spectrum(coeffs)

    print(f"  系数矩阵形状: {coeffs.shape}")
    print(f"  周期范围: {periods[0]:.1f}d ~ {periods[-1]:.1f}d")

    # ---- 3. 影响锥 ----
    coi_periods = compute_coi(n_signal, dt=1.0, wavelet=wavelet)

    # ---- 4. 全局小波谱 ----
    print(f"\n[全局小波谱]")
    global_spectrum = compute_global_wavelet_spectrum(power)

    # ---- 5. AR(1) 红噪声 Monte Carlo 显著性检验 ----
    print(f"\n[Monte Carlo 显著性检验]")
    significance_threshold, surrogate_spectra = significance_test_monte_carlo(
        signal, periods, dt=1.0, wavelet=wavelet,
        n_surrogates=n_surrogates, significance_level=SIGNIFICANCE_LEVEL,
    )

    # ---- 6. 找出显著周期 ----
    significant_periods = find_significant_periods(
        global_spectrum, significance_threshold, periods,
    )

    print(f"\n[显著周期（超过95%置信水平）]")
    if significant_periods:
        for sp in significant_periods:
            days = sp['period']
            years = days / 365.25
            print(f"  * {days:7.0f} 天 ({years:5.2f} 年) | "
                  f"功率={sp['power']:.4f} | 阈值={sp['threshold']:.4f} | "
                  f"比值={sp['ratio']:.2f}x")
    else:
        print("  未发现超过95%显著性水平的周期")

    # ---- 7. 关键周期功率时间演化 ----
    print(f"\n[关键周期功率追踪]")
    key_power = extract_power_at_periods(power, periods, key_periods)
    for kp, info in key_power.items():
        print(f"  {kp}d -> 实际匹配周期: {info['actual_period']:.1f}d, "
              f"平均功率: {np.mean(info['power']):.4f}")

    # ---- 8. 可视化 ----
    print(f"\n[生成图表]")

    # 8.1 CWT Scalogram
    plot_cwt_scalogram(
        power, periods, signal_dates, coi_periods,
        output_dir / 'wavelet_scalogram.png',
    )

    # 8.2 全局小波谱 + 显著性
    plot_global_spectrum(
        global_spectrum, significance_threshold, periods, significant_periods,
        output_dir / 'wavelet_global_spectrum.png',
    )

    # 8.3 关键周期功率演化
    plot_key_period_power(
        key_power, signal_dates, coi_periods,
        output_dir / 'wavelet_key_periods.png',
    )

    # ---- 9. 汇总结果 ----
    ar1_alpha = _estimate_ar1(signal)

    results = {
        'coeffs': coeffs,
        'power': power,
        'periods': periods,
        'scales': scales,
        'global_spectrum': global_spectrum,
        'significance_threshold': significance_threshold,
        'significant_periods': significant_periods,
        'key_period_power': key_power,
        'coi_periods': coi_periods,
        'ar1_alpha': ar1_alpha,
        'dates': signal_dates,
        'wavelet': wavelet,
        'signal_length': n_signal,
    }

    print(f"\n{'=' * 70}")
    print(f"小波分析完成。共生成 3 张图表，保存至: {output_dir}")
    print(f"{'=' * 70}")

    return results


# ============================================================================
# 独立运行入口
# ============================================================================

if __name__ == '__main__':
    from src.data_loader import load_daily

    print("加载 BTC/USDT 日线数据...")
    df = load_daily()
    print(f"数据加载完成: {len(df)} 行\n")

    results = run_wavelet_analysis(df, output_dir='outputs/wavelet')
