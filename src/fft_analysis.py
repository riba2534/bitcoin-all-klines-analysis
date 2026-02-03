"""FFT 频谱分析模块 - BTC价格周期性检测与频域特征提取"""

import matplotlib
matplotlib.use("Agg")

from src.font_config import configure_chinese_font
configure_chinese_font()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import find_peaks, butter, sosfiltfilt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.data_loader import load_klines
from src.preprocessing import log_returns, detrend_linear


# ============================================================
# 常量定义
# ============================================================

# 多时间框架比较所用的K线粒度及其对应采样周期（天）
MULTI_TF_INTERVALS = {
    "1m": 1 / (24 * 60),    # 分钟线
    "3m": 3 / (24 * 60),
    "5m": 5 / (24 * 60),
    "15m": 15 / (24 * 60),
    "30m": 30 / (24 * 60),
    "1h": 1 / 24,            # 小时线
    "2h": 2 / 24,
    "4h": 4 / 24,
    "6h": 6 / 24,
    "8h": 8 / 24,
    "12h": 12 / 24,
    "1d": 1.0,               # 日线
    "3d": 3.0,
    "1w": 7.0,               # 周线
    "1mo": 30.0,             # 月线（近似30天）
}

# 带通滤波目标周期（天）
BANDPASS_PERIODS_DAYS = [7, 30, 90, 365, 1400]

# 峰值检测阈值：功率必须超过背景噪声的倍数
PEAK_THRESHOLD_RATIO = 5.0

# 图表保存参数
SAVE_KW = dict(dpi=150, bbox_inches="tight")


# ============================================================
# 核心FFT计算函数
# ============================================================

def compute_fft_spectrum(
    signal: np.ndarray,
    sampling_period_days: float,
    apply_window: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算信号的FFT功率谱

    Parameters
    ----------
    signal : np.ndarray
        输入时域信号（需已去趋势/取对数收益率）
    sampling_period_days : float
        采样周期，单位为天（日线=1.0, 4h线=4/24）
    apply_window : bool
        是否应用Hann窗函数以抑制频谱泄漏

    Returns
    -------
    freqs : np.ndarray
        频率数组（仅正频率部分），单位 cycles/day
    periods : np.ndarray
        周期数组（天），即 1/freqs
    power : np.ndarray
        功率谱（振幅平方的归一化值）
    """
    n = len(signal)
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    # 应用Hann窗减少频谱泄漏
    if apply_window:
        window = np.hanning(n)
        windowed = signal * window
        # 窗函数能量补偿：保持总功率不变
        window_energy = np.sum(window ** 2) / n
    else:
        windowed = signal.copy()
        window_energy = 1.0

    # FFT计算
    yf = fft(windowed)
    freqs = fftfreq(n, d=sampling_period_days)

    # 仅取正频率部分（排除直流分量 freq=0）
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    yf_pos = yf[pos_mask]

    # 功率谱密度：单边谱乘2，加入采样频率 fs 归一化
    fs = 1.0 / sampling_period_days  # 采样频率 (cycles/day)
    power = 2.0 * (np.abs(yf_pos) ** 2) / (n * fs * window_energy)

    # 对应周期
    periods = 1.0 / freqs_pos

    return freqs_pos, periods, power


# ============================================================
# AR(1) 红噪声基线模型
# ============================================================

def ar1_red_noise_spectrum(
    signal: np.ndarray,
    freqs: np.ndarray,
    sampling_period_days: float,
    confidence_percentile: float = 95.0,
    power: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于AR(1)模型估算红噪声理论功率谱

    AR(1)模型的功率谱密度公式：
        S(f) = S0 * (1 - rho^2) / (1 - 2*rho*cos(2*pi*f*dt) + rho^2)

    Parameters
    ----------
    signal : np.ndarray
        原始信号
    freqs : np.ndarray
        频率数组
    sampling_period_days : float
        采样周期
    confidence_percentile : float
        置信水平百分位数（默认95%）
    power : np.ndarray, optional
        信号功率谱，用于经验缩放使理论谱均值匹配信号谱均值

    Returns
    -------
    noise_mean : np.ndarray
        红噪声理论均值功率谱
    noise_threshold : np.ndarray
        指定置信水平的功率阈值
    """
    n = len(signal)
    if n < 3:
        return np.zeros_like(freqs), np.zeros_like(freqs)

    # 估计AR(1)系数 rho（滞后1自相关）
    signal_centered = signal - np.mean(signal)
    autocov_0 = np.sum(signal_centered ** 2) / n
    autocov_1 = np.sum(signal_centered[:-1] * signal_centered[1:]) / n
    rho = autocov_1 / autocov_0 if autocov_0 > 0 else 0.0
    rho = np.clip(rho, -0.999, 0.999)  # 防止数值不稳定

    # AR(1)理论功率谱
    variance = autocov_0
    s0 = variance * (1 - rho ** 2)
    cos_term = np.cos(2 * np.pi * freqs * sampling_period_days)
    denominator = 1 - 2 * rho * cos_term + rho ** 2
    noise_mean = s0 / denominator

    # 经验缩放：使理论谱均值匹配信号谱均值
    if power is not None and np.mean(noise_mean) > 0:
        scale_factor_empirical = np.mean(power) / np.mean(noise_mean)
        noise_mean = noise_mean * scale_factor_empirical

    # 在chi-squared分布下，FFT功率近似服从指数分布（自由度2）
    # 95%置信上界 = 均值 * chi2_ppf(0.95, 2) / 2 ≈ 均值 * 2.996
    from scipy.stats import chi2
    scale_factor = chi2.ppf(confidence_percentile / 100.0, df=2) / 2.0
    noise_threshold = noise_mean * scale_factor

    return noise_mean, noise_threshold


# ============================================================
# 峰值检测
# ============================================================

def detect_spectral_peaks(
    freqs: np.ndarray,
    periods: np.ndarray,
    power: np.ndarray,
    noise_mean: np.ndarray,
    noise_threshold: np.ndarray,
    threshold_ratio: float = PEAK_THRESHOLD_RATIO,
    min_period_days: float = 2.0,
) -> pd.DataFrame:
    """
    在功率谱中检测显著峰值

    峰值判定标准：
    1. scipy.signal.find_peaks 局部峰值
    2. 功率 > threshold_ratio * 背景噪声均值
    3. 周期 > min_period_days（过滤高频噪声）

    Parameters
    ----------
    freqs, periods, power : np.ndarray
        频率、周期、功率数组
    noise_mean, noise_threshold : np.ndarray
        红噪声均值和置信阈值
    threshold_ratio : float
        峰值必须超过噪声均值的倍数
    min_period_days : float
        最小周期阈值（天）

    Returns
    -------
    pd.DataFrame
        检测到的峰值信息表，包含 period_days, frequency, power, noise_level, snr 列
    """
    if len(power) == 0:
        return pd.DataFrame(columns=["period_days", "frequency", "power", "noise_level", "snr"])

    # 使用scipy检测局部峰值
    peak_indices, properties = find_peaks(power, height=0)

    results = []
    for idx in peak_indices:
        period_d = periods[idx]
        pwr = power[idx]
        noise_lvl = noise_mean[idx] if idx < len(noise_mean) else 1.0
        snr = pwr / noise_lvl if noise_lvl > 0 else 0.0

        # 筛选：周期足够长且功率显著超过噪声
        if period_d >= min_period_days and snr >= threshold_ratio:
            results.append({
                "period_days": period_d,
                "frequency": freqs[idx],
                "power": pwr,
                "noise_level": noise_lvl,
                "snr": snr,
            })

    df_peaks = pd.DataFrame(results)
    if not df_peaks.empty:
        df_peaks = df_peaks.sort_values("snr", ascending=False).reset_index(drop=True)

    return df_peaks


# ============================================================
# 带通滤波器
# ============================================================

def bandpass_filter(
    signal: np.ndarray,
    sampling_period_days: float,
    center_period_days: float,
    bandwidth_ratio: float = 0.3,
    order: int = 4,
) -> np.ndarray:
    """
    带通滤波提取特定周期分量

    对于长周期（归一化低频 < 0.01）自动使用FFT域滤波以避免
    Butterworth滤波器的数值不稳定问题。其余情况使用SOS格式的
    Butterworth带通滤波（sosfiltfilt），保证数值稳定性。

    Parameters
    ----------
    signal : np.ndarray
        输入信号
    sampling_period_days : float
        采样周期（天）
    center_period_days : float
        目标中心周期（天）
    bandwidth_ratio : float
        带宽比例：实际带宽 = center_period * (1 +/- bandwidth_ratio)
    order : int
        Butterworth滤波器阶数

    Returns
    -------
    np.ndarray
        滤波后的信号分量
    """
    fs = 1.0 / sampling_period_days  # 采样频率 (cycles/day)
    nyquist = fs / 2.0

    # 带通频率范围
    low_period = center_period_days * (1 + bandwidth_ratio)
    high_period = center_period_days * (1 - bandwidth_ratio)

    if high_period <= 0:
        high_period = sampling_period_days * 2.1  # 保证物理意义

    low_freq = 1.0 / low_period
    high_freq = 1.0 / high_period

    # 归一化到Nyquist频率
    low_norm = low_freq / nyquist
    high_norm = high_freq / nyquist

    # 确保归一化频率在有效范围 (0, 1) 内
    low_norm = np.clip(low_norm, 1e-6, 0.9999)
    high_norm = np.clip(high_norm, low_norm + 1e-6, 0.9999)

    if low_norm >= high_norm:
        return np.zeros_like(signal)

    # 对于长周期（归一化低频极小），Butterworth滤波器数值不稳定
    # 直接使用FFT域带通滤波作为可靠替代
    if low_norm < 0.01:
        return _fft_bandpass_fallback(signal, sampling_period_days,
                                      center_period_days, bandwidth_ratio)

    # 信号长度检查：sosfiltfilt 需要足够的样本点
    min_samples = 3 * (2 * order + 1)
    if len(signal) < min_samples:
        return np.zeros_like(signal)

    try:
        # 使用SOS格式（二阶节）保证数值稳定性
        sos = butter(order, [low_norm, high_norm], btype="band", output="sos")
        filtered = sosfiltfilt(sos, signal)
        return filtered
    except (ValueError, np.linalg.LinAlgError):
        # 若滤波失败，回退到FFT方式
        return _fft_bandpass_fallback(signal, sampling_period_days,
                                      center_period_days, bandwidth_ratio)


def _fft_bandpass_fallback(
    signal: np.ndarray,
    sampling_period_days: float,
    center_period_days: float,
    bandwidth_ratio: float,
) -> np.ndarray:
    """FFT域带通滤波备选方案"""
    n = len(signal)
    freqs = fftfreq(n, d=sampling_period_days)
    yf = fft(signal)

    center_freq = 1.0 / center_period_days
    low_freq = center_freq / (1 + bandwidth_ratio)
    high_freq = center_freq / (1 - bandwidth_ratio) if bandwidth_ratio < 1 else center_freq * 10

    # 频域掩码：保留目标频段
    mask = (np.abs(freqs) >= low_freq) & (np.abs(freqs) <= high_freq)
    yf_filtered = np.zeros_like(yf)
    yf_filtered[mask] = yf[mask]

    return np.real(ifft(yf_filtered))


# ============================================================
# 可视化函数
# ============================================================

def plot_power_spectrum(
    periods: np.ndarray,
    power: np.ndarray,
    noise_mean: np.ndarray,
    noise_threshold: np.ndarray,
    peaks_df: pd.DataFrame,
    title: str = "BTC Log Returns - FFT Power Spectrum",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    功率谱图：包含峰值标注和红噪声置信带

    Parameters
    ----------
    periods, power : np.ndarray
        周期和功率数组
    noise_mean, noise_threshold : np.ndarray
        红噪声均值和置信阈值
    peaks_df : pd.DataFrame
        检测到的峰值表
    title : str
        图表标题
    save_path : Path, optional
        保存路径

    Returns
    -------
    fig : plt.Figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # 功率谱（对数坐标）
    ax.loglog(periods, power, color="#2196F3", linewidth=0.6, alpha=0.8, label="Power Spectrum")

    # 红噪声基线
    ax.loglog(periods, noise_mean, color="#FF9800", linewidth=1.5,
              linestyle="--", label="AR(1) Red Noise Mean")

    # 95%置信带
    ax.fill_between(periods, 0, noise_threshold,
                    alpha=0.15, color="#FF9800", label="95% Confidence Band")
    ax.loglog(periods, noise_threshold, color="#FF5722", linewidth=1.0,
              linestyle=":", alpha=0.7, label="95% Confidence Threshold")

    # 5x噪声阈值线
    noise_5x = noise_mean * PEAK_THRESHOLD_RATIO
    ax.loglog(periods, noise_5x, color="#F44336", linewidth=1.0,
              linestyle="-.", alpha=0.5, label=f"{PEAK_THRESHOLD_RATIO:.0f}x Noise Threshold")

    # 峰值标注
    if not peaks_df.empty:
        for _, row in peaks_df.iterrows():
            period_d = row["period_days"]
            pwr = row["power"]
            snr = row["snr"]

            ax.plot(period_d, pwr, "rv", markersize=10, zorder=5)

            # 周期标签格式化
            if period_d >= 365:
                label_str = f"{period_d / 365:.1f}y (SNR={snr:.1f})"
            elif period_d >= 30:
                label_str = f"{period_d:.0f}d (SNR={snr:.1f})"
            else:
                label_str = f"{period_d:.1f}d (SNR={snr:.1f})"

            ax.annotate(
                label_str,
                xy=(period_d, pwr),
                xytext=(0, 15),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                color="#D32F2F",
                ha="center",
                arrowprops=dict(arrowstyle="-", color="#D32F2F", lw=0.5),
            )

    ax.set_xlabel("Period (days)", fontsize=12)
    ax.set_ylabel("Power", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # X轴标记关键周期
    key_periods = [7, 14, 30, 60, 90, 180, 365, 730, 1460]
    ax.set_xticks(key_periods)
    ax.set_xticklabels([str(p) for p in key_periods], fontsize=8)
    ax.set_xlim(left=max(2, periods.min()), right=periods.max())

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, **SAVE_KW)
        print(f"  [保存] 功率谱图 -> {save_path}")

    return fig


def plot_multi_timeframe(
    tf_results: Dict[str, dict],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    多时间框架FFT频谱对比图

    Parameters
    ----------
    tf_results : dict
        键为时间框架标签，值为包含 periods/power/noise_mean 的dict
    save_path : Path, optional
        保存路径

    Returns
    -------
    fig : plt.Figure
    """
    n_tf = len(tf_results)

    # 根据时间框架数量决定布局：超过6个使用2列布局
    if n_tf > 6:
        ncols = 2
        nrows = (n_tf + 1) // 2
        figsize = (16, 4 * nrows)
    else:
        ncols = 1
        nrows = n_tf
        figsize = (14, 5 * n_tf)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=False)

    # 统一处理axes为一维数组
    if n_tf == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_tf > 1 else [axes]

    # 使用colormap生成足够多的颜色
    if n_tf <= 10:
        cmap = plt.cm.tab10
    else:
        cmap = plt.cm.tab20
    colors = [cmap(i % cmap.N) for i in range(n_tf)]

    for idx, ((label, data), color) in enumerate(zip(tf_results.items(), colors)):
        ax = axes[idx]
        periods = data["periods"]
        power = data["power"]
        noise_mean = data["noise_mean"]

        # 转换颜色为hex格式
        if isinstance(color, tuple):
            import matplotlib.colors as mcolors
            color_hex = mcolors.rgb2hex(color[:3])
        else:
            color_hex = color

        ax.loglog(periods, power, color=color_hex, linewidth=0.6, alpha=0.8,
                  label=f"{label} Spectrum")
        ax.loglog(periods, noise_mean, color="#FF9800", linewidth=1.2,
                  linestyle="--", alpha=0.7, label="AR(1) Noise")

        # 标注峰值
        peaks_df = data.get("peaks", pd.DataFrame())
        if not peaks_df.empty:
            for _, row in peaks_df.head(5).iterrows():
                period_d = row["period_days"]
                pwr = row["power"]
                ax.plot(period_d, pwr, "rv", markersize=8, zorder=5)
                if period_d >= 365:
                    lbl = f"{period_d / 365:.1f}y"
                elif period_d >= 30:
                    lbl = f"{period_d:.0f}d"
                else:
                    lbl = f"{period_d:.1f}d"
                ax.annotate(lbl, xy=(period_d, pwr), xytext=(0, 10),
                            textcoords="offset points", fontsize=8,
                            color="#D32F2F", ha="center", fontweight="bold")

        ax.set_ylabel("Power", fontsize=11)
        ax.set_title(f"BTC FFT Spectrum - {label}", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

    # 隐藏多余的子图
    for idx in range(n_tf, len(axes)):
        axes[idx].set_visible(False)

    # 设置xlabel（最底行的子图）
    if ncols == 2:
        # 2列布局：设置最后一行的xlabel
        for idx in range(max(0, len(axes) - ncols), len(axes)):
            if idx < n_tf:
                axes[idx].set_xlabel("Period (days)", fontsize=12)
    else:
        # 单列布局
        axes[n_tf - 1].set_xlabel("Period (days)", fontsize=12)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, **SAVE_KW)
        print(f"  [保存] 多时间框架对比图 -> {save_path}")

    return fig


def plot_spectral_waterfall(
    tf_results: Dict[str, dict],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    15尺度频谱瀑布图 - 热力图展示不同时间框架的功率谱

    Parameters
    ----------
    tf_results : dict
        键为时间框架标签，值为包含 periods/power 的dict
    save_path : Path, optional
        保存路径

    Returns
    -------
    fig : plt.Figure
    """
    if not tf_results:
        print("  [警告] 无有效时间框架数据，跳过瀑布图")
        return None

    # 按采样频率排序时间框架（从高频到低频）
    sorted_tfs = sorted(
        tf_results.items(),
        key=lambda x: MULTI_TF_INTERVALS.get(x[0], 1.0)
    )

    # 统一周期网格（对数空间）
    all_periods = []
    for _, data in sorted_tfs:
        all_periods.extend(data["periods"])

    # 创建对数均匀分布的周期网格
    min_period = max(1.0, min(all_periods))
    max_period = max(all_periods)
    period_grid = np.logspace(np.log10(min_period), np.log10(max_period), 500)

    # 插值每个时间框架的功率谱到统一网格
    n_tf = len(sorted_tfs)
    power_matrix = np.zeros((n_tf, len(period_grid)))
    tf_labels = []

    for i, (label, data) in enumerate(sorted_tfs):
        periods = data["periods"]
        power = data["power"]

        # 对数插值
        log_periods = np.log10(periods)
        log_power = np.log10(power + 1e-20)  # 避免log(0)
        log_period_grid = np.log10(period_grid)

        # 使用numpy插值
        log_power_interp = np.interp(log_period_grid, log_periods, log_power)
        power_matrix[i, :] = log_power_interp
        tf_labels.append(label)

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(16, 10))

    # 使用pcolormesh绘制
    X, Y = np.meshgrid(period_grid, np.arange(n_tf))
    im = ax.pcolormesh(X, Y, power_matrix, cmap="viridis", shading="auto")

    # 颜色条
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("log10(Power)", fontsize=12)

    # Y轴标签（时间框架）
    ax.set_yticks(np.arange(n_tf))
    ax.set_yticklabels(tf_labels, fontsize=10)
    ax.set_ylabel("Timeframe", fontsize=12, fontweight="bold")

    # X轴对数刻度
    ax.set_xscale("log")
    ax.set_xlabel("Period (days)", fontsize=12, fontweight="bold")
    ax.set_xlim(min_period, max_period)

    # 关键周期参考线
    key_periods = [7, 30, 90, 365, 1460]
    for kp in key_periods:
        if min_period <= kp <= max_period:
            ax.axvline(kp, color="white", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.text(kp, n_tf + 0.5, f"{kp}d", fontsize=8, color="white",
                   ha="center", va="bottom", fontweight="bold")

    ax.set_title("BTC Price FFT Spectral Waterfall - Multi-Timeframe Comparison",
                fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, which="both", alpha=0.2, color="white", linewidth=0.5)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, **SAVE_KW)
        print(f"  [保存] 频谱瀑布图 -> {save_path}")

    return fig


def plot_bandpass_components(
    dates: pd.DatetimeIndex,
    original_signal: np.ndarray,
    components: Dict[str, np.ndarray],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    带通滤波分量子图

    Parameters
    ----------
    dates : pd.DatetimeIndex
        日期索引
    original_signal : np.ndarray
        原始信号（对数收益率）
    components : dict
        键为周期标签（如 "7d"），值为滤波后的信号数组
    save_path : Path, optional
        保存路径

    Returns
    -------
    fig : plt.Figure
    """
    n_comp = len(components) + 1  # +1 for original
    fig, axes = plt.subplots(n_comp, 1, figsize=(14, 3 * n_comp), sharex=True)

    # 原始信号
    axes[0].plot(dates, original_signal, color="#455A64", linewidth=0.5, alpha=0.8)
    axes[0].set_title("Original Log Returns", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Log Return", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # 各周期分量
    colors_bp = ["#E91E63", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    for i, ((label, comp), color) in enumerate(zip(components.items(), colors_bp)):
        ax = axes[i + 1]
        ax.plot(dates, comp, color=color, linewidth=0.8, alpha=0.9)
        ax.set_title(f"Bandpass Component: {label} cycle", fontsize=11, fontweight="bold")
        ax.set_ylabel("Amplitude", fontsize=9)
        ax.grid(True, alpha=0.3)

        # 显示该分量的方差占比
        if np.var(original_signal) > 0:
            var_ratio = np.var(comp) / np.var(original_signal) * 100
            ax.text(0.02, 0.92, f"Variance ratio: {var_ratio:.2f}%",
                    transform=ax.transAxes, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.15))

    axes[-1].set_xlabel("Date", fontsize=11)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, **SAVE_KW)
        print(f"  [保存] 带通滤波分量图 -> {save_path}")

    return fig


# ============================================================
# 单时间框架FFT分析流水线
# ============================================================

def _analyze_single_timeframe(
    df: pd.DataFrame,
    sampling_period_days: float,
    label: str = "1d",
) -> dict:
    """
    对单个时间框架执行完整FFT分析

    Returns
    -------
    dict
        包含 freqs, periods, power, noise_mean, noise_threshold, peaks, log_ret 等
    """
    prices = df["close"].dropna()
    if len(prices) < 10:
        print(f"  [警告] {label} 数据量不足 ({len(prices)} 条)，跳过分析")
        return {}

    # 计算对数收益率
    log_ret = np.log(prices / prices.shift(1)).dropna().values

    # FFT频谱计算（Hann窗）
    freqs, periods, power = compute_fft_spectrum(
        log_ret, sampling_period_days, apply_window=True
    )

    if len(freqs) == 0:
        return {}

    # AR(1)红噪声基线
    noise_mean, noise_threshold = ar1_red_noise_spectrum(
        log_ret, freqs, sampling_period_days, confidence_percentile=95.0,
        power=power,
    )

    # 峰值检测
    # 对于低频数据（如周线），放宽最小周期约束
    min_period = max(2.0, sampling_period_days * 3)
    peaks_df = detect_spectral_peaks(
        freqs, periods, power, noise_mean, noise_threshold,
        threshold_ratio=PEAK_THRESHOLD_RATIO,
        min_period_days=min_period,
    )

    return {
        "freqs": freqs,
        "periods": periods,
        "power": power,
        "noise_mean": noise_mean,
        "noise_threshold": noise_threshold,
        "peaks": peaks_df,
        "log_ret": log_ret,
        "label": label,
    }


# ============================================================
# 主入口函数
# ============================================================

def run_fft_analysis(
    df: pd.DataFrame,
    output_dir: str,
) -> Dict:
    """
    BTC价格FFT频谱分析主入口

    执行以下分析并保存可视化结果：
    1. 日线对数收益率FFT频谱分析（Hann窗 + AR1红噪声基线）
    2. 功率谱峰值检测（5x噪声阈值）
    3. 多时间框架（全部15个粒度）频谱对比 + 频谱瀑布图
    4. 带通滤波提取关键周期分量（7d/30d/90d/365d/1400d）

    Parameters
    ----------
    df : pd.DataFrame
        日线K线数据，DatetimeIndex，需包含 close 列
    output_dir : str
        图表输出目录路径

    Returns
    -------
    dict
        分析结果汇总：
        - daily_peaks: 日线显著周期峰值表
        - multi_tf_peaks: 各时间框架峰值字典
        - bandpass_variance_ratios: 各带通分量方差占比
        - ar1_rho: AR(1)自相关系数
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BTC FFT 频谱分析")
    print("=" * 70)

    # ----------------------------------------------------------
    # 第一部分：日线对数收益率FFT分析
    # ----------------------------------------------------------
    print("\n[1/4] 日线对数收益率FFT分析 (Hann窗)")
    daily_result = _analyze_single_timeframe(df, sampling_period_days=1.0, label="1d")

    if not daily_result:
        print("  [错误] 日线分析失败，数据不足")
        return {}

    log_ret = daily_result["log_ret"]
    periods = daily_result["periods"]
    power = daily_result["power"]
    noise_mean = daily_result["noise_mean"]
    noise_threshold = daily_result["noise_threshold"]
    peaks_df = daily_result["peaks"]

    # 打印AR(1)参数
    signal_centered = log_ret - np.mean(log_ret)
    autocov_0 = np.sum(signal_centered ** 2) / len(log_ret)
    autocov_1 = np.sum(signal_centered[:-1] * signal_centered[1:]) / len(log_ret)
    ar1_rho = autocov_1 / autocov_0 if autocov_0 > 0 else 0.0
    print(f"  AR(1) 自相关系数 rho = {ar1_rho:.4f}")
    print(f"  数据长度: {len(log_ret)} 个交易日")
    print(f"  频率分辨率: {1.0 / len(log_ret):.6f} cycles/day (最大可分辨周期: {len(log_ret):.0f} 天)")

    # 打印显著峰值
    if not peaks_df.empty:
        print(f"\n  检测到 {len(peaks_df)} 个显著周期峰值 (SNR > {PEAK_THRESHOLD_RATIO:.0f}x):")
        print("  " + "-" * 60)
        print(f"  {'周期(天)':>10} | {'周期':>12} | {'SNR':>8} | {'功率':>12}")
        print("  " + "-" * 60)
        for _, row in peaks_df.iterrows():
            pd_days = row["period_days"]
            snr = row["snr"]
            pwr = row["power"]
            if pd_days >= 365:
                human_period = f"{pd_days / 365:.1f} 年"
            elif pd_days >= 30:
                human_period = f"{pd_days / 30:.1f} 月"
            else:
                human_period = f"{pd_days:.1f} 天"
            print(f"  {pd_days:>10.1f} | {human_period:>12} | {snr:>8.2f} | {pwr:>12.6e}")
        print("  " + "-" * 60)
    else:
        print("  未检测到显著超过红噪声基线的周期峰值")

    # 功率谱图
    fig_spectrum = plot_power_spectrum(
        periods, power, noise_mean, noise_threshold, peaks_df,
        title="BTC Daily Log Returns - FFT Power Spectrum (Hann Window)",
        save_path=output_path / "fft_power_spectrum.png",
    )
    plt.close(fig_spectrum)

    # ----------------------------------------------------------
    # 第二部分：多时间框架FFT对比
    # ----------------------------------------------------------
    print("\n[2/4] 多时间框架FFT对比 (全部15个粒度)")
    print(f"  时间框架列表: {list(MULTI_TF_INTERVALS.keys())}")
    tf_results = {}

    for interval, sp_days in MULTI_TF_INTERVALS.items():
        try:
            if interval == "1d":
                tf_df = df
            else:
                tf_df = load_klines(interval)
            result = _analyze_single_timeframe(tf_df, sp_days, label=interval)
            if result:
                tf_results[interval] = result
                n_peaks = len(result["peaks"]) if not result["peaks"].empty else 0
                print(f"  {interval:>4}: {len(result['log_ret']):>8} 样本, {n_peaks:>2} 个显著峰值")
        except FileNotFoundError:
            print(f"  [警告] {interval} 数据文件未找到，跳过")
        except Exception as e:
            print(f"  [警告] {interval} 分析失败: {e}")

    print(f"\n  成功分析 {len(tf_results)}/{len(MULTI_TF_INTERVALS)} 个时间框架")

    # 多时间框架对比图
    if len(tf_results) > 1:
        fig_mtf = plot_multi_timeframe(
            tf_results,
            save_path=output_path / "fft_multi_timeframe.png",
        )
        plt.close(fig_mtf)

        # 新增：频谱瀑布图
        fig_waterfall = plot_spectral_waterfall(
            tf_results,
            save_path=output_path / "fft_spectral_waterfall.png",
        )
        if fig_waterfall:
            plt.close(fig_waterfall)
    else:
        print("  [警告] 可用时间框架不足，跳过对比图")

    # ----------------------------------------------------------
    # 第三部分：带通滤波提取周期分量
    # ----------------------------------------------------------
    print(f"\n[3/4] 带通滤波提取周期分量: {BANDPASS_PERIODS_DAYS}")
    prices = df["close"].dropna()
    dates = prices.index[1:]  # 与log_ret对齐（差分损失1个点）
    # 确保dates和log_ret长度一致
    if len(dates) > len(log_ret):
        dates = dates[:len(log_ret)]
    elif len(dates) < len(log_ret):
        log_ret = log_ret[:len(dates)]

    components = {}
    variance_ratios = {}
    original_var = np.var(log_ret)

    for period_days in BANDPASS_PERIODS_DAYS:
        # 检查Nyquist条件：目标周期必须大于2倍采样周期
        if period_days < 2.0 * 1.0:
            print(f"  [跳过] {period_days}d 周期低于Nyquist极限")
            continue
        # 检查信号长度是否覆盖至少2个完整周期
        if len(log_ret) < period_days * 2:
            print(f"  [跳过] {period_days}d 周期：数据长度不足 ({len(log_ret)} < {period_days * 2:.0f})")
            continue

        filtered = bandpass_filter(
            log_ret,
            sampling_period_days=1.0,
            center_period_days=float(period_days),
            bandwidth_ratio=0.3,
            order=4,
        )

        label = f"{period_days}d"
        components[label] = filtered
        var_ratio = np.var(filtered) / original_var * 100 if original_var > 0 else 0
        variance_ratios[label] = var_ratio
        print(f"  {label:>6} 分量方差占比: {var_ratio:.3f}%")

    # 带通分量图
    if components:
        fig_bp = plot_bandpass_components(
            dates, log_ret, components,
            save_path=output_path / "fft_bandpass_components.png",
        )
        plt.close(fig_bp)
    else:
        print("  [警告] 无有效带通分量可绘制")

    # ----------------------------------------------------------
    # 第四部分：汇总输出
    # ----------------------------------------------------------
    print("\n[4/4] 分析汇总")

    # 收集多时间框架峰值
    multi_tf_peaks = {}
    for tf_label, tf_data in tf_results.items():
        if not tf_data["peaks"].empty:
            multi_tf_peaks[tf_label] = tf_data["peaks"]

    # 跨时间框架一致性检验
    print("\n  跨时间框架周期一致性检查:")
    if len(multi_tf_peaks) >= 2:
        # 收集所有检测到的周期
        all_detected_periods = []
        for tf_label, p_df in multi_tf_peaks.items():
            for _, row in p_df.iterrows():
                all_detected_periods.append({
                    "timeframe": tf_label,
                    "period_days": row["period_days"],
                    "snr": row["snr"],
                })

        if all_detected_periods:
            all_periods_df = pd.DataFrame(all_detected_periods)
            # 按周期分组（允许20%误差范围），寻找多时间框架确认的周期
            confirmed = []
            used = set()
            for i, row_i in all_periods_df.iterrows():
                if i in used:
                    continue
                p_i = row_i["period_days"]
                group = [row_i]
                used.add(i)
                for j, row_j in all_periods_df.iterrows():
                    if j in used:
                        continue
                    if row_j["timeframe"] != row_i["timeframe"]:
                        if abs(row_j["period_days"] - p_i) / p_i < 0.2:
                            group.append(row_j)
                            used.add(j)
                if len(group) > 1:
                    tfs = [g["timeframe"] for g in group]
                    avg_period = np.mean([g["period_days"] for g in group])
                    avg_snr = np.mean([g["snr"] for g in group])
                    confirmed.append({
                        "period_days": avg_period,
                        "confirmed_by": tfs,
                        "avg_snr": avg_snr,
                    })

            if confirmed:
                for c in confirmed:
                    tfs_str = " & ".join(c["confirmed_by"])
                    print(f"    {c['period_days']:.1f}d 周期被 {tfs_str} 共同确认 (平均SNR={c['avg_snr']:.2f})")
            else:
                print("    未发现跨时间框架一致确认的周期")
        else:
            print("    各时间框架均未检测到显著峰值")
    else:
        print("    可用时间框架不足，无法进行一致性检查")

    print("\n" + "=" * 70)
    print("FFT分析完成")
    print(f"图表已保存至: {output_path.resolve()}")
    print("=" * 70)

    # ----------------------------------------------------------
    # 返回结果字典
    # ----------------------------------------------------------
    results = {
        "daily_peaks": peaks_df,
        "multi_tf_peaks": multi_tf_peaks,
        "bandpass_variance_ratios": variance_ratios,
        "bandpass_components": components,
        "ar1_rho": ar1_rho,
        "daily_spectrum": {
            "freqs": daily_result["freqs"],
            "periods": daily_result["periods"],
            "power": daily_result["power"],
            "noise_mean": daily_result["noise_mean"],
            "noise_threshold": daily_result["noise_threshold"],
        },
        "multi_tf_results": tf_results,
    }

    return results


# ============================================================
# 独立运行入口
# ============================================================

if __name__ == "__main__":
    from src.data_loader import load_daily

    print("加载BTC日线数据...")
    df = load_daily()
    print(f"数据范围: {df.index.min()} ~ {df.index.max()}, 共 {len(df)} 条")

    results = run_fft_analysis(df, output_dir="output/fft")
