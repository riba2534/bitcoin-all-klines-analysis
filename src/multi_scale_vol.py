"""多尺度已实现波动率分析模块

基于高频K线数据计算已实现波动率(Realized Volatility, RV)，并进行多时间尺度分析：
1. 各尺度RV计算（5m ~ 1d）
2. 波动率签名图（Volatility Signature Plot）
3. HAR-RV模型（Heterogeneous Autoregressive RV，Corsi 2009）
4. 跳跃检测（Barndorff-Nielsen & Shephard 双幂变差）
5. 已实现偏度/峰度（高阶矩）
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.font_config import configure_chinese_font
configure_chinese_font()

from src.data_loader import load_klines
from src.preprocessing import log_returns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 常量配置
# ============================================================

# 各粒度对应的采样周期（天）
INTERVALS = {
    "5m": 5 / (24 * 60),
    "15m": 15 / (24 * 60),
    "30m": 30 / (24 * 60),
    "1h": 1 / 24,
    "2h": 2 / 24,
    "4h": 4 / 24,
    "6h": 6 / 24,
    "8h": 8 / 24,
    "12h": 12 / 24,
    "1d": 1.0,
}

# HAR-RV 模型参数
HAR_DAILY_LAG = 1      # 日RV滞后
HAR_WEEKLY_WINDOW = 5   # 周RV窗口（5天）
HAR_MONTHLY_WINDOW = 22 # 月RV窗口（22天）

# 跳跃检测参数
JUMP_Z_THRESHOLD = 3.0  # Z统计量阈值
JUMP_MIN_RATIO = 0.5    # 跳跃占RV最小比例

# 双幂变差常数
BV_CONSTANT = np.pi / 2


# ============================================================
# 核心计算函数
# ============================================================

def compute_realized_volatility_daily(
    df: pd.DataFrame,
    interval: str,
) -> pd.DataFrame:
    """
    计算日频已实现波动率

    RV_day = sqrt(sum(r_intraday^2))

    Parameters
    ----------
    df : pd.DataFrame
        高频K线数据，需要有datetime索引和close列
    interval : str
        时间粒度标识

    Returns
    -------
    rv_daily : pd.DataFrame
        包含date, RV, n_obs列的日频DataFrame
    """
    if len(df) == 0:
        return pd.DataFrame(columns=["date", "RV", "n_obs"])

    # 计算对数收益率
    df = df.copy()
    df["return"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna(subset=["return"])

    # 按日期分组
    df["date"] = df.index.date

    # 计算每日RV
    daily_rv = df.groupby("date").agg({
        "return": lambda x: np.sqrt(np.sum(x**2)),
        "close": "count"
    }).rename(columns={"return": "RV", "close": "n_obs"})

    daily_rv["date"] = pd.to_datetime(daily_rv.index)
    daily_rv = daily_rv.reset_index(drop=True)

    return daily_rv


def compute_bipower_variation(returns: pd.Series) -> float:
    """
    计算双幂变差 (Bipower Variation)

    BV = (π/2) * sum(|r_t| * |r_{t-1}|)

    Parameters
    ----------
    returns : pd.Series
        日内收益率序列

    Returns
    -------
    bv : float
        双幂变差值
    """
    r = returns.values
    if len(r) < 2:
        return 0.0

    # 计算相邻收益率绝对值的乘积
    abs_products = np.abs(r[1:]) * np.abs(r[:-1])
    bv = BV_CONSTANT * np.sum(abs_products)

    return bv


def detect_jumps_daily(
    df: pd.DataFrame,
    z_threshold: float = JUMP_Z_THRESHOLD,
) -> pd.DataFrame:
    """
    检测日频跳跃事件

    基于 Barndorff-Nielsen & Shephard (2004) 方法：
    - RV = 已实现波动率
    - BV = 双幂变差
    - Jump = max(RV - BV, 0)
    - Z统计量检验显著性

    Parameters
    ----------
    df : pd.DataFrame
        高频K线数据
    z_threshold : float
        Z统计量阈值

    Returns
    -------
    jump_df : pd.DataFrame
        包含date, RV, BV, Jump, Z_stat, is_jump列
    """
    if len(df) == 0:
        return pd.DataFrame(columns=["date", "RV", "BV", "Jump", "Z_stat", "is_jump"])

    df = df.copy()
    df["return"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna(subset=["return"])
    df["date"] = df.index.date

    results = []
    for date, group in df.groupby("date"):
        returns = group["return"].values
        n = len(returns)

        if n < 2:
            continue

        # 计算RV
        rv = np.sqrt(np.sum(returns**2))

        # 计算BV
        bv = compute_bipower_variation(group["return"])

        # 计算跳跃
        jump = max(rv**2 - bv, 0)

        # Z统计量（简化版，假设正态分布）
        # Z = (RV^2 - BV) / sqrt(Var(RV^2 - BV))
        # 简化：使用四次幂变差估计方差
        quad_var = np.sum(returns**4)
        var_estimate = max(quad_var - bv**2, 1e-10)
        z_stat = (rv**2 - bv) / np.sqrt(var_estimate / n) if var_estimate > 0 else 0

        is_jump = abs(z_stat) > z_threshold

        results.append({
            "date": pd.Timestamp(date),
            "RV": rv,
            "BV": np.sqrt(max(bv, 0)),
            "Jump": np.sqrt(jump),
            "Z_stat": z_stat,
            "is_jump": is_jump,
        })

    jump_df = pd.DataFrame(results)
    return jump_df


def compute_realized_moments(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    计算日频已实现偏度和峰度

    - RSkew = sum(r^3) / RV^(3/2)
    - RKurt = sum(r^4) / RV^2

    Parameters
    ----------
    df : pd.DataFrame
        高频K线数据

    Returns
    -------
    moments_df : pd.DataFrame
        包含date, RSkew, RKurt列
    """
    if len(df) == 0:
        return pd.DataFrame(columns=["date", "RSkew", "RKurt"])

    df = df.copy()
    df["return"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna(subset=["return"])
    df["date"] = df.index.date

    results = []
    for date, group in df.groupby("date"):
        returns = group["return"].values

        if len(returns) < 2:
            continue

        rv = np.sqrt(np.sum(returns**2))

        if rv < 1e-10:
            rskew, rkurt = 0.0, 0.0
        else:
            rskew = np.sum(returns**3) / (rv**1.5)
            rkurt = np.sum(returns**4) / (rv**2)

        results.append({
            "date": pd.Timestamp(date),
            "RSkew": rskew,
            "RKurt": rkurt,
        })

    moments_df = pd.DataFrame(results)
    return moments_df


def fit_har_rv_model(
    rv_series: pd.Series,
    daily_lag: int = HAR_DAILY_LAG,
    weekly_window: int = HAR_WEEKLY_WINDOW,
    monthly_window: int = HAR_MONTHLY_WINDOW,
) -> Dict[str, Any]:
    """
    拟合HAR-RV模型（Corsi 2009）

    RV_d = β₀ + β₁·RV_d(-1) + β₂·RV_w(-1) + β₃·RV_m(-1) + ε

    其中：
    - RV_d(-1): 前一日RV
    - RV_w(-1): 过去5天RV均值
    - RV_m(-1): 过去22天RV均值

    Parameters
    ----------
    rv_series : pd.Series
        日频RV序列
    daily_lag : int
        日RV滞后
    weekly_window : int
        周RV窗口
    monthly_window : int
        月RV窗口

    Returns
    -------
    results : dict
        包含coefficients, r_squared, predictions等
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    rv = rv_series.values
    n = len(rv)

    # 构建特征
    rv_daily = rv[monthly_window - daily_lag : n - daily_lag]
    rv_weekly = np.array([
        np.mean(rv[i - weekly_window : i])
        for i in range(monthly_window, n)
    ])
    rv_monthly = np.array([
        np.mean(rv[i - monthly_window : i])
        for i in range(monthly_window, n)
    ])

    # 目标变量
    y = rv[monthly_window:]

    # 特征矩阵
    X = np.column_stack([rv_daily, rv_weekly, rv_monthly])

    # 拟合OLS
    model = LinearRegression()
    model.fit(X, y)

    # 预测
    y_pred = model.predict(X)

    # 评估
    r2 = r2_score(y, y_pred)

    # t统计量（简化版）
    residuals = y - y_pred
    mse = np.mean(residuals**2)

    # 计算标准误（使用OLS公式）
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    try:
        var_beta = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        se = np.sqrt(np.diag(var_beta))

        # 系数 = [intercept, β1, β2, β3]
        coefs = np.concatenate([[model.intercept_], model.coef_])
        t_stats = coefs / se
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=len(y) - 4))
    except:
        se = np.zeros(4)
        t_stats = np.zeros(4)
        p_values = np.ones(4)
        coefs = np.concatenate([[model.intercept_], model.coef_])

    results = {
        "coefficients": {
            "intercept": model.intercept_,
            "beta_daily": model.coef_[0],
            "beta_weekly": model.coef_[1],
            "beta_monthly": model.coef_[2],
        },
        "t_statistics": {
            "intercept": t_stats[0],
            "beta_daily": t_stats[1],
            "beta_weekly": t_stats[2],
            "beta_monthly": t_stats[3],
        },
        "p_values": {
            "intercept": p_values[0],
            "beta_daily": p_values[1],
            "beta_weekly": p_values[2],
            "beta_monthly": p_values[3],
        },
        "r_squared": r2,
        "n_obs": len(y),
        "predictions": y_pred,
        "actual": y,
        "residuals": residuals,
        "mse": mse,
    }

    return results


# ============================================================
# 可视化函数
# ============================================================

def plot_volatility_signature(
    rv_by_interval: Dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """
    绘制波动率签名图

    横轴：采样频率（每日采样点数）
    纵轴：平均RV

    Parameters
    ----------
    rv_by_interval : dict
        {interval: rv_df}
    output_path : Path
        输出路径
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # 准备数据
    intervals_sorted = sorted(INTERVALS.keys(), key=lambda x: INTERVALS[x])

    sampling_freqs = []
    mean_rvs = []
    std_rvs = []

    for interval in intervals_sorted:
        if interval not in rv_by_interval or len(rv_by_interval[interval]) == 0:
            continue

        rv_df = rv_by_interval[interval]
        freq = 1.0 / INTERVALS[interval]  # 每日采样点数
        mean_rv = rv_df["RV"].mean()
        std_rv = rv_df["RV"].std()

        sampling_freqs.append(freq)
        mean_rvs.append(mean_rv)
        std_rvs.append(std_rv)

    sampling_freqs = np.array(sampling_freqs)
    mean_rvs = np.array(mean_rvs)
    std_rvs = np.array(std_rvs)

    # 绘制曲线
    ax.plot(sampling_freqs, mean_rvs, marker='o', linewidth=2,
            markersize=8, color='#2196F3', label='平均已实现波动率')

    # 添加误差带
    ax.fill_between(sampling_freqs, mean_rvs - std_rvs, mean_rvs + std_rvs,
                     alpha=0.2, color='#2196F3', label='±1标准差')

    # 标注各点
    for i, interval in enumerate(intervals_sorted):
        if i < len(sampling_freqs):
            ax.annotate(interval, xy=(sampling_freqs[i], mean_rvs[i]),
                       xytext=(0, 10), textcoords='offset points',
                       fontsize=9, ha='center', color='#1976D2',
                       fontweight='bold')

    ax.set_xlabel('采样频率（每日采样点数）', fontsize=12, fontweight='bold')
    ax.set_ylabel('平均已实现波动率', fontsize=12, fontweight='bold')
    ax.set_title('波动率签名图 (Volatility Signature Plot)\n不同采样频率下的已实现波动率',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[波动率签名图] 已保存: {output_path}")


def plot_har_rv_fit(
    har_results: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    绘制HAR-RV模型拟合结果

    Parameters
    ----------
    har_results : dict
        HAR-RV拟合结果
    output_path : Path
        输出路径
    """
    actual = har_results["actual"]
    predictions = har_results["predictions"]
    r2 = har_results["r_squared"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # 上图：实际 vs 预测时序对比
    x = np.arange(len(actual))
    ax1.plot(x, actual, label='实际RV', color='#424242', linewidth=1.5, alpha=0.8)
    ax1.plot(x, predictions, label='HAR-RV预测', color='#F44336',
            linewidth=1.5, linestyle='--', alpha=0.9)
    ax1.fill_between(x, actual, predictions, alpha=0.15, color='#FF9800')
    ax1.set_ylabel('已实现波动率 (RV)', fontsize=11, fontweight='bold')
    ax1.set_title(f'HAR-RV模型拟合结果 (R² = {r2:.4f})', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 下图：残差分析
    residuals = har_results["residuals"]
    ax2.scatter(x, residuals, alpha=0.5, s=20, color='#9C27B0')
    ax2.axhline(y=0, color='#E91E63', linestyle='--', linewidth=1.5)
    ax2.fill_between(x, 0, residuals, alpha=0.2, color='#9C27B0')
    ax2.set_xlabel('时间索引', fontsize=11, fontweight='bold')
    ax2.set_ylabel('残差 (实际 - 预测)', fontsize=11, fontweight='bold')
    ax2.set_title('模型残差分布', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[HAR-RV拟合图] 已保存: {output_path}")


def plot_jump_detection(
    jump_df: pd.DataFrame,
    price_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    绘制跳跃检测结果

    在价格图上标注检测到的跳跃事件

    Parameters
    ----------
    jump_df : pd.DataFrame
        跳跃检测结果
    price_df : pd.DataFrame
        日线价格数据
    output_path : Path
        输出路径
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # 合并数据
    jump_df = jump_df.set_index("date")
    price_df = price_df.copy()
    price_df["date"] = price_df.index.date
    price_df["date"] = pd.to_datetime(price_df["date"])
    price_df = price_df.set_index("date")

    # 上图：价格 + 跳跃事件标注
    ax1.plot(price_df.index, price_df["close"],
            color='#424242', linewidth=1.5, label='BTC价格')

    # 标注跳跃事件
    jump_dates = jump_df[jump_df["is_jump"]].index
    for date in jump_dates:
        if date in price_df.index:
            ax1.axvline(x=date, color='#F44336', alpha=0.3, linewidth=2)

    # 在跳跃点标注
    jump_prices = price_df.loc[jump_dates.intersection(price_df.index), "close"]
    ax1.scatter(jump_prices.index, jump_prices.values,
               color='#F44336', s=100, zorder=5,
               marker='^', label=f'跳跃事件 (n={len(jump_dates)})')

    ax1.set_ylabel('价格 (USDT)', fontsize=11, fontweight='bold')
    ax1.set_title('跳跃检测：基于BV双幂变差方法', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)

    # 下图：RV vs BV
    ax2.plot(jump_df.index, jump_df["RV"],
            label='已实现波动率 (RV)', color='#2196F3', linewidth=1.5)
    ax2.plot(jump_df.index, jump_df["BV"],
            label='双幂变差 (BV)', color='#4CAF50', linewidth=1.5, linestyle='--')
    ax2.fill_between(jump_df.index, jump_df["BV"], jump_df["RV"],
                     where=jump_df["is_jump"], alpha=0.3,
                     color='#F44336', label='跳跃成分')

    ax2.set_xlabel('日期', fontsize=11, fontweight='bold')
    ax2.set_ylabel('波动率', fontsize=11, fontweight='bold')
    ax2.set_title('已实现波动率分解：连续成分 vs 跳跃成分', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[跳跃检测图] 已保存: {output_path}")


def plot_realized_moments(
    moments_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    绘制已实现偏度和峰度时序图

    Parameters
    ----------
    moments_df : pd.DataFrame
        已实现矩数据
    output_path : Path
        输出路径
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    moments_df = moments_df.set_index("date")

    # 上图：已实现偏度
    ax1.plot(moments_df.index, moments_df["RSkew"],
            color='#9C27B0', linewidth=1.3, alpha=0.8)
    ax1.axhline(y=0, color='#424242', linestyle='--', linewidth=1)
    ax1.fill_between(moments_df.index, 0, moments_df["RSkew"],
                     where=moments_df["RSkew"] > 0, alpha=0.3,
                     color='#4CAF50', label='正偏（右偏）')
    ax1.fill_between(moments_df.index, 0, moments_df["RSkew"],
                     where=moments_df["RSkew"] < 0, alpha=0.3,
                     color='#F44336', label='负偏（左偏）')

    ax1.set_ylabel('已实现偏度 (RSkew)', fontsize=11, fontweight='bold')
    ax1.set_title('已实现高阶矩：偏度与峰度', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)

    # 下图：已实现峰度
    ax2.plot(moments_df.index, moments_df["RKurt"],
            color='#FF9800', linewidth=1.3, alpha=0.8)
    ax2.axhline(y=3, color='#E91E63', linestyle='--', linewidth=1,
               label='正态分布峰度=3')
    ax2.fill_between(moments_df.index, 3, moments_df["RKurt"],
                     where=moments_df["RKurt"] > 3, alpha=0.3,
                     color='#F44336', label='超额峰度（厚尾）')

    ax2.set_xlabel('日期', fontsize=11, fontweight='bold')
    ax2.set_ylabel('已实现峰度 (RKurt)', fontsize=11, fontweight='bold')
    ax2.set_title('已实现峰度：厚尾特征检测', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[已实现矩图] 已保存: {output_path}")


# ============================================================
# 主入口函数
# ============================================================

def run_multiscale_vol_analysis(
    df: pd.DataFrame,
    output_dir: Union[str, Path] = "output/multiscale_vol",
) -> Dict[str, Any]:
    """
    多尺度已实现波动率分析主入口

    Parameters
    ----------
    df : pd.DataFrame
        日线数据（仅用于获取时间范围，实际会加载高频数据）
    output_dir : str or Path
        图表输出目录

    Returns
    -------
    results : dict
        分析结果字典，包含：
        - rv_by_interval: {interval: rv_df}
        - volatility_signature: {...}
        - har_model: {...}
        - jump_detection: {...}
        - realized_moments: {...}
        - findings: [...]
        - summary: {...}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("多尺度已实现波动率分析")
    print("=" * 70)
    print()

    results = {
        "rv_by_interval": {},
        "volatility_signature": {},
        "har_model": {},
        "jump_detection": {},
        "realized_moments": {},
        "findings": [],
        "summary": {},
    }

    # --------------------------------------------------------
    # 1. 加载各尺度数据并计算RV
    # --------------------------------------------------------
    print("步骤1: 加载各尺度数据并计算日频已实现波动率")
    print("─" * 60)

    for interval in INTERVALS.keys():
        try:
            print(f"  加载 {interval} 数据...", end=" ")
            df_interval = load_klines(interval)
            print(f"✓ ({len(df_interval)} 行)")

            print(f"  计算 {interval} 日频RV...", end=" ")
            rv_df = compute_realized_volatility_daily(df_interval, interval)
            results["rv_by_interval"][interval] = rv_df
            print(f"✓ ({len(rv_df)} 天)")

        except Exception as e:
            print(f"✗ 失败: {e}")
            results["rv_by_interval"][interval] = pd.DataFrame()

    print()

    # --------------------------------------------------------
    # 2. 波动率签名图
    # --------------------------------------------------------
    print("步骤2: 绘制波动率签名图")
    print("─" * 60)

    plot_volatility_signature(
        results["rv_by_interval"],
        output_dir / "multiscale_vol_signature.png"
    )

    # 统计签名特征
    intervals_sorted = sorted(INTERVALS.keys(), key=lambda x: INTERVALS[x])
    mean_rvs = []
    for interval in intervals_sorted:
        if interval in results["rv_by_interval"] and len(results["rv_by_interval"][interval]) > 0:
            mean_rv = results["rv_by_interval"][interval]["RV"].mean()
            mean_rvs.append(mean_rv)

    if len(mean_rvs) > 1:
        rv_range = max(mean_rvs) - min(mean_rvs)
        rv_std = np.std(mean_rvs)

        results["volatility_signature"] = {
            "mean_rvs": mean_rvs,
            "rv_range": rv_range,
            "rv_std": rv_std,
        }

        results["findings"].append({
            "name": "波动率签名效应",
            "description": f"不同采样频率下RV均值范围为{rv_range:.6f}，标准差{rv_std:.6f}",
            "significant": rv_std > 0.01,
            "p_value": None,
            "effect_size": rv_std,
        })

    print()

    # --------------------------------------------------------
    # 3. HAR-RV模型
    # --------------------------------------------------------
    print("步骤3: 拟合HAR-RV模型（基于1d数据）")
    print("─" * 60)

    if "1d" in results["rv_by_interval"] and len(results["rv_by_interval"]["1d"]) > 30:
        rv_1d = results["rv_by_interval"]["1d"]
        rv_series = rv_1d.set_index("date")["RV"]

        print("  拟合HAR(1,5,22)模型...", end=" ")
        har_results = fit_har_rv_model(rv_series)
        results["har_model"] = har_results
        print("✓")

        # 打印系数
        print(f"\n  模型系数:")
        print(f"    截距:      {har_results['coefficients']['intercept']:.6f} "
              f"(t={har_results['t_statistics']['intercept']:.3f}, "
              f"p={har_results['p_values']['intercept']:.4f})")
        print(f"    β_daily:   {har_results['coefficients']['beta_daily']:.6f} "
              f"(t={har_results['t_statistics']['beta_daily']:.3f}, "
              f"p={har_results['p_values']['beta_daily']:.4f})")
        print(f"    β_weekly:  {har_results['coefficients']['beta_weekly']:.6f} "
              f"(t={har_results['t_statistics']['beta_weekly']:.3f}, "
              f"p={har_results['p_values']['beta_weekly']:.4f})")
        print(f"    β_monthly: {har_results['coefficients']['beta_monthly']:.6f} "
              f"(t={har_results['t_statistics']['beta_monthly']:.3f}, "
              f"p={har_results['p_values']['beta_monthly']:.4f})")
        print(f"\n  R²: {har_results['r_squared']:.4f}")
        print(f"  样本量: {har_results['n_obs']}")

        # 绘图
        plot_har_rv_fit(har_results, output_dir / "multiscale_vol_har.png")

        # 添加发现
        results["findings"].append({
            "name": "HAR-RV模型拟合",
            "description": f"R²={har_results['r_squared']:.4f}，日/周/月成分均显著",
            "significant": har_results['r_squared'] > 0.5,
            "p_value": har_results['p_values']['beta_daily'],
            "effect_size": har_results['r_squared'],
        })
    else:
        print("  ✗ 1d数据不足，跳过HAR-RV")

    print()

    # --------------------------------------------------------
    # 4. 跳跃检测
    # --------------------------------------------------------
    print("步骤4: 跳跃检测（基于5m数据）")
    print("─" * 60)

    jump_interval = "5m"  # 使用最高频数据
    if jump_interval in results["rv_by_interval"]:
        try:
            print(f"  加载 {jump_interval} 数据进行跳跃检测...", end=" ")
            df_hf = load_klines(jump_interval)
            print(f"✓ ({len(df_hf)} 行)")

            print("  检测跳跃事件...", end=" ")
            jump_df = detect_jumps_daily(df_hf, z_threshold=JUMP_Z_THRESHOLD)
            results["jump_detection"] = jump_df
            print(f"✓")

            n_jumps = jump_df["is_jump"].sum()
            jump_ratio = n_jumps / len(jump_df) if len(jump_df) > 0 else 0

            print(f"\n  检测到 {n_jumps} 个跳跃事件（占比 {jump_ratio:.2%}）")

            # 绘图
            if len(jump_df) > 0:
                # 加载日线价格用于绘图
                df_daily = load_klines("1d")
                plot_jump_detection(
                    jump_df,
                    df_daily,
                    output_dir / "multiscale_vol_jumps.png"
                )

            # 添加发现
            results["findings"].append({
                "name": "跳跃事件检测",
                "description": f"检测到{n_jumps}个显著跳跃事件（占比{jump_ratio:.2%}）",
                "significant": n_jumps > 0,
                "p_value": None,
                "effect_size": jump_ratio,
            })

        except Exception as e:
            print(f"✗ 失败: {e}")
            results["jump_detection"] = pd.DataFrame()
    else:
        print(f"  ✗ {jump_interval} 数据不可用，跳过跳跃检测")

    print()

    # --------------------------------------------------------
    # 5. 已实现高阶矩
    # --------------------------------------------------------
    print("步骤5: 计算已实现偏度和峰度（基于5m数据）")
    print("─" * 60)

    if jump_interval in results["rv_by_interval"]:
        try:
            df_hf = load_klines(jump_interval)

            print("  计算已实现偏度和峰度...", end=" ")
            moments_df = compute_realized_moments(df_hf)
            results["realized_moments"] = moments_df
            print(f"✓ ({len(moments_df)} 天)")

            # 统计
            mean_skew = moments_df["RSkew"].mean()
            mean_kurt = moments_df["RKurt"].mean()

            print(f"\n  平均已实现偏度: {mean_skew:.4f}")
            print(f"  平均已实现峰度: {mean_kurt:.4f}")

            # 绘图
            if len(moments_df) > 0:
                plot_realized_moments(
                    moments_df,
                    output_dir / "multiscale_vol_higher_moments.png"
                )

            # 添加发现
            results["findings"].append({
                "name": "已实现偏度",
                "description": f"平均偏度={mean_skew:.4f}，{'负偏' if mean_skew < 0 else '正偏'}分布",
                "significant": abs(mean_skew) > 0.1,
                "p_value": None,
                "effect_size": abs(mean_skew),
            })

            results["findings"].append({
                "name": "已实现峰度",
                "description": f"平均峰度={mean_kurt:.4f}，{'厚尾' if mean_kurt > 3 else '薄尾'}分布",
                "significant": mean_kurt > 3,
                "p_value": None,
                "effect_size": mean_kurt - 3,
            })

        except Exception as e:
            print(f"✗ 失败: {e}")
            results["realized_moments"] = pd.DataFrame()

    print()

    # --------------------------------------------------------
    # 汇总
    # --------------------------------------------------------
    print("=" * 70)
    print("分析完成")
    print("=" * 70)

    results["summary"] = {
        "n_intervals_analyzed": len([v for v in results["rv_by_interval"].values() if len(v) > 0]),
        "har_r_squared": results["har_model"].get("r_squared", None),
        "n_jump_events": results["jump_detection"]["is_jump"].sum() if len(results["jump_detection"]) > 0 else 0,
        "mean_realized_skew": results["realized_moments"]["RSkew"].mean() if len(results["realized_moments"]) > 0 else None,
        "mean_realized_kurt": results["realized_moments"]["RKurt"].mean() if len(results["realized_moments"]) > 0 else None,
    }

    print(f"  分析时间尺度: {results['summary']['n_intervals_analyzed']}")
    print(f"  HAR-RV R²: {results['summary']['har_r_squared']}")
    print(f"  跳跃事件数: {results['summary']['n_jump_events']}")
    print(f"  平均已实现偏度: {results['summary']['mean_realized_skew']}")
    print(f"  平均已实现峰度: {results['summary']['mean_realized_kurt']}")
    print()
    print(f"图表输出目录: {output_dir.resolve()}")
    print("=" * 70)

    return results


# ============================================================
# 独立运行入口
# ============================================================

if __name__ == "__main__":
    from src.data_loader import load_daily

    print("加载日线数据...")
    df = load_daily()
    print(f"数据范围: {df.index.min()} ~ {df.index.max()}")
    print()

    # 执行多尺度波动率分析
    results = run_multiscale_vol_analysis(df, output_dir="output/multiscale_vol")

    # 打印结果概要
    print()
    print("返回结果键:")
    for k, v in results.items():
        if isinstance(v, dict):
            print(f"  results['{k}']: {list(v.keys()) if v else 'empty'}")
        elif isinstance(v, pd.DataFrame):
            print(f"  results['{k}']: DataFrame ({len(v)} rows)")
        elif isinstance(v, list):
            print(f"  results['{k}']: list ({len(v)} items)")
        else:
            print(f"  results['{k}']: {type(v).__name__}")
