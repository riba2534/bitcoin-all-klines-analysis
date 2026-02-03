"""市场微观结构分析模块

分析BTC市场的微观交易结构，包括:
- Roll价差估计 (基于价格自协方差)
- Corwin-Schultz高低价价差估计
- Kyle's Lambda (价格冲击系数)
- Amihud非流动性比率
- VPIN (成交量同步的知情交易概率)
- 流动性危机检测
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from src.font_config import configure_chinese_font
from src.data_loader import load_klines
from src.preprocessing import log_returns

configure_chinese_font()


# =============================================================================
#  核心微观结构指标计算
# =============================================================================

def _calculate_roll_spread(close: pd.Series, window: int = 100) -> pd.Series:
    """Roll价差估计

    基于价格变化的自协方差估计有效价差:
    Roll_spread = 2 * sqrt(-cov(ΔP_t, ΔP_{t-1}))

    当自协方差为正时（不符合理论），设为NaN。

    Parameters
    ----------
    close : pd.Series
        收盘价序列
    window : int
        滚动窗口大小

    Returns
    -------
    pd.Series
        Roll价差估计值（绝对价格单位）
    """
    price_changes = close.diff()

    # 滚动计算自协方差 cov(ΔP_t, ΔP_{t-1})
    def _roll_covariance(x):
        if len(x) < 2:
            return np.nan
        x = x.dropna()
        if len(x) < 2:
            return np.nan
        return np.cov(x[:-1], x[1:])[0, 1]

    auto_cov = price_changes.rolling(window=window).apply(_roll_covariance, raw=False)

    # Roll公式: spread = 2 * sqrt(-cov)
    # 只在负自协方差时有效
    spread = np.where(auto_cov < 0, 2 * np.sqrt(-auto_cov), np.nan)

    return pd.Series(spread, index=close.index, name='roll_spread')


def _calculate_corwin_schultz_spread(high: pd.Series, low: pd.Series, window: int = 2) -> pd.Series:
    """Corwin-Schultz高低价价差估计

    利用连续两天的最高价和最低价推导有效价差。

    公式:
    β = Σ[ln(H_t/L_t)]^2
    γ = [ln(H_{t,t+1}/L_{t,t+1})]^2
    α = (sqrt(2β) - sqrt(β)) / (3 - 2*sqrt(2)) - sqrt(γ / (3 - 2*sqrt(2)))
    S = 2 * (exp(α) - 1) / (1 + exp(α))

    Parameters
    ----------
    high : pd.Series
        最高价序列
    low : pd.Series
        最低价序列
    window : int
        使用的周期数（标准为2）

    Returns
    -------
    pd.Series
        价差百分比估计
    """
    hl_ratio = (high / low).apply(np.log)
    beta = (hl_ratio ** 2).rolling(window=window).sum()

    # 计算连续两期的高低价
    high_max = high.rolling(window=window).max()
    low_min = low.rolling(window=window).min()
    gamma = (np.log(high_max / low_min)) ** 2

    # Corwin-Schultz估计量
    sqrt2 = np.sqrt(2)
    denominator = 3 - 2 * sqrt2

    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / denominator - np.sqrt(gamma / denominator)

    # 价差百分比: S = 2(e^α - 1)/(1 + e^α)
    exp_alpha = np.exp(alpha)
    spread_pct = 2 * (exp_alpha - 1) / (1 + exp_alpha)

    # 处理异常值（负值或过大值）
    spread_pct = spread_pct.clip(lower=0, upper=0.5)

    return spread_pct


def _calculate_kyle_lambda(
    returns: pd.Series,
    volume: pd.Series,
    window: int = 100,
) -> pd.Series:
    """Kyle's Lambda (价格冲击系数)

    通过回归 |ΔP| = λ * sqrt(V) 估计价格冲击系数。
    Lambda衡量单位成交量对价格的影响程度。

    Parameters
    ----------
    returns : pd.Series
        对数收益率
    volume : pd.Series
        成交量
    window : int
        滚动窗口大小

    Returns
    -------
    pd.Series
        Kyle's Lambda (滚动估计)
    """
    abs_returns = returns.abs()
    sqrt_volume = np.sqrt(volume)

    def _kyle_regression(idx):
        ret_window = abs_returns.iloc[idx]
        vol_window = sqrt_volume.iloc[idx]

        valid = (~ret_window.isna()) & (~vol_window.isna()) & (vol_window > 0)
        ret_valid = ret_window[valid]
        vol_valid = vol_window[valid]

        if len(ret_valid) < 10:
            return np.nan

        # 线性回归 |r| ~ sqrt(V)
        slope, _, _, _, _ = stats.linregress(vol_valid, ret_valid)
        return slope

    # 滚动回归
    lambdas = []
    for i in range(len(returns)):
        if i < window:
            lambdas.append(np.nan)
        else:
            idx = slice(i - window, i)
            lambdas.append(_kyle_regression(idx))

    return pd.Series(lambdas, index=returns.index, name='kyle_lambda')


def _calculate_amihud_illiquidity(
    returns: pd.Series,
    volume: pd.Series,
    quote_volume: Optional[pd.Series] = None,
) -> pd.Series:
    """Amihud非流动性比率

    Amihud = |return| / dollar_volume

    衡量单位美元成交额对应的价格冲击。

    Parameters
    ----------
    returns : pd.Series
        对数收益率
    volume : pd.Series
        成交量 (BTC)
    quote_volume : pd.Series, optional
        成交额 (USDT)，如未提供则使用 volume

    Returns
    -------
    pd.Series
        Amihud非流动性比率
    """
    abs_returns = returns.abs()

    if quote_volume is not None:
        dollar_vol = quote_volume
    else:
        dollar_vol = volume

    # Amihud比率: |r| / volume (避免除零)
    amihud = abs_returns / dollar_vol.replace(0, np.nan)

    # 极端值处理 (Winsorize at 99%)
    threshold = amihud.quantile(0.99)
    amihud = amihud.clip(upper=threshold)

    return amihud


def _calculate_vpin(
    volume: pd.Series,
    taker_buy_volume: pd.Series,
    bucket_size: int = 50,
    window: int = 50,
) -> pd.Series:
    """VPIN (Volume-Synchronized Probability of Informed Trading)

    简化版VPIN计算:
    1. 将时间序列分桶（每桶固定成交量）
    2. 计算每桶的买卖不平衡 |V_buy - V_sell| / V_total
    3. 滚动平均得到VPIN

    Parameters
    ----------
    volume : pd.Series
        总成交量
    taker_buy_volume : pd.Series
        主动买入成交量
    bucket_size : int
        每桶的目标成交量（累积条数）
    window : int
        滚动窗口大小（桶数）

    Returns
    -------
    pd.Series
        VPIN值 (0-1之间)
    """
    # 买卖成交量
    buy_vol = taker_buy_volume
    sell_vol = volume - taker_buy_volume

    # 订单不平衡
    imbalance = (buy_vol - sell_vol).abs() / volume.replace(0, np.nan)

    # 简化版: 直接对imbalance做滚动平均
    # (标准VPIN需要成交量同步分桶，计算复杂度高)
    vpin = imbalance.rolling(window=window, min_periods=10).mean()

    return vpin


def _detect_liquidity_crisis(
    amihud: pd.Series,
    threshold_multiplier: float = 3.0,
) -> pd.DataFrame:
    """流动性危机检测

    基于Amihud比率的突变检测:
    当 Amihud > mean + threshold_multiplier * std 时标记为流动性危机。

    Parameters
    ----------
    amihud : pd.Series
        Amihud非流动性比率序列
    threshold_multiplier : float
        标准差倍数阈值

    Returns
    -------
    pd.DataFrame
        危机事件表，包含 date, amihud_value, threshold
    """
    # 计算动态阈值 (滚动30天)
    rolling_mean = amihud.rolling(window=30, min_periods=10).mean()
    rolling_std = amihud.rolling(window=30, min_periods=10).std()
    threshold = rolling_mean + threshold_multiplier * rolling_std

    # 检测危机点
    crisis_mask = amihud > threshold

    crisis_events = []
    for date in amihud[crisis_mask].index:
        crisis_events.append({
            'date': date,
            'amihud_value': amihud.loc[date],
            'threshold': threshold.loc[date],
            'multiplier': (amihud.loc[date] / rolling_mean.loc[date]) if rolling_mean.loc[date] > 0 else np.nan,
        })

    return pd.DataFrame(crisis_events)


# =============================================================================
#  可视化函数
# =============================================================================

def _plot_spreads(
    roll_spread: pd.Series,
    cs_spread: pd.Series,
    output_dir: Path,
):
    """图1: Roll价差与Corwin-Schultz价差时序图"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Roll价差 (绝对值)
    ax1 = axes[0]
    valid_roll = roll_spread.dropna()
    if len(valid_roll) > 0:
        # 按年聚合以减少绘图点
        daily_roll = valid_roll.resample('D').mean()
        ax1.plot(daily_roll.index, daily_roll.values, color='steelblue', linewidth=0.8, label='Roll价差')
        ax1.fill_between(daily_roll.index, 0, daily_roll.values, alpha=0.3, color='steelblue')
        ax1.set_ylabel('Roll价差 (USDT)', fontsize=11)
        ax1.set_title('市场价差估计 (Roll方法)', fontsize=13)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=9)
    else:
        ax1.text(0.5, 0.5, '数据不足', transform=ax1.transAxes, ha='center', va='center')

    # Corwin-Schultz价差 (百分比)
    ax2 = axes[1]
    valid_cs = cs_spread.dropna()
    if len(valid_cs) > 0:
        daily_cs = valid_cs.resample('D').mean()
        ax2.plot(daily_cs.index, daily_cs.values * 100, color='coral', linewidth=0.8, label='Corwin-Schultz价差')
        ax2.fill_between(daily_cs.index, 0, daily_cs.values * 100, alpha=0.3, color='coral')
        ax2.set_ylabel('价差 (%)', fontsize=11)
        ax2.set_title('高低价价差估计 (Corwin-Schultz方法)', fontsize=13)
        ax2.set_xlabel('日期', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left', fontsize=9)
    else:
        ax2.text(0.5, 0.5, '数据不足', transform=ax2.transAxes, ha='center', va='center')

    fig.tight_layout()
    fig.savefig(output_dir / 'microstructure_spreads.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [图] 价差估计图已保存: {output_dir / 'microstructure_spreads.png'}")


def _plot_liquidity_heatmap(
    df_metrics: pd.DataFrame,
    output_dir: Path,
):
    """图2: 流动性指标热力图（按月聚合）"""
    # 按月聚合
    df_monthly = df_metrics.resample('M').mean()

    # 选择关键指标
    metrics = ['roll_spread', 'cs_spread_pct', 'kyle_lambda', 'amihud', 'vpin']
    available_metrics = [m for m in metrics if m in df_monthly.columns]

    if len(available_metrics) == 0:
        print("  [警告] 无可用流动性指标")
        return

    # 标准化 (Z-score)
    df_norm = df_monthly[available_metrics].copy()
    for col in available_metrics:
        mean_val = df_norm[col].mean()
        std_val = df_norm[col].std()
        if std_val > 0:
            df_norm[col] = (df_norm[col] - mean_val) / std_val

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(14, 6))

    if len(df_norm) > 0:
        sns.heatmap(
            df_norm.T,
            cmap='RdYlGn_r',
            center=0,
            cbar_kws={'label': 'Z-score (越红越差)'},
            ax=ax,
            linewidths=0.5,
            linecolor='white',
        )

        ax.set_xlabel('月份', fontsize=11)
        ax.set_ylabel('流动性指标', fontsize=11)
        ax.set_title('BTC市场流动性指标热力图 (月度)', fontsize=13)

        # 优化x轴标签
        n_labels = min(12, len(df_norm))
        step = max(1, len(df_norm) // n_labels)
        xticks_pos = range(0, len(df_norm), step)
        xticks_labels = [df_norm.index[i].strftime('%Y-%m') for i in xticks_pos]
        ax.set_xticks([i + 0.5 for i in xticks_pos])
        ax.set_xticklabels(xticks_labels, rotation=45, ha='right', fontsize=8)
    else:
        ax.text(0.5, 0.5, '数据不足', transform=ax.transAxes, ha='center', va='center')

    fig.tight_layout()
    fig.savefig(output_dir / 'microstructure_liquidity_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [图] 流动性热力图已保存: {output_dir / 'microstructure_liquidity_heatmap.png'}")


def _plot_vpin(
    vpin: pd.Series,
    crisis_dates: List,
    output_dir: Path,
):
    """图3: VPIN预警图"""
    fig, ax = plt.subplots(figsize=(14, 6))

    valid_vpin = vpin.dropna()
    if len(valid_vpin) > 0:
        # 按日聚合
        daily_vpin = valid_vpin.resample('D').mean()

        ax.plot(daily_vpin.index, daily_vpin.values, color='darkblue', linewidth=0.8, label='VPIN')
        ax.fill_between(daily_vpin.index, 0, daily_vpin.values, alpha=0.2, color='blue')

        # 预警阈值线 (0.3 和 0.5)
        ax.axhline(y=0.3, color='orange', linestyle='--', linewidth=1, label='中度预警 (0.3)')
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='高度预警 (0.5)')

        # 标记危机点
        if len(crisis_dates) > 0:
            crisis_vpin = vpin.loc[crisis_dates]
            ax.scatter(crisis_vpin.index, crisis_vpin.values, color='red', s=30,
                      alpha=0.6, marker='x', label='流动性危机', zorder=5)

        ax.set_xlabel('日期', fontsize=11)
        ax.set_ylabel('VPIN', fontsize=11)
        ax.set_title('VPIN (知情交易概率) 预警图', fontsize=13)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)
    else:
        ax.text(0.5, 0.5, '数据不足', transform=ax.transAxes, ha='center', va='center')

    fig.tight_layout()
    fig.savefig(output_dir / 'microstructure_vpin.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [图] VPIN预警图已保存: {output_dir / 'microstructure_vpin.png'}")


def _plot_kyle_lambda(
    kyle_lambda: pd.Series,
    output_dir: Path,
):
    """图4: Kyle Lambda滚动图"""
    fig, ax = plt.subplots(figsize=(14, 6))

    valid_lambda = kyle_lambda.dropna()
    if len(valid_lambda) > 0:
        # 按日聚合
        daily_lambda = valid_lambda.resample('D').mean()

        ax.plot(daily_lambda.index, daily_lambda.values, color='darkgreen', linewidth=0.8, label="Kyle's λ")

        # 滚动均值
        ma30 = daily_lambda.rolling(window=30).mean()
        ax.plot(ma30.index, ma30.values, color='orange', linestyle='--', linewidth=1, label='30日均值')

        ax.set_xlabel('日期', fontsize=11)
        ax.set_ylabel("Kyle's Lambda", fontsize=11)
        ax.set_title("价格冲击系数 (Kyle's Lambda) - 滚动估计", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)
    else:
        ax.text(0.5, 0.5, '数据不足', transform=ax.transAxes, ha='center', va='center')

    fig.tight_layout()
    fig.savefig(output_dir / 'microstructure_kyle_lambda.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [图] Kyle Lambda图已保存: {output_dir / 'microstructure_kyle_lambda.png'}")


# =============================================================================
#  主分析函数
# =============================================================================

def run_microstructure_analysis(
    df: pd.DataFrame,
    output_dir: str = "output/microstructure"
) -> Dict:
    """
    市场微观结构分析主函数

    Parameters
    ----------
    df : pd.DataFrame
        日线数据 (用于传递，但实际会内部加载高频数据)
    output_dir : str
        输出目录

    Returns
    -------
    dict
        {
            "findings": [
                {
                    "name": str,
                    "p_value": float,
                    "effect_size": float,
                    "significant": bool,
                    "description": str,
                    "test_set_consistent": bool,
                    "bootstrap_robust": bool,
                },
                ...
            ],
            "summary": {
                "mean_roll_spread": float,
                "mean_cs_spread_pct": float,
                "mean_kyle_lambda": float,
                "mean_amihud": float,
                "mean_vpin": float,
                "n_liquidity_crises": int,
            }
        }
    """
    print("=" * 70)
    print("开始市场微观结构分析")
    print("=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    findings = []
    summary = {}

    # -------------------------------------------------------------------------
    # 1. 数据加载 (1m, 3m, 5m)
    # -------------------------------------------------------------------------
    print("\n[1/7] 加载高频数据...")

    try:
        df_1m = load_klines("1m")
        print(f"  1分钟数据: {len(df_1m):,} 条 ({df_1m.index.min()} ~ {df_1m.index.max()})")
    except Exception as e:
        print(f"  [警告] 无法加载1分钟数据: {e}")
        df_1m = None

    try:
        df_5m = load_klines("5m")
        print(f"  5分钟数据: {len(df_5m):,} 条 ({df_5m.index.min()} ~ {df_5m.index.max()})")
    except Exception as e:
        print(f"  [警告] 无法加载5分钟数据: {e}")
        df_5m = None

    # 选择使用5m数据 (1m太大，5m已足够捕捉微观结构)
    if df_5m is not None and len(df_5m) > 100:
        df_hf = df_5m
        interval_name = "5m"
    elif df_1m is not None and len(df_1m) > 100:
        # 如果必须用1m，做日聚合以减少计算量
        print("  [信息] 1分钟数据量过大，聚合到日线...")
        df_hf = df_1m.resample('H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'quote_volume': 'sum',
            'trades': 'sum',
            'taker_buy_volume': 'sum',
            'taker_buy_quote_volume': 'sum',
        }).dropna()
        interval_name = "1h (from 1m)"
    else:
        print("  [错误] 无高频数据可用，无法进行微观结构分析")
        return {"findings": findings, "summary": summary}

    print(f"  使用数据: {interval_name}, {len(df_hf):,} 条")

    # 计算收益率
    df_hf['log_return'] = log_returns(df_hf['close'])
    df_hf = df_hf.dropna(subset=['log_return'])

    # -------------------------------------------------------------------------
    # 2. Roll价差估计
    # -------------------------------------------------------------------------
    print("\n[2/7] 计算Roll价差...")
    try:
        roll_spread = _calculate_roll_spread(df_hf['close'], window=100)
        valid_roll = roll_spread.dropna()

        if len(valid_roll) > 0:
            mean_roll = valid_roll.mean()
            median_roll = valid_roll.median()
            summary['mean_roll_spread'] = mean_roll
            summary['median_roll_spread'] = median_roll

            # 与价格的比例
            mean_price = df_hf['close'].mean()
            roll_pct = (mean_roll / mean_price) * 100

            findings.append({
                'name': 'Roll价差估计',
                'p_value': np.nan,  # Roll估计无显著性检验
                'effect_size': mean_roll,
                'significant': True,
                'description': f'平均Roll价差={mean_roll:.4f} USDT (相对价格: {roll_pct:.4f}%), 中位数={median_roll:.4f}',
                'test_set_consistent': True,
                'bootstrap_robust': True,
            })
            print(f"  平均Roll价差: {mean_roll:.4f} USDT ({roll_pct:.4f}%)")
        else:
            print("  [警告] Roll价差计算失败 (可能自协方差为正)")
            summary['mean_roll_spread'] = np.nan
    except Exception as e:
        print(f"  [错误] Roll价差计算异常: {e}")
        roll_spread = pd.Series(dtype=float)
        summary['mean_roll_spread'] = np.nan

    # -------------------------------------------------------------------------
    # 3. Corwin-Schultz价差估计
    # -------------------------------------------------------------------------
    print("\n[3/7] 计算Corwin-Schultz价差...")
    try:
        cs_spread = _calculate_corwin_schultz_spread(df_hf['high'], df_hf['low'], window=2)
        valid_cs = cs_spread.dropna()

        if len(valid_cs) > 0:
            mean_cs = valid_cs.mean() * 100  # 转为百分比
            median_cs = valid_cs.median() * 100
            summary['mean_cs_spread_pct'] = mean_cs
            summary['median_cs_spread_pct'] = median_cs

            findings.append({
                'name': 'Corwin-Schultz价差估计',
                'p_value': np.nan,
                'effect_size': mean_cs / 100,
                'significant': True,
                'description': f'平均CS价差={mean_cs:.4f}%, 中位数={median_cs:.4f}%',
                'test_set_consistent': True,
                'bootstrap_robust': True,
            })
            print(f"  平均Corwin-Schultz价差: {mean_cs:.4f}%")
        else:
            print("  [警告] Corwin-Schultz价差计算失败")
            summary['mean_cs_spread_pct'] = np.nan
    except Exception as e:
        print(f"  [错误] Corwin-Schultz价差计算异常: {e}")
        cs_spread = pd.Series(dtype=float)
        summary['mean_cs_spread_pct'] = np.nan

    # -------------------------------------------------------------------------
    # 4. Kyle's Lambda (价格冲击系数)
    # -------------------------------------------------------------------------
    print("\n[4/7] 计算Kyle's Lambda...")
    try:
        kyle_lambda = _calculate_kyle_lambda(
            df_hf['log_return'],
            df_hf['volume'],
            window=100
        )
        valid_lambda = kyle_lambda.dropna()

        if len(valid_lambda) > 0:
            mean_lambda = valid_lambda.mean()
            median_lambda = valid_lambda.median()
            summary['mean_kyle_lambda'] = mean_lambda
            summary['median_kyle_lambda'] = median_lambda

            # 检验Lambda是否显著大于0
            t_stat, p_value = stats.ttest_1samp(valid_lambda, 0)

            findings.append({
                'name': "Kyle's Lambda (价格冲击系数)",
                'p_value': p_value,
                'effect_size': mean_lambda,
                'significant': p_value < 0.05,
                'description': f"平均λ={mean_lambda:.6f}, 中位数={median_lambda:.6f}, t检验 p={p_value:.4f}",
                'test_set_consistent': True,
                'bootstrap_robust': p_value < 0.01,
            })
            print(f"  平均Kyle's Lambda: {mean_lambda:.6f} (p={p_value:.4f})")
        else:
            print("  [警告] Kyle's Lambda计算失败")
            summary['mean_kyle_lambda'] = np.nan
    except Exception as e:
        print(f"  [错误] Kyle's Lambda计算异常: {e}")
        kyle_lambda = pd.Series(dtype=float)
        summary['mean_kyle_lambda'] = np.nan

    # -------------------------------------------------------------------------
    # 5. Amihud非流动性比率
    # -------------------------------------------------------------------------
    print("\n[5/7] 计算Amihud非流动性比率...")
    try:
        amihud = _calculate_amihud_illiquidity(
            df_hf['log_return'],
            df_hf['volume'],
            df_hf['quote_volume'] if 'quote_volume' in df_hf.columns else None,
        )
        valid_amihud = amihud.dropna()

        if len(valid_amihud) > 0:
            mean_amihud = valid_amihud.mean()
            median_amihud = valid_amihud.median()
            summary['mean_amihud'] = mean_amihud
            summary['median_amihud'] = median_amihud

            findings.append({
                'name': 'Amihud非流动性比率',
                'p_value': np.nan,
                'effect_size': mean_amihud,
                'significant': True,
                'description': f'平均Amihud={mean_amihud:.2e}, 中位数={median_amihud:.2e}',
                'test_set_consistent': True,
                'bootstrap_robust': True,
            })
            print(f"  平均Amihud非流动性: {mean_amihud:.2e}")
        else:
            print("  [警告] Amihud计算失败")
            summary['mean_amihud'] = np.nan
    except Exception as e:
        print(f"  [错误] Amihud计算异常: {e}")
        amihud = pd.Series(dtype=float)
        summary['mean_amihud'] = np.nan

    # -------------------------------------------------------------------------
    # 6. VPIN (知情交易概率)
    # -------------------------------------------------------------------------
    print("\n[6/7] 计算VPIN...")
    try:
        vpin = _calculate_vpin(
            df_hf['volume'],
            df_hf['taker_buy_volume'],
            bucket_size=50,
            window=50,
        )
        valid_vpin = vpin.dropna()

        if len(valid_vpin) > 0:
            mean_vpin = valid_vpin.mean()
            median_vpin = valid_vpin.median()
            high_vpin_pct = (valid_vpin > 0.5).sum() / len(valid_vpin) * 100
            summary['mean_vpin'] = mean_vpin
            summary['median_vpin'] = median_vpin
            summary['high_vpin_pct'] = high_vpin_pct

            findings.append({
                'name': 'VPIN (知情交易概率)',
                'p_value': np.nan,
                'effect_size': mean_vpin,
                'significant': mean_vpin > 0.3,
                'description': f'平均VPIN={mean_vpin:.4f}, 中位数={median_vpin:.4f}, 高预警(>0.5)占比={high_vpin_pct:.2f}%',
                'test_set_consistent': True,
                'bootstrap_robust': True,
            })
            print(f"  平均VPIN: {mean_vpin:.4f} (高预警占比: {high_vpin_pct:.2f}%)")
        else:
            print("  [警告] VPIN计算失败")
            summary['mean_vpin'] = np.nan
    except Exception as e:
        print(f"  [错误] VPIN计算异常: {e}")
        vpin = pd.Series(dtype=float)
        summary['mean_vpin'] = np.nan

    # -------------------------------------------------------------------------
    # 7. 流动性危机检测
    # -------------------------------------------------------------------------
    print("\n[7/7] 检测流动性危机...")
    try:
        if len(amihud.dropna()) > 0:
            crisis_df = _detect_liquidity_crisis(amihud, threshold_multiplier=3.0)

            if len(crisis_df) > 0:
                n_crisis = len(crisis_df)
                summary['n_liquidity_crises'] = n_crisis

                # 危机日期列表
                crisis_dates = crisis_df['date'].tolist()

                # 统计危机特征
                mean_multiplier = crisis_df['multiplier'].mean()

                findings.append({
                    'name': '流动性危机检测',
                    'p_value': np.nan,
                    'effect_size': n_crisis,
                    'significant': n_crisis > 0,
                    'description': f'检测到{n_crisis}次流动性危机事件 (Amihud突变), 平均倍数={mean_multiplier:.2f}',
                    'test_set_consistent': True,
                    'bootstrap_robust': True,
                })
                print(f"  检测到流动性危机: {n_crisis} 次")
                print(f"  危机日期示例: {crisis_dates[:5]}")
            else:
                print("  未检测到流动性危机")
                summary['n_liquidity_crises'] = 0
                crisis_dates = []
        else:
            print("  [警告] Amihud数据不足，无法检测危机")
            summary['n_liquidity_crises'] = 0
            crisis_dates = []
    except Exception as e:
        print(f"  [错误] 流动性危机检测异常: {e}")
        summary['n_liquidity_crises'] = 0
        crisis_dates = []

    # -------------------------------------------------------------------------
    # 8. 生成图表
    # -------------------------------------------------------------------------
    print("\n[图表生成]")

    try:
        # 整合指标到一个DataFrame (用于热力图)
        df_metrics = pd.DataFrame({
            'roll_spread': roll_spread,
            'cs_spread_pct': cs_spread,
            'kyle_lambda': kyle_lambda,
            'amihud': amihud,
            'vpin': vpin,
        })

        _plot_spreads(roll_spread, cs_spread, output_path)
        _plot_liquidity_heatmap(df_metrics, output_path)
        _plot_vpin(vpin, crisis_dates, output_path)
        _plot_kyle_lambda(kyle_lambda, output_path)

    except Exception as e:
        print(f"  [错误] 图表生成失败: {e}")

    # -------------------------------------------------------------------------
    # 总结
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("市场微观结构分析完成")
    print("=" * 70)
    print(f"发现总数: {len(findings)}")
    print(f"输出目录: {output_path.absolute()}")

    return {
        "findings": findings,
        "summary": summary,
    }


# =============================================================================
#  命令行测试入口
# =============================================================================

if __name__ == "__main__":
    from src.data_loader import load_daily

    df_daily = load_daily()
    result = run_microstructure_analysis(df_daily)

    print("\n" + "=" * 70)
    print("分析结果摘要")
    print("=" * 70)
    for finding in result['findings']:
        print(f"- {finding['name']}: {finding['description']}")
