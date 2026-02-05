"""历史跌幅与熊市周期分析

识别所有熊市周期，统计历史跌幅分布，
提取底部特征（波动率收缩、RSI持续超卖等）。
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.signal import argrelextrema
from scipy import stats

from src.font_config import configure_chinese_font
configure_chinese_font()


def identify_bear_markets(df: pd.DataFrame, min_drawdown_pct: float = 20) -> List[Dict]:
    """
    识别历史上的熊市周期

    Parameters
    ----------
    df : pd.DataFrame
        日线数据
    min_drawdown_pct : float
        最小跌幅阈值（百分比）

    Returns
    -------
    List[Dict]
        熊市周期列表
    """
    prices = df['close'].values
    dates = df.index

    # 计算滚动最大值
    rolling_max = df['close'].cummax()
    drawdown = (df['close'] - rolling_max) / rolling_max * 100

    bear_markets = []
    in_bear = False
    peak_date = None
    peak_price = None

    for i in range(len(df)):
        current_drawdown = drawdown.iloc[i]

        if not in_bear and current_drawdown < -min_drawdown_pct:
            # 进入熊市
            in_bear = True
            # 找到之前的高点
            peak_idx = rolling_max[:i+1].idxmax()
            peak_date = peak_idx
            peak_price = rolling_max.loc[peak_idx]
            trough_date = dates[i]
            trough_price = prices[i]
            max_drawdown = current_drawdown

        elif in_bear:
            if current_drawdown < max_drawdown:
                # 更新最大跌幅
                max_drawdown = current_drawdown
                trough_date = dates[i]
                trough_price = prices[i]

            # 检查是否从低点反弹超过 20%
            recovery = (prices[i] - trough_price) / trough_price * 100
            if recovery > 20:
                # 熊市结束
                in_bear = False
                bear_markets.append({
                    'peak_date': peak_date,
                    'peak_price': peak_price,
                    'trough_date': trough_date,
                    'trough_price': trough_price,
                    'max_drawdown_pct': max_drawdown,
                    'duration_days': (trough_date - peak_date).days,
                    'recovery_date': dates[i],
                })

    # 如果当前仍在熊市中
    if in_bear:
        bear_markets.append({
            'peak_date': peak_date,
            'peak_price': peak_price,
            'trough_date': trough_date,
            'trough_price': trough_price,
            'max_drawdown_pct': max_drawdown,
            'duration_days': (trough_date - peak_date).days,
            'recovery_date': None,  # 尚未恢复
        })

    return bear_markets


def calc_drawdown_stats(bear_markets: List[Dict]) -> Dict:
    """计算跌幅统计"""
    if len(bear_markets) == 0:
        return {}

    drawdowns = [abs(b['max_drawdown_pct']) for b in bear_markets]
    durations = [b['duration_days'] for b in bear_markets]

    return {
        'count': len(bear_markets),
        'avg_drawdown': np.mean(drawdowns),
        'median_drawdown': np.median(drawdowns),
        'max_drawdown': np.max(drawdowns),
        'min_drawdown': np.min(drawdowns),
        'std_drawdown': np.std(drawdowns),
        'avg_duration_days': np.mean(durations),
        'median_duration_days': np.median(durations),
        'percentiles': {
            '25%': np.percentile(drawdowns, 25),
            '50%': np.percentile(drawdowns, 50),
            '75%': np.percentile(drawdowns, 75),
            '90%': np.percentile(drawdowns, 90),
        }
    }


def extract_bottom_features(df: pd.DataFrame, bear_market: Dict, lookback: int = 30) -> Dict:
    """
    提取底部特征

    Parameters
    ----------
    df : pd.DataFrame
        日线数据
    bear_market : Dict
        熊市周期信息
    lookback : int
        底部前的观察天数

    Returns
    -------
    Dict
        底部特征
    """
    trough_date = bear_market['trough_date']
    trough_idx = df.index.get_loc(trough_date)

    # 获取底部前后的数据
    start_idx = max(0, trough_idx - lookback)
    end_idx = min(len(df) - 1, trough_idx + lookback)

    pre_bottom = df.iloc[start_idx:trough_idx]
    post_bottom = df.iloc[trough_idx:end_idx + 1]

    features = {}

    # 1. 波动率特征
    if len(pre_bottom) > 5:
        returns = pre_bottom['close'].pct_change().dropna()
        features['pre_volatility'] = returns.std() * np.sqrt(252) * 100

        # 波动率是否收缩
        if len(returns) >= 20:
            early_vol = returns[:10].std()
            late_vol = returns[-10:].std()
            features['volatility_squeeze'] = (late_vol < early_vol * 0.7)
        else:
            features['volatility_squeeze'] = None

    # 2. RSI 特征
    if 'close' in df.columns and len(pre_bottom) >= 14:
        delta = pre_bottom['close'].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)

        features['rsi_at_bottom'] = rsi.iloc[-1] if len(rsi) > 0 and not np.isnan(rsi.iloc[-1]) else None
        features['rsi_below_30_days'] = (rsi < 30).sum()
        features['rsi_min'] = rsi.min() if len(rsi) > 0 else None

    # 3. 成交量特征
    if 'volume' in df.columns:
        vol_at_bottom = df['volume'].iloc[trough_idx]
        avg_vol = pre_bottom['volume'].mean() if len(pre_bottom) > 0 else vol_at_bottom
        features['volume_ratio_at_bottom'] = vol_at_bottom / avg_vol if avg_vol > 0 else None

        # 成交量是否枯竭后放量
        if len(pre_bottom) >= 10:
            late_vol = pre_bottom['volume'].iloc[-5:].mean()
            early_vol = pre_bottom['volume'].iloc[:5].mean()
            features['volume_exhaustion'] = (late_vol < early_vol * 0.5)
        else:
            features['volume_exhaustion'] = None

    # 4. 下跌速度特征
    if len(pre_bottom) > 0:
        total_drop = (pre_bottom['close'].iloc[-1] / pre_bottom['close'].iloc[0] - 1) * 100
        features['pre_bottom_drop_pct'] = total_drop
        features['avg_daily_drop'] = total_drop / len(pre_bottom)

    return features


def calc_current_drawdown(df: pd.DataFrame) -> Dict:
    """计算当前回撤情况"""
    rolling_max = df['close'].cummax()
    current_price = df['close'].iloc[-1]
    ath_price = rolling_max.iloc[-1]
    ath_date = rolling_max.idxmax()

    drawdown_pct = (current_price - ath_price) / ath_price * 100

    # 找当前周期的低点
    post_ath = df[df.index > ath_date]
    if len(post_ath) > 0:
        trough_price = post_ath['close'].min()
        trough_date = post_ath['close'].idxmin()
    else:
        trough_price = current_price
        trough_date = df.index[-1]

    return {
        'current_price': current_price,
        'ath_price': ath_price,
        'ath_date': ath_date,
        'current_drawdown_pct': drawdown_pct,
        'trough_price': trough_price,
        'trough_date': trough_date,
        'days_since_ath': (df.index[-1] - ath_date).days,
    }


def estimate_potential_bottom(bear_stats: Dict, current_info: Dict) -> Dict:
    """
    根据历史跌幅统计估算潜在底部

    Returns
    -------
    Dict
        包含乐观/中性/悲观情景的底部估计
    """
    ath_price = current_info['ath_price']

    # 基于历史跌幅分位数估算
    scenarios = {}

    if 'percentiles' in bear_stats:
        # 乐观情景：25% 分位跌幅
        optimistic_drawdown = bear_stats['percentiles']['25%']
        scenarios['optimistic'] = {
            'drawdown_pct': optimistic_drawdown,
            'price': ath_price * (1 - optimistic_drawdown / 100),
            'description': '25% 分位历史跌幅'
        }

        # 中性情景：中位数跌幅
        median_drawdown = bear_stats['percentiles']['50%']
        scenarios['neutral'] = {
            'drawdown_pct': median_drawdown,
            'price': ath_price * (1 - median_drawdown / 100),
            'description': '历史中位数跌幅'
        }

        # 悲观情景：75% 分位跌幅
        pessimistic_drawdown = bear_stats['percentiles']['75%']
        scenarios['pessimistic'] = {
            'drawdown_pct': pessimistic_drawdown,
            'price': ath_price * (1 - pessimistic_drawdown / 100),
            'description': '75% 分位历史跌幅'
        }

        # 极端情景：90% 分位或历史最大
        extreme_drawdown = max(bear_stats['percentiles']['90%'], bear_stats.get('max_drawdown', 85))
        scenarios['extreme'] = {
            'drawdown_pct': extreme_drawdown,
            'price': ath_price * (1 - extreme_drawdown / 100),
            'description': '历史极端跌幅'
        }

    return scenarios


def _plot_drawdown_history(df: pd.DataFrame, bear_markets: List[Dict], output_dir: Path):
    """图1: 历史回撤图"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [2, 1]})

    # 上图：价格 + 熊市标记
    ax1 = axes[0]
    ax1.semilogy(df.index, df['close'], color='black', linewidth=0.8, label='BTC 收盘价')

    # 标记熊市区间
    for i, bear in enumerate(bear_markets):
        ax1.axvspan(bear['peak_date'], bear['trough_date'], alpha=0.2, color='red')
        ax1.scatter([bear['peak_date']], [bear['peak_price']], color='red', s=50, marker='v', zorder=5)
        ax1.scatter([bear['trough_date']], [bear['trough_price']], color='green', s=50, marker='^', zorder=5)
        ax1.annotate(f"{bear['max_drawdown_pct']:.0f}%",
                    xy=(bear['trough_date'], bear['trough_price']),
                    fontsize=8, color='red')

    ax1.set_ylabel('价格 (USDT, 对数尺度)', fontsize=12)
    ax1.set_title('BTC 历史熊市周期', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')

    # 下图：回撤曲线
    ax2 = axes[1]
    rolling_max = df['close'].cummax()
    drawdown = (df['close'] - rolling_max) / rolling_max * 100

    ax2.fill_between(df.index, 0, drawdown, color='red', alpha=0.5)
    ax2.set_ylabel('回撤 (%)', fontsize=12)
    ax2.set_xlabel('日期', fontsize=12)
    ax2.set_title('历史回撤曲线', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 标记重要跌幅线
    for pct in [-20, -50, -80]:
        ax2.axhline(y=pct, color='gray', linestyle='--', alpha=0.5)
        ax2.text(df.index[0], pct, f'{pct}%', fontsize=8, va='bottom')

    plt.tight_layout()
    fig.savefig(output_dir / 'drawdown_history.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [图] 历史回撤: {output_dir / 'drawdown_history.png'}")


def _plot_drawdown_distribution(bear_markets: List[Dict], current_info: Dict, output_dir: Path):
    """图2: 跌幅分布"""
    if len(bear_markets) == 0:
        print("  [跳过] 无熊市数据")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：跌幅直方图
    ax1 = axes[0]
    drawdowns = [abs(b['max_drawdown_pct']) for b in bear_markets]

    ax1.hist(drawdowns, bins=10, color='steelblue', alpha=0.7, edgecolor='white')
    ax1.axvline(x=abs(current_info['current_drawdown_pct']), color='red', linestyle='--',
                linewidth=2, label=f"当前: {abs(current_info['current_drawdown_pct']):.1f}%")
    ax1.axvline(x=np.median(drawdowns), color='green', linestyle='--',
                linewidth=2, label=f"中位数: {np.median(drawdowns):.1f}%")

    ax1.set_xlabel('最大跌幅 (%)', fontsize=12)
    ax1.set_ylabel('频次', fontsize=12)
    ax1.set_title('历史熊市跌幅分布', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # 右图：跌幅 vs 持续时间
    ax2 = axes[1]
    durations = [b['duration_days'] for b in bear_markets]

    scatter = ax2.scatter(durations, drawdowns, s=100, c=range(len(bear_markets)),
                          cmap='viridis', alpha=0.7, edgecolors='black')

    # 标注各个熊市
    for i, bear in enumerate(bear_markets):
        ax2.annotate(f'{bear["peak_date"].year}',
                    xy=(bear['duration_days'], abs(bear['max_drawdown_pct'])),
                    fontsize=8, xytext=(5, 5), textcoords='offset points')

    ax2.set_xlabel('持续时间 (天)', fontsize=12)
    ax2.set_ylabel('最大跌幅 (%)', fontsize=12)
    ax2.set_title('跌幅 vs 持续时间', fontsize=13)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'drawdown_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [图] 跌幅分布: {output_dir / 'drawdown_distribution.png'}")


def _plot_bottom_scenarios(scenarios: Dict, current_info: Dict, output_dir: Path):
    """图3: 底部预测情景"""
    if not scenarios:
        print("  [跳过] 无预测情景")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    scenario_names = list(scenarios.keys())
    prices = [scenarios[s]['price'] for s in scenario_names]
    drawdowns = [scenarios[s]['drawdown_pct'] for s in scenario_names]

    colors = ['green', 'gold', 'orange', 'red']

    y_pos = np.arange(len(scenario_names))
    bars = ax.barh(y_pos, prices, color=colors[:len(scenario_names)], alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{s.capitalize()}\n(-{d:.0f}%)" for s, d in zip(scenario_names, drawdowns)])
    ax.set_xlabel('预估底部价格 (USDT)', fontsize=12)
    ax.set_title(f'底部预测情景 (基于 ATH ${current_info["ath_price"]:,.0f})', fontsize=14)

    # 标注价格
    for bar, price in zip(bars, prices):
        ax.text(price + 1000, bar.get_y() + bar.get_height()/2,
                f'${price:,.0f}', va='center', fontsize=10)

    # 当前价格线
    ax.axvline(x=current_info['current_price'], color='blue', linestyle='--',
               linewidth=2, label=f"当前价格: ${current_info['current_price']:,.0f}")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    fig.savefig(output_dir / 'bottom_scenarios.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [图] 底部预测情景: {output_dir / 'bottom_scenarios.png'}")


def run_drawdown_analysis(df: pd.DataFrame, output_dir: str = "output") -> Dict:
    """历史跌幅分析 - 主入口函数"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  历史跌幅与熊市周期分析")
    print("=" * 60)

    # 识别熊市周期
    print("\n--- 熊市周期识别 ---")
    bear_markets = identify_bear_markets(df, min_drawdown_pct=20)
    print(f"  识别到 {len(bear_markets)} 个熊市周期")

    for i, bear in enumerate(bear_markets, 1):
        print(f"\n  [{i}] {bear['peak_date'].strftime('%Y-%m-%d')} → {bear['trough_date'].strftime('%Y-%m-%d')}")
        print(f"      高点: ${bear['peak_price']:,.0f} → 低点: ${bear['trough_price']:,.0f}")
        print(f"      跌幅: {bear['max_drawdown_pct']:.1f}%, 持续: {bear['duration_days']} 天")

    # 跌幅统计
    print("\n--- 跌幅统计 ---")
    bear_stats = calc_drawdown_stats(bear_markets)

    if bear_stats:
        print(f"  平均跌幅: {bear_stats['avg_drawdown']:.1f}%")
        print(f"  中位数跌幅: {bear_stats['median_drawdown']:.1f}%")
        print(f"  最大跌幅: {bear_stats['max_drawdown']:.1f}%")
        print(f"  平均持续天数: {bear_stats['avg_duration_days']:.0f} 天")
        print(f"\n  跌幅分位数:")
        for pct, val in bear_stats['percentiles'].items():
            print(f"    {pct}: {val:.1f}%")

    # 当前回撤
    print("\n--- 当前回撤 ---")
    current_info = calc_current_drawdown(df)
    print(f"  当前价格: ${current_info['current_price']:,.2f}")
    print(f"  历史高点: ${current_info['ath_price']:,.2f} ({current_info['ath_date'].strftime('%Y-%m-%d')})")
    print(f"  当前回撤: {current_info['current_drawdown_pct']:.1f}%")
    print(f"  距离 ATH: {current_info['days_since_ath']} 天")

    # 底部特征提取
    print("\n--- 底部特征分析 ---")
    bottom_features = []
    for bear in bear_markets[-3:]:  # 分析最近3次熊市
        features = extract_bottom_features(df, bear)
        features['bear_market'] = f"{bear['peak_date'].year}-{bear['trough_date'].year}"
        bottom_features.append(features)

        print(f"\n  {features['bear_market']} 熊市底部特征:")
        if features.get('rsi_at_bottom'):
            print(f"    底部 RSI: {features['rsi_at_bottom']:.1f}")
        if features.get('rsi_below_30_days'):
            print(f"    RSI < 30 持续: {features['rsi_below_30_days']} 天")
        if features.get('volatility_squeeze') is not None:
            print(f"    波动率收缩: {'是' if features['volatility_squeeze'] else '否'}")

    # 底部预测
    print("\n--- 底部预测 ---")
    scenarios = estimate_potential_bottom(bear_stats, current_info)

    for name, scenario in scenarios.items():
        print(f"  {name.capitalize()}: ${scenario['price']:,.0f} ({scenario['description']})")

    # 生成图表
    print("\n--- 生成图表 ---")
    _plot_drawdown_history(df, bear_markets, output_dir)
    _plot_drawdown_distribution(bear_markets, current_info, output_dir)
    _plot_bottom_scenarios(scenarios, current_info, output_dir)

    print("\n" + "=" * 60)
    print("  历史跌幅分析完成")
    print("=" * 60)

    return {
        'bear_markets': [{
            'peak_date': str(b['peak_date'].date()),
            'peak_price': b['peak_price'],
            'trough_date': str(b['trough_date'].date()),
            'trough_price': b['trough_price'],
            'max_drawdown_pct': b['max_drawdown_pct'],
            'duration_days': b['duration_days'],
        } for b in bear_markets],
        'bear_stats': bear_stats,
        'current_info': {
            'current_price': current_info['current_price'],
            'ath_price': current_info['ath_price'],
            'ath_date': str(current_info['ath_date'].date()),
            'current_drawdown_pct': current_info['current_drawdown_pct'],
            'days_since_ath': current_info['days_since_ath'],
        },
        'bottom_scenarios': scenarios,
        'bottom_features': bottom_features,
        'findings': [
            f"历史熊市: {len(bear_markets)} 次",
            f"平均跌幅: {bear_stats.get('avg_drawdown', 0):.1f}%",
            f"中位数跌幅: {bear_stats.get('median_drawdown', 0):.1f}%",
            f"当前回撤: {current_info['current_drawdown_pct']:.1f}%",
            f"乐观底部: ${scenarios.get('optimistic', {}).get('price', 0):,.0f}",
            f"中性底部: ${scenarios.get('neutral', {}).get('price', 0):,.0f}",
            f"悲观底部: ${scenarios.get('pessimistic', {}).get('price', 0):,.0f}",
        ]
    }


if __name__ == '__main__':
    from data_loader import load_daily
    df = load_daily()
    results = run_drawdown_analysis(df, output_dir='../output/drawdown')
