"""斐波那契回撤分析

自动识别重要高点/低点，计算标准回撤位，
统计历史上各回撤位的支撑效果。
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.signal import argrelextrema

from src.font_config import configure_chinese_font
configure_chinese_font()


# 标准斐波那契回撤位
FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
FIB_NAMES = ['0%', '23.6%', '38.2%', '50%', '61.8%', '78.6%', '100%']


def find_swing_points(df: pd.DataFrame, order: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    找出显著的波段高点和低点

    Parameters
    ----------
    df : pd.DataFrame
        K线数据
    order : int
        用于确定局部极值的窗口大小

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (高点DataFrame, 低点DataFrame)
    """
    prices = df['close'].values

    # 找局部极大值（高点）
    high_idx = argrelextrema(prices, np.greater, order=order)[0]

    # 找局部极小值（低点）
    low_idx = argrelextrema(prices, np.less, order=order)[0]

    swing_highs = df.iloc[high_idx][['close']].copy()
    swing_highs['type'] = 'high'

    swing_lows = df.iloc[low_idx][['close']].copy()
    swing_lows['type'] = 'low'

    return swing_highs, swing_lows


def find_major_cycles(df: pd.DataFrame, min_move_pct: float = 50) -> List[Dict]:
    """
    找出主要的牛熊周期（涨幅/跌幅超过阈值的波段）

    Parameters
    ----------
    df : pd.DataFrame
        K线数据
    min_move_pct : float
        最小波动百分比阈值

    Returns
    -------
    List[Dict]
        周期列表，每个包含 start_date, end_date, start_price, end_price, type, change_pct
    """
    swing_highs, swing_lows = find_swing_points(df, order=30)

    # 合并所有波段点
    all_swings = pd.concat([swing_highs, swing_lows]).sort_index()

    cycles = []

    for i in range(len(all_swings) - 1):
        start = all_swings.iloc[i]
        end = all_swings.iloc[i + 1]

        start_price = start['close']
        end_price = end['close']
        change_pct = (end_price - start_price) / start_price * 100

        if abs(change_pct) >= min_move_pct:
            cycle_type = 'bull' if change_pct > 0 else 'bear'
            cycles.append({
                'start_date': all_swings.index[i],
                'end_date': all_swings.index[i + 1],
                'start_price': start_price,
                'end_price': end_price,
                'type': cycle_type,
                'change_pct': change_pct
            })

    return cycles


def calc_fib_retracement(high: float, low: float) -> Dict[str, float]:
    """
    计算斐波那契回撤位价格

    Parameters
    ----------
    high : float
        高点价格
    low : float
        低点价格

    Returns
    -------
    Dict[str, float]
        各回撤位对应的价格
    """
    diff = high - low
    levels = {}

    for level, name in zip(FIB_LEVELS, FIB_NAMES):
        # 从高点回撤
        price = high - diff * level
        levels[name] = price

    return levels


def analyze_fib_support(df: pd.DataFrame, fib_levels: Dict[str, float],
                        start_date: pd.Timestamp, threshold_pct: float = 0.02) -> Dict[str, Dict]:
    """
    分析各斐波那契位的支撑/阻力效果

    Parameters
    ----------
    df : pd.DataFrame
        K线数据
    fib_levels : Dict[str, float]
        斐波那契位价格
    start_date : pd.Timestamp
        分析起始日期（回撤开始日期）
    threshold_pct : float
        触及阈值

    Returns
    -------
    Dict[str, Dict]
        各位的支撑分析结果
    """
    analysis_df = df[df.index >= start_date]

    results = {}

    for name, level_price in fib_levels.items():
        if name in ['0%', '100%']:  # 跳过起点和终点
            continue

        # 找出价格触及该位的时点
        distance_pct = (analysis_df['close'] - level_price) / level_price
        near_level = np.abs(distance_pct) < threshold_pct

        # 统计触及次数和反弹情况
        touches = analysis_df[near_level]

        if len(touches) > 0:
            # 计算首次触及后的反弹
            first_touch_idx = touches.index[0]
            first_touch_loc = df.index.get_loc(first_touch_idx)

            # 后续表现
            returns = {}
            for days in [7, 30, 90]:
                future_loc = first_touch_loc + days
                if future_loc < len(df):
                    future_price = df['close'].iloc[future_loc]
                    returns[f'return_{days}d'] = (future_price / level_price - 1) * 100

            results[name] = {
                'level_price': level_price,
                'touch_count': len(touches),
                'first_touch_date': first_touch_idx,
                'bounced': analysis_df['close'].iloc[-1] > level_price,
                **returns
            }
        else:
            results[name] = {
                'level_price': level_price,
                'touch_count': 0,
                'first_touch_date': None,
                'bounced': None
            }

    return results


def calc_current_retracement(df: pd.DataFrame) -> Dict:
    """
    计算当前从最高点的回撤情况

    Returns
    -------
    Dict
        包含高点、低点、当前价格、回撤比例等信息
    """
    # 找历史最高点
    ath_idx = df['close'].idxmax()
    ath_price = df['close'].max()

    # 找最高点后的最低点
    post_ath = df[df.index > ath_idx]
    if len(post_ath) > 0:
        low_after_ath_idx = post_ath['close'].idxmin()
        low_after_ath = post_ath['close'].min()
    else:
        low_after_ath_idx = ath_idx
        low_after_ath = ath_price

    current_price = df['close'].iloc[-1]

    # 计算回撤比例
    drawdown_from_ath = (ath_price - current_price) / ath_price * 100
    retracement_ratio = (ath_price - current_price) / (ath_price - low_after_ath) if ath_price != low_after_ath else 0

    # 计算斐波那契位
    fib_levels = calc_fib_retracement(ath_price, low_after_ath)

    return {
        'ath_date': ath_idx,
        'ath_price': ath_price,
        'low_after_ath_date': low_after_ath_idx,
        'low_after_ath': low_after_ath,
        'current_price': current_price,
        'drawdown_from_ath_pct': drawdown_from_ath,
        'retracement_ratio': retracement_ratio,
        'fib_levels': fib_levels
    }


def _plot_fib_retracement(df: pd.DataFrame, retracement_info: Dict, output_dir: Path):
    """图1: 当前斐波那契回撤图"""
    fig, ax = plt.subplots(figsize=(16, 10))

    ath_date = retracement_info['ath_date']

    # 只显示从高点前一段时间开始的数据
    lookback = pd.Timedelta(days=365)
    plot_df = df[df.index >= (ath_date - lookback)]

    ax.semilogy(plot_df.index, plot_df['close'], color='black', linewidth=1, label='BTC 收盘价')

    # 绘制斐波那契位
    fib_levels = retracement_info['fib_levels']
    colors = ['green', 'lime', 'yellow', 'gold', 'orange', 'red', 'darkred']

    for (name, price), color in zip(fib_levels.items(), colors):
        ax.axhline(y=price, color=color, linestyle='--', alpha=0.7, linewidth=1.5)
        ax.text(plot_df.index[-1], price, f' {name}: ${price:,.0f}',
                fontsize=9, va='center', color=color)

    # 标记高点和低点
    ax.scatter([ath_date], [retracement_info['ath_price']],
               color='red', s=100, zorder=5, marker='v', label=f"ATH: ${retracement_info['ath_price']:,.0f}")

    if retracement_info['low_after_ath_date'] != ath_date:
        ax.scatter([retracement_info['low_after_ath_date']], [retracement_info['low_after_ath']],
                   color='green', s=100, zorder=5, marker='^',
                   label=f"周期低点: ${retracement_info['low_after_ath']:,.0f}")

    # 当前价格
    current = retracement_info['current_price']
    ax.axhline(y=current, color='blue', linestyle='-', alpha=0.8, linewidth=2)
    ax.text(plot_df.index[0], current, f'当前: ${current:,.0f}', fontsize=10, va='bottom', color='blue')

    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('价格 (USDT, 对数尺度)', fontsize=12)
    ax.set_title(f'BTC 斐波那契回撤分析 (从 ATH ${retracement_info["ath_price"]:,.0f} 回撤 {retracement_info["drawdown_from_ath_pct"]:.1f}%)', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')

    fig.savefig(output_dir / 'fib_current_retracement.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [图] 当前斐波那契回撤: {output_dir / 'fib_current_retracement.png'}")


def _plot_historical_fib(df: pd.DataFrame, cycles: List[Dict], output_dir: Path):
    """图2: 历史周期斐波那契分析"""
    # 找最近的几个主要熊市周期
    bear_cycles = [c for c in cycles if c['type'] == 'bear' and c['change_pct'] < -50]

    if len(bear_cycles) == 0:
        print("  [跳过] 无显著熊市周期")
        return

    # 最多显示4个周期
    bear_cycles = bear_cycles[-4:]

    n_cycles = len(bear_cycles)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, cycle in enumerate(bear_cycles):
        ax = axes[idx]

        # 获取周期数据
        start_date = cycle['start_date']
        end_date = cycle['end_date']
        high_price = cycle['start_price']
        low_price = cycle['end_price']

        # 扩展显示范围
        plot_start = start_date - pd.Timedelta(days=30)
        plot_end = end_date + pd.Timedelta(days=180)
        plot_df = df[(df.index >= plot_start) & (df.index <= plot_end)]

        if len(plot_df) == 0:
            continue

        ax.semilogy(plot_df.index, plot_df['close'], color='black', linewidth=1)

        # 绘制斐波那契位
        fib_levels = calc_fib_retracement(high_price, low_price)
        colors = ['green', 'lime', 'yellow', 'gold', 'orange', 'red', 'darkred']

        for (name, price), color in zip(fib_levels.items(), colors):
            ax.axhline(y=price, color=color, linestyle='--', alpha=0.5, linewidth=1)

        ax.set_title(f'{start_date.strftime("%Y-%m")} → {end_date.strftime("%Y-%m")}: {cycle["change_pct"]:.0f}%', fontsize=11)
        ax.grid(True, alpha=0.3, which='both')

    # 隐藏多余的子图
    for idx in range(n_cycles, 4):
        axes[idx].axis('off')

    plt.suptitle('历史熊市周期斐波那契回撤', fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / 'fib_historical_cycles.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [图] 历史周期斐波那契: {output_dir / 'fib_historical_cycles.png'}")


def _plot_fib_support_stats(support_analysis: Dict, output_dir: Path):
    """图3: 斐波那契位支撑效果统计"""
    if not support_analysis:
        print("  [跳过] 无支撑分析数据")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图: 各位触及次数
    levels = []
    touches = []
    bounces = []

    for name, data in support_analysis.items():
        levels.append(name)
        touches.append(data.get('touch_count', 0))
        bounced = data.get('bounced')
        bounces.append(1 if bounced else 0)

    ax1 = axes[0]
    x = np.arange(len(levels))
    bars = ax1.bar(x, touches, color='steelblue', alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(levels)
    ax1.set_ylabel('触及次数')
    ax1.set_title('各斐波那契位触及次数')
    ax1.grid(True, alpha=0.3, axis='y')

    # 右图: 各位的价格
    ax2 = axes[1]
    prices = [data['level_price'] for data in support_analysis.values()]
    ax2.barh(x, prices, color='green', alpha=0.7)
    ax2.set_yticks(x)
    ax2.set_yticklabels(levels)
    ax2.set_xlabel('价格 (USDT)')
    ax2.set_title('各斐波那契位对应价格')
    ax2.grid(True, alpha=0.3, axis='x')

    for i, price in enumerate(prices):
        ax2.text(price + 1000, i, f'${price:,.0f}', va='center', fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / 'fib_support_stats.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [图] 支撑效果统计: {output_dir / 'fib_support_stats.png'}")


def run_fibonacci_analysis(df: pd.DataFrame, output_dir: str = "output") -> Dict:
    """斐波那契回撤分析 - 主入口函数"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  斐波那契回撤分析")
    print("=" * 60)

    # 计算当前回撤情况
    print("\n--- 当前回撤分析 ---")
    retracement_info = calc_current_retracement(df)

    print(f"  历史最高点: ${retracement_info['ath_price']:,.2f} ({retracement_info['ath_date'].strftime('%Y-%m-%d')})")
    print(f"  周期低点: ${retracement_info['low_after_ath']:,.2f}")
    print(f"  当前价格: ${retracement_info['current_price']:,.2f}")
    print(f"  从 ATH 回撤: {retracement_info['drawdown_from_ath_pct']:.1f}%")

    print("\n  斐波那契回撤位:")
    for name, price in retracement_info['fib_levels'].items():
        status = "<<< 当前" if abs(price - retracement_info['current_price']) / price < 0.03 else ""
        print(f"    {name}: ${price:,.0f} {status}")

    # 找出主要周期
    print("\n--- 历史周期分析 ---")
    cycles = find_major_cycles(df, min_move_pct=50)
    print(f"  识别到 {len(cycles)} 个主要周期 (波动 > 50%)")

    bull_cycles = [c for c in cycles if c['type'] == 'bull']
    bear_cycles = [c for c in cycles if c['type'] == 'bear']
    print(f"  牛市周期: {len(bull_cycles)} 个")
    print(f"  熊市周期: {len(bear_cycles)} 个")

    # 分析支撑效果
    print("\n--- 斐波那契支撑分析 ---")
    support_analysis = analyze_fib_support(
        df,
        retracement_info['fib_levels'],
        retracement_info['ath_date']
    )

    for name, data in support_analysis.items():
        if data['touch_count'] > 0:
            print(f"  {name} (${data['level_price']:,.0f}): 触及 {data['touch_count']} 次")

    # 生成图表
    print("\n--- 生成图表 ---")
    _plot_fib_retracement(df, retracement_info, output_dir)
    _plot_historical_fib(df, cycles, output_dir)
    _plot_fib_support_stats(support_analysis, output_dir)

    print("\n" + "=" * 60)
    print("  斐波那契回撤分析完成")
    print("=" * 60)

    # 计算关键支撑位
    key_supports = []
    for level in ['38.2%', '50%', '61.8%', '78.6%']:
        if level in retracement_info['fib_levels']:
            key_supports.append({
                'level': level,
                'price': retracement_info['fib_levels'][level]
            })

    return {
        'ath_price': retracement_info['ath_price'],
        'ath_date': str(retracement_info['ath_date'].date()),
        'current_price': retracement_info['current_price'],
        'drawdown_from_ath_pct': retracement_info['drawdown_from_ath_pct'],
        'fib_levels': retracement_info['fib_levels'],
        'key_supports': key_supports,
        'cycles': cycles,
        'support_analysis': support_analysis,
        'findings': [
            f"ATH: ${retracement_info['ath_price']:,.0f}",
            f"当前从 ATH 回撤: {retracement_info['drawdown_from_ath_pct']:.1f}%",
            f"38.2% 回撤位: ${retracement_info['fib_levels'].get('38.2%', 0):,.0f}",
            f"61.8% 回撤位: ${retracement_info['fib_levels'].get('61.8%', 0):,.0f}",
            f"78.6% 回撤位: ${retracement_info['fib_levels'].get('78.6%', 0):,.0f}",
        ]
    }


if __name__ == '__main__':
    from data_loader import load_daily
    df = load_daily()
    results = run_fibonacci_analysis(df, output_dir='../output/fibonacci')
