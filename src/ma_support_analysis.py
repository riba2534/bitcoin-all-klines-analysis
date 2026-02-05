"""200周均线支撑分析

分析比特币价格与 200 周均线（约 1400 天 MA）的关系，
统计历史上价格触及 200 周 MA 的次数和后续表现。
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats

from src.font_config import configure_chinese_font
configure_chinese_font()


def calc_200_week_ma(df: pd.DataFrame) -> pd.Series:
    """计算 200 周均线（1400 天 MA）"""
    return df['close'].rolling(window=1400, min_periods=200).mean()


def calc_weekly_ma(df: pd.DataFrame, weeks: int = 200) -> pd.Series:
    """从周线数据计算 N 周均线"""
    # 将日线重采样为周线
    weekly = df['close'].resample('W').last()
    ma = weekly.rolling(window=weeks, min_periods=50).mean()
    # 向前填充到日线
    return ma.reindex(df.index, method='ffill')


def find_ma_touches(df: pd.DataFrame, ma: pd.Series, threshold_pct: float = 0.03) -> pd.DataFrame:
    """
    找出价格触及均线的时点

    Parameters
    ----------
    df : pd.DataFrame
        日线数据
    ma : pd.Series
        均线序列
    threshold_pct : float
        距离阈值百分比（默认 3%）

    Returns
    -------
    pd.DataFrame
        触及事件详情
    """
    # 计算价格与均线的距离百分比
    distance_pct = (df['close'] - ma) / ma * 100

    # 找出价格在均线附近的点（距离 < threshold）
    near_ma = np.abs(distance_pct) < threshold_pct * 100

    # 找出从上方触及（下跌到均线）
    touch_from_above = (distance_pct.shift(1) > threshold_pct * 100) & near_ma

    # 找出从下方触及（反弹到均线）
    touch_from_below = (distance_pct.shift(1) < -threshold_pct * 100) & near_ma

    touches = []

    for date in df.index[touch_from_above | touch_from_below]:
        idx = df.index.get_loc(date)
        direction = 'from_above' if touch_from_above.iloc[idx] else 'from_below'

        # 计算后续收益
        returns = {}
        for days in [7, 30, 90, 180, 365]:
            future_idx = idx + days
            if future_idx < len(df):
                future_price = df['close'].iloc[future_idx]
                returns[f'return_{days}d'] = (future_price / df['close'].iloc[idx] - 1) * 100
            else:
                returns[f'return_{days}d'] = np.nan

        touches.append({
            'date': date,
            'price': df['close'].iloc[idx],
            'ma_value': ma.iloc[idx],
            'distance_pct': distance_pct.iloc[idx],
            'direction': direction,
            **returns
        })

    return pd.DataFrame(touches)


def analyze_ma_support(touches_df: pd.DataFrame) -> Dict:
    """分析均线支撑效果"""
    if len(touches_df) == 0:
        return {}

    results = {
        'total_touches': len(touches_df),
        'from_above': len(touches_df[touches_df['direction'] == 'from_above']),
        'from_below': len(touches_df[touches_df['direction'] == 'from_below']),
    }

    # 统计各时间段收益
    for days in [7, 30, 90, 180, 365]:
        col = f'return_{days}d'
        if col in touches_df.columns:
            valid = touches_df[col].dropna()
            if len(valid) > 0:
                results[f'avg_return_{days}d'] = valid.mean()
                results[f'median_return_{days}d'] = valid.median()
                results[f'win_rate_{days}d'] = (valid > 0).mean() * 100
                results[f'count_{days}d'] = len(valid)

    return results


def backtest_ma_strategy(df: pd.DataFrame, ma: pd.Series, threshold_pct: float = 0.05) -> Dict:
    """
    回测策略：在价格触及 200 周均线附近买入

    策略规则：
    - 当价格从上方首次跌破均线 threshold% 范围内时买入
    - 持有 N 天后卖出
    """
    distance_pct = (df['close'] - ma) / ma

    # 进入买入区间（价格在均线 ±threshold% 以内）
    in_buy_zone = np.abs(distance_pct) < threshold_pct

    # 只在首次进入买入区间时买入（避免重复）
    zone_entry = in_buy_zone & (~in_buy_zone.shift(1).fillna(False))

    trades = []
    holding_periods = [30, 90, 180, 365]

    for date in df.index[zone_entry]:
        idx = df.index.get_loc(date)
        entry_price = df['close'].iloc[idx]

        for hold_days in holding_periods:
            exit_idx = idx + hold_days
            if exit_idx < len(df):
                exit_price = df['close'].iloc[exit_idx]
                ret = (exit_price / entry_price - 1) * 100
                trades.append({
                    'entry_date': date,
                    'entry_price': entry_price,
                    'hold_days': hold_days,
                    'exit_price': exit_price,
                    'return_pct': ret
                })

    trades_df = pd.DataFrame(trades)

    # 按持有期汇总统计
    results = {}
    for hold_days in holding_periods:
        subset = trades_df[trades_df['hold_days'] == hold_days]
        if len(subset) > 0:
            results[f'hold_{hold_days}d'] = {
                'num_trades': len(subset),
                'avg_return': subset['return_pct'].mean(),
                'median_return': subset['return_pct'].median(),
                'win_rate': (subset['return_pct'] > 0).mean() * 100,
                'max_return': subset['return_pct'].max(),
                'max_loss': subset['return_pct'].min(),
            }

    return results


def _plot_ma_support(df: pd.DataFrame, ma: pd.Series, touches_df: pd.DataFrame, output_dir: Path):
    """图1: 价格与 200 周均线"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [2, 1]})

    # 上图：价格与均线
    ax1 = axes[0]
    ax1.semilogy(df.index, df['close'], color='black', linewidth=0.8, label='BTC 收盘价')
    ax1.semilogy(ma.index, ma, color='blue', linewidth=2, label='200 周均线 (1400d MA)')

    # 标记触及点
    if len(touches_df) > 0:
        touch_dates = touches_df['date']
        touch_prices = touches_df['price']
        ax1.scatter(touch_dates, touch_prices, color='red', s=50, zorder=5, label='触及均线', alpha=0.7)

    # 当前值标注
    current_price = df['close'].iloc[-1]
    current_ma = ma.iloc[-1]
    if not np.isnan(current_ma):
        ax1.axhline(y=current_ma, color='blue', linestyle='--', alpha=0.5)
        ax1.text(df.index[-1], current_ma, f'MA: ${current_ma:,.0f}', fontsize=10, va='bottom')

    ax1.set_ylabel('价格 (USDT, 对数尺度)', fontsize=12)
    ax1.set_title('BTC 价格与 200 周均线支撑分析', fontsize=14)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3, which='both')

    # 下图：价格相对均线的偏离度
    ax2 = axes[1]
    deviation = (df['close'] - ma) / ma * 100
    ax2.fill_between(df.index, 0, deviation, where=deviation >= 0, color='green', alpha=0.3, label='高于均线')
    ax2.fill_between(df.index, 0, deviation, where=deviation < 0, color='red', alpha=0.3, label='低于均线')
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='偏离 +50%')
    ax2.axhline(y=-20, color='purple', linestyle='--', alpha=0.5, label='偏离 -20%')

    ax2.set_xlabel('日期', fontsize=12)
    ax2.set_ylabel('偏离度 (%)', fontsize=12)
    ax2.set_title('价格相对 200 周均线偏离度', fontsize=12)
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'ma_support_price_ma.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [图] 价格与均线: {output_dir / 'ma_support_price_ma.png'}")


def _plot_touch_returns(touches_df: pd.DataFrame, output_dir: Path):
    """图2: 触及均线后的收益分布"""
    if len(touches_df) == 0:
        print("  [跳过] 无触及事件，无法绘制收益分布图")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    periods = [7, 30, 90, 180, 365]
    period_names = ['7天', '30天', '90天', '180天', '365天']

    for idx, (days, name) in enumerate(zip(periods, period_names)):
        ax = axes[idx // 3, idx % 3]
        col = f'return_{days}d'

        if col in touches_df.columns:
            data = touches_df[col].dropna()
            if len(data) > 0:
                colors = ['green' if x > 0 else 'red' for x in data]
                ax.bar(range(len(data)), data, color=colors, alpha=0.7)
                ax.axhline(y=0, color='black', linewidth=1)
                ax.axhline(y=data.mean(), color='blue', linestyle='--', label=f'均值: {data.mean():.1f}%')
                ax.set_title(f'触及后 {name} 收益', fontsize=11)
                ax.set_xlabel('触及事件编号')
                ax.set_ylabel('收益率 (%)')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

    # 最后一个子图显示汇总统计
    ax = axes[1, 2]
    ax.axis('off')

    stats_text = "触及均线后收益汇总:\n\n"
    for days, name in zip(periods, period_names):
        col = f'return_{days}d'
        if col in touches_df.columns:
            data = touches_df[col].dropna()
            if len(data) > 0:
                stats_text += f"{name}: 均值={data.mean():.1f}%, 胜率={((data > 0).mean() * 100):.0f}%\n"

    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    fig.savefig(output_dir / 'ma_support_touch_returns.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [图] 触及收益分布: {output_dir / 'ma_support_touch_returns.png'}")


def _plot_backtest_results(backtest: Dict, output_dir: Path):
    """图3: 回测结果"""
    if not backtest:
        print("  [跳过] 无回测数据")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图: 各持有期收益
    hold_periods = []
    avg_returns = []
    win_rates = []

    for key, stats in backtest.items():
        days = int(key.split('_')[1].replace('d', ''))
        hold_periods.append(days)
        avg_returns.append(stats['avg_return'])
        win_rates.append(stats['win_rate'])

    x = np.arange(len(hold_periods))
    width = 0.35

    ax1 = axes[0]
    bars = ax1.bar(x, avg_returns, width, color=['green' if r > 0 else 'red' for r in avg_returns], alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{d}天' for d in hold_periods])
    ax1.set_ylabel('平均收益率 (%)')
    ax1.set_title('200周均线策略 - 各持有期平均收益')
    ax1.axhline(y=0, color='black', linewidth=1)
    ax1.grid(True, alpha=0.3, axis='y')

    for bar, ret in zip(bars, avg_returns):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{ret:.1f}%', ha='center', fontsize=10)

    # 右图: 胜率
    ax2 = axes[1]
    bars2 = ax2.bar(x, win_rates, width, color='steelblue', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{d}天' for d in hold_periods])
    ax2.set_ylabel('胜率 (%)')
    ax2.set_title('200周均线策略 - 各持有期胜率')
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50% 基准')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, wr in zip(bars2, win_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{wr:.0f}%', ha='center', fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / 'ma_support_backtest.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [图] 回测结果: {output_dir / 'ma_support_backtest.png'}")


def run_ma_support_analysis(df: pd.DataFrame, output_dir: str = "output") -> Dict:
    """200周均线支撑分析 - 主入口函数"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  200 周均线支撑分析")
    print("=" * 60)

    # 计算 200 周均线
    print("\n--- 计算均线 ---")
    ma_1400d = calc_200_week_ma(df)
    ma_weekly = calc_weekly_ma(df, weeks=200)

    # 使用 1400 日 MA 作为主要分析对象
    ma = ma_1400d

    current_price = df['close'].iloc[-1]
    current_ma = ma.iloc[-1]

    print(f"  当前价格: ${current_price:,.2f}")
    print(f"  当前 200 周 MA: ${current_ma:,.2f}" if not np.isnan(current_ma) else "  当前 200 周 MA: 数据不足")

    if not np.isnan(current_ma):
        deviation = (current_price - current_ma) / current_ma * 100
        print(f"  当前偏离度: {deviation:+.1f}%")

        if deviation > 0:
            print(f"  >> 当前价格高于 200 周 MA {deviation:.1f}%")
        else:
            print(f"  >> 当前价格低于 200 周 MA {abs(deviation):.1f}%")

    # 找出触及事件
    print("\n--- 触及事件分析 ---")
    touches_df = find_ma_touches(df, ma, threshold_pct=0.05)
    print(f"  历史触及次数: {len(touches_df)}")

    if len(touches_df) > 0:
        support_stats = analyze_ma_support(touches_df)
        print(f"  从上方触及: {support_stats.get('from_above', 0)} 次")
        print(f"  从下方触及: {support_stats.get('from_below', 0)} 次")

        # 打印各时间段收益
        print("\n  触及后收益统计:")
        for days in [30, 90, 180, 365]:
            avg_key = f'avg_return_{days}d'
            wr_key = f'win_rate_{days}d'
            if avg_key in support_stats:
                print(f"    {days:>3}天: 平均收益 {support_stats[avg_key]:+.1f}%, 胜率 {support_stats.get(wr_key, 0):.0f}%")
    else:
        support_stats = {}
        print("  未找到触及事件")

    # 回测
    print("\n--- 策略回测 ---")
    backtest = backtest_ma_strategy(df, ma, threshold_pct=0.05)

    if backtest:
        for key, stats in backtest.items():
            days = key.split('_')[1]
            print(f"  持有 {days}: 平均收益 {stats['avg_return']:.1f}%, 胜率 {stats['win_rate']:.0f}%, 交易次数 {stats['num_trades']}")

    # 生成图表
    print("\n--- 生成图表 ---")
    _plot_ma_support(df, ma, touches_df, output_dir)
    _plot_touch_returns(touches_df, output_dir)
    _plot_backtest_results(backtest, output_dir)

    print("\n" + "=" * 60)
    print("  200 周均线支撑分析完成")
    print("=" * 60)

    return {
        'current_price': current_price,
        'current_ma_200w': current_ma if not np.isnan(current_ma) else None,
        'current_deviation_pct': deviation if not np.isnan(current_ma) else None,
        'touches': touches_df.to_dict('records') if len(touches_df) > 0 else [],
        'support_stats': support_stats,
        'backtest_results': backtest,
        'findings': [
            f"200周均线当前值: ${current_ma:,.0f}" if not np.isnan(current_ma) else "200周均线数据不足",
            f"当前价格偏离均线: {deviation:+.1f}%" if not np.isnan(current_ma) else "",
            f"历史触及均线后平均收益: 30天 {support_stats.get('avg_return_30d', 0):.1f}%, 90天 {support_stats.get('avg_return_90d', 0):.1f}%" if support_stats else "无触及数据",
        ]
    }


if __name__ == '__main__':
    from data_loader import load_daily
    df = load_daily()
    results = run_ma_support_analysis(df, output_dir='../output/ma_support')
