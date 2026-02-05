"""综合底部预测模块

整合所有分析结果，加权计算底部区间估计，
生成概率分布和触底信号清单。
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Any
from scipy import stats as scipy_stats

from src.font_config import configure_chinese_font
configure_chinese_font()


class BottomPredictor:
    """综合底部预测器"""

    # 各模型权重（可调整）
    MODEL_WEIGHTS = {
        'power_law_5pct': 0.20,      # 幂律走廊 5% 下界
        'ma_200w': 0.25,              # 200 周均线
        'fib_618': 0.10,              # 斐波那契 61.8% 回撤
        'fib_786': 0.10,              # 斐波那契 78.6% 回撤
        'historical_median': 0.15,    # 历史中位数跌幅
        'historical_75pct': 0.10,     # 历史 75% 分位跌幅
        'mvrv_implied': 0.10,         # MVRV < 1 隐含价格（如有）
    }

    def __init__(self):
        self.estimates = {}
        self.signals = []
        self.current_price = None
        self.ath_price = None

    def add_estimate(self, name: str, price: float, confidence: float = 1.0):
        """添加一个底部估计"""
        self.estimates[name] = {
            'price': price,
            'confidence': confidence,
        }

    def add_signal(self, signal_name: str, is_triggered: bool, description: str, weight: float = 1.0):
        """添加一个触底信号"""
        self.signals.append({
            'name': signal_name,
            'triggered': is_triggered,
            'description': description,
            'weight': weight,
        })

    def calc_weighted_bottom(self) -> Dict:
        """计算加权底部估计"""
        if len(self.estimates) == 0:
            return {}

        # 收集所有估计
        prices = []
        weights = []

        for name, data in self.estimates.items():
            price = data['price']
            weight = self.MODEL_WEIGHTS.get(name, 0.1) * data['confidence']

            if price > 0 and not np.isnan(price):
                prices.append(price)
                weights.append(weight)

        if len(prices) == 0:
            return {}

        prices = np.array(prices)
        weights = np.array(weights)
        weights = weights / weights.sum()  # 归一化

        # 加权平均
        weighted_avg = np.average(prices, weights=weights)

        # 加权标准差
        weighted_var = np.average((prices - weighted_avg) ** 2, weights=weights)
        weighted_std = np.sqrt(weighted_var)

        # 置信区间
        lower_bound = weighted_avg - 1.5 * weighted_std
        upper_bound = weighted_avg + 1.5 * weighted_std

        return {
            'point_estimate': weighted_avg,
            'std': weighted_std,
            'lower_bound': max(lower_bound, min(prices) * 0.8),
            'upper_bound': min(upper_bound, max(prices) * 1.2),
            'estimates': dict(zip(self.estimates.keys(), prices)),
            'weights': dict(zip(self.estimates.keys(), weights)),
        }

    def calc_signal_score(self) -> Dict:
        """计算触底信号得分"""
        if len(self.signals) == 0:
            return {'score': 0, 'max_score': 0, 'pct': 0}

        total_weight = sum(s['weight'] for s in self.signals)
        triggered_weight = sum(s['weight'] for s in self.signals if s['triggered'])

        return {
            'score': triggered_weight,
            'max_score': total_weight,
            'pct': triggered_weight / total_weight * 100 if total_weight > 0 else 0,
            'triggered_signals': [s for s in self.signals if s['triggered']],
            'pending_signals': [s for s in self.signals if not s['triggered']],
        }


def collect_analysis_results(all_results: Dict) -> Dict:
    """从各模块分析结果中收集底部预测相关数据"""
    collected = {
        'power_law': {},
        'ma_support': {},
        'fibonacci': {},
        'drawdown': {},
        'clustering': {},
        'indicators': {},
        'volatility': {},
    }

    # 幂律分析
    if 'power_law' in all_results:
        pl = all_results['power_law']
        collected['power_law'] = {
            'corridor_5pct': pl.get('corridor_prices', {}).get(0.05),
            'corridor_50pct': pl.get('corridor_prices', {}).get(0.5),
            'current_percentile': pl.get('current_percentile'),
        }

    # 200周均线
    if 'ma_support' in all_results:
        ma = all_results['ma_support']
        collected['ma_support'] = {
            'ma_200w': ma.get('current_ma_200w'),
            'current_deviation': ma.get('current_deviation_pct'),
        }

    # 斐波那契
    if 'fibonacci' in all_results:
        fib = all_results['fibonacci']
        levels = fib.get('fib_levels', {})
        collected['fibonacci'] = {
            'fib_382': levels.get('38.2%'),
            'fib_500': levels.get('50%'),
            'fib_618': levels.get('61.8%'),
            'fib_786': levels.get('78.6%'),
            'current_drawdown': fib.get('drawdown_from_ath_pct'),
        }

    # 历史跌幅
    if 'drawdown' in all_results:
        dd = all_results['drawdown']
        scenarios = dd.get('bottom_scenarios', {})
        bear_stats = dd.get('bear_stats', {})
        collected['drawdown'] = {
            'optimistic': scenarios.get('optimistic', {}).get('price'),
            'neutral': scenarios.get('neutral', {}).get('price'),
            'pessimistic': scenarios.get('pessimistic', {}).get('price'),
            'extreme': scenarios.get('extreme', {}).get('price'),
            'avg_drawdown': bear_stats.get('avg_drawdown'),
            'median_drawdown': bear_stats.get('median_drawdown'),
        }

    # 市场聚类
    if 'clustering' in all_results:
        cl = all_results['clustering']
        collected['clustering'] = {
            'current_state': cl.get('current_state'),
            'state_probabilities': cl.get('state_probabilities'),
        }

    # 技术指标
    if 'indicators' in all_results:
        ind = all_results['indicators']
        # 提取 RSI 相关信息
        collected['indicators'] = {
            'train_results': ind.get('train_results'),
        }

    return collected


def generate_bottom_signals(df: pd.DataFrame, collected: Dict) -> List[Dict]:
    """生成触底信号清单"""
    signals = []

    # 计算当前 RSI
    close = df['close']
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    current_rsi = rsi.iloc[-1]

    # 1. RSI 超卖信号
    rsi_oversold = current_rsi < 30
    signals.append({
        'name': 'RSI 超卖',
        'triggered': rsi_oversold,
        'description': f'RSI = {current_rsi:.1f} {"< 30" if rsi_oversold else ">= 30"}',
        'weight': 1.5,
    })

    # 2. RSI 持续超卖（周线 RSI < 30 持续 2 周）
    weekly_close = close.resample('W').last()
    if len(weekly_close) >= 3:
        weekly_delta = weekly_close.diff()
        weekly_gain = weekly_delta.clip(lower=0)
        weekly_loss = (-weekly_delta).clip(lower=0)
        weekly_avg_gain = weekly_gain.ewm(alpha=1/14, min_periods=2, adjust=False).mean()
        weekly_avg_loss = weekly_loss.ewm(alpha=1/14, min_periods=2, adjust=False).mean()
        weekly_rs = weekly_avg_gain / weekly_avg_loss.replace(0, np.nan)
        weekly_rsi = 100 - 100 / (1 + weekly_rs)

        rsi_below_30_weeks = (weekly_rsi.tail(4) < 30).sum()
        persistent_oversold = rsi_below_30_weeks >= 2
        signals.append({
            'name': 'RSI 持续超卖',
            'triggered': persistent_oversold,
            'description': f'周线 RSI < 30 持续 {rsi_below_30_weeks} 周',
            'weight': 2.0,
        })

    # 3. 波动率收缩
    returns = close.pct_change()
    recent_vol = returns.tail(30).std() * np.sqrt(252)
    historical_vol = returns.tail(365).std() * np.sqrt(252)
    vol_squeeze = recent_vol < historical_vol * 0.6
    signals.append({
        'name': '波动率收缩',
        'triggered': vol_squeeze,
        'description': f'近期波动率 {recent_vol*100:.1f}% vs 历史 {historical_vol*100:.1f}%',
        'weight': 1.5,
    })

    # 4. 成交量枯竭
    if 'volume' in df.columns:
        recent_vol_avg = df['volume'].tail(30).mean()
        historical_vol_avg = df['volume'].tail(365).mean()
        vol_exhaustion = recent_vol_avg < historical_vol_avg * 0.5
        signals.append({
            'name': '成交量枯竭',
            'triggered': vol_exhaustion,
            'description': f'近期成交量 / 历史均值 = {recent_vol_avg/historical_vol_avg:.1%}',
            'weight': 1.0,
        })

    # 5. 价格接近 200 周均线
    if collected.get('ma_support', {}).get('ma_200w'):
        ma_200w = collected['ma_support']['ma_200w']
        current_price = close.iloc[-1]
        near_ma = abs(current_price - ma_200w) / ma_200w < 0.1
        signals.append({
            'name': '接近 200 周均线',
            'triggered': near_ma,
            'description': f'当前 ${current_price:,.0f} vs MA ${ma_200w:,.0f}',
            'weight': 2.0,
        })

    # 6. 幂律走廊低位
    if collected.get('power_law', {}).get('current_percentile'):
        pct = collected['power_law']['current_percentile']
        low_percentile = pct < 20
        signals.append({
            'name': '幂律走廊低位',
            'triggered': low_percentile,
            'description': f'当前位于 {pct:.1f}% 分位',
            'weight': 1.5,
        })

    # 7. 市场聚类状态
    if collected.get('clustering', {}).get('current_state'):
        state = collected['clustering']['current_state']
        # 假设聚类状态中有"底部积累"或类似状态
        bottom_state = 'accumulation' in str(state).lower() or 'bottom' in str(state).lower()
        signals.append({
            'name': '市场状态',
            'triggered': bottom_state,
            'description': f'当前状态: {state}',
            'weight': 1.0,
        })

    return signals


def _plot_bottom_prediction(predictor: BottomPredictor, df: pd.DataFrame, output_dir: Path):
    """图1: 底部预测综合图"""
    result = predictor.calc_weighted_bottom()

    if not result:
        print("  [跳过] 无足够数据绘制底部预测图")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 左上：价格与预测区间
    ax1 = axes[0, 0]

    # 只显示最近一年的数据
    recent_df = df.tail(365)
    ax1.semilogy(recent_df.index, recent_df['close'], color='black', linewidth=1, label='BTC 收盘价')

    # 预测区间
    point = result['point_estimate']
    lower = result['lower_bound']
    upper = result['upper_bound']

    ax1.axhline(y=point, color='green', linestyle='-', linewidth=2, label=f'预测底部: ${point:,.0f}')
    ax1.axhspan(lower, upper, alpha=0.2, color='green', label=f'置信区间: ${lower:,.0f} - ${upper:,.0f}')

    # 各模型估计点
    colors = plt.cm.Set2(np.linspace(0, 1, len(predictor.estimates)))
    for (name, data), color in zip(predictor.estimates.items(), colors):
        price = data['price']
        if price > 0 and not np.isnan(price):
            ax1.axhline(y=price, color=color, linestyle='--', alpha=0.5, linewidth=1)

    ax1.set_ylabel('价格 (USDT, 对数尺度)', fontsize=11)
    ax1.set_title('底部预测综合图', fontsize=13)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3, which='both')

    # 右上：各模型估计对比
    ax2 = axes[0, 1]

    names = list(predictor.estimates.keys())
    prices = [predictor.estimates[n]['price'] for n in names]
    weights = [result['weights'].get(n, 0) * 100 for n in names]

    y_pos = np.arange(len(names))
    bars = ax2.barh(y_pos, prices, color='steelblue', alpha=0.7)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"{n}\n({w:.0f}%)" for n, w in zip(names, weights)], fontsize=9)
    ax2.set_xlabel('预测价格 (USDT)', fontsize=11)
    ax2.set_title('各模型底部估计', fontsize=13)
    ax2.axvline(x=point, color='green', linestyle='--', linewidth=2, label='加权平均')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')

    for bar, price in zip(bars, prices):
        ax2.text(price + 500, bar.get_y() + bar.get_height()/2,
                f'${price:,.0f}', va='center', fontsize=9)

    # 左下：触底信号清单
    ax3 = axes[1, 0]
    ax3.axis('off')

    signal_result = predictor.calc_signal_score()

    text = f"触底信号得分: {signal_result['score']:.1f} / {signal_result['max_score']:.1f} ({signal_result['pct']:.0f}%)\n\n"
    text += "已触发信号:\n"
    for s in signal_result['triggered_signals']:
        text += f"  [x] {s['name']}: {s['description']}\n"

    text += "\n待触发信号:\n"
    for s in signal_result['pending_signals']:
        text += f"  [ ] {s['name']}: {s['description']}\n"

    ax3.text(0.05, 0.95, text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    ax3.set_title('触底信号清单', fontsize=13)

    # 右下：概率分布
    ax4 = axes[1, 1]

    # 基于各模型估计生成概率分布
    prices_arr = np.array([p for p in predictor.estimates.values() if p['price'] > 0])
    if len(prices_arr) > 0:
        prices_only = np.array([p['price'] for p in prices_arr])

        # 使用核密度估计
        x_range = np.linspace(prices_only.min() * 0.8, prices_only.max() * 1.2, 100)

        # 简单的高斯混合
        pdf = np.zeros_like(x_range)
        for price in prices_only:
            pdf += scipy_stats.norm.pdf(x_range, loc=price, scale=price * 0.1)
        pdf = pdf / pdf.max()

        ax4.fill_between(x_range, 0, pdf, alpha=0.5, color='steelblue')
        ax4.axvline(x=point, color='green', linestyle='--', linewidth=2, label=f'加权估计: ${point:,.0f}')
        ax4.axvline(x=predictor.current_price, color='red', linestyle='-', linewidth=2,
                   label=f'当前价格: ${predictor.current_price:,.0f}' if predictor.current_price else '')

        ax4.set_xlabel('价格 (USDT)', fontsize=11)
        ax4.set_ylabel('相对概率', fontsize=11)
        ax4.set_title('底部价格概率分布', fontsize=13)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'bottom_prediction_summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [图] 底部预测综合图: {output_dir / 'bottom_prediction_summary.png'}")


def run_bottom_prediction(df: pd.DataFrame, all_results: Dict, output_dir: str = "output") -> Dict:
    """综合底部预测 - 主入口函数

    Parameters
    ----------
    df : pd.DataFrame
        日线数据
    all_results : Dict
        所有分析模块的结果
    output_dir : str
        输出目录

    Returns
    -------
    Dict
        底部预测结果
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  综合底部预测")
    print("=" * 60)

    # 收集各模块结果
    print("\n--- 收集分析结果 ---")
    collected = collect_analysis_results(all_results)

    # 创建预测器
    predictor = BottomPredictor()
    predictor.current_price = df['close'].iloc[-1]
    predictor.ath_price = df['close'].max()

    print(f"  当前价格: ${predictor.current_price:,.2f}")
    print(f"  历史最高: ${predictor.ath_price:,.2f}")

    # 添加各模型估计
    print("\n--- 添加模型估计 ---")

    # 幂律 5% 走廊
    if collected['power_law'].get('corridor_5pct'):
        price = collected['power_law']['corridor_5pct']
        predictor.add_estimate('power_law_5pct', price, confidence=0.9)
        print(f"  幂律 5% 走廊: ${price:,.0f}")

    # 200 周均线
    if collected['ma_support'].get('ma_200w'):
        price = collected['ma_support']['ma_200w']
        predictor.add_estimate('ma_200w', price, confidence=0.95)
        print(f"  200 周均线: ${price:,.0f}")

    # 斐波那契
    if collected['fibonacci'].get('fib_618'):
        price = collected['fibonacci']['fib_618']
        predictor.add_estimate('fib_618', price, confidence=0.8)
        print(f"  斐波那契 61.8%: ${price:,.0f}")

    if collected['fibonacci'].get('fib_786'):
        price = collected['fibonacci']['fib_786']
        predictor.add_estimate('fib_786', price, confidence=0.7)
        print(f"  斐波那契 78.6%: ${price:,.0f}")

    # 历史跌幅
    if collected['drawdown'].get('neutral'):
        price = collected['drawdown']['neutral']
        predictor.add_estimate('historical_median', price, confidence=0.85)
        print(f"  历史中位数跌幅: ${price:,.0f}")

    if collected['drawdown'].get('pessimistic'):
        price = collected['drawdown']['pessimistic']
        predictor.add_estimate('historical_75pct', price, confidence=0.75)
        print(f"  历史 75% 分位跌幅: ${price:,.0f}")

    # 生成触底信号
    print("\n--- 生成触底信号 ---")
    signals = generate_bottom_signals(df, collected)
    for signal in signals:
        predictor.add_signal(
            signal['name'],
            signal['triggered'],
            signal['description'],
            signal['weight']
        )
        status = "[x]" if signal['triggered'] else "[ ]"
        print(f"  {status} {signal['name']}: {signal['description']}")

    # 计算综合预测
    print("\n--- 综合预测结果 ---")
    bottom_result = predictor.calc_weighted_bottom()
    signal_result = predictor.calc_signal_score()

    if bottom_result:
        print(f"\n  加权底部估计: ${bottom_result['point_estimate']:,.0f}")
        print(f"  置信区间: ${bottom_result['lower_bound']:,.0f} - ${bottom_result['upper_bound']:,.0f}")
        print(f"  标准差: ${bottom_result['std']:,.0f}")

        # 计算潜在下跌空间
        downside = (predictor.current_price - bottom_result['point_estimate']) / predictor.current_price * 100
        print(f"\n  当前价格: ${predictor.current_price:,.0f}")
        print(f"  潜在下跌空间: {downside:.1f}%")

    print(f"\n  触底信号得分: {signal_result['score']:.1f} / {signal_result['max_score']:.1f} ({signal_result['pct']:.0f}%)")

    # 生成图表
    print("\n--- 生成图表 ---")
    _plot_bottom_prediction(predictor, df, output_dir)

    print("\n" + "=" * 60)
    print("  综合底部预测完成")
    print("=" * 60)

    # 返回结果
    result = {
        'current_price': predictor.current_price,
        'ath_price': predictor.ath_price,
        'estimates': {k: v['price'] for k, v in predictor.estimates.items()},
        'bottom_prediction': bottom_result,
        'signal_score': signal_result,
        'collected_data': collected,
        'findings': [
            f"当前价格: ${predictor.current_price:,.0f}",
            f"加权底部估计: ${bottom_result.get('point_estimate', 0):,.0f}" if bottom_result else "数据不足",
            f"置信区间: ${bottom_result.get('lower_bound', 0):,.0f} - ${bottom_result.get('upper_bound', 0):,.0f}" if bottom_result else "",
            f"潜在下跌空间: {downside:.1f}%" if bottom_result else "",
            f"触底信号: {signal_result['pct']:.0f}% 已触发",
        ]
    }

    return result


if __name__ == '__main__':
    # 测试用
    from data_loader import load_daily
    df = load_daily()

    # 模拟一些分析结果
    mock_results = {
        'power_law': {
            'corridor_prices': {0.05: 45000, 0.5: 70000, 0.95: 120000},
            'current_percentile': 35,
        },
        'ma_support': {
            'current_ma_200w': 55000,
            'current_deviation_pct': 10,
        },
        'fibonacci': {
            'fib_levels': {'38.2%': 65000, '50%': 58000, '61.8%': 50000, '78.6%': 40000},
            'drawdown_from_ath_pct': 30,
        },
        'drawdown': {
            'bottom_scenarios': {
                'optimistic': {'price': 55000},
                'neutral': {'price': 45000},
                'pessimistic': {'price': 35000},
            },
            'bear_stats': {
                'avg_drawdown': 65,
                'median_drawdown': 70,
            },
        },
    }

    results = run_bottom_prediction(df, mock_results, output_dir='../output/bottom_prediction')
