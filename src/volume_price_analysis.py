"""成交量-价格关系与OBV分析

分析BTC成交量与价格变动的关系，包括Spearman相关性、
Taker买入比例领先分析、Granger因果检验和OBV背离检测。
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
from pathlib import Path
from typing import Dict, List, Tuple

# 中文显示支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
#  核心分析函数
# =============================================================================

def _spearman_volume_returns(volume: pd.Series, returns: pd.Series) -> Dict:
    """Spearman秩相关: 成交量 vs |收益率|

    使用Spearman而非Pearson，因为量价关系通常是非线性的。

    Returns
    -------
    dict
        包含 correlation, p_value, n_samples
    """
    # 对齐索引并去除NaN
    abs_ret = returns.abs()
    aligned = pd.concat([volume, abs_ret], axis=1, keys=['volume', 'abs_return']).dropna()

    corr, p_val = stats.spearmanr(aligned['volume'], aligned['abs_return'])

    return {
        'correlation': corr,
        'p_value': p_val,
        'n_samples': len(aligned),
    }


def _taker_buy_ratio_lead_lag(
    taker_buy_ratio: pd.Series,
    returns: pd.Series,
    max_lag: int = 20,
) -> pd.DataFrame:
    """Taker买入比例领先-滞后分析

    计算 taker_buy_ratio(t) 与 returns(t+lag) 的互相关，
    检验买入比例对未来收益的预测能力。

    Parameters
    ----------
    taker_buy_ratio : pd.Series
        Taker买入占比序列
    returns : pd.Series
        对数收益率序列
    max_lag : int
        最大领先天数

    Returns
    -------
    pd.DataFrame
        包含 lag, correlation, p_value, significant 列
    """
    results = []
    for lag in range(1, max_lag + 1):
        # taker_buy_ratio(t) vs returns(t+lag)
        ratio_shifted = taker_buy_ratio.shift(lag)
        aligned = pd.concat([ratio_shifted, returns], axis=1).dropna()
        aligned.columns = ['ratio', 'return']

        if len(aligned) < 30:
            continue

        corr, p_val = stats.spearmanr(aligned['ratio'], aligned['return'])
        results.append({
            'lag': lag,
            'correlation': corr,
            'p_value': p_val,
            'significant': p_val < 0.05,
        })

    return pd.DataFrame(results)


def _granger_causality(
    volume: pd.Series,
    returns: pd.Series,
    max_lag: int = 10,
) -> Dict[str, pd.DataFrame]:
    """双向Granger因果检验: 成交量 ↔ 收益率

    Parameters
    ----------
    volume : pd.Series
        成交量序列
    returns : pd.Series
        收益率序列
    max_lag : int
        最大滞后阶数

    Returns
    -------
    dict
        'volume_to_returns': 成交量→收益率 的p值表
        'returns_to_volume': 收益率→成交量 的p值表
    """
    # 对齐并去除NaN
    aligned = pd.concat([volume, returns], axis=1, keys=['volume', 'returns']).dropna()

    results = {}

    # 方向1: 成交量 → 收益率 (检验成交量是否Granger-cause收益率)
    # grangercausalitytests 的数据格式: [被预测变量, 预测变量]
    try:
        data_v2r = aligned[['returns', 'volume']].values
        gc_v2r = grangercausalitytests(data_v2r, maxlag=max_lag, verbose=False)
        rows_v2r = []
        for lag_order in range(1, max_lag + 1):
            test_results = gc_v2r[lag_order][0]
            rows_v2r.append({
                'lag': lag_order,
                'ssr_ftest_pval': test_results['ssr_ftest'][1],
                'ssr_chi2test_pval': test_results['ssr_chi2test'][1],
                'lrtest_pval': test_results['lrtest'][1],
                'params_ftest_pval': test_results['params_ftest'][1],
            })
        results['volume_to_returns'] = pd.DataFrame(rows_v2r)
    except Exception as e:
        print(f"  [警告] 成交量→收益率 Granger检验失败: {e}")
        results['volume_to_returns'] = pd.DataFrame()

    # 方向2: 收益率 → 成交量
    try:
        data_r2v = aligned[['volume', 'returns']].values
        gc_r2v = grangercausalitytests(data_r2v, maxlag=max_lag, verbose=False)
        rows_r2v = []
        for lag_order in range(1, max_lag + 1):
            test_results = gc_r2v[lag_order][0]
            rows_r2v.append({
                'lag': lag_order,
                'ssr_ftest_pval': test_results['ssr_ftest'][1],
                'ssr_chi2test_pval': test_results['ssr_chi2test'][1],
                'lrtest_pval': test_results['lrtest'][1],
                'params_ftest_pval': test_results['params_ftest'][1],
            })
        results['returns_to_volume'] = pd.DataFrame(rows_r2v)
    except Exception as e:
        print(f"  [警告] 收益率→成交量 Granger检验失败: {e}")
        results['returns_to_volume'] = pd.DataFrame()

    return results


def _compute_obv(df: pd.DataFrame) -> pd.Series:
    """计算OBV (On-Balance Volume)

    规则:
    - 收盘价上涨: OBV += volume
    - 收盘价下跌: OBV -= volume
    - 收盘价持平: OBV 不变
    """
    close = df['close']
    volume = df['volume']

    direction = np.sign(close.diff())
    obv = (direction * volume).fillna(0).cumsum()
    obv.name = 'obv'
    return obv


def _detect_obv_divergences(
    prices: pd.Series,
    obv: pd.Series,
    window: int = 60,
    lookback: int = 5,
) -> pd.DataFrame:
    """检测OBV-价格背离

    背离类型:
    - 顶背离 (bearish): 价格创新高但OBV未创新高 → 潜在下跌信号
    - 底背离 (bullish): 价格创新低但OBV未创新低 → 潜在上涨信号

    Parameters
    ----------
    prices : pd.Series
        收盘价序列
    obv : pd.Series
        OBV序列
    window : int
        滚动窗口大小，用于判断"新高"/"新低"
    lookback : int
        新高/新低确认回看天数

    Returns
    -------
    pd.DataFrame
        背离事件表，包含 date, type, price, obv 列
    """
    divergences = []

    # 滚动最高/最低
    price_rolling_max = prices.rolling(window=window, min_periods=window).max()
    price_rolling_min = prices.rolling(window=window, min_periods=window).min()
    obv_rolling_max = obv.rolling(window=window, min_periods=window).max()
    obv_rolling_min = obv.rolling(window=window, min_periods=window).min()

    for i in range(window + lookback, len(prices)):
        idx = prices.index[i]
        price_val = prices.iloc[i]
        obv_val = obv.iloc[i]

        # 价格创近期新高 (最近lookback天内触及滚动最高)
        recent_prices = prices.iloc[i - lookback:i + 1]
        recent_obv = obv.iloc[i - lookback:i + 1]
        rolling_max_price = price_rolling_max.iloc[i]
        rolling_max_obv = obv_rolling_max.iloc[i]
        rolling_min_price = price_rolling_min.iloc[i]
        rolling_min_obv = obv_rolling_min.iloc[i]

        # 顶背离: 价格 == 滚动最高 且 OBV 未达到滚动最高的95%
        if price_val >= rolling_max_price * 0.998:
            if obv_val < rolling_max_obv * 0.95:
                divergences.append({
                    'date': idx,
                    'type': 'bearish',  # 顶背离
                    'price': price_val,
                    'obv': obv_val,
                })

        # 底背离: 价格 == 滚动最低 且 OBV 未达到滚动最低(更高)
        if price_val <= rolling_min_price * 1.002:
            if obv_val > rolling_min_obv * 1.05:
                divergences.append({
                    'date': idx,
                    'type': 'bullish',  # 底背离
                    'price': price_val,
                    'obv': obv_val,
                })

    df_div = pd.DataFrame(divergences)

    # 去除密集重复信号 (同类型信号间隔至少10天)
    if not df_div.empty:
        df_div = df_div.sort_values('date')
        filtered = [df_div.iloc[0]]
        for _, row in df_div.iloc[1:].iterrows():
            last = filtered[-1]
            if row['type'] != last['type'] or (row['date'] - last['date']).days >= 10:
                filtered.append(row)
        df_div = pd.DataFrame(filtered).reset_index(drop=True)

    return df_div


# =============================================================================
#  可视化函数
# =============================================================================

def _plot_volume_return_scatter(
    volume: pd.Series,
    returns: pd.Series,
    spearman_result: Dict,
    output_dir: Path,
):
    """图1: 成交量 vs |收益率| 散点图"""
    fig, ax = plt.subplots(figsize=(10, 7))

    abs_ret = returns.abs()
    aligned = pd.concat([volume, abs_ret], axis=1, keys=['volume', 'abs_return']).dropna()

    ax.scatter(aligned['volume'], aligned['abs_return'],
               s=5, alpha=0.3, color='steelblue')

    rho = spearman_result['correlation']
    p_val = spearman_result['p_value']
    ax.set_xlabel('成交量', fontsize=12)
    ax.set_ylabel('|对数收益率|', fontsize=12)
    ax.set_title(f'成交量 vs |收益率| 散点图\nSpearman ρ={rho:.4f}, p={p_val:.2e}', fontsize=13)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / 'volume_return_scatter.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [图] 量价散点图已保存: {output_dir / 'volume_return_scatter.png'}")


def _plot_lead_lag_correlation(
    lead_lag_df: pd.DataFrame,
    output_dir: Path,
):
    """图2: Taker买入比例领先-滞后相关性柱状图"""
    fig, ax = plt.subplots(figsize=(12, 6))

    if lead_lag_df.empty:
        ax.text(0.5, 0.5, '数据不足，无法计算领先-滞后相关性',
                transform=ax.transAxes, ha='center', va='center', fontsize=14)
        fig.savefig(output_dir / 'taker_buy_lead_lag.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        return

    colors = ['red' if sig else 'steelblue'
              for sig in lead_lag_df['significant']]

    bars = ax.bar(lead_lag_df['lag'], lead_lag_df['correlation'],
                   color=colors, alpha=0.8, edgecolor='white')

    # 显著性水平线
    ax.axhline(y=0, color='black', linewidth=0.5)

    ax.set_xlabel('领先天数 (lag)', fontsize=12)
    ax.set_ylabel('Spearman 相关系数', fontsize=12)
    ax.set_title('Taker买入比例对未来收益的领先相关性\n(红色=p<0.05 显著)', fontsize=13)
    ax.set_xticks(lead_lag_df['lag'])
    ax.grid(True, alpha=0.3, axis='y')

    fig.savefig(output_dir / 'taker_buy_lead_lag.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [图] Taker买入比例领先分析已保存: {output_dir / 'taker_buy_lead_lag.png'}")


def _plot_granger_heatmap(
    granger_results: Dict[str, pd.DataFrame],
    output_dir: Path,
):
    """图3: Granger因果检验p值热力图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    titles = {
        'volume_to_returns': '成交量 → 收益率',
        'returns_to_volume': '收益率 → 成交量',
    }

    for ax, (direction, df_gc) in zip(axes, granger_results.items()):
        if df_gc.empty:
            ax.text(0.5, 0.5, '检验失败', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)
            ax.set_title(titles[direction], fontsize=13)
            continue

        # 构建热力图矩阵
        test_names = ['ssr_ftest_pval', 'ssr_chi2test_pval', 'lrtest_pval', 'params_ftest_pval']
        test_labels = ['SSR F-test', 'SSR Chi2', 'LR test', 'Params F-test']
        lags = df_gc['lag'].values

        heatmap_data = df_gc[test_names].values.T  # shape: (4, n_lags)

        im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn',
                        vmin=0, vmax=0.1, interpolation='nearest')

        ax.set_xticks(range(len(lags)))
        ax.set_xticklabels(lags, fontsize=9)
        ax.set_yticks(range(len(test_labels)))
        ax.set_yticklabels(test_labels, fontsize=9)
        ax.set_xlabel('滞后阶数', fontsize=11)
        ax.set_title(f'Granger因果: {titles[direction]}', fontsize=13)

        # 标注p值
        for i in range(len(test_labels)):
            for j in range(len(lags)):
                val = heatmap_data[i, j]
                color = 'white' if val < 0.03 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=7, color=color)

    fig.colorbar(im, ax=axes, label='p-value', shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / 'granger_causality_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [图] Granger因果热力图已保存: {output_dir / 'granger_causality_heatmap.png'}")


def _plot_obv_with_divergences(
    df: pd.DataFrame,
    obv: pd.Series,
    divergences: pd.DataFrame,
    output_dir: Path,
):
    """图4: OBV vs 价格 + 背离标记"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]})

    # 上图: 价格
    ax1.plot(df.index, df['close'], color='black', linewidth=0.8, label='BTC 收盘价')
    ax1.set_ylabel('价格 (USDT)', fontsize=12)
    ax1.set_title('BTC 价格与OBV背离分析', fontsize=14)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, which='both')

    # 下图: OBV
    ax2.plot(obv.index, obv.values, color='steelblue', linewidth=0.8, label='OBV')
    ax2.set_ylabel('OBV', fontsize=12)
    ax2.set_xlabel('日期', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 标记背离
    if not divergences.empty:
        bearish = divergences[divergences['type'] == 'bearish']
        bullish = divergences[divergences['type'] == 'bullish']

        if not bearish.empty:
            ax1.scatter(bearish['date'], bearish['price'],
                        marker='v', s=60, color='red', zorder=5,
                        label=f'顶背离 ({len(bearish)}次)', alpha=0.7)
            for _, row in bearish.iterrows():
                ax2.axvline(row['date'], color='red', alpha=0.2, linewidth=0.5)

        if not bullish.empty:
            ax1.scatter(bullish['date'], bullish['price'],
                        marker='^', s=60, color='green', zorder=5,
                        label=f'底背离 ({len(bullish)}次)', alpha=0.7)
            for _, row in bullish.iterrows():
                ax2.axvline(row['date'], color='green', alpha=0.2, linewidth=0.5)

    ax1.legend(fontsize=10, loc='upper left')
    ax2.legend(fontsize=10, loc='upper left')

    fig.tight_layout()
    fig.savefig(output_dir / 'obv_divergence.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [图] OBV背离分析已保存: {output_dir / 'obv_divergence.png'}")


# =============================================================================
#  主入口
# =============================================================================

def run_volume_price_analysis(df: pd.DataFrame, output_dir: str = "output") -> Dict:
    """成交量-价格关系与OBV分析 — 主入口函数

    Parameters
    ----------
    df : pd.DataFrame
        由 data_loader.load_daily() 返回的日线数据，含 DatetimeIndex,
        close, volume, taker_buy_volume 等列
    output_dir : str
        图表输出目录

    Returns
    -------
    dict
        分析结果摘要
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  BTC 成交量-价格关系分析")
    print("=" * 60)

    # 准备数据
    prices = df['close'].dropna()
    volume = df['volume'].dropna()
    log_ret = np.log(prices / prices.shift(1)).dropna()

    # 计算taker买入比例
    taker_buy_ratio = (df['taker_buy_volume'] / df['volume'].replace(0, np.nan)).dropna()

    print(f"\n数据范围: {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"样本数量: {len(df)}")

    # ---- 步骤1: Spearman相关性 ----
    print("\n--- Spearman 成交量-|收益率| 相关性 ---")
    spearman_result = _spearman_volume_returns(volume, log_ret)
    print(f"  Spearman ρ:  {spearman_result['correlation']:.4f}")
    print(f"  p-value:     {spearman_result['p_value']:.2e}")
    print(f"  样本量:       {spearman_result['n_samples']}")
    if spearman_result['p_value'] < 0.01:
        print("  >> 结论: 成交量与|收益率|存在显著正相关（成交量放大伴随大幅波动）")
    else:
        print("  >> 结论: 成交量与|收益率|相关性不显著")

    # ---- 步骤2: Taker买入比例领先分析 ----
    print("\n--- Taker买入比例领先分析 ---")
    lead_lag_df = _taker_buy_ratio_lead_lag(taker_buy_ratio, log_ret, max_lag=20)
    if not lead_lag_df.empty:
        sig_lags = lead_lag_df[lead_lag_df['significant']]
        if not sig_lags.empty:
            print(f"  显著领先期 (p<0.05):")
            for _, row in sig_lags.iterrows():
                print(f"    lag={int(row['lag']):>2d}天: ρ={row['correlation']:.4f}, p={row['p_value']:.4f}")
            best = sig_lags.loc[sig_lags['correlation'].abs().idxmax()]
            print(f"  >> 最强领先信号: lag={int(best['lag'])}天, ρ={best['correlation']:.4f}")
        else:
            print("  未发现显著的领先关系 (所有lag的p>0.05)")
    else:
        print("  数据不足，无法进行领先-滞后分析")

    # ---- 步骤3: Granger因果检验 ----
    print("\n--- Granger 因果检验 (双向, lag 1-10) ---")
    granger_results = _granger_causality(volume, log_ret, max_lag=10)

    for direction, label in [('volume_to_returns', '成交量→收益率'),
                              ('returns_to_volume', '收益率→成交量')]:
        df_gc = granger_results[direction]
        if not df_gc.empty:
            # 使用SSR F-test的p值
            sig_gc = df_gc[df_gc['ssr_ftest_pval'] < 0.05]
            if not sig_gc.empty:
                print(f"  {label}: 在以下滞后阶显著 (SSR F-test p<0.05):")
                for _, row in sig_gc.iterrows():
                    print(f"    lag={int(row['lag'])}: p={row['ssr_ftest_pval']:.4f}")
            else:
                print(f"  {label}: 在所有滞后阶均不显著")
        else:
            print(f"  {label}: 检验失败")

    # ---- 步骤4: OBV计算与背离检测 ----
    print("\n--- OBV 与 价格背离分析 ---")
    obv = _compute_obv(df)
    divergences = _detect_obv_divergences(prices, obv, window=60, lookback=5)

    if not divergences.empty:
        bearish_count = len(divergences[divergences['type'] == 'bearish'])
        bullish_count = len(divergences[divergences['type'] == 'bullish'])
        print(f"  检测到 {len(divergences)} 个背离信号:")
        print(f"    顶背离 (看跌): {bearish_count} 次")
        print(f"    底背离 (看涨): {bullish_count} 次")

        # 最近的背离
        recent = divergences.tail(5)
        print(f"  最近 {len(recent)} 个背离:")
        for _, row in recent.iterrows():
            div_type = '顶背离' if row['type'] == 'bearish' else '底背离'
            date_str = row['date'].strftime('%Y-%m-%d')
            print(f"    {date_str}: {div_type}, 价格=${row['price']:,.0f}")
    else:
        bearish_count = 0
        bullish_count = 0
        print("  未检测到明显的OBV-价格背离")

    # ---- 步骤5: 生成可视化 ----
    print("\n--- 生成可视化图表 ---")
    _plot_volume_return_scatter(volume, log_ret, spearman_result, output_dir)
    _plot_lead_lag_correlation(lead_lag_df, output_dir)
    _plot_granger_heatmap(granger_results, output_dir)
    _plot_obv_with_divergences(df, obv, divergences, output_dir)

    print("\n" + "=" * 60)
    print("  成交量-价格分析完成")
    print("=" * 60)

    # 返回结果摘要
    return {
        'spearman': spearman_result,
        'lead_lag': {
            'significant_lags': lead_lag_df[lead_lag_df['significant']]['lag'].tolist()
            if not lead_lag_df.empty else [],
        },
        'granger': {
            'volume_to_returns_sig_lags': granger_results['volume_to_returns'][
                granger_results['volume_to_returns']['ssr_ftest_pval'] < 0.05
            ]['lag'].tolist() if not granger_results['volume_to_returns'].empty else [],
            'returns_to_volume_sig_lags': granger_results['returns_to_volume'][
                granger_results['returns_to_volume']['ssr_ftest_pval'] < 0.05
            ]['lag'].tolist() if not granger_results['returns_to_volume'].empty else [],
        },
        'obv_divergences': {
            'total': len(divergences),
            'bearish': bearish_count,
            'bullish': bullish_count,
        },
    }


if __name__ == '__main__':
    from data_loader import load_daily
    df = load_daily()
    results = run_volume_price_analysis(df, output_dir='../output/volume_price')
