"""
技术指标有效性验证模块

手动实现常见技术指标（MA/EMA交叉、RSI、MACD、布林带），
在训练集上进行统计显著性检验，并在验证集上验证。
包含反数据窥探措施：Benjamini-Hochberg FDR 校正 + 置换检验。
"""

import matplotlib
matplotlib.use('Agg')

from src.font_config import configure_chinese_font
configure_chinese_font()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from src.data_loader import split_data
from src.preprocessing import log_returns


# ============================================================
# 1. 手动实现技术指标
# ============================================================

def calc_sma(series: pd.Series, window: int) -> pd.Series:
    """简单移动平均线"""
    return series.rolling(window=window, min_periods=window).mean()


def calc_ema(series: pd.Series, span: int) -> pd.Series:
    """指数移动平均线"""
    return series.ewm(span=span, adjust=False).mean()


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    相对强弱指标 (RSI)
    RSI = 100 - 100 / (1 + RS)
    RS = 平均上涨幅度 / 平均下跌幅度
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    # 使用 EMA 计算平均涨跌
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi


def calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD 指标
    返回: (macd_line, signal_line, histogram)
    """
    ema_fast = calc_ema(close, fast)
    ema_slow = calc_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    布林带
    返回: (upper, middle, lower)
    """
    middle = calc_sma(close, window)
    rolling_std = close.rolling(window=window, min_periods=window).std()
    upper = middle + num_std * rolling_std
    lower = middle - num_std * rolling_std
    return upper, middle, lower


# ============================================================
# 2. 信号生成
# ============================================================

def generate_ma_crossover_signals(close: pd.Series, short_w: int, long_w: int, use_ema: bool = False) -> pd.Series:
    """
    均线交叉信号
    金叉 = +1（短期上穿长期），死叉 = -1（短期下穿长期），无信号 = 0
    """
    func = calc_ema if use_ema else calc_sma
    short_ma = func(close, short_w)
    long_ma = func(close, long_w)
    # 当前短>长 且 前一根短<=长 => 金叉(+1)
    # 当前短<长 且 前一根短>=长 => 死叉(-1)
    cross_up = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
    cross_down = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
    signal = pd.Series(0, index=close.index)
    signal[cross_up] = 1
    signal[cross_down] = -1
    return signal


def generate_rsi_signals(close: pd.Series, period: int, oversold: float = 30, overbought: float = 70) -> pd.Series:
    """
    RSI 超买超卖信号
    RSI 从超卖区回升 => +1 (买入信号)
    RSI 从超买区回落 => -1 (卖出信号)
    """
    rsi = calc_rsi(close, period)
    rsi_prev = rsi.shift(1)
    signal = pd.Series(0, index=close.index)
    # 从超卖回升
    signal[(rsi_prev <= oversold) & (rsi > oversold)] = 1
    # 从超买回落
    signal[(rsi_prev >= overbought) & (rsi < overbought)] = -1
    return signal


def generate_macd_signals(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.Series:
    """
    MACD 交叉信号
    MACD线上穿信号线 => +1
    MACD线下穿信号线 => -1
    """
    macd_line, signal_line, _ = calc_macd(close, fast, slow, sig)
    cross_up = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    cross_down = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
    signal = pd.Series(0, index=close.index)
    signal[cross_up] = 1
    signal[cross_down] = -1
    return signal


def generate_bollinger_signals(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """
    布林带信号
    价格触及下轨后回升 => +1 (买入)
    价格触及上轨后回落 => -1 (卖出)
    """
    upper, middle, lower = calc_bollinger_bands(close, window, num_std)
    # 前一根在下轨以下，当前回到下轨以上
    cross_up = (close.shift(1) <= lower.shift(1)) & (close > lower)
    # 前一根在上轨以上，当前回到上轨以下
    cross_down = (close.shift(1) >= upper.shift(1)) & (close < upper)
    signal = pd.Series(0, index=close.index)
    signal[cross_up] = 1
    signal[cross_down] = -1
    return signal


def build_all_signals(close: pd.Series) -> Dict[str, pd.Series]:
    """
    构建所有技术指标信号
    返回字典: {指标名称: 信号序列}
    """
    signals = {}

    # --- MA / EMA 交叉 ---
    ma_pairs = [(5, 20), (10, 50), (20, 100), (50, 200)]
    for short_w, long_w in ma_pairs:
        signals[f"SMA_{short_w}_{long_w}"] = generate_ma_crossover_signals(close, short_w, long_w, use_ema=False)
        signals[f"EMA_{short_w}_{long_w}"] = generate_ma_crossover_signals(close, short_w, long_w, use_ema=True)

    # --- RSI ---
    rsi_configs = [
        (7, 30, 70), (7, 25, 75), (7, 20, 80),
        (14, 30, 70), (14, 25, 75), (14, 20, 80),
        (21, 30, 70), (21, 25, 75), (21, 20, 80),
    ]
    for period, oversold, overbought in rsi_configs:
        signals[f"RSI_{period}_{oversold}_{overbought}"] = generate_rsi_signals(close, period, oversold, overbought)

    # --- MACD ---
    macd_configs = [(12, 26, 9), (8, 17, 9), (5, 35, 5)]
    for fast, slow, sig in macd_configs:
        signals[f"MACD_{fast}_{slow}_{sig}"] = generate_macd_signals(close, fast, slow, sig)

    # --- 布林带 ---
    signals["BB_20_2"] = generate_bollinger_signals(close, 20, 2.0)

    return signals


# ============================================================
# 3. 统计检验
# ============================================================

def calc_forward_returns(close: pd.Series, periods: int = 1) -> pd.Series:
    """计算未来N日收益率（对数收益率）"""
    return np.log(close.shift(-periods) / close)


def test_signal_returns(signal: pd.Series, returns: pd.Series) -> Dict:
    """
    对单个指标信号进行统计检验

    - Welch t-test：比较信号日 vs 非信号日收益均值差异
    - Mann-Whitney U：非参数检验
    - 二项检验：方向准确率是否显著高于50%
    - 信息系数 (IC)：Spearman秩相关
    """
    # 买入信号日（signal == 1）的收益
    buy_returns = returns[signal == 1].dropna()
    # 卖出信号日（signal == -1）的收益
    sell_returns = returns[signal == -1].dropna()
    # 非信号日收益
    no_signal_returns = returns[signal == 0].dropna()

    result = {
        'n_buy': len(buy_returns),
        'n_sell': len(sell_returns),
        'n_no_signal': len(no_signal_returns),
        'buy_mean': buy_returns.mean() if len(buy_returns) > 0 else np.nan,
        'sell_mean': sell_returns.mean() if len(sell_returns) > 0 else np.nan,
        'no_signal_mean': no_signal_returns.mean() if len(no_signal_returns) > 0 else np.nan,
    }

    # --- Welch t-test (买入信号 vs 非信号) ---
    if len(buy_returns) >= 5 and len(no_signal_returns) >= 5:
        t_stat, t_pval = stats.ttest_ind(buy_returns, no_signal_returns, equal_var=False)
        result['welch_t_stat'] = t_stat
        result['welch_t_pval'] = t_pval
    else:
        result['welch_t_stat'] = np.nan
        result['welch_t_pval'] = np.nan

    # --- Mann-Whitney U (买入信号 vs 非信号) ---
    if len(buy_returns) >= 5 and len(no_signal_returns) >= 5:
        u_stat, u_pval = stats.mannwhitneyu(buy_returns, no_signal_returns, alternative='two-sided')
        result['mwu_stat'] = u_stat
        result['mwu_pval'] = u_pval
    else:
        result['mwu_stat'] = np.nan
        result['mwu_pval'] = np.nan

    # --- 二项检验：买入信号日收益>0的比例 vs 50% ---
    if len(buy_returns) >= 5:
        n_positive = (buy_returns > 0).sum()
        binom_pval = stats.binomtest(n_positive, len(buy_returns), 0.5).pvalue
        result['buy_hit_rate'] = n_positive / len(buy_returns)
        result['binom_pval'] = binom_pval
    else:
        result['buy_hit_rate'] = np.nan
        result['binom_pval'] = np.nan

    # --- 信息系数 (IC)：Spearman秩相关 ---
    # 用信号值（-1, 0, 1）与未来收益的秩相关
    valid_mask = signal.notna() & returns.notna()
    if valid_mask.sum() >= 30:
        ic, ic_pval = stats.spearmanr(signal[valid_mask], returns[valid_mask])
        result['ic'] = ic
        result['ic_pval'] = ic_pval
    else:
        result['ic'] = np.nan
        result['ic_pval'] = np.nan

    return result


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg FDR 校正

    参数:
        p_values: 原始 p 值数组
        alpha: 显著性水平

    返回:
        (rejected, adjusted_p): 是否拒绝原假设, 校正后p值
    """
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=bool), np.array([])

    # 处理 NaN
    valid_mask = ~np.isnan(p_values)
    adjusted = np.full(n, np.nan)
    rejected = np.full(n, False)

    valid_pvals = p_values[valid_mask]
    n_valid = len(valid_pvals)
    if n_valid == 0:
        return rejected, adjusted

    # 排序
    sorted_idx = np.argsort(valid_pvals)
    sorted_pvals = valid_pvals[sorted_idx]

    # BH校正
    rank = np.arange(1, n_valid + 1)
    adjusted_sorted = sorted_pvals * n_valid / rank
    # 从后往前取累积最小值，确保单调性
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0, 1)

    # 填回
    valid_indices = np.where(valid_mask)[0]
    for i, idx in enumerate(sorted_idx):
        adjusted[valid_indices[idx]] = adjusted_sorted[i]
        rejected[valid_indices[idx]] = adjusted_sorted[i] <= alpha

    return rejected, adjusted


def permutation_test(signal: pd.Series, returns: pd.Series, n_permutations: int = 1000, stat_func=None) -> Tuple[float, float]:
    """
    置换检验

    随机打乱信号与收益的对应关系，评估原始统计量的显著性
    返回: (observed_stat, p_value)
    """
    if stat_func is None:
        # 默认统计量：买入信号日均值 - 非信号日均值
        def stat_func(sig, ret):
            buy_ret = ret[sig == 1]
            no_sig_ret = ret[sig == 0]
            if len(buy_ret) < 2 or len(no_sig_ret) < 2:
                return 0.0
            return buy_ret.mean() - no_sig_ret.mean()

    valid_mask = signal.notna() & returns.notna()
    sig_valid = signal[valid_mask].values
    ret_valid = returns[valid_mask].values

    observed = stat_func(pd.Series(sig_valid), pd.Series(ret_valid))

    # 置换
    count_extreme = 0
    rng = np.random.RandomState(42)
    for _ in range(n_permutations):
        perm_sig = rng.permutation(sig_valid)
        perm_stat = stat_func(pd.Series(perm_sig), pd.Series(ret_valid))
        if abs(perm_stat) >= abs(observed):
            count_extreme += 1

    perm_pval = (count_extreme + 1) / (n_permutations + 1)
    return observed, perm_pval


# ============================================================
# 4. 可视化
# ============================================================

def plot_ic_distribution(results_df: pd.DataFrame, output_dir: Path, prefix: str = "train"):
    """绘制信息系数 (IC) 分布图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ic_vals = results_df['ic'].dropna()
    ax.barh(range(len(ic_vals)), ic_vals.values, color=['green' if v > 0 else 'red' for v in ic_vals.values])
    ax.set_yticks(range(len(ic_vals)))
    ax.set_yticklabels(ic_vals.index, fontsize=7)
    ax.set_xlabel('Information Coefficient (Spearman)')
    ax.set_title(f'IC Distribution - {prefix.upper()} Set')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    fig.savefig(output_dir / f"ic_distribution_{prefix}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [saved] ic_distribution_{prefix}.png")


def plot_pvalue_heatmap(results_df: pd.DataFrame, output_dir: Path, prefix: str = "train"):
    """绘制 p 值热力图：原始 vs FDR 校正后"""
    pval_cols = ['welch_t_pval', 'mwu_pval', 'binom_pval', 'ic_pval']
    adj_cols = ['welch_t_adj_pval', 'mwu_adj_pval', 'binom_adj_pval', 'ic_adj_pval']

    # 只取存在的列
    existing_pval = [c for c in pval_cols if c in results_df.columns]
    existing_adj = [c for c in adj_cols if c in results_df.columns]

    if not existing_pval:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(results_df) * 0.35)))

    # 原始 p 值
    pval_data = results_df[existing_pval].values.astype(float)
    im1 = axes[0].imshow(pval_data, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=0.1)
    axes[0].set_yticks(range(len(results_df)))
    axes[0].set_yticklabels(results_df.index, fontsize=6)
    axes[0].set_xticks(range(len(existing_pval)))
    axes[0].set_xticklabels([c.replace('_pval', '') for c in existing_pval], fontsize=8, rotation=45)
    axes[0].set_title('Raw p-values')
    plt.colorbar(im1, ax=axes[0], shrink=0.6)

    # FDR 校正后 p 值
    if existing_adj:
        adj_data = results_df[existing_adj].values.astype(float)
        im2 = axes[1].imshow(adj_data, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=0.1)
        axes[1].set_yticks(range(len(results_df)))
        axes[1].set_yticklabels(results_df.index, fontsize=6)
        axes[1].set_xticks(range(len(existing_adj)))
        axes[1].set_xticklabels([c.replace('_adj_pval', '') for c in existing_adj], fontsize=8, rotation=45)
        axes[1].set_title('FDR-adjusted p-values')
        plt.colorbar(im2, ax=axes[1], shrink=0.6)
    else:
        axes[1].text(0.5, 0.5, 'No adjusted p-values', ha='center', va='center')
        axes[1].set_title('FDR-adjusted p-values (N/A)')

    plt.suptitle(f'P-value Heatmap - {prefix.upper()} Set', fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / f"pvalue_heatmap_{prefix}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [saved] pvalue_heatmap_{prefix}.png")


def plot_best_indicator_signal(close: pd.Series, signal: pd.Series, returns: pd.Series,
                                indicator_name: str, output_dir: Path, prefix: str = "train"):
    """绘制最佳指标的信号 vs 收益散点图"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})

    # 上图：价格 + 信号标记
    axes[0].plot(close.index, close.values, color='gray', alpha=0.7, linewidth=0.8, label='BTC Close')
    buy_mask = signal == 1
    sell_mask = signal == -1
    axes[0].scatter(close.index[buy_mask], close.values[buy_mask],
                    marker='^', color='green', s=40, label='Buy Signal', zorder=5)
    axes[0].scatter(close.index[sell_mask], close.values[sell_mask],
                    marker='v', color='red', s=40, label='Sell Signal', zorder=5)
    axes[0].set_title(f'Best Indicator: {indicator_name} - {prefix.upper()} Set')
    axes[0].set_ylabel('Price (USDT)')
    axes[0].legend(fontsize=8)

    # 下图：信号日收益分布
    buy_returns = returns[buy_mask].dropna()
    sell_returns = returns[sell_mask].dropna()
    if len(buy_returns) > 0:
        axes[1].hist(buy_returns, bins=30, alpha=0.6, color='green', label=f'Buy ({len(buy_returns)})')
    if len(sell_returns) > 0:
        axes[1].hist(sell_returns, bins=30, alpha=0.6, color='red', label=f'Sell ({len(sell_returns)})')
    axes[1].axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    axes[1].set_xlabel('Forward 1-day Log Return')
    axes[1].set_ylabel('Count')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / f"best_indicator_{prefix}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [saved] best_indicator_{prefix}.png")


# ============================================================
# 5. 主流程
# ============================================================

def evaluate_signals_on_set(close: pd.Series, signals: Dict[str, pd.Series], set_name: str) -> pd.DataFrame:
    """
    在给定数据集上评估所有信号

    返回包含所有统计指标的 DataFrame
    """
    # 未来1日收益
    fwd_ret = calc_forward_returns(close, periods=1)

    results = {}
    for name, signal in signals.items():
        # 只取当前数据集范围内的信号
        sig = signal.reindex(close.index).fillna(0)
        ret = fwd_ret.reindex(close.index)
        results[name] = test_signal_returns(sig, ret)

    results_df = pd.DataFrame(results).T
    results_df.index.name = 'indicator'

    print(f"\n{'='*60}")
    print(f"  {set_name} 数据集评估结果")
    print(f"{'='*60}")
    print(f"  总指标数: {len(results_df)}")
    print(f"  数据点数: {len(close)}")

    return results_df


def apply_fdr_correction(results_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    对所有 p 值列进行 Benjamini-Hochberg FDR 校正
    """
    pval_cols = ['welch_t_pval', 'mwu_pval', 'binom_pval', 'ic_pval']

    for col in pval_cols:
        if col not in results_df.columns:
            continue
        pvals = results_df[col].values.astype(float)
        rejected, adjusted = benjamini_hochberg(pvals, alpha)
        adj_col = col.replace('_pval', '_adj_pval')
        rej_col = col.replace('_pval', '_rejected')
        results_df[adj_col] = adjusted
        results_df[rej_col] = rejected

    return results_df


def run_indicators_analysis(df: pd.DataFrame, output_dir: str) -> Dict:
    """
    技术指标有效性验证主入口

    参数:
        df: 完整的日线 DataFrame（含 open/high/low/close/volume 等列，DatetimeIndex）
        output_dir: 图表输出目录

    返回:
        包含训练集和验证集结果的字典
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  技术指标有效性验证")
    print("=" * 60)

    # --- 数据切分 ---
    train, val, test = split_data(df)
    print(f"\n训练集: {train.index.min()} ~ {train.index.max()}  ({len(train)} bars)")
    print(f"验证集: {val.index.min()} ~ {val.index.max()}  ({len(val)} bars)")

    # --- 构建全部信号（在全量数据上计算，避免前导NaN问题） ---
    all_signals = build_all_signals(df['close'])
    print(f"\n共构建 {len(all_signals)} 个技术指标信号")

    # ============ 训练集评估 ============
    train_results = evaluate_signals_on_set(train['close'], all_signals, "训练集 (TRAIN)")

    # FDR 校正
    train_results = apply_fdr_correction(train_results, alpha=0.05)

    # 找出通过 FDR 校正的指标
    reject_cols = [c for c in train_results.columns if c.endswith('_rejected')]
    if reject_cols:
        train_results['any_fdr_pass'] = train_results[reject_cols].any(axis=1)
        fdr_passed = train_results[train_results['any_fdr_pass']].index.tolist()
    else:
        fdr_passed = []

    print(f"\n--- FDR 校正结果 (训练集) ---")
    if fdr_passed:
        print(f"  通过 FDR 校正的指标 ({len(fdr_passed)} 个):")
        for name in fdr_passed:
            row = train_results.loc[name]
            ic_val = row.get('ic', np.nan)
            print(f"    - {name}: IC={ic_val:.4f}" if not np.isnan(ic_val) else f"    - {name}")
    else:
        print("  没有指标通过 FDR 校正（alpha=0.05）")

    # --- 置换检验（仅对 IC 排名前5的指标） ---
    fwd_ret_train = calc_forward_returns(train['close'], periods=1)
    ic_series = train_results['ic'].dropna().abs().sort_values(ascending=False)
    top_indicators = ic_series.head(5).index.tolist()

    print(f"\n--- 置换检验 (训练集, top-5 IC 指标, 1000次置换) ---")
    perm_results = {}
    for name in top_indicators:
        sig = all_signals[name].reindex(train.index).fillna(0)
        ret = fwd_ret_train.reindex(train.index)
        obs, pval = permutation_test(sig, ret, n_permutations=1000)
        perm_results[name] = {'observed_diff': obs, 'perm_pval': pval}
        perm_pass = "PASS" if pval < 0.05 else "FAIL"
        print(f"  {name}: obs_diff={obs:.6f}, perm_p={pval:.4f} [{perm_pass}]")

    # --- 训练集可视化 ---
    print("\n--- 训练集可视化 ---")
    plot_ic_distribution(train_results, output_dir, prefix="train")
    plot_pvalue_heatmap(train_results, output_dir, prefix="train")

    # 最佳指标（IC绝对值最大）
    if len(ic_series) > 0:
        best_name = ic_series.index[0]
        best_signal = all_signals[best_name].reindex(train.index).fillna(0)
        best_ret = fwd_ret_train.reindex(train.index)
        plot_best_indicator_signal(train['close'], best_signal, best_ret, best_name, output_dir, prefix="train")

    # ============ 验证集评估 ============
    val_results = evaluate_signals_on_set(val['close'], all_signals, "验证集 (VAL)")
    val_results = apply_fdr_correction(val_results, alpha=0.05)

    reject_cols_val = [c for c in val_results.columns if c.endswith('_rejected')]
    if reject_cols_val:
        val_results['any_fdr_pass'] = val_results[reject_cols_val].any(axis=1)
        val_fdr_passed = val_results[val_results['any_fdr_pass']].index.tolist()
    else:
        val_fdr_passed = []

    print(f"\n--- FDR 校正结果 (验证集) ---")
    if val_fdr_passed:
        print(f"  通过 FDR 校正的指标 ({len(val_fdr_passed)} 个):")
        for name in val_fdr_passed:
            row = val_results.loc[name]
            ic_val = row.get('ic', np.nan)
            print(f"    - {name}: IC={ic_val:.4f}" if not np.isnan(ic_val) else f"    - {name}")
    else:
        print("  没有指标通过 FDR 校正（alpha=0.05）")

    # 训练集 vs 验证集 IC 对比
    if 'ic' in train_results.columns and 'ic' in val_results.columns:
        print(f"\n--- 训练集 vs 验证集 IC 对比 (Top-10) ---")
        merged_ic = pd.DataFrame({
            'train_ic': train_results['ic'],
            'val_ic': val_results['ic']
        }).dropna()
        merged_ic['consistent'] = (merged_ic['train_ic'] * merged_ic['val_ic']) > 0  # 同号
        merged_ic = merged_ic.reindex(merged_ic['train_ic'].abs().sort_values(ascending=False).index)
        for name in merged_ic.head(10).index:
            row = merged_ic.loc[name]
            cons = "OK" if row['consistent'] else "FLIP"
            print(f"  {name}: train_IC={row['train_ic']:.4f}, val_IC={row['val_ic']:.4f} [{cons}]")

    # --- 验证集可视化 ---
    print("\n--- 验证集可视化 ---")
    plot_ic_distribution(val_results, output_dir, prefix="val")
    plot_pvalue_heatmap(val_results, output_dir, prefix="val")

    val_ic_series = val_results['ic'].dropna().abs().sort_values(ascending=False)
    if len(val_ic_series) > 0:
        fwd_ret_val = calc_forward_returns(val['close'], periods=1)
        best_val_name = val_ic_series.index[0]
        best_val_signal = all_signals[best_val_name].reindex(val.index).fillna(0)
        best_val_ret = fwd_ret_val.reindex(val.index)
        plot_best_indicator_signal(val['close'], best_val_signal, best_val_ret, best_val_name, output_dir, prefix="val")

    print(f"\n{'='*60}")
    print("  技术指标有效性验证完成")
    print(f"{'='*60}")

    return {
        'train_results': train_results,
        'val_results': val_results,
        'fdr_passed_train': fdr_passed,
        'fdr_passed_val': val_fdr_passed,
        'permutation_results': perm_results,
        'all_signals': all_signals,
    }
