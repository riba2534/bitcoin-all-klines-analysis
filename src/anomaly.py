"""异常检测与前兆模式提取模块

分析内容：
- 集成异常检测（Isolation Forest + LOF + COPOD，≥2/3 一致判定）
- GARCH 条件波动率异常检测（标准化残差 > 3）
- 异常前兆模式提取（Random Forest 分类器）
- 事件对齐分析（比特币减半等重大事件）
- 可视化：异常标记价格图、特征分布对比、ROC 曲线、特征重要性
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, roc_curve

from src.data_loader import load_klines
from src.preprocessing import add_derived_features

try:
    from pyod.models.copod import COPOD
    HAS_COPOD = True
except ImportError:
    HAS_COPOD = False
    print("[警告] pyod 未安装，COPOD 检测将跳过，使用 2/2 一致判定")


# ============================================================
# 1. 检测特征定义
# ============================================================

# 用于异常检测的特征列
DETECTION_FEATURES = [
    'log_return',
    'abs_return',
    'volume_ratio',
    'range_pct',
    'taker_buy_ratio',
    'vol_7d',
]

# 比特币减半及其他重大事件日期
KNOWN_EVENTS = {
    '2012-11-28': '第一次减半',
    '2016-07-09': '第二次减半',
    '2020-05-11': '第三次减半',
    '2024-04-20': '第四次减半',
    '2017-12-17': '2017年牛市顶点',
    '2018-12-15': '2018年熊市底部',
    '2020-03-12': '新冠黑色星期四',
    '2021-04-14': '2021年牛市中期高点',
    '2021-11-10': '2021年牛市顶点',
    '2022-06-18': 'Luna/3AC 暴跌',
    '2022-11-09': 'FTX 崩盘',
    '2024-01-11': 'BTC ETF 获批',
}


# ============================================================
# 2. 集成异常检测
# ============================================================

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    准备异常检测特征矩阵

    Parameters
    ----------
    df : pd.DataFrame
        含衍生特征的日线数据

    Returns
    -------
    features_df : pd.DataFrame
        特征子集（已去除 NaN）
    X_scaled : np.ndarray
        标准化后的特征矩阵
    """
    # 选取可用特征
    available = [f for f in DETECTION_FEATURES if f in df.columns]
    if len(available) < 3:
        raise ValueError(f"可用特征不足: {available}，至少需要 3 个")

    features_df = df[available].dropna()

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df.values)

    return features_df, X_scaled


def detect_isolation_forest(X: np.ndarray, contamination: float = 0.05) -> np.ndarray:
    """Isolation Forest 异常检测"""
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    # -1 = 异常, 1 = 正常
    labels = model.fit_predict(X)
    return (labels == -1).astype(int)


def detect_lof(X: np.ndarray, contamination: float = 0.05) -> np.ndarray:
    """Local Outlier Factor 异常检测"""
    model = LocalOutlierFactor(
        n_neighbors=20,
        contamination=contamination,
        novelty=False,
        n_jobs=-1,
    )
    labels = model.fit_predict(X)
    return (labels == -1).astype(int)


def detect_copod(X: np.ndarray, contamination: float = 0.05) -> np.ndarray:
    """COPOD 异常检测（基于 Copula）"""
    if not HAS_COPOD:
        return None

    model = COPOD(contamination=contamination)
    labels = model.fit_predict(X)
    return labels.astype(int)


def ensemble_anomaly_detection(
    df: pd.DataFrame,
    contamination: float = 0.05,
    min_agreement: int = 2,
) -> pd.DataFrame:
    """
    集成异常检测：要求 ≥ min_agreement / n_methods 一致判定

    Parameters
    ----------
    df : pd.DataFrame
        含衍生特征的日线数据
    contamination : float
        预期异常比例
    min_agreement : int
        最少多少个方法一致才标记为异常

    Returns
    -------
    pd.DataFrame
        添加了各方法检测结果及集成结果的数据
    """
    features_df, X_scaled = prepare_features(df)

    print(f"  特征矩阵: {X_scaled.shape[0]} 样本 x {X_scaled.shape[1]} 特征")

    # 执行各方法检测
    print("  [1/3] Isolation Forest...")
    if_labels = detect_isolation_forest(X_scaled, contamination)

    print("  [2/3] Local Outlier Factor...")
    lof_labels = detect_lof(X_scaled, contamination)

    n_methods = 2
    vote_matrix = np.column_stack([if_labels, lof_labels])
    method_names = ['iforest', 'lof']

    print("  [3/3] COPOD...")
    copod_labels = detect_copod(X_scaled, contamination)
    if copod_labels is not None:
        vote_matrix = np.column_stack([vote_matrix, copod_labels])
        method_names.append('copod')
        n_methods = 3
    else:
        print("    COPOD 不可用，使用 2 方法集成")

    # 投票
    vote_sum = vote_matrix.sum(axis=1)
    ensemble_label = (vote_sum >= min_agreement).astype(int)

    # 构建结果 DataFrame
    result = features_df.copy()
    for i, name in enumerate(method_names):
        result[f'anomaly_{name}'] = vote_matrix[:, i]
    result['anomaly_votes'] = vote_sum
    result['anomaly_ensemble'] = ensemble_label

    # 打印各方法统计
    print(f"\n  异常检测统计:")
    for name in method_names:
        n_anom = result[f'anomaly_{name}'].sum()
        print(f"    {name:>12}: {n_anom} 个异常 ({n_anom / len(result) * 100:.2f}%)")
    n_ensemble = ensemble_label.sum()
    print(f"    {'集成(≥' + str(min_agreement) + ')':>12}: {n_ensemble} 个异常 ({n_ensemble / len(result) * 100:.2f}%)")

    # 方法间重叠度
    print(f"\n  方法间重叠:")
    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            overlap = ((vote_matrix[:, i] == 1) & (vote_matrix[:, j] == 1)).sum()
            n_i = vote_matrix[:, i].sum()
            n_j = vote_matrix[:, j].sum()
            if min(n_i, n_j) > 0:
                jaccard = overlap / ((vote_matrix[:, i] == 1) | (vote_matrix[:, j] == 1)).sum()
            else:
                jaccard = 0.0
            print(f"    {method_names[i]} ∩ {method_names[j]}: "
                  f"{overlap} 个 (Jaccard={jaccard:.3f})")

    return result


# ============================================================
# 3. GARCH 条件波动率异常
# ============================================================

def garch_anomaly_detection(
    df: pd.DataFrame,
    threshold: float = 3.0,
) -> pd.Series:
    """
    基于 GARCH(1,1) 的条件波动率异常检测

    标准化残差 |ε_t / σ_t| > threshold 的日期标记为异常

    Parameters
    ----------
    df : pd.DataFrame
        含 log_return 列的数据
    threshold : float
        标准化残差阈值

    Returns
    -------
    pd.Series
        异常标记（1 = 异常，0 = 正常），索引与输入对齐
    """
    from arch import arch_model

    returns = df['log_return'].dropna()
    r_pct = returns * 100  # arch 库使用百分比收益率

    # 拟合 GARCH(1,1)
    model = arch_model(r_pct, vol='Garch', p=1, q=1, mean='Constant', dist='Normal')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(disp='off')

    # 计算标准化残差
    std_resid = result.resid / result.conditional_volatility
    anomaly = (std_resid.abs() > threshold).astype(int)

    n_anom = anomaly.sum()
    print(f"  GARCH 异常: {n_anom} 个 (|标准化残差| > {threshold})")
    print(f"  GARCH 模型: α={result.params.get('alpha[1]', np.nan):.4f}, "
          f"β={result.params.get('beta[1]', np.nan):.4f}, "
          f"持续性={result.params.get('alpha[1]', 0) + result.params.get('beta[1]', 0):.4f}")

    return anomaly


# ============================================================
# 4. 前兆模式提取
# ============================================================

def extract_precursor_features(
    df: pd.DataFrame,
    anomaly_labels: pd.Series,
    lookback_windows: List[int] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    提取异常日前若干天的特征作为前兆信号

    Parameters
    ----------
    df : pd.DataFrame
        含衍生特征的数据
    anomaly_labels : pd.Series
        异常标记（1 = 异常）
    lookback_windows : list of int
        向前回溯的天数窗口

    Returns
    -------
    X : pd.DataFrame
        前兆特征矩阵
    y : pd.Series
        标签（1 = 后续发生异常, 0 = 正常）
    """
    if lookback_windows is None:
        lookback_windows = [5, 10, 20]

    # 确保对齐
    common_idx = df.index.intersection(anomaly_labels.index)
    df_aligned = df.loc[common_idx]
    labels_aligned = anomaly_labels.loc[common_idx]

    base_features = [f for f in DETECTION_FEATURES if f in df.columns]
    precursor_features = {}

    for window in lookback_windows:
        for feat in base_features:
            if feat not in df_aligned.columns:
                continue
            series = df_aligned[feat]

            # 滚动统计作为前兆特征
            precursor_features[f'{feat}_mean_{window}d'] = series.rolling(window).mean()
            precursor_features[f'{feat}_std_{window}d'] = series.rolling(window).std()
            precursor_features[f'{feat}_max_{window}d'] = series.rolling(window).max()
            precursor_features[f'{feat}_min_{window}d'] = series.rolling(window).min()

            # 趋势特征（最近值 vs 窗口均值的偏离）
            rolling_mean = series.rolling(window).mean()
            precursor_features[f'{feat}_deviation_{window}d'] = series - rolling_mean

    X = pd.DataFrame(precursor_features, index=df_aligned.index)

    # 标签: 预测次日是否出现异常（前瞻1天）
    y = labels_aligned.shift(-1).dropna()
    X = X.loc[y.index]  # 对齐特征和标签

    # 去除 NaN
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X[valid_mask]
    y = y[valid_mask]

    return X, y


def train_precursor_classifier(
    X: pd.DataFrame,
    y: pd.Series,
) -> Dict:
    """
    训练前兆模式分类器（Random Forest）

    使用分层 K 折交叉验证评估

    Parameters
    ----------
    X : pd.DataFrame
        前兆特征矩阵
    y : pd.Series
        标签

    Returns
    -------
    dict
        AUC、特征重要性等结果
    """
    if len(X) < 50 or y.sum() < 10:
        print(f"  [警告] 样本不足 (n={len(X)}, 正例={y.sum()})，跳过分类器训练")
        return {}

    # 时间序列交叉验证
    n_splits = min(5, int(y.sum()))
    if n_splits < 2:
        print("  [警告] 正例数过少，无法进行交叉验证")
        return {}

    cv = TimeSeriesSplit(n_splits=n_splits)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )

    # 手动交叉验证（每折单独 fit scaler，防止数据泄漏）
    try:
        y_prob = np.full(len(y), np.nan)
        for train_idx, val_idx in cv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            clf.fit(X_train_scaled, y_train)
            y_prob[val_idx] = clf.predict_proba(X_val_scaled)[:, 1]
        # 去除未被验证的样本（如有）
        valid_prob_mask = ~np.isnan(y_prob)
        y_eval = y[valid_prob_mask]
        y_prob_eval = y_prob[valid_prob_mask]
        auc = roc_auc_score(y_eval, y_prob_eval)
    except Exception as e:
        print(f"  [错误] 交叉验证失败: {e}")
        return {}

    # 在全量数据上训练获取特征重要性
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf.fit(X_scaled, y)
    importances = pd.Series(clf.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)

    # ROC 曲线数据
    fpr, tpr, thresholds = roc_curve(y_eval, y_prob_eval)

    results = {
        'auc': auc,
        'feature_importances': importances,
        'y_true': y_eval,
        'y_prob': y_prob_eval,
        'fpr': fpr,
        'tpr': tpr,
    }

    print(f"\n  前兆分类器结果:")
    print(f"    AUC: {auc:.4f}")
    print(f"    样本: {len(y)} (异常: {y.sum()}, 正常: {(y == 0).sum()})")
    print(f"    Top-10 重要特征:")
    for feat, imp in importances.head(10).items():
        print(f"      {feat:<40} {imp:.4f}")

    return results


# ============================================================
# 5. 事件对齐分析
# ============================================================

def align_with_events(
    anomaly_dates: pd.DatetimeIndex,
    tolerance_days: int = 5,
) -> pd.DataFrame:
    """
    将异常日期与已知事件对齐

    Parameters
    ----------
    anomaly_dates : pd.DatetimeIndex
        异常日期列表
    tolerance_days : int
        容差天数（异常日期与事件日期相差 ≤ tolerance_days 天即视为匹配）

    Returns
    -------
    pd.DataFrame
        匹配结果
    """
    matches = []

    for event_date_str, event_name in KNOWN_EVENTS.items():
        event_date = pd.Timestamp(event_date_str)

        for anom_date in anomaly_dates:
            diff_days = abs((anom_date - event_date).days)
            if diff_days <= tolerance_days:
                matches.append({
                    'anomaly_date': anom_date,
                    'event_date': event_date,
                    'event_name': event_name,
                    'diff_days': diff_days,
                })

    if matches:
        result = pd.DataFrame(matches)
        print(f"\n  事件对齐 (容差 {tolerance_days} 天):")
        for _, row in result.iterrows():
            print(f"    异常 {row['anomaly_date'].strftime('%Y-%m-%d')} ↔ "
                  f"{row['event_name']} ({row['event_date'].strftime('%Y-%m-%d')}, "
                  f"差 {row['diff_days']} 天)")
        return result
    else:
        print(f"  [信息] 无异常日期与已知事件匹配 (容差 {tolerance_days} 天)")
        return pd.DataFrame()


# ============================================================
# 6. 可视化
# ============================================================

def plot_price_with_anomalies(
    df: pd.DataFrame,
    anomaly_result: pd.DataFrame,
    garch_anomaly: Optional[pd.Series],
    output_dir: Path,
):
    """绘制价格图，标注异常点"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})

    # 上图：价格 + 异常标记
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], linewidth=0.6, color='steelblue', alpha=0.8, label='BTC 收盘价')

    # 集成异常
    ensemble_anom = anomaly_result[anomaly_result['anomaly_ensemble'] == 1]
    if not ensemble_anom.empty:
        # 获取异常日期对应的收盘价
        anom_prices = df.loc[df.index.isin(ensemble_anom.index), 'close']
        ax1.scatter(anom_prices.index, anom_prices.values,
                    color='red', s=30, zorder=5, label=f'集成异常 (n={len(anom_prices)})',
                    alpha=0.7, edgecolors='darkred', linewidths=0.5)

    # GARCH 异常
    if garch_anomaly is not None:
        garch_anom_dates = garch_anomaly[garch_anomaly == 1].index
        garch_prices = df.loc[df.index.isin(garch_anom_dates), 'close']
        if not garch_prices.empty:
            ax1.scatter(garch_prices.index, garch_prices.values,
                        color='orange', s=20, zorder=4, marker='^',
                        label=f'GARCH 异常 (n={len(garch_prices)})',
                        alpha=0.7, edgecolors='darkorange', linewidths=0.5)

    ax1.set_ylabel('价格 (USDT)', fontsize=12)
    ax1.set_title('BTC 价格与异常检测结果', fontsize=14)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # 下图：成交量 + 异常标记
    ax2 = axes[1]
    if 'volume' in df.columns:
        ax2.bar(df.index, df['volume'], width=1, color='steelblue', alpha=0.4, label='成交量')
        if not ensemble_anom.empty:
            anom_vol = df.loc[df.index.isin(ensemble_anom.index), 'volume']
            ax2.bar(anom_vol.index, anom_vol.values, width=1, color='red', alpha=0.7, label='异常日成交量')
    ax2.set_ylabel('成交量', fontsize=12)
    ax2.set_xlabel('日期', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / 'anomaly_price_chart.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] {output_dir / 'anomaly_price_chart.png'}")


def plot_anomaly_feature_distributions(
    anomaly_result: pd.DataFrame,
    output_dir: Path,
):
    """绘制异常日 vs 正常日的特征分布对比"""
    features_to_plot = [f for f in DETECTION_FEATURES if f in anomaly_result.columns]
    n_feats = len(features_to_plot)
    if n_feats == 0:
        print("  [警告] 无可绘制特征")
        return

    n_cols = 3
    n_rows = (n_feats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    normal = anomaly_result[anomaly_result['anomaly_ensemble'] == 0]
    anomaly = anomaly_result[anomaly_result['anomaly_ensemble'] == 1]

    for idx, feat in enumerate(features_to_plot):
        ax = axes[idx]

        # 正常分布
        vals_normal = normal[feat].dropna()
        vals_anomaly = anomaly[feat].dropna()

        ax.hist(vals_normal, bins=50, density=True, alpha=0.6,
                color='steelblue', label=f'正常 (n={len(vals_normal)})', edgecolor='white', linewidth=0.3)
        if len(vals_anomaly) > 0:
            ax.hist(vals_anomaly, bins=30, density=True, alpha=0.6,
                    color='red', label=f'异常 (n={len(vals_anomaly)})', edgecolor='white', linewidth=0.3)

        ax.set_title(feat, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 隐藏多余子图
    for idx in range(n_feats, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('异常日 vs 正常日 特征分布对比', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / 'anomaly_feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] {output_dir / 'anomaly_feature_distributions.png'}")


def plot_precursor_roc(precursor_results: Dict, output_dir: Path):
    """绘制前兆分类器 ROC 曲线"""
    if not precursor_results or 'fpr' not in precursor_results:
        print("  [警告] 无前兆分类器结果，跳过 ROC 曲线")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    fpr = precursor_results['fpr']
    tpr = precursor_results['tpr']
    auc = precursor_results['auc']

    ax.plot(fpr, tpr, color='steelblue', linewidth=2,
            label=f'Random Forest (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='随机基线')

    ax.set_xlabel('假阳性率 (FPR)', fontsize=12)
    ax.set_ylabel('真阳性率 (TPR)', fontsize=12)
    ax.set_title('异常前兆分类器 ROC 曲线', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    fig.savefig(output_dir / 'precursor_roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] {output_dir / 'precursor_roc_curve.png'}")


def plot_feature_importance(precursor_results: Dict, output_dir: Path, top_n: int = 20):
    """绘制前兆特征重要性条形图"""
    if not precursor_results or 'feature_importances' not in precursor_results:
        print("  [警告] 无特征重要性数据，跳过")
        return

    importances = precursor_results['feature_importances'].head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))

    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(importances)))
    ax.barh(range(len(importances)), importances.values[::-1],
            color=colors[::-1], edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels(importances.index[::-1], fontsize=9)
    ax.set_xlabel('特征重要性', fontsize=12)
    ax.set_title(f'异常前兆 Top-{top_n} 特征重要性 (Random Forest)', fontsize=13)
    ax.grid(True, alpha=0.3, axis='x')

    fig.savefig(output_dir / 'precursor_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] {output_dir / 'precursor_feature_importance.png'}")


# ============================================================
# 9. 多尺度异常检测
# ============================================================

def multi_scale_anomaly_detection(intervals=None, contamination=0.05) -> Dict:
    """多尺度异常检测"""
    if intervals is None:
        intervals = ['1h', '4h', '1d']

    results = {}
    for interval in intervals:
        try:
            print(f"\n  加载 {interval} 数据进行异常检测...")
            df_tf = load_klines(interval)
            df_tf = add_derived_features(df_tf)

            # 截断大数据
            if len(df_tf) > 50000:
                df_tf = df_tf.iloc[-50000:]

            if len(df_tf) < 200:
                print(f"    {interval} 数据不足，跳过")
                continue

            # 集成异常检测
            anomaly_result = ensemble_anomaly_detection(df_tf, contamination=contamination, min_agreement=2)

            # 提取异常日期
            anomaly_dates = anomaly_result[anomaly_result['anomaly_ensemble'] == 1].index

            results[interval] = {
                'anomaly_dates': anomaly_dates,
                'n_anomalies': len(anomaly_dates),
                'n_total': len(anomaly_result),
                'anomaly_pct': len(anomaly_dates) / len(anomaly_result) * 100,
            }

            print(f"    {interval}: {len(anomaly_dates)} 个异常 ({len(anomaly_dates)/len(anomaly_result)*100:.2f}%)")

        except FileNotFoundError:
            print(f"    {interval} 数据文件不存在，跳过")
        except Exception as e:
            print(f"    {interval} 异常检测失败: {e}")

    return results


def cross_scale_anomaly_consensus(ms_results: Dict, tolerance_hours: int = 24) -> pd.DataFrame:
    """
    跨尺度异常共识：多个尺度在同一时间窗口内同时报异常 → 高置信度

    Parameters
    ----------
    ms_results : Dict
        多尺度异常检测结果字典
    tolerance_hours : int
        时间容差（小时）

    Returns
    -------
    pd.DataFrame
        共识异常数据
    """
    # 将所有尺度的异常日期映射到日频
    all_dates = []
    for interval, result in ms_results.items():
        dates = result['anomaly_dates']
        # 转换为日期（去除时间部分）
        daily_dates = pd.to_datetime(dates.date).unique()
        for date in daily_dates:
            all_dates.append({'date': date, 'interval': interval})

    if not all_dates:
        return pd.DataFrame()

    df_dates = pd.DataFrame(all_dates)

    # 统计每个日期被多少个尺度报为异常
    consensus_counts = df_dates.groupby('date').size().reset_index(name='n_scales')
    consensus_counts = consensus_counts.sort_values('date')

    # >=2 个尺度报异常 = "共识异常"
    consensus_counts['is_consensus'] = (consensus_counts['n_scales'] >= 2).astype(int)

    # 添加参与的尺度列表
    scale_groups = df_dates.groupby('date')['interval'].apply(list).reset_index()
    consensus_counts = consensus_counts.merge(scale_groups, on='date')

    n_consensus = consensus_counts['is_consensus'].sum()
    print(f"\n  跨尺度共识异常: {n_consensus} 天 (≥2 个尺度同时报异常)")

    return consensus_counts


def plot_multi_scale_anomaly_timeline(df: pd.DataFrame, ms_results: Dict, consensus: pd.DataFrame, output_dir: Path):
    """多尺度异常共识时间线"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [2, 1]})

    # 上图: 价格图（对数尺度）+ 共识异常点标注
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], linewidth=0.6, color='steelblue', alpha=0.8, label='BTC 收盘价')

    if not consensus.empty:
        # 标注共识异常点
        consensus_dates = consensus[consensus['is_consensus'] == 1]['date']
        if len(consensus_dates) > 0:
            # 获取对应的价格
            consensus_prices = df.loc[df.index.isin(consensus_dates), 'close']
            if not consensus_prices.empty:
                ax1.scatter(consensus_prices.index, consensus_prices.values,
                           color='red', s=50, zorder=5, label=f'共识异常 (n={len(consensus_prices)})',
                           alpha=0.8, edgecolors='darkred', linewidths=1, marker='*')

    ax1.set_ylabel('价格 (USDT)', fontsize=12)
    ax1.set_title('多尺度异常检测：价格与共识异常', fontsize=14)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # 下图: 各尺度异常时间线（类似甘特图）
    ax2 = axes[1]

    interval_labels = list(ms_results.keys())
    y_positions = range(len(interval_labels))

    colors = {'1h': 'lightcoral', '4h': 'orange', '1d': 'steelblue'}

    for idx, interval in enumerate(interval_labels):
        anomaly_dates = ms_results[interval]['anomaly_dates']
        # 转换为日期
        daily_dates = pd.to_datetime(anomaly_dates.date).unique()

        # 绘制时间线（每个异常日期用竖线表示）
        for date in daily_dates:
            ax2.axvline(x=date, ymin=idx/len(interval_labels), ymax=(idx+0.8)/len(interval_labels),
                       color=colors.get(interval, 'gray'), alpha=0.6, linewidth=2)

    # 标注共识异常区域
    if not consensus.empty:
        consensus_dates = consensus[consensus['is_consensus'] == 1]['date']
        for date in consensus_dates:
            ax2.axvspan(date, date + pd.Timedelta(days=1),
                       color='red', alpha=0.15, zorder=0)

    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(interval_labels)
    ax2.set_ylabel('时间尺度', fontsize=12)
    ax2.set_xlabel('日期', fontsize=12)
    ax2.set_title('各尺度异常时间线（红色背景 = 共识异常）', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_xlim(df.index.min(), df.index.max())

    fig.tight_layout()
    fig.savefig(output_dir / 'anomaly_multi_scale_timeline.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] {output_dir / 'anomaly_multi_scale_timeline.png'}")


# ============================================================
# 7. 结果打印
# ============================================================

def print_anomaly_summary(
    anomaly_result: pd.DataFrame,
    garch_anomaly: Optional[pd.Series],
    precursor_results: Dict,
):
    """打印异常检测汇总"""
    print("\n" + "=" * 70)
    print("异常检测结果汇总")
    print("=" * 70)

    # 集成异常统计
    n_total = len(anomaly_result)
    n_ensemble = anomaly_result['anomaly_ensemble'].sum()
    print(f"\n  总样本数: {n_total}")
    print(f"  集成异常数: {n_ensemble} ({n_ensemble / n_total * 100:.2f}%)")

    # 各方法统计
    method_cols = [c for c in anomaly_result.columns if c.startswith('anomaly_') and c != 'anomaly_ensemble' and c != 'anomaly_votes']
    for col in method_cols:
        method_name = col.replace('anomaly_', '')
        n_anom = anomaly_result[col].sum()
        print(f"    {method_name:>12}: {n_anom} ({n_anom / n_total * 100:.2f}%)")

    # GARCH 异常
    if garch_anomaly is not None:
        n_garch = garch_anomaly.sum()
        print(f"    {'GARCH':>12}: {n_garch} ({n_garch / len(garch_anomaly) * 100:.2f}%)")

        # 集成异常与 GARCH 异常的重叠
        common_idx = anomaly_result.index.intersection(garch_anomaly.index)
        if len(common_idx) > 0:
            ensemble_set = set(anomaly_result.loc[common_idx][anomaly_result.loc[common_idx, 'anomaly_ensemble'] == 1].index)
            garch_set = set(garch_anomaly[garch_anomaly == 1].index)
            overlap = len(ensemble_set & garch_set)
            print(f"\n  集成 ∩ GARCH 重叠: {overlap} 个")

    # 前兆分类器
    if precursor_results and 'auc' in precursor_results:
        print(f"\n  前兆分类器 AUC: {precursor_results['auc']:.4f}")
        print(f"  Top-5 前兆特征:")
        for feat, imp in precursor_results['feature_importances'].head(5).items():
            print(f"    {feat:<40} {imp:.4f}")


# ============================================================
# 8. 主入口
# ============================================================

def run_anomaly_analysis(
    df: pd.DataFrame,
    output_dir: str = "output/anomaly",
) -> Dict:
    """
    异常检测与前兆模式分析主函数

    Parameters
    ----------
    df : pd.DataFrame
        日线数据（已通过 add_derived_features 添加衍生特征）
    output_dir : str
        图表输出目录

    Returns
    -------
    dict
        包含所有分析结果的字典
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BTC 异常检测与前兆模式分析")
    print("=" * 70)
    print(f"数据范围: {df.index.min()} ~ {df.index.max()}")
    print(f"样本数量: {len(df)}")

    from src.font_config import configure_chinese_font
    configure_chinese_font()

    # --- 集成异常检测 ---
    print("\n>>> [1/5] 执行集成异常检测...")
    anomaly_result = ensemble_anomaly_detection(df, contamination=0.05, min_agreement=2)

    # --- GARCH 条件波动率异常 ---
    print("\n>>> [2/5] 执行 GARCH 条件波动率异常检测...")
    garch_anomaly = None
    try:
        garch_anomaly = garch_anomaly_detection(df, threshold=3.0)
    except Exception as e:
        print(f"  [错误] GARCH 异常检测失败: {e}")

    # --- 事件对齐 ---
    print("\n>>> [3/5] 执行事件对齐分析...")
    ensemble_anom_dates = anomaly_result[anomaly_result['anomaly_ensemble'] == 1].index
    event_alignment = align_with_events(ensemble_anom_dates, tolerance_days=5)

    # --- 前兆模式提取 ---
    print("\n>>> [4/5] 提取前兆模式并训练分类器...")
    precursor_results = {}
    try:
        X_precursor, y_precursor = extract_precursor_features(
            df, anomaly_result['anomaly_ensemble'], lookback_windows=[5, 10, 20]
        )
        print(f"  前兆特征矩阵: {X_precursor.shape[0]} 样本 x {X_precursor.shape[1]} 特征")
        precursor_results = train_precursor_classifier(X_precursor, y_precursor)
    except Exception as e:
        print(f"  [错误] 前兆模式提取失败: {e}")

    # --- 可视化 ---
    print("\n>>> [5/5] 生成可视化图表...")
    plot_price_with_anomalies(df, anomaly_result, garch_anomaly, output_dir)
    plot_anomaly_feature_distributions(anomaly_result, output_dir)
    plot_precursor_roc(precursor_results, output_dir)
    plot_feature_importance(precursor_results, output_dir)

    # --- 汇总打印 ---
    print_anomaly_summary(anomaly_result, garch_anomaly, precursor_results)

    # --- 多尺度异常检测 ---
    print("\n>>> [额外] 多尺度异常检测与共识分析...")
    ms_anomaly = multi_scale_anomaly_detection(['1h', '4h', '1d'])
    consensus = None
    if len(ms_anomaly) >= 2:
        consensus = cross_scale_anomaly_consensus(ms_anomaly)
        plot_multi_scale_anomaly_timeline(df, ms_anomaly, consensus, output_dir)

    print("\n" + "=" * 70)
    print("异常检测与前兆模式分析完成！")
    print(f"图表已保存至: {output_dir.resolve()}")
    print("=" * 70)

    return {
        'anomaly_result': anomaly_result,
        'garch_anomaly': garch_anomaly,
        'event_alignment': event_alignment,
        'precursor_results': precursor_results,
        'multi_scale_anomaly': ms_anomaly,
        'cross_scale_consensus': consensus,
    }


# ============================================================
# 独立运行入口
# ============================================================

if __name__ == '__main__':
    from src.data_loader import load_daily
    from src.preprocessing import add_derived_features

    df = load_daily()
    df = add_derived_features(df)
    run_anomaly_analysis(df)
