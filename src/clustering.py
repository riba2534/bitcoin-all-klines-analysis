"""市场状态聚类与马尔可夫链分析模块

基于K-Means、GMM、HDBSCAN对BTC日线特征进行聚类，
构建状态转移矩阵并计算平稳分布。
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional, Tuple, Dict, List

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    warnings.warn("hdbscan 未安装，将跳过 HDBSCAN 聚类。pip install hdbscan")


# ============================================================
# 特征工程
# ============================================================

FEATURE_COLS = [
    "log_return", "abs_return", "vol_7d", "vol_30d",
    "volume_ratio", "taker_buy_ratio", "range_pct", "body_pct",
    "log_return_lag1", "log_return_lag2",
]


def _prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    """
    准备聚类特征：添加滞后收益率、标准化、去除NaN行

    Returns
    -------
    df_clean : 清洗后的DataFrame（保留索引用于后续映射）
    X_scaled : 标准化后的特征矩阵
    scaler   : 标准化器（可用于逆变换）
    """
    out = df.copy()

    # 添加滞后收益率特征
    out["log_return_lag1"] = out["log_return"].shift(1)
    out["log_return_lag2"] = out["log_return"].shift(2)

    # 只保留所需特征列，删除含NaN的行
    df_feat = out[FEATURE_COLS].copy()
    mask = df_feat.notna().all(axis=1)
    df_clean = out.loc[mask].copy()
    X_raw = df_feat.loc[mask].values

    # Z-score标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    print(f"[特征准备] 有效样本数: {X_scaled.shape[0]}, 特征维度: {X_scaled.shape[1]}")
    return df_clean, X_scaled, scaler


# ============================================================
# K-Means 聚类
# ============================================================

def _run_kmeans(X: np.ndarray, k_range: List[int] = None) -> Tuple[int, np.ndarray, Dict]:
    """
    K-Means聚类，通过轮廓系数选择最优k

    Returns
    -------
    best_k : 最优聚类数
    labels : 最优k对应的聚类标签
    info   : 包含每个k的轮廓系数、惯性等
    """
    if k_range is None:
        k_range = [3, 4, 5, 6, 7]

    results = {}
    best_score = -1
    best_k = k_range[0]
    best_labels = None

    print("\n" + "=" * 60)
    print("K-Means 聚类分析")
    print("=" * 60)

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, max_iter=500, random_state=42)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        inertia = km.inertia_
        results[k] = {"silhouette": sil, "inertia": inertia, "labels": labels, "model": km}
        print(f"  k={k}: 轮廓系数={sil:.4f}, 惯性={inertia:.1f}")

        if sil > best_score:
            best_score = sil
            best_k = k
            best_labels = labels

    print(f"\n  >>> 最优 k = {best_k} (轮廓系数 = {best_score:.4f})")
    return best_k, best_labels, results


# ============================================================
# GMM (高斯混合模型)
# ============================================================

def _run_gmm(X: np.ndarray, k_range: List[int] = None) -> Tuple[int, np.ndarray, Dict]:
    """
    GMM聚类，通过BIC选择最优组件数

    Returns
    -------
    best_k : BIC最低的组件数
    labels : 对应的聚类标签
    info   : 每个k的BIC、AIC、标签等
    """
    if k_range is None:
        k_range = [3, 4, 5, 6, 7]

    results = {}
    best_bic = np.inf
    best_k = k_range[0]
    best_labels = None

    print("\n" + "=" * 60)
    print("GMM (高斯混合模型) 聚类分析")
    print("=" * 60)

    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type='full',
                               n_init=5, max_iter=500, random_state=42)
        gmm.fit(X)
        labels = gmm.predict(X)
        bic = gmm.bic(X)
        aic = gmm.aic(X)
        sil = silhouette_score(X, labels)
        results[k] = {"bic": bic, "aic": aic, "silhouette": sil,
                       "labels": labels, "model": gmm}
        print(f"  k={k}: BIC={bic:.1f}, AIC={aic:.1f}, 轮廓系数={sil:.4f}")

        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_labels = labels

    print(f"\n  >>> 最优 k = {best_k} (BIC = {best_bic:.1f})")
    return best_k, best_labels, results


# ============================================================
# HDBSCAN (密度聚类)
# ============================================================

def _run_hdbscan(X: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    HDBSCAN密度聚类

    Returns
    -------
    labels : 聚类标签 (-1表示噪声)
    info   : 聚类统计信息
    """
    if not HAS_HDBSCAN:
        print("\n[HDBSCAN] 跳过 - hdbscan 未安装")
        return None, {}

    print("\n" + "=" * 60)
    print("HDBSCAN 密度聚类分析")
    print("=" * 60)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=30,
        min_samples=10,
        metric='euclidean',
        cluster_selection_method='eom',
    )
    labels = clusterer.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    noise_pct = n_noise / len(labels) * 100

    info = {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_pct": noise_pct,
        "labels": labels,
        "model": clusterer,
    }

    print(f"  聚类数: {n_clusters}")
    print(f"  噪声点: {n_noise} ({noise_pct:.1f}%)")

    # 排除噪声点后计算轮廓系数
    if n_clusters >= 2:
        mask = labels >= 0
        if mask.sum() > n_clusters:
            sil = silhouette_score(X[mask], labels[mask])
            info["silhouette"] = sil
            print(f"  轮廓系数(去噪): {sil:.4f}")

    return labels, info


# ============================================================
# 聚类解释与标签映射
# ============================================================

# 状态标签定义
STATE_LABELS = {
    "sideways": "横盘整理",
    "mild_up": "温和上涨",
    "mild_down": "温和下跌",
    "surge": "强势上涨",
    "crash": "急剧下跌",
    "high_vol": "高波动",
    "low_vol": "低波动",
}


def _interpret_clusters(df_clean: pd.DataFrame, labels: np.ndarray,
                        method_name: str = "K-Means") -> pd.DataFrame:
    """
    解释聚类结果：计算每个簇的特征均值，并自动标注状态名称

    Returns
    -------
    cluster_desc : 每个聚类的特征均值表 + state_label列
    """
    df_work = df_clean.copy()
    col_name = f"cluster_{method_name}"
    df_work[col_name] = labels

    # 计算每个聚类的特征均值
    cluster_means = df_work.groupby(col_name)[FEATURE_COLS].mean()

    print(f"\n{'=' * 60}")
    print(f"{method_name} 聚类特征均值")
    print("=" * 60)

    # 自动标注状态（基于数据分布的自适应阈值）
    state_labels = {}

    # 计算自适应阈值：基于聚类均值的标准差
    lr_values = cluster_means["log_return"]
    abs_r_values = cluster_means["abs_return"]
    lr_std = lr_values.std() if len(lr_values) > 1 else 0.02
    abs_r_std = abs_r_values.std() if len(abs_r_values) > 1 else 0.02
    high_lr_threshold = max(0.005, lr_std)  # 至少 0.5% 作为下限
    high_abs_threshold = max(0.005, abs_r_std)
    mild_lr_threshold = max(0.002, high_lr_threshold * 0.25)

    for cid in cluster_means.index:
        row = cluster_means.loc[cid]
        lr = row["log_return"]
        vol = row["vol_7d"]
        abs_r = row["abs_return"]

        # 基于自适应阈值的规则判断
        if lr > high_lr_threshold and abs_r > high_abs_threshold:
            label = "surge"
        elif lr < -high_lr_threshold and abs_r > high_abs_threshold:
            label = "crash"
        elif lr > mild_lr_threshold:
            label = "mild_up"
        elif lr < -mild_lr_threshold:
            label = "mild_down"
        elif abs_r > high_abs_threshold * 0.75 or vol > cluster_means["vol_7d"].median() * 1.5:
            label = "high_vol"
        else:
            label = "sideways"

        state_labels[cid] = label

    cluster_means["state_label"] = pd.Series(state_labels)
    cluster_means["state_cn"] = cluster_means["state_label"].map(STATE_LABELS)

    # 统计每个聚类的样本数和占比
    counts = df_work[col_name].value_counts().sort_index()
    cluster_means["count"] = counts
    cluster_means["pct"] = (counts / counts.sum() * 100).round(1)

    for cid in cluster_means.index:
        row = cluster_means.loc[cid]
        print(f"\n  聚类 {cid} [{row['state_cn']}] (n={int(row['count'])}, {row['pct']:.1f}%)")
        print(f"    log_return: {row['log_return']:.5f}, abs_return: {row['abs_return']:.5f}")
        print(f"    vol_7d: {row['vol_7d']:.4f}, vol_30d: {row['vol_30d']:.4f}")
        print(f"    volume_ratio: {row['volume_ratio']:.3f}, taker_buy_ratio: {row['taker_buy_ratio']:.4f}")
        print(f"    range_pct: {row['range_pct']:.5f}, body_pct: {row['body_pct']:.5f}")

    return cluster_means


# ============================================================
# 马尔可夫转移矩阵
# ============================================================

def _compute_transition_matrix(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算状态转移概率矩阵、平稳分布和平均持有时间

    Parameters
    ----------
    labels : 时间序列的聚类标签

    Returns
    -------
    trans_matrix : 转移概率矩阵 (n_states x n_states)
    stationary   : 平稳分布向量
    holding_time : 各状态平均持有时间
    """
    states = np.sort(np.unique(labels))
    n_states = len(states)

    # 状态映射到连续索引
    state_to_idx = {s: i for i, s in enumerate(states)}

    # 计数矩阵
    count_matrix = np.zeros((n_states, n_states), dtype=np.float64)
    for t in range(len(labels) - 1):
        i = state_to_idx[labels[t]]
        j = state_to_idx[labels[t + 1]]
        count_matrix[i, j] += 1

    # 转移概率矩阵（行归一化）
    row_sums = count_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除零
    trans_matrix = count_matrix / row_sums

    # 平稳分布：求转移矩阵的左特征向量（特征值=1对应的）
    # π * P = π  =>  P^T * π^T = π^T
    eigenvalues, eigenvectors = np.linalg.eig(trans_matrix.T)

    # 找最接近1的特征值对应的特征向量
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    stationary = np.real(eigenvectors[:, idx])
    stationary = stationary / stationary.sum()  # 归一化为概率

    # 确保非负（数值误差可能导致微小负值）
    stationary = np.abs(stationary)
    stationary = stationary / stationary.sum()

    # 平均持有时间 = 1 / (1 - p_ii)
    diag = np.diag(trans_matrix)
    holding_time = np.where(diag < 1.0, 1.0 / (1.0 - diag), np.inf)

    return trans_matrix, stationary, holding_time


def _print_markov_results(trans_matrix: np.ndarray, stationary: np.ndarray,
                          holding_time: np.ndarray, cluster_desc: pd.DataFrame):
    """打印马尔可夫链分析结果"""
    states = cluster_desc.index.tolist()
    state_names = cluster_desc["state_cn"].tolist()

    print("\n" + "=" * 60)
    print("马尔可夫链状态转移分析")
    print("=" * 60)

    # 转移概率矩阵
    print("\n转移概率矩阵:")
    header = "      " + "  ".join([f"  {state_names[j][:4]:>4s}" for j in range(len(states))])
    print(header)
    for i, s in enumerate(states):
        row_str = f"  {state_names[i][:4]:>4s}"
        for j in range(len(states)):
            row_str += f"  {trans_matrix[i, j]:6.3f}"
        print(row_str)

    # 平稳分布
    print("\n平稳分布 (长期均衡概率):")
    for i, s in enumerate(states):
        print(f"  {state_names[i]}: {stationary[i]:.4f} ({stationary[i]*100:.1f}%)")

    # 平均持有时间
    print("\n平均持有时间 (天):")
    for i, s in enumerate(states):
        if np.isinf(holding_time[i]):
            print(f"  {state_names[i]}: ∞ (吸收态)")
        else:
            print(f"  {state_names[i]}: {holding_time[i]:.2f} 天")


# ============================================================
# 可视化
# ============================================================

def _plot_pca_scatter(X: np.ndarray, labels: np.ndarray,
                      cluster_desc: pd.DataFrame, method_name: str,
                      output_dir: Path):
    """2D PCA散点图，按聚类着色"""
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(12, 8))
    states = np.sort(np.unique(labels))
    colors = plt.cm.Set2(np.linspace(0, 1, len(states)))

    for i, s in enumerate(states):
        mask = labels == s
        label_name = cluster_desc.loc[s, "state_cn"] if s in cluster_desc.index else f"Cluster {s}"
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors[i]], label=label_name,
                   alpha=0.5, s=15, edgecolors='none')

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
    ax.set_title(f"{method_name} 聚类结果 - PCA 2D投影", fontsize=14)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / f"cluster_pca_{method_name.lower().replace(' ', '_')}.png",
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] cluster_pca_{method_name.lower().replace(' ', '_')}.png")


def _plot_silhouette(X: np.ndarray, labels: np.ndarray, method_name: str, output_dir: Path):
    """轮廓系数分析图"""
    n_clusters = len(set(labels) - {-1})
    if n_clusters < 2:
        return

    # 排除噪声点
    mask = labels >= 0
    if mask.sum() < n_clusters + 1:
        return

    sil_vals = silhouette_samples(X[mask], labels[mask])
    avg_sil = silhouette_score(X[mask], labels[mask])

    fig, ax = plt.subplots(figsize=(10, 7))
    y_lower = 10
    valid_labels = np.sort(np.unique(labels[mask]))
    colors = plt.cm.Set2(np.linspace(0, 1, len(valid_labels)))

    for i, c in enumerate(valid_labels):
        c_sil = sil_vals[labels[mask] == c]
        c_sil.sort()
        size = c_sil.shape[0]
        y_upper = y_lower + size

        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, c_sil,
                         facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size, str(c), fontsize=10)
        y_lower = y_upper + 10

    ax.axvline(x=avg_sil, color="red", linestyle="--", label=f"平均={avg_sil:.3f}")
    ax.set_xlabel("轮廓系数", fontsize=12)
    ax.set_ylabel("聚类标签", fontsize=12)
    ax.set_title(f"{method_name} 轮廓系数分析 (平均={avg_sil:.3f})", fontsize=14)
    ax.legend(fontsize=10)

    fig.savefig(output_dir / f"cluster_silhouette_{method_name.lower().replace(' ', '_')}.png",
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] cluster_silhouette_{method_name.lower().replace(' ', '_')}.png")


def _plot_cluster_heatmap(cluster_desc: pd.DataFrame, method_name: str, output_dir: Path):
    """聚类特征热力图"""
    # 只选择数值型特征列
    feat_cols = [c for c in FEATURE_COLS if c in cluster_desc.columns]
    data = cluster_desc[feat_cols].copy()

    # 对每列进行Z-score标准化（便于比较不同量纲的特征）
    data_norm = (data - data.mean()) / (data.std() + 1e-10)

    fig, ax = plt.subplots(figsize=(14, max(6, len(data) * 1.2)))

    # 行标签用中文状态名
    row_labels = [f"{idx}-{cluster_desc.loc[idx, 'state_cn']}" for idx in data.index]

    im = ax.imshow(data_norm.values, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(feat_cols)))
    ax.set_xticklabels(feat_cols, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=11)

    # 在格子中显示原始数值
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.iloc[i, j]
            ax.text(j, i, f"{val:.4f}", ha='center', va='center', fontsize=8,
                    color='black' if abs(data_norm.iloc[i, j]) < 1.5 else 'white')

    plt.colorbar(im, ax=ax, shrink=0.8, label="标准化值")
    ax.set_title(f"{method_name} 各聚类特征热力图", fontsize=14)

    fig.savefig(output_dir / f"cluster_heatmap_{method_name.lower().replace(' ', '_')}.png",
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] cluster_heatmap_{method_name.lower().replace(' ', '_')}.png")


def _plot_transition_heatmap(trans_matrix: np.ndarray, cluster_desc: pd.DataFrame,
                             output_dir: Path):
    """状态转移概率矩阵热力图"""
    state_names = [cluster_desc.loc[idx, "state_cn"] for idx in cluster_desc.index]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(trans_matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')

    n = len(state_names)
    ax.set_xticks(range(n))
    ax.set_xticklabels(state_names, rotation=45, ha='right', fontsize=11)
    ax.set_yticks(range(n))
    ax.set_yticklabels(state_names, fontsize=11)

    # 标注概率值
    for i in range(n):
        for j in range(n):
            color = 'white' if trans_matrix[i, j] > 0.5 else 'black'
            ax.text(j, i, f"{trans_matrix[i, j]:.3f}", ha='center', va='center',
                    fontsize=11, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, shrink=0.8, label="转移概率")
    ax.set_xlabel("下一状态", fontsize=12)
    ax.set_ylabel("当前状态", fontsize=12)
    ax.set_title("马尔可夫状态转移概率矩阵", fontsize=14)

    fig.savefig(output_dir / "cluster_transition_matrix.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] cluster_transition_matrix.png")


def _plot_state_timeseries(df_clean: pd.DataFrame, labels: np.ndarray,
                           cluster_desc: pd.DataFrame, output_dir: Path):
    """状态随时间变化的时间序列图"""
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), height_ratios=[2, 1], sharex=True)

    dates = df_clean.index
    close = df_clean["close"].values

    states = np.sort(np.unique(labels))
    colors = plt.cm.Set2(np.linspace(0, 1, len(states)))
    color_map = {s: colors[i] for i, s in enumerate(states)}

    # 上图：价格走势，按状态着色
    ax1 = axes[0]
    for i in range(len(dates) - 1):
        ax1.plot([dates[i], dates[i + 1]], [close[i], close[i + 1]],
                 color=color_map[labels[i]], linewidth=0.8)

    # 添加图例
    from matplotlib.patches import Patch
    legend_patches = []
    for s in states:
        name = cluster_desc.loc[s, "state_cn"] if s in cluster_desc.index else f"Cluster {s}"
        legend_patches.append(Patch(color=color_map[s], label=name))
    ax1.legend(handles=legend_patches, fontsize=9, loc='upper left')
    ax1.set_ylabel("BTC 价格 (USDT)", fontsize=12)
    ax1.set_title("BTC 价格与市场状态时间序列", fontsize=14)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # 下图：状态标签时间线
    ax2 = axes[1]
    state_colors = [color_map[l] for l in labels]
    ax2.bar(dates, np.ones(len(dates)), color=state_colors, width=1.5, edgecolor='none')
    ax2.set_yticks([])
    ax2.set_ylabel("市场状态", fontsize=12)
    ax2.set_xlabel("日期", fontsize=12)

    plt.tight_layout()
    fig.savefig(output_dir / "cluster_state_timeseries.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] cluster_state_timeseries.png")


def _plot_kmeans_selection(kmeans_results: Dict, gmm_results: Dict, output_dir: Path):
    """K选择对比图：轮廓系数 + BIC"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. K-Means 轮廓系数
    ks_km = sorted(kmeans_results.keys())
    sils_km = [kmeans_results[k]["silhouette"] for k in ks_km]
    axes[0].plot(ks_km, sils_km, 'bo-', linewidth=2, markersize=8)
    best_k_km = ks_km[np.argmax(sils_km)]
    axes[0].axvline(x=best_k_km, color='red', linestyle='--', alpha=0.7)
    axes[0].set_xlabel("k", fontsize=12)
    axes[0].set_ylabel("轮廓系数", fontsize=12)
    axes[0].set_title("K-Means 轮廓系数", fontsize=13)
    axes[0].grid(True, alpha=0.3)

    # 2. K-Means 惯性 (Elbow)
    inertias = [kmeans_results[k]["inertia"] for k in ks_km]
    axes[1].plot(ks_km, inertias, 'gs-', linewidth=2, markersize=8)
    axes[1].set_xlabel("k", fontsize=12)
    axes[1].set_ylabel("惯性 (Inertia)", fontsize=12)
    axes[1].set_title("K-Means 肘部法则", fontsize=13)
    axes[1].grid(True, alpha=0.3)

    # 3. GMM BIC
    ks_gmm = sorted(gmm_results.keys())
    bics = [gmm_results[k]["bic"] for k in ks_gmm]
    axes[2].plot(ks_gmm, bics, 'r^-', linewidth=2, markersize=8)
    best_k_gmm = ks_gmm[np.argmin(bics)]
    axes[2].axvline(x=best_k_gmm, color='blue', linestyle='--', alpha=0.7)
    axes[2].set_xlabel("k", fontsize=12)
    axes[2].set_ylabel("BIC", fontsize=12)
    axes[2].set_title("GMM BIC 选择", fontsize=13)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "cluster_k_selection.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] cluster_k_selection.png")


# ============================================================
# 主入口
# ============================================================

def run_clustering_analysis(df: pd.DataFrame, output_dir: "str | Path" = "output/clustering") -> Dict:
    """
    市场状态聚类与马尔可夫链分析 - 主入口

    Parameters
    ----------
    df : pd.DataFrame
        已经通过 add_derived_features() 添加了衍生特征的日线数据
    output_dir : str or Path
        图表输出目录

    Returns
    -------
    results : dict
        包含聚类结果、转移矩阵、平稳分布等
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.font_config import configure_chinese_font
    configure_chinese_font()

    print("=" * 60)
    print("  BTC 市场状态聚类与马尔可夫链分析")
    print("=" * 60)

    # ---- 1. 特征准备 ----
    df_clean, X_scaled, scaler = _prepare_features(df)

    # ---- 2. K-Means 聚类 ----
    best_k_km, km_labels, kmeans_results = _run_kmeans(X_scaled)

    # ---- 3. GMM 聚类 ----
    best_k_gmm, gmm_labels, gmm_results = _run_gmm(X_scaled)

    # ---- 4. HDBSCAN 聚类 ----
    hdbscan_labels, hdbscan_info = _run_hdbscan(X_scaled)

    # ---- 5. K选择对比图 ----
    print("\n[可视化] 生成K选择对比图...")
    _plot_kmeans_selection(kmeans_results, gmm_results, output_dir)

    # ---- 6. K-Means 聚类解释 ----
    km_desc = _interpret_clusters(df_clean, km_labels, "K-Means")

    # ---- 7. GMM 聚类解释 ----
    gmm_desc = _interpret_clusters(df_clean, gmm_labels, "GMM")

    # ---- 8. 马尔可夫链分析（基于K-Means结果）----
    trans_matrix, stationary, holding_time = _compute_transition_matrix(km_labels)
    _print_markov_results(trans_matrix, stationary, holding_time, km_desc)

    # ---- 9. 可视化 ----
    print("\n[可视化] 生成分析图表...")

    # PCA散点图
    _plot_pca_scatter(X_scaled, km_labels, km_desc, "K-Means", output_dir)
    _plot_pca_scatter(X_scaled, gmm_labels, gmm_desc, "GMM", output_dir)
    if hdbscan_labels is not None and hdbscan_info.get("n_clusters", 0) >= 2:
        # 为HDBSCAN创建简易描述
        hdb_states = np.sort(np.unique(hdbscan_labels[hdbscan_labels >= 0]))
        hdb_desc = _interpret_clusters(df_clean, hdbscan_labels, "HDBSCAN")
        _plot_pca_scatter(X_scaled, hdbscan_labels, hdb_desc, "HDBSCAN", output_dir)

    # 轮廓系数图
    _plot_silhouette(X_scaled, km_labels, "K-Means", output_dir)

    # 聚类特征热力图
    _plot_cluster_heatmap(km_desc, "K-Means", output_dir)
    _plot_cluster_heatmap(gmm_desc, "GMM", output_dir)

    # 转移矩阵热力图
    _plot_transition_heatmap(trans_matrix, km_desc, output_dir)

    # 状态时间序列图
    _plot_state_timeseries(df_clean, km_labels, km_desc, output_dir)

    # ---- 10. 汇总结果 ----
    results = {
        "kmeans": {
            "best_k": best_k_km,
            "labels": km_labels,
            "cluster_desc": km_desc,
            "all_results": kmeans_results,
        },
        "gmm": {
            "best_k": best_k_gmm,
            "labels": gmm_labels,
            "cluster_desc": gmm_desc,
            "all_results": gmm_results,
        },
        "hdbscan": {
            "labels": hdbscan_labels,
            "info": hdbscan_info,
        },
        "markov": {
            "transition_matrix": trans_matrix,
            "stationary_distribution": stationary,
            "holding_time": holding_time,
        },
        "features": {
            "df_clean": df_clean,
            "X_scaled": X_scaled,
            "scaler": scaler,
        },
    }

    print("\n" + "=" * 60)
    print("  聚类与马尔可夫链分析完成！")
    print("=" * 60)

    return results


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    from data_loader import load_daily
    from preprocessing import add_derived_features

    df = load_daily()
    df = add_derived_features(df)

    results = run_clustering_analysis(df, output_dir="output/clustering")
