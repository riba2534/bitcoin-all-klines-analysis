"""统一可视化工具模块

提供跨模块共用的绑图辅助函数与综合结果仪表盘。
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import warnings

# ── 全局样式 ──────────────────────────────────────────────

STYLE_CONFIG = {
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
}

COLOR_PALETTE = {
    "primary":   "#2563eb",
    "secondary": "#7c3aed",
    "success":   "#059669",
    "danger":    "#dc2626",
    "warning":   "#d97706",
    "info":      "#0891b2",
    "muted":     "#6b7280",
    "bg_light":  "#f8fafc",
}

EVIDENCE_COLORS = {
    "strong":  "#059669",   # 绿
    "moderate": "#d97706",  # 橙
    "weak":    "#dc2626",   # 红
    "none":    "#6b7280",   # 灰
}


def apply_style():
    """应用全局matplotlib样式"""
    plt.rcParams.update(STYLE_CONFIG)
    from src.font_config import configure_chinese_font
    configure_chinese_font()


def ensure_dir(path):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


# ── 证据评分框架 ───────────────────────────────────────────

EVIDENCE_CRITERIA = """
"真正有规律" 判定标准（必须同时满足）：
  1. FDR校正后 p < 0.05
  2. 排列检验 p < 0.01（如适用）
  3. 测试集上效果方向一致且显著
  4. >80% bootstrap子样本中成立（如适用）
  5. Cohen's d > 0.2 或经济意义显著
  6. 有合理的经济/市场直觉解释
"""


def score_evidence(result: Dict) -> Dict:
    """
    对单个分析模块的结果打分

    Parameters
    ----------
    result : dict
        模块返回的结果字典，应包含 'findings' 列表

    Returns
    -------
    dict
        包含 score, level, summary
    """
    findings = result.get("findings", [])
    if not findings:
        return {"score": 0, "level": "none", "summary": "无可评估的发现",
                "n_findings": 0, "total_score": 0, "details": []}

    total_score = 0
    details = []

    for f in findings:
        s = 0
        name = f.get("name", "未命名")
        p_value = f.get("p_value")
        effect_size = f.get("effect_size")
        significant = f.get("significant", False)
        description = f.get("description", "")

        if significant:
            s += 2
        if p_value is not None and p_value < 0.01:
            s += 1
        if effect_size is not None and abs(effect_size) > 0.2:
            s += 1
        if f.get("test_set_consistent", False):
            s += 2
        if f.get("bootstrap_robust", False):
            s += 1

        total_score += s
        details.append({"name": name, "score": s, "description": description})

    avg = total_score / len(findings) if findings else 0

    if avg >= 5:
        level = "strong"
    elif avg >= 3:
        level = "moderate"
    elif avg >= 1:
        level = "weak"
    else:
        level = "none"

    return {
        "score": round(avg, 2),
        "level": level,
        "n_findings": len(findings),
        "total_score": total_score,
        "details": details,
    }


# ── 综合仪表盘 ─────────────────────────────────────────────

def generate_summary_dashboard(all_results: Dict[str, Dict], output_dir: str = "output"):
    """
    生成综合分析仪表盘

    Parameters
    ----------
    all_results : dict
        {module_name: module_result_dict}
    output_dir : str
        输出目录
    """
    apply_style()
    out = ensure_dir(output_dir)

    # ── 1. 汇总各模块证据强度 ──
    summary_rows = []
    for module, result in all_results.items():
        ev = score_evidence(result)
        summary_rows.append({
            "module": module,
            "score": ev["score"],
            "level": ev["level"],
            "n_findings": ev["n_findings"],
            "total_score": ev["total_score"],
        })

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        print("[visualization] 无模块结果可汇总")
        return {}

    summary_df.sort_values("score", ascending=True, inplace=True)

    # ── 2. 证据强度横向柱状图 ──
    fig, ax = plt.subplots(figsize=(10, max(6, len(summary_df) * 0.5)))
    colors = [EVIDENCE_COLORS.get(row["level"], "#6b7280") for _, row in summary_df.iterrows()]
    bars = ax.barh(summary_df["module"], summary_df["score"], color=colors, edgecolor="white", linewidth=0.5)

    for bar, (_, row) in zip(bars, summary_df.iterrows()):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{row["score"]:.1f} ({row["level"]})',
                va='center', fontsize=9)

    ax.set_xlabel("Evidence Score")
    ax.set_title("BTC/USDT Analysis - Evidence Strength by Module")
    ax.axvline(x=3, color="#d97706", linestyle="--", alpha=0.5, label="Moderate threshold")
    ax.axvline(x=5, color="#059669", linestyle="--", alpha=0.5, label="Strong threshold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(out / "evidence_dashboard.png")
    plt.close(fig)

    # ── 3. 综合结论文本报告 ──
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("BTC/USDT 价格规律性分析 — 综合结论报告")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append(EVIDENCE_CRITERIA)
    report_lines.append("")
    report_lines.append("-" * 70)
    report_lines.append(f"{'模块':<30} {'得分':>6} {'强度':>10} {'发现数':>8}")
    report_lines.append("-" * 70)

    for _, row in summary_df.sort_values("score", ascending=False).iterrows():
        report_lines.append(
            f"{row['module']:<30} {row['score']:>6.2f} {row['level']:>10} {row['n_findings']:>8}"
        )

    report_lines.append("-" * 70)
    report_lines.append("")

    # 分级汇总
    strong = summary_df[summary_df["level"] == "strong"]["module"].tolist()
    moderate = summary_df[summary_df["level"] == "moderate"]["module"].tolist()
    weak = summary_df[summary_df["level"] == "weak"]["module"].tolist()
    none_found = summary_df[summary_df["level"] == "none"]["module"].tolist()

    report_lines.append("## 强证据规律（可重复、有经济意义）:")
    if strong:
        for m in strong:
            report_lines.append(f"  * {m}")
    else:
        report_lines.append("  （无）")

    report_lines.append("")
    report_lines.append("## 中等证据规律（统计显著但效果有限）:")
    if moderate:
        for m in moderate:
            report_lines.append(f"  * {m}")
    else:
        report_lines.append("  （无）")

    report_lines.append("")
    report_lines.append("## 弱证据/不显著:")
    for m in weak + none_found:
        report_lines.append(f"  * {m}")

    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("注: 得分基于各模块自报告的统计检验结果。")
    report_lines.append("    具体参数和图表请参见各子目录的输出。")
    report_lines.append("=" * 70)

    report_text = "\n".join(report_lines)

    with open(out / "综合结论报告.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    # ── 4. JSON 格式结果存储 ──
    json_results = {}
    for module, result in all_results.items():
        # 去除不可序列化的对象
        clean = {}
        for k, v in result.items():
            try:
                json.dumps(v)
                clean[k] = v
            except (TypeError, ValueError):
                clean[k] = str(v)
        json_results[module] = clean

    with open(out / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2, default=str)

    print(report_text)

    return {
        "summary_df": summary_df,
        "report_path": str(out / "综合结论报告.txt"),
        "dashboard_path": str(out / "evidence_dashboard.png"),
        "json_path": str(out / "all_results.json"),
    }


def plot_price_overview(df: pd.DataFrame, output_dir: str = "output"):
    """生成价格概览图（对数尺度 + 成交量 + 关键事件标注）"""
    apply_style()
    out = ensure_dir(output_dir)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1],
                                    sharex=True, gridspec_kw={"hspace": 0.05})

    # 价格（对数尺度）
    ax1.semilogy(df.index, df["close"], color=COLOR_PALETTE["primary"], linewidth=0.8)
    ax1.set_ylabel("Price (USDT, log scale)")
    ax1.set_title("BTC/USDT Price & Volume Overview")

    # 标注减半事件
    halvings = [
        ("2020-05-11", "3rd Halving"),
        ("2024-04-20", "4th Halving"),
    ]
    for date_str, label in halvings:
        dt = pd.Timestamp(date_str)
        if df.index.min() <= dt <= df.index.max():
            ax1.axvline(x=dt, color=COLOR_PALETTE["danger"], linestyle="--", alpha=0.6)
            ax1.text(dt, ax1.get_ylim()[1] * 0.9, label, rotation=90,
                     va="top", fontsize=8, color=COLOR_PALETTE["danger"])

    # 成交量
    ax2.bar(df.index, df["volume"], width=1, color=COLOR_PALETTE["info"], alpha=0.5)
    ax2.set_ylabel("Volume")
    ax2.set_xlabel("Date")

    fig.savefig(out / "price_overview.png")
    plt.close(fig)
    print(f"[visualization] 价格概览图 -> {out / 'price_overview.png'}")
