#!/usr/bin/env python3
"""BTC/USDT 价格规律性全面分析 — 主入口

串联执行所有分析模块，输出结果到 output/ 目录。
每个模块独立运行，单个模块失败不影响其他模块。

用法:
    python3 main.py              # 运行全部模块
    python3 main.py --modules fft wavelet  # 只运行指定模块
    python3 main.py --list       # 列出所有可用模块
"""

import sys
import time
import argparse
import traceback
from pathlib import Path
from collections import OrderedDict

# 确保 src 在路径中
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_klines, load_daily, load_hourly, validate_data
from src.preprocessing import add_derived_features


# ── 模块注册表 ─────────────────────────────────────────────

def _import_module(name):
    """延迟导入分析模块，避免启动时全部加载"""
    import importlib
    return importlib.import_module(f"src.{name}")


# (模块key, 显示名称, 源模块名, 入口函数名, 是否需要hourly数据)
MODULE_REGISTRY = OrderedDict([
    ("fft",          ("FFT频谱分析",           "fft_analysis",          "run_fft_analysis",          False)),
    ("wavelet",      ("小波变换分析",           "wavelet_analysis",      "run_wavelet_analysis",      False)),
    ("acf",          ("ACF/PACF分析",          "acf_analysis",          "run_acf_analysis",          False)),
    ("returns",      ("收益率分布分析",          "returns_analysis",      "run_returns_analysis",      False)),
    ("volatility",   ("波动率聚集分析",          "volatility_analysis",   "run_volatility_analysis",   False)),
    ("hurst",        ("Hurst指数分析",          "hurst_analysis",        "run_hurst_analysis",        False)),
    ("fractal",      ("分形维度分析",            "fractal_analysis",      "run_fractal_analysis",      False)),
    ("power_law",    ("幂律增长分析",            "power_law_analysis",    "run_power_law_analysis",    False)),
    ("volume_price", ("量价关系分析",            "volume_price_analysis", "run_volume_price_analysis", False)),
    ("calendar",     ("日历效应分析",            "calendar_analysis",     "run_calendar_analysis",     True)),
    ("halving",      ("减半周期分析",            "halving_analysis",      "run_halving_analysis",      False)),
    ("indicators",   ("技术指标验证",            "indicators",            "run_indicators_analysis",   False)),
    ("patterns",     ("K线形态分析",             "patterns",              "run_patterns_analysis",     False)),
    ("clustering",   ("市场状态聚类",            "clustering",            "run_clustering_analysis",   False)),
    ("time_series",  ("时序预测",               "time_series",           "run_time_series_analysis",  False)),
    ("causality",    ("因果检验",               "causality",             "run_causality_analysis",    False)),
    ("anomaly",      ("异常检测",               "anomaly",               "run_anomaly_analysis",      False)),
])


OUTPUT_DIR = ROOT / "output"


def run_single_module(key, df, df_hourly, output_base):
    """
    运行单个分析模块

    Returns
    -------
    dict or None
        模块返回的结果字典，失败返回 None
    """
    display_name, mod_name, func_name, needs_hourly = MODULE_REGISTRY[key]
    module_output = str(output_base / key)
    Path(module_output).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  [{key}] {display_name}")
    print(f"{'='*60}")

    try:
        mod = _import_module(mod_name)
        func = getattr(mod, func_name)

        if needs_hourly:
            result = func(df, df_hourly, module_output)
        else:
            result = func(df, module_output)

        if result is None:
            result = {"status": "completed", "findings": []}

        result["status"] = "success"
        print(f"  [{key}] 完成 ✓")
        return result

    except Exception as e:
        print(f"  [{key}] 失败 ✗: {e}")
        traceback.print_exc()
        return {"status": "error", "error": str(e), "findings": []}


def main():
    parser = argparse.ArgumentParser(description="BTC/USDT 价格规律性全面分析")
    parser.add_argument("--modules", nargs="*", default=None,
                        help="指定要运行的模块 (默认运行全部)")
    parser.add_argument("--list", action="store_true",
                        help="列出所有可用模块")
    parser.add_argument("--start", type=str, default=None,
                        help="数据起始日期, 如 2020-01-01")
    parser.add_argument("--end", type=str, default=None,
                        help="数据结束日期, 如 2025-12-31")
    args = parser.parse_args()

    if args.list:
        print("\n可用分析模块:")
        print("-" * 50)
        for key, (name, _, _, _) in MODULE_REGISTRY.items():
            print(f"  {key:<15} {name}")
        print()
        return

    # ── 1. 加载数据 ──────────────────────────────────────
    print("=" * 60)
    print("  BTC/USDT 价格规律性全面分析")
    print("=" * 60)

    print("\n[1/3] 加载日线数据...")
    df_daily = load_daily(start=args.start, end=args.end)
    report = validate_data(df_daily, "1d")
    print(f"  行数: {report['rows']}")
    print(f"  日期范围: {report['date_range']}")
    print(f"  价格范围: {report['price_range']}")

    print("\n[2/3] 添加衍生特征...")
    df = add_derived_features(df_daily)
    print(f"  特征列: {list(df.columns)}")

    print("\n[3/3] 加载小时数据 (日历效应需要)...")
    try:
        df_hourly_raw = load_hourly(start=args.start, end=args.end)
        df_hourly = add_derived_features(df_hourly_raw)
        print(f"  小时数据行数: {len(df_hourly)}")
    except Exception as e:
        print(f"  小时数据加载失败 (日历效应小时分析将跳过): {e}")
        df_hourly = None

    # ── 2. 确定要运行的模块 ──────────────────────────────
    if args.modules:
        modules_to_run = []
        for m in args.modules:
            if m in MODULE_REGISTRY:
                modules_to_run.append(m)
            else:
                print(f"  警告: 未知模块 '{m}', 跳过")
    else:
        modules_to_run = list(MODULE_REGISTRY.keys())

    print(f"\n将运行 {len(modules_to_run)} 个分析模块:")
    for m in modules_to_run:
        print(f"  - {m}: {MODULE_REGISTRY[m][0]}")

    # ── 3. 逐一运行模块 ─────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    timings = {}

    for key in modules_to_run:
        t0 = time.time()
        result = run_single_module(key, df, df_hourly, OUTPUT_DIR)
        elapsed = time.time() - t0
        timings[key] = elapsed
        if result is not None:
            all_results[key] = result
        print(f"  耗时: {elapsed:.1f}s")

    # ── 4. 生成综合报告 ──────────────────────────────────
    print(f"\n{'='*60}")
    print("  生成综合分析报告")
    print(f"{'='*60}")

    from src.visualization import generate_summary_dashboard, plot_price_overview

    # 价格概览图
    plot_price_overview(df_daily, str(OUTPUT_DIR))

    # 综合仪表盘
    dashboard_result = generate_summary_dashboard(all_results, str(OUTPUT_DIR))

    # ── 5. 打印执行摘要 ──────────────────────────────────
    print(f"\n{'='*60}")
    print("  执行摘要")
    print(f"{'='*60}")

    success = sum(1 for r in all_results.values() if r.get("status") == "success")
    failed = sum(1 for r in all_results.values() if r.get("status") == "error")
    total_time = sum(timings.values())

    print(f"\n  模块总数: {len(modules_to_run)}")
    print(f"  成功: {success}")
    print(f"  失败: {failed}")
    print(f"  总耗时: {total_time:.1f}s")

    print(f"\n  各模块耗时:")
    for key, t in sorted(timings.items(), key=lambda x: -x[1]):
        status = all_results.get(key, {}).get("status", "unknown")
        mark = "✓" if status == "success" else "✗"
        print(f"    {mark} {key:<15} {t:>8.1f}s")

    print(f"\n  输出目录: {OUTPUT_DIR.resolve()}")
    if dashboard_result:
        print(f"  综合报告: {dashboard_result.get('report_path', 'N/A')}")
        print(f"  仪表盘图: {dashboard_result.get('dashboard_path', 'N/A')}")
        print(f"  JSON结果: {dashboard_result.get('json_path', 'N/A')}")

    print(f"\n{'='*60}")
    print("  分析完成!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
