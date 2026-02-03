#!/usr/bin/env python3
"""
测试脚本：验证Hurst分析增强功能
- 15个时间粒度的多尺度分析
- Hurst vs log(Δt) 标度关系图
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.hurst_analysis import multi_timeframe_hurst, plot_multi_timeframe, plot_hurst_vs_scale

def test_15_scales():
    """测试15个时间尺度的Hurst分析"""
    print("=" * 70)
    print("测试15个时间尺度Hurst分析")
    print("=" * 70)

    # 定义全部15个粒度
    ALL_INTERVALS = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1mo']

    print(f"\n将测试以下 {len(ALL_INTERVALS)} 个时间粒度：")
    print(f"  {', '.join(ALL_INTERVALS)}")

    # 执行多时间框架分析
    print("\n开始计算Hurst指数...")
    mt_results = multi_timeframe_hurst(ALL_INTERVALS)

    # 输出结果统计
    print("\n" + "=" * 70)
    print(f"分析完成：成功分析 {len(mt_results)}/{len(ALL_INTERVALS)} 个粒度")
    print("=" * 70)

    if mt_results:
        print("\n各粒度Hurst指数汇总：")
        print("-" * 70)
        for interval, data in mt_results.items():
            print(f"  {interval:5s} | R/S: {data['R/S Hurst']:.4f} | DFA: {data['DFA Hurst']:.4f} | "
                  f"平均: {data['平均Hurst']:.4f} | 数据量: {data['数据量']:>7}")

        # 生成可视化
        output_dir = Path("output/hurst_test")
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print("生成可视化图表...")
        print("=" * 70)

        # 1. 多时间框架对比图
        plot_multi_timeframe(mt_results, output_dir, "test_15scales_comparison.png")

        # 2. Hurst vs 时间尺度标度关系图
        plot_hurst_vs_scale(mt_results, output_dir, "test_hurst_vs_scale.png")

        print(f"\n图表已保存至: {output_dir.resolve()}")
        print("  - test_15scales_comparison.png (15尺度对比柱状图)")
        print("  - test_hurst_vs_scale.png (标度关系图)")
    else:
        print("\n⚠ 警告：没有成功分析任何粒度")

    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)

if __name__ == "__main__":
    try:
        test_15_scales()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
