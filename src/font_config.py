"""
统一 matplotlib 中文字体配置。

所有绘图模块在创建图表前应调用 configure_chinese_font()。
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_configured = False

# 按优先级排列的中文字体候选列表
_CHINESE_FONT_CANDIDATES = [
    'Noto Sans SC',       # Google 思源黑体（最佳渲染质量）
    'Hiragino Sans GB',   # macOS 系统自带
    'STHeiti',            # macOS 系统自带
    'Arial Unicode MS',   # macOS/Windows 通用
    'SimHei',             # Windows 黑体
    'WenQuanYi Micro Hei',  # Linux 文泉驿
    'DejaVu Sans',        # 最终回退（不支持中文，但不会崩溃）
]


def _find_available_chinese_fonts():
    """检测系统中实际可用的中文字体。"""
    available = []
    for font_name in _CHINESE_FONT_CANDIDATES:
        try:
            path = fm.findfont(
                fm.FontProperties(family=font_name),
                fallback_to_default=False
            )
            if path and 'LastResort' not in path:
                available.append(font_name)
        except Exception:
            continue
    return available if available else ['DejaVu Sans']


def configure_chinese_font():
    """
    配置 matplotlib 使用中文字体。

    - 自动检测系统可用的中文字体
    - 设置 sans-serif 字体族
    - 修复负号显示问题
    - 仅在首次调用时执行，后续调用为空操作
    """
    global _configured
    if _configured:
        return

    available = _find_available_chinese_fonts()

    plt.rcParams['font.sans-serif'] = available
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'sans-serif'

    _configured = True
