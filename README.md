# BTC/USDT 价格分析框架

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

一个全面的 BTC/USDT 价格量化分析框架，涵盖 25 个分析维度，从统计分布到分形几何。框架处理 Binance 多时间粒度 K 线数据（1 分钟至月线），时间跨度 2017-08 至 2026-02，生成可复现的研究级可视化图表和统计报告。

## 特性

- **多时间粒度数据管道** — 15 种粒度（1m ~ 1M），统一加载器，含数据校验
- **25 个分析模块** — 各模块独立运行，单模块失败不影响其余模块
- **统计严谨性** — 训练/验证集划分、多重假设检验校正、Bootstrap 置信区间
- **出版级输出** — 53 张图表（支持中文字体）+ 1300 行 Markdown 研究报告
- **模块化架构** — 可一键运行全部模块，也可通过 CLI 参数选择指定模块

## 项目结构

```
btc_price_anany/
├── main.py                 # CLI 入口
├── requirements.txt        # Python 依赖
├── LICENSE                 # MIT 许可证
├── data/                   # 15 个 BTC/USDT K线 CSV（1m ~ 1M）
├── src/                    # 30 个分析与工具模块
│   ├── data_loader.py      # 数据加载与校验
│   ├── preprocessing.py    # 衍生特征工程
│   ├── font_config.py      # 中文字体渲染
│   ├── visualization.py    # 综合仪表盘生成
│   └── ...                 # 26 个分析模块
├── output/                 # 生成的图表（53 张 PNG）
├── docs/
│   └── REPORT.md           # 完整研究报告
└── tests/
    └── test_hurst_15scales.py  # Hurst 指数多尺度测试
```

## 快速开始

### 环境要求

- Python 3.10+
- 约 1 GB 磁盘空间（K 线数据）

### 安装

```bash
git clone https://github.com/riba2534/btc_price_anany.git
cd btc_price_anany
pip install -r requirements.txt
```

### 使用

```bash
# 运行全部 25 个分析模块
python main.py

# 查看可用模块列表
python main.py --list

# 运行指定模块
python main.py --modules fft wavelet hurst

# 限定日期范围
python main.py --start 2020-01-01 --end 2025-12-31
```

## 数据说明

| 文件 | 时间粒度 | 行数（约） |
|------|---------|-----------|
| `btcusdt_1m.csv` | 1 分钟 | ~4,500,000 |
| `btcusdt_3m.csv` | 3 分钟 | ~1,500,000 |
| `btcusdt_5m.csv` | 5 分钟 | ~900,000 |
| `btcusdt_15m.csv` | 15 分钟 | ~300,000 |
| `btcusdt_30m.csv` | 30 分钟 | ~150,000 |
| `btcusdt_1h.csv` | 1 小时 | ~75,000 |
| `btcusdt_2h.csv` | 2 小时 | ~37,000 |
| `btcusdt_4h.csv` | 4 小时 | ~19,000 |
| `btcusdt_6h.csv` | 6 小时 | ~12,500 |
| `btcusdt_8h.csv` | 8 小时 | ~9,500 |
| `btcusdt_12h.csv` | 12 小时 | ~6,300 |
| `btcusdt_1d.csv` | 1 天 | ~3,100 |
| `btcusdt_3d.csv` | 3 天 | ~1,000 |
| `btcusdt_1w.csv` | 1 周 | ~450 |
| `btcusdt_1mo.csv` | 1 月 | ~100 |

全部数据来源于 Binance 公开 API，时间范围 2017-08 至 2026-02。

> **数据未包含在仓库中**，请从 Binance 官方数据源下载后放入 `data/` 目录：
>
> - K 线数据下载页面：<https://data.binance.vision/?prefix=data/spot/daily/klines/BTCUSDT/1m/>
> - 将 URL 中的 `1m` 替换为所需粒度（`3m`、`5m`、`15m`、`30m`、`1h`、`2h`、`4h`、`6h`、`8h`、`12h`、`1d`、`3d`、`1w`、`1mo`）即可下载对应时间粒度的数据
> - 下载后合并为单个 CSV 文件，命名格式：`btcusdt_{interval}.csv`，放入 `data/` 目录

## 分析模块

| 模块 | 说明 |
|------|------|
| `fft` | FFT 功率谱、多时间粒度频谱分析、带通滤波 |
| `wavelet` | 连续小波变换时频图、全局谱、关键周期追踪 |
| `acf` | ACF/PACF 网格分析，自相关结构识别 |
| `returns` | 收益率分布拟合、QQ 图、多尺度矩分析 |
| `volatility` | 波动率聚集、GARCH 建模、杠杆效应量化 |
| `hurst` | R/S 和 DFA Hurst 指数估计、滚动窗口分析 |
| `fractal` | 盒计数维度、Monte Carlo 基准、自相似性检验 |
| `power_law` | 双对数回归、幂律增长通道、模型比较 |
| `volume_price` | 量价散点分析、OBV 背离检测 |
| `calendar` | 星期、月份、小时、季度边界效应 |
| `halving` | 减半周期分析与归一化轨迹对比 |
| `indicators` | 技术指标 IC 检验（训练/验证集划分） |
| `patterns` | K 线形态识别与前瞻收益验证 |
| `clustering` | 市场状态聚类（K-Means、GMM）与转移矩阵 |
| `time_series` | ARIMA、Prophet、LSTM 预测与方向准确率 |
| `causality` | 量价特征间 Granger 因果检验 |
| `anomaly` | 异常检测与前兆特征分析 |
| `microstructure` | 市场微观结构：价差、Kyle's lambda、VPIN |
| `intraday` | 日内交易时段模式与成交量热力图 |
| `scaling` | 统计标度律与峰度衰减 |
| `multiscale_vol` | HAR 波动率、跳跃检测、高阶矩分析 |
| `entropy` | 样本熵与排列熵的多尺度分析 |
| `extreme` | 极端值理论：Hill 估计量、VaR 回测 |
| `cross_tf` | 跨时间粒度相关性与领先滞后分析 |
| `momentum_rev` | 动量 vs 均值回归：方差比率、OU 半衰期 |

## 核心发现

完整分析报告见 [`docs/REPORT.md`](docs/REPORT.md)，主要结论包括：

- **非高斯收益率**：BTC 日收益率呈现显著厚尾（峰度 ~10），Student-t 分布拟合最优，而非高斯分布
- **波动率聚集**：强 GARCH 效应，具有长记忆特征（d ≈ 0.4），波动率持续性跨时间尺度成立
- **Hurst 指数 H ≈ 0.55**：弱但统计显著的长程依赖，短期趋势性向长期均值回归过渡
- **分形维度 D ≈ 1.4**：价格序列比布朗运动更粗糙，呈现多重分形特征
- **减半周期效应**：减半后牛市统计显著，但每轮周期收益递减
- **日历效应**：可检测到微弱的星期和月度季节性；日内模式在扣除交易成本后不具可利用性

## 许可证

本项目基于 [MIT 许可证](LICENSE) 开源。
