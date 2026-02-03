# BTC 全数据深度分析扩展计划

## 目标
充分利用全部 15 个 K 线数据文件（1m~1mo），新增 8 个分析模块 + 增强 5 个现有模块，覆盖目前完全未触及的分钟级微观结构、多尺度统计标度律、极端风险等领域。

---

## 一、新增 8 个分析模块

### 1. `microstructure.py` — 市场微观结构分析
**使用数据**: 1m, 3m, 5m
- Roll 价差估计（基于收盘价序列相关性）
- Corwin-Schultz 高低价价差估计
- Kyle's Lambda（价格冲击系数）
- Amihud 非流动性比率
- VPIN（基于成交量同步的知情交易概率）
- 图表: 价差时序、流动性热力图、VPIN 预警图

### 2. `intraday_patterns.py` — 日内模式分析
**使用数据**: 1m, 5m, 15m, 30m, 1h
- 日内成交量 U 型曲线（按小时/分钟聚合）
- 日内波动率微笑模式
- 亚洲/欧洲/美洲交易时段对比
- 日内收益率自相关结构
- 图表: 时段热力图、成交量/波动率日内模式、三时区对比

### 3. `scaling_laws.py` — 统计标度律分析
**使用数据**: 全部 15 个文件
- 波动率标度: σ(Δt) ∝ (Δt)^H，拟合 H 指数
- Taylor 效应: |r|^q 的自相关衰减与 q 的关系
- 收益率聚合特性（正态化速度）
- Epps 效应（高频相关性衰减）
- 图表: 标度律拟合、Taylor 效应矩阵、正态性 vs 时间尺度

### 4. `multi_scale_vol.py` — 多尺度已实现波动率
**使用数据**: 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d
- 已实现波动率 (RV) 在各尺度上的计算
- 波动率签名图 (Volatility Signature Plot)
- HAR-RV 模型 (Corsi 2009) — 用 5m RV 预测日/周/月 RV
- 多尺度波动率溢出 (Diebold-Yilmaz)
- 图表: 签名图、HAR-RV 拟合、波动率溢出网络

### 5. `entropy_analysis.py` — 信息熵分析
**使用数据**: 1m, 5m, 15m, 1h, 4h, 1d
- Shannon 熵跨时间尺度比较
- 样本熵 (SampEn) / 近似熵 (ApEn)
- 排列熵 (Permutation Entropy) 多尺度
- 转移熵 (Transfer Entropy) — 时间尺度间信息流方向
- 图表: 熵 vs 时间尺度、滚动熵时序、信息流向图

### 6. `extreme_value.py` — 极端值与尾部风险
**使用数据**: 1h, 4h, 1d, 1w
- 广义极值分布 (GEV) 区组极大值拟合
- 广义 Pareto 分布 (GPD) 超阈值拟合
- 多尺度 VaR / CVaR 计算
- 尾部指数估计 (Hill estimator)
- 极端事件聚集检验
- 图表: 尾部拟合 QQ 图、VaR 回测、尾部指数时序

### 7. `cross_timeframe.py` — 跨时间尺度关联分析
**使用数据**: 5m, 15m, 1h, 4h, 1d, 1w
- 跨尺度收益率相关矩阵
- Lead-lag 领先/滞后关系检测
- 多尺度 Granger 因果检验
- 信息流方向（粗粒度 → 细粒度 or 反向？）
- 图表: 跨尺度相关热力图、领先滞后矩阵、信息流向图

### 8. `momentum_reversion.py` — 动量与均值回归多尺度检验
**使用数据**: 1m, 5m, 15m, 1h, 4h, 1d, 1w, 1mo
- 各尺度收益率自相关符号分析
- 方差比检验 (Lo-MacKinlay)
- 均值回归半衰期 (Ornstein-Uhlenbeck 拟合)
- 动量/反转盈利能力回测
- 图表: 方差比 vs 尺度、自相关衰减、策略 PnL 对比

---

## 二、增强 5 个现有模块

### 9. `fft_analysis.py` 增强
- 当前: 仅用 4h, 1d, 1w
- 扩展: 加入 1m, 5m, 15m, 30m, 1h, 2h, 6h, 8h, 12h, 3d, 1mo
- 新增: 全 15 尺度频谱瀑布图

### 10. `hurst_analysis.py` 增强
- 当前: 仅用 1h, 4h, 1d, 1w
- 扩展: 全部 15 个粒度的 Hurst 指数
- 新增: Hurst 指数 vs 时间尺度的标度关系图

### 11. `returns_analysis.py` 增强
- 当前: 仅用 1h, 4h, 1d, 1w
- 扩展: 加入 1m, 5m, 15m, 30m, 2h, 6h, 8h, 12h, 3d, 1mo
- 新增: 峰度/偏度 vs 时间尺度图，正态化收敛速度

### 12. `acf_analysis.py` 增强
- 当前: 仅用 1d
- 扩展: 加入 1h, 4h, 1w 的 ACF/PACF 多尺度对比
- 新增: 自相关衰减速度 vs 时间尺度

### 13. `volatility_analysis.py` 增强
- 当前: 仅用 1d
- 扩展: 加入 5m, 1h, 4h 的波动率聚集分析
- 新增: 波动率长记忆参数 d vs 时间尺度

---

## 三、main.py 更新

在 MODULE_REGISTRY 中注册全部 8 个新模块:

```python
("microstructure", ("市场微观结构",   "microstructure",       "run_microstructure_analysis",   False)),
("intraday",       ("日内模式分析",   "intraday_patterns",    "run_intraday_analysis",         False)),
("scaling",        ("统计标度律",     "scaling_laws",         "run_scaling_analysis",           False)),
("multiscale_vol", ("多尺度波动率",   "multi_scale_vol",      "run_multiscale_vol_analysis",    False)),
("entropy",        ("信息熵分析",     "entropy_analysis",     "run_entropy_analysis",           False)),
("extreme",        ("极端值分析",     "extreme_value",        "run_extreme_value_analysis",     False)),
("cross_tf",       ("跨尺度关联",     "cross_timeframe",      "run_cross_timeframe_analysis",   False)),
("momentum_rev",   ("动量均值回归",   "momentum_reversion",   "run_momentum_reversion_analysis",False)),
```

---

## 四、实施策略

- 8 个新模块并行开发（各模块独立无依赖）
- 5 个模块增强并行开发
- 全部完成后更新 main.py 注册 + 运行全量测试
- 每个模块遵循现有 `run_xxx(df, output_dir) -> Dict` 签名
- 需要多尺度数据的模块内部调用 `load_klines(interval)` 自行加载

## 五、数据覆盖验证

| 数据文件 | 当前使用 | 扩展后使用 |
|---------|---------|----------|
| 1m | - | microstructure, intraday, scaling, momentum_rev, fft(增) |
| 3m | - | microstructure, scaling |
| 5m | - | microstructure, intraday, scaling, multi_scale_vol, entropy, cross_tf, momentum_rev, returns(增), volatility(增) |
| 15m | - | intraday, scaling, entropy, cross_tf, momentum_rev, returns(增) |
| 30m | - | intraday, scaling, multi_scale_vol, returns(增), fft(增) |
| 1h | hurst,returns,causality,calendar | +intraday, scaling, multi_scale_vol, entropy, cross_tf, momentum_rev, acf(增), volatility(增) |
| 2h | - | multi_scale_vol, scaling, fft(增), returns(增) |
| 4h | fft,hurst,returns | +multi_scale_vol, entropy, cross_tf, momentum_rev, acf(增), volatility(增), extreme |
| 6h | - | multi_scale_vol, scaling, fft(增), returns(增) |
| 8h | - | multi_scale_vol, scaling, fft(增), returns(增) |
| 12h | - | multi_scale_vol, scaling, fft(增), returns(增) |
| 1d | 全部17模块 | +所有新增模块 |
| 3d | - | scaling, fft(增), returns(增) |
| 1w | fft,hurst,returns | +extreme, cross_tf, momentum_rev, acf(增) |
| 1mo | - | momentum_rev, scaling, fft(增), returns(增) |

**结果: 全部 15 个数据文件 100% 覆盖使用**
