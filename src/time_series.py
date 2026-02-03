"""时间序列预测模块 - ARIMA、Prophet、LSTM/GRU

对BTC日线数据进行多模型预测与对比评估。
每个模型独立运行，单个模型失败不影响其他模型。
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from scipy import stats

from src.data_loader import split_data


# ============================================================
# 评估指标
# ============================================================

def _direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """方向准确率：预测涨跌方向正确的比例"""
    if len(y_true) < 2:
        return np.nan
    true_dir = np.sign(y_true)
    pred_dir = np.sign(y_pred)
    return np.mean(true_dir == pred_dir)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方根误差"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def _diebold_mariano_test(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> Tuple[float, float]:
    """
    Diebold-Mariano检验：比较两个预测的损失差异是否显著

    H0: 两个模型预测精度无差异
    e1, e2: 两个模型的预测误差序列

    Returns
    -------
    dm_stat : DM统计量
    p_value : 双侧p值
    """
    d = e1 ** 2 - e2 ** 2  # 平方损失差
    n = len(d)
    if n < 10:
        return np.nan, np.nan

    mean_d = np.mean(d)

    # Newey-West方差估计（考虑自相关）
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0
    for k in range(1, h):
        gamma_k = np.cov(d[k:], d[:-k])[0, 1] if len(d[k:]) > 1 else 0
        gamma_sum += 2 * gamma_k

    var_d = (gamma_0 + gamma_sum) / n
    if var_d <= 0:
        return np.nan, np.nan

    dm_stat = mean_d / np.sqrt(var_d)
    p_value = 2 * stats.norm.sf(np.abs(dm_stat))
    return dm_stat, p_value


def _evaluate_model(name: str, y_true: np.ndarray, y_pred: np.ndarray,
                    rw_errors: np.ndarray) -> Dict:
    """统一评估单个模型"""
    errors = y_true - y_pred
    rmse_val = _rmse(y_true, y_pred)
    rw_rmse = _rmse(y_true, np.zeros_like(y_true))  # Random Walk RMSE
    rmse_ratio = rmse_val / rw_rmse if rw_rmse > 0 else np.nan
    dir_acc = _direction_accuracy(y_true, y_pred)

    # DM检验 vs Random Walk
    dm_stat, dm_pval = _diebold_mariano_test(errors, rw_errors)

    result = {
        "name": name,
        "rmse": rmse_val,
        "rmse_ratio_vs_rw": rmse_ratio,
        "direction_accuracy": dir_acc,
        "dm_stat_vs_rw": dm_stat,
        "dm_pval_vs_rw": dm_pval,
        "predictions": y_pred,
        "errors": errors,
    }
    return result


# ============================================================
# 基准模型
# ============================================================

def _baseline_random_walk(y_true: np.ndarray) -> np.ndarray:
    """随机游走基准：预测收益率=0"""
    return np.zeros_like(y_true)


def _baseline_historical_mean(train_returns: np.ndarray, n_pred: int) -> np.ndarray:
    """历史均值基准：预测收益率=训练集均值"""
    return np.full(n_pred, np.mean(train_returns))


# ============================================================
# ARIMA 模型
# ============================================================

def _run_arima(train_returns: pd.Series, val_returns: pd.Series) -> Dict:
    """
    ARIMA模型：使用auto_arima自动选参 + walk-forward预测

    Returns
    -------
    dict : 包含预测结果和诊断信息
    """
    try:
        import pmdarima as pm
        from statsmodels.stats.diagnostic import acorr_ljungbox
    except ImportError:
        print("  [ARIMA] 跳过 - pmdarima 未安装。pip install pmdarima")
        return None

    print("\n" + "=" * 60)
    print("ARIMA 模型")
    print("=" * 60)

    # 自动选择ARIMA参数
    print("  [1/3] auto_arima 参数搜索...")
    model = pm.auto_arima(
        train_returns.values,
        start_p=0, max_p=5,
        start_q=0, max_q=5,
        d=0,  # 对数收益率已经是平稳的
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        trace=False,
        information_criterion='aic',
    )
    print(f"  最优模型: ARIMA{model.order}")
    print(f"  AIC: {model.aic():.2f}")

    # Ljung-Box 残差诊断
    print("  [2/3] Ljung-Box 残差白噪声检验...")
    residuals = model.resid()
    lb_result = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
    print(f"  Ljung-Box 检验 (lag=10): 统计量={lb_result.iloc[0]['lb_stat']:.2f}, "
          f"p值={lb_result.iloc[0]['lb_pvalue']:.4f}")
    print(f"  Ljung-Box 检验 (lag=20): 统计量={lb_result.iloc[1]['lb_stat']:.2f}, "
          f"p值={lb_result.iloc[1]['lb_pvalue']:.4f}")

    if lb_result.iloc[0]['lb_pvalue'] > 0.05:
        print("  残差通过白噪声检验 (p>0.05)，模型拟合充分")
    else:
        print("  残差未通过白噪声检验 (p<=0.05)，可能存在未捕获的自相关结构")

    # Walk-forward 预测
    print("  [3/3] Walk-forward 验证集预测...")
    val_values = val_returns.values
    n_val = len(val_values)
    predictions = np.zeros(n_val)

    # 使用滚动窗口预测
    history = list(train_returns.values)
    for i in range(n_val):
        # 一步预测
        fc = model.predict(n_periods=1)
        predictions[i] = fc[0]
        # 更新模型（添加真实观测值）
        model.update(val_values[i:i+1])
        if (i + 1) % 100 == 0:
            print(f"    进度: {i+1}/{n_val}")

    print(f"  Walk-forward 预测完成，共{n_val}步")

    return {
        "predictions": predictions,
        "order": model.order,
        "aic": model.aic(),
        "ljung_box": lb_result,
    }


# ============================================================
# Prophet 模型
# ============================================================

def _run_prophet(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict:
    """
    Prophet模型：基于日收盘价的时间序列预测

    Returns
    -------
    dict : 包含预测结果
    """
    try:
        from prophet import Prophet
    except ImportError:
        print("  [Prophet] 跳过 - prophet 未安装。pip install prophet")
        return None

    print("\n" + "=" * 60)
    print("Prophet 模型")
    print("=" * 60)

    # 准备Prophet格式数据
    prophet_train = pd.DataFrame({
        'ds': train_df.index,
        'y': train_df['close'].values,
    })

    print("  [1/3] 构建Prophet模型并添加自定义季节性...")

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False,
        changepoint_prior_scale=0.05,
    )

    # 添加自定义季节性
    model.add_seasonality(name='weekly', period=7, fourier_order=3)
    model.add_seasonality(name='monthly', period=30, fourier_order=5)
    model.add_seasonality(name='yearly', period=365, fourier_order=10)
    model.add_seasonality(name='halving_cycle', period=1458, fourier_order=5)

    print("  [2/3] 拟合模型...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(prophet_train)

    # 预测验证期
    print("  [3/3] 预测验证期...")
    future_dates = pd.DataFrame({'ds': val_df.index})
    forecast = model.predict(future_dates)

    # 转换为对数收益率预测（与其他模型对齐）
    pred_close = forecast['yhat'].values
    # 用前一天的真实收盘价计算预测收益率
    # 第一天用训练集最后一天的价格
    prev_close = np.concatenate([[train_df['close'].iloc[-1]], val_df['close'].values[:-1]])
    pred_returns = np.log(pred_close / prev_close)

    print(f"  预测完成，验证期: {val_df.index[0]} ~ {val_df.index[-1]}")
    print(f"  预测价格范围: {pred_close.min():.0f} ~ {pred_close.max():.0f}")

    return {
        "predictions_return": pred_returns,
        "predictions_close": pred_close,
        "forecast": forecast,
        "model": model,
    }


# ============================================================
# LSTM/GRU 模型 (PyTorch)
# ============================================================

def _run_lstm(train_df: pd.DataFrame, val_df: pd.DataFrame,
              lookback: int = 60, hidden_size: int = 128,
              num_layers: int = 2, max_epochs: int = 100,
              patience: int = 10, batch_size: int = 64) -> Dict:
    """
    LSTM/GRU 模型：基于PyTorch的深度学习时间序列预测

    Returns
    -------
    dict : 包含预测结果和训练历史
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("  [LSTM] 跳过 - PyTorch 未安装。pip install torch")
        return None

    print("\n" + "=" * 60)
    print("LSTM 模型 (PyTorch)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"  设备: {device}")

    # ---- 数据准备 ----
    # 使用收盘价的对数收益率作为目标
    feature_cols = ['log_return', 'volume_ratio', 'taker_buy_ratio']
    available_cols = [c for c in feature_cols if c in train_df.columns]

    if not available_cols:
        # 降级到只用收盘价
        print("  [警告] 特征列不可用，仅使用收盘价收益率")
        available_cols = ['log_return']

    print(f"  特征: {available_cols}")

    # 合并训练和验证数据以创建连续序列
    all_data = pd.concat([train_df, val_df])
    features = all_data[available_cols].values
    target = all_data['log_return'].values

    # 处理NaN
    mask = ~np.isnan(features).any(axis=1) & ~np.isnan(target)
    features_clean = features[mask]
    target_clean = target[mask]

    # 特征标准化（基于训练集统计量）
    train_len = mask[:len(train_df)].sum()
    feat_mean = features_clean[:train_len].mean(axis=0)
    feat_std = features_clean[:train_len].std(axis=0) + 1e-10
    features_norm = (features_clean - feat_mean) / feat_std

    target_mean = target_clean[:train_len].mean()
    target_std = target_clean[:train_len].std() + 1e-10
    target_norm = (target_clean - target_mean) / target_std

    # 创建序列样本
    def create_sequences(feat, tgt, seq_len):
        X, y = [], []
        for i in range(seq_len, len(feat)):
            X.append(feat[i - seq_len:i])
            y.append(tgt[i])
        return np.array(X), np.array(y)

    X_all, y_all = create_sequences(features_norm, target_norm, lookback)

    # 划分训练和验证（根据原始训练集长度调整）
    train_samples = max(0, train_len - lookback)
    X_train = X_all[:train_samples]
    y_train = y_all[:train_samples]
    X_val = X_all[train_samples:]
    y_val = y_all[train_samples:]

    if len(X_train) == 0 or len(X_val) == 0:
        print("  [LSTM] 跳过 - 数据不足以创建训练/验证序列")
        return None

    print(f"  训练样本: {len(X_train)}, 验证样本: {len(X_val)}")
    print(f"  回看窗口: {lookback}, 隐藏维度: {hidden_size}, 层数: {num_layers}")

    # 转换为Tensor
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # ---- 模型定义 ----
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            # 取最后一个时间步的输出
            last_out = lstm_out[:, -1, :]
            return self.fc(last_out).squeeze(-1)

    input_size = len(available_cols)
    model = LSTMModel(input_size, hidden_size, num_layers).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )

    # ---- 训练 ----
    print(f"  开始训练 (最多{max_epochs}轮, 早停耐心={patience})...")
    best_val_loss = np.inf
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(max_epochs):
        # 训练
        model.train()
        epoch_loss = 0
        n_batches = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch+1}/{max_epochs}: "
                  f"train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f}, lr={lr:.1e}")

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    早停触发 (epoch {epoch+1})")
                break

    # 加载最佳模型
    model.load_state_dict(best_state)
    model.eval()

    # ---- 预测 ----
    with torch.no_grad():
        val_pred_norm = model(X_val_t).cpu().numpy()

    # 逆标准化
    val_pred_returns = val_pred_norm * target_std + target_mean
    val_true_returns = y_val * target_std + target_mean

    print(f"  训练完成，最佳验证损失: {best_val_loss:.6f}")

    return {
        "predictions_return": val_pred_returns,
        "true_returns": val_true_returns,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "model": model,
        "device": str(device),
    }


# ============================================================
# 可视化
# ============================================================

def _plot_predictions(val_dates, y_true, model_preds: Dict[str, np.ndarray],
                      output_dir: Path):
    """各模型实际 vs 预测对比图"""
    n_models = len(model_preds)
    fig, axes = plt.subplots(n_models, 1, figsize=(16, 4 * n_models), sharex=True)
    if n_models == 1:
        axes = [axes]

    for i, (name, y_pred) in enumerate(model_preds.items()):
        ax = axes[i]
        # 对齐长度（LSTM可能因lookback导致长度不同）
        n = min(len(y_true), len(y_pred))
        dates = val_dates[:n] if len(val_dates) >= n else val_dates

        ax.plot(dates, y_true[:n], 'b-', alpha=0.6, linewidth=0.8, label='实际收益率')
        ax.plot(dates, y_pred[:n], 'r-', alpha=0.6, linewidth=0.8, label='预测收益率')
        ax.set_title(f"{name} - 实际 vs 预测", fontsize=13)
        ax.set_ylabel("对数收益率", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    axes[-1].set_xlabel("日期", fontsize=11)
    plt.tight_layout()
    fig.savefig(output_dir / "ts_predictions_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] ts_predictions_comparison.png")


def _plot_direction_accuracy(metrics: Dict[str, Dict], output_dir: Path):
    """方向准确率对比柱状图"""
    names = list(metrics.keys())
    accs = [metrics[n]["direction_accuracy"] * 100 for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    bars = ax.bar(names, accs, color=colors, edgecolor='gray', linewidth=0.5)

    # 标注数值
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='随机基准 (50%)')
    ax.set_ylabel("方向准确率 (%)", fontsize=12)
    ax.set_title("各模型方向预测准确率对比", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(accs) * 1.2 if accs else 100)

    fig.savefig(output_dir / "ts_direction_accuracy.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] ts_direction_accuracy.png")


def _plot_cumulative_error(val_dates, metrics: Dict[str, Dict], output_dir: Path):
    """累计误差对比图"""
    fig, ax = plt.subplots(figsize=(16, 7))

    for name, m in metrics.items():
        errors = m.get("errors")
        if errors is None:
            continue
        n = len(errors)
        dates = val_dates[:n]
        cum_sq_err = np.cumsum(errors ** 2)
        ax.plot(dates, cum_sq_err, linewidth=1.2, label=f"{name}")

    ax.set_xlabel("日期", fontsize=12)
    ax.set_ylabel("累计平方误差", fontsize=12)
    ax.set_title("各模型累计预测误差对比", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / "ts_cumulative_error.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] ts_cumulative_error.png")


def _plot_lstm_training(train_losses: List, val_losses: List, output_dir: Path):
    """LSTM训练损失曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, 'b-', label='训练损失', linewidth=1.5)
    ax.plot(val_losses, 'r-', label='验证损失', linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("LSTM 训练过程", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / "ts_lstm_training.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] ts_lstm_training.png")


def _plot_prophet_components(prophet_result: Dict, output_dir: Path):
    """Prophet预测 - 实际价格 vs 预测价格"""
    try:
        from prophet import Prophet
    except ImportError:
        return

    forecast = prophet_result.get("forecast")
    if forecast is None:
        return

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(forecast['ds'], forecast['yhat'], 'r-', linewidth=1.2, label='Prophet预测')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                    alpha=0.15, color='red', label='置信区间')
    ax.set_xlabel("日期", fontsize=12)
    ax.set_ylabel("BTC 价格 (USDT)", fontsize=12)
    ax.set_title("Prophet 价格预测（验证期）", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / "ts_prophet_forecast.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] ts_prophet_forecast.png")


# ============================================================
# 结果打印
# ============================================================

def _print_metrics_table(all_metrics: Dict[str, Dict]):
    """打印所有模型的评估指标表"""
    print("\n" + "=" * 80)
    print("  模型评估汇总")
    print("=" * 80)
    print(f"  {'模型':<20s} {'RMSE':>10s} {'RMSE/RW':>10s} {'方向准确率':>10s} "
          f"{'DM统计量':>10s} {'DM p值':>10s}")
    print("-" * 80)

    for name, m in all_metrics.items():
        rmse_str = f"{m['rmse']:.6f}"
        ratio_str = f"{m['rmse_ratio_vs_rw']:.4f}" if not np.isnan(m['rmse_ratio_vs_rw']) else "N/A"
        dir_str = f"{m['direction_accuracy']*100:.1f}%"
        dm_str = f"{m['dm_stat_vs_rw']:.3f}" if not np.isnan(m['dm_stat_vs_rw']) else "N/A"
        pv_str = f"{m['dm_pval_vs_rw']:.4f}" if not np.isnan(m['dm_pval_vs_rw']) else "N/A"
        print(f"  {name:<20s} {rmse_str:>10s} {ratio_str:>10s} {dir_str:>10s} "
              f"{dm_str:>10s} {pv_str:>10s}")

    print("-" * 80)

    # 解读
    print("\n  [解读]")
    print("  - RMSE/RW < 1.0 表示优于随机游走基准")
    print("  - 方向准确率 > 50% 表示有一定方向预测能力")
    print("  - DM检验 p值 < 0.05 表示与随机游走有显著差异")


# ============================================================
# 主入口
# ============================================================

def run_time_series_analysis(df: pd.DataFrame, output_dir: "str | Path" = "output/time_series") -> Dict:
    """
    时间序列预测分析 - 主入口

    Parameters
    ----------
    df : pd.DataFrame
        已经通过 add_derived_features() 添加了衍生特征的日线数据
    output_dir : str or Path
        图表输出目录

    Returns
    -------
    results : dict
        包含所有模型的预测结果和评估指标
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.font_config import configure_chinese_font
    configure_chinese_font()

    print("=" * 60)
    print("  BTC 时间序列预测分析")
    print("=" * 60)

    # ---- 数据划分 ----
    train_df, val_df, test_df = split_data(df)
    print(f"\n  训练集: {train_df.index[0]} ~ {train_df.index[-1]} ({len(train_df)}天)")
    print(f"  验证集: {val_df.index[0]} ~ {val_df.index[-1]} ({len(val_df)}天)")
    print(f"  测试集: {test_df.index[0]} ~ {test_df.index[-1]} ({len(test_df)}天)")

    # 对数收益率序列
    train_returns = train_df['log_return'].dropna()
    val_returns = val_df['log_return'].dropna()
    val_dates = val_returns.index
    y_true = val_returns.values

    # ---- 基准模型 ----
    print("\n" + "=" * 60)
    print("基准模型")
    print("=" * 60)

    # Random Walk基准
    rw_pred = _baseline_random_walk(y_true)
    rw_errors = y_true - rw_pred
    print(f"  Random Walk (预测收益=0): RMSE = {_rmse(y_true, rw_pred):.6f}")

    # 历史均值基准
    hm_pred = _baseline_historical_mean(train_returns.values, len(y_true))
    print(f"  Historical Mean (收益={train_returns.mean():.6f}): RMSE = {_rmse(y_true, hm_pred):.6f}")

    # 存储所有模型结果
    all_metrics = {}
    model_preds = {}

    # 评估基准模型
    all_metrics["Random Walk"] = _evaluate_model("Random Walk", y_true, rw_pred, rw_errors)
    model_preds["Random Walk"] = rw_pred

    all_metrics["Historical Mean"] = _evaluate_model("Historical Mean", y_true, hm_pred, rw_errors)
    model_preds["Historical Mean"] = hm_pred

    # ---- ARIMA ----
    try:
        arima_result = _run_arima(train_returns, val_returns)
        if arima_result is not None:
            arima_pred = arima_result["predictions"]
            all_metrics["ARIMA"] = _evaluate_model("ARIMA", y_true, arima_pred, rw_errors)
            model_preds["ARIMA"] = arima_pred
            print(f"\n  ARIMA 验证集: RMSE={all_metrics['ARIMA']['rmse']:.6f}, "
                  f"方向准确率={all_metrics['ARIMA']['direction_accuracy']*100:.1f}%")
    except Exception as e:
        print(f"\n  [ARIMA] 运行失败: {e}")

    # ---- Prophet ----
    try:
        prophet_result = _run_prophet(train_df, val_df)
        if prophet_result is not None:
            prophet_pred = prophet_result["predictions_return"]
            # 对齐长度
            n = min(len(y_true), len(prophet_pred))
            all_metrics["Prophet"] = _evaluate_model(
                "Prophet", y_true[:n], prophet_pred[:n], rw_errors[:n]
            )
            model_preds["Prophet"] = prophet_pred[:n]
            print(f"\n  Prophet 验证集: RMSE={all_metrics['Prophet']['rmse']:.6f}, "
                  f"方向准确率={all_metrics['Prophet']['direction_accuracy']*100:.1f}%")

            # Prophet专属图表
            _plot_prophet_components(prophet_result, output_dir)
    except Exception as e:
        print(f"\n  [Prophet] 运行失败: {e}")
        prophet_result = None

    # ---- LSTM ----
    try:
        lstm_result = _run_lstm(train_df, val_df)
        if lstm_result is not None:
            lstm_pred = lstm_result["predictions_return"]
            lstm_true = lstm_result["true_returns"]
            n_lstm = len(lstm_pred)

            # LSTM因lookback导致样本数不同，使用其自身的true_returns评估
            lstm_rw_errors = lstm_true - np.zeros_like(lstm_true)
            all_metrics["LSTM"] = _evaluate_model(
                "LSTM", lstm_true, lstm_pred, lstm_rw_errors
            )
            model_preds["LSTM"] = lstm_pred
            print(f"\n  LSTM 验证集: RMSE={all_metrics['LSTM']['rmse']:.6f}, "
                  f"方向准确率={all_metrics['LSTM']['direction_accuracy']*100:.1f}%")

            # LSTM训练曲线
            _plot_lstm_training(lstm_result["train_losses"],
                                lstm_result["val_losses"], output_dir)
    except Exception as e:
        print(f"\n  [LSTM] 运行失败: {e}")
        lstm_result = None

    # ---- 评估汇总 ----
    _print_metrics_table(all_metrics)

    # ---- 可视化 ----
    print("\n[可视化] 生成分析图表...")

    # 预测对比图（仅使用与y_true等长的预测，排除LSTM）
    aligned_preds = {k: v for k, v in model_preds.items()
                     if k != "LSTM" and len(v) == len(y_true)}
    if aligned_preds:
        _plot_predictions(val_dates, y_true, aligned_preds, output_dir)

    # LSTM单独画图（长度不同）
    if "LSTM" in model_preds and lstm_result is not None:
        lstm_dates = val_dates[-len(lstm_result["predictions_return"]):]
        _plot_predictions(lstm_dates, lstm_result["true_returns"],
                          {"LSTM": lstm_result["predictions_return"]}, output_dir)

    # 方向准确率对比
    _plot_direction_accuracy(all_metrics, output_dir)

    # 累计误差对比
    _plot_cumulative_error(val_dates, all_metrics, output_dir)

    # ---- 汇总 ----
    results = {
        "metrics": all_metrics,
        "model_predictions": model_preds,
        "val_dates": val_dates,
        "y_true": y_true,
    }

    if 'arima_result' in dir() and arima_result is not None:
        results["arima"] = arima_result
    if prophet_result is not None:
        results["prophet"] = prophet_result
    if lstm_result is not None:
        results["lstm"] = lstm_result

    print("\n" + "=" * 60)
    print("  时间序列预测分析完成！")
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

    results = run_time_series_analysis(df, output_dir="output/time_series")
