"""统一数据加载模块 - 处理毫秒/微秒时间戳差异"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "data"

AVAILABLE_INTERVALS = [
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1mo"
]

NUMERIC_COLS = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume"
]


def _adaptive_timestamp(ts_series: pd.Series) -> pd.DatetimeIndex:
    """自适应处理毫秒(13位)和微秒(16位)时间戳"""
    ts = pd.to_numeric(ts_series, errors="coerce").astype(np.int64)
    # 16位时间戳(微秒) -> 转为毫秒
    mask = ts > 1e15
    ts = ts.copy()
    ts[mask] = ts[mask] // 1000
    return pd.to_datetime(ts, unit="ms")


def load_klines(
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    加载指定时间粒度的K线数据

    Parameters
    ----------
    interval : str
        K线粒度，如 '1d', '1h', '4h', '1w', '1mo'
    start : str, optional
        起始日期，如 '2020-01-01'
    end : str, optional
        结束日期，如 '2025-12-31'
    data_dir : Path, optional
        数据目录，默认使用 data/

    Returns
    -------
    pd.DataFrame
        以 DatetimeIndex 为索引的K线数据
    """
    if data_dir is None:
        data_dir = DATA_DIR

    filepath = data_dir / f"btcusdt_{interval}.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"数据文件不存在: {filepath}")

    df = pd.read_csv(filepath)

    # 类型转换
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 自适应时间戳处理
    df.index = _adaptive_timestamp(df["open_time"])
    df.index.name = "datetime"

    # close_time 也做处理
    if "close_time" in df.columns:
        df["close_time"] = _adaptive_timestamp(df["close_time"])

    # 删除原始时间戳列和ignore列
    df.drop(columns=["open_time", "ignore"], inplace=True, errors="ignore")

    # 排序去重
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="first")]

    # 时间范围过滤
    if start:
        try:
            df = df[df.index >= pd.Timestamp(start)]
        except ValueError:
            print(f"[警告] 无效的起始日期 '{start}'，忽略")
    if end:
        try:
            df = df[df.index <= pd.Timestamp(end)]
        except ValueError:
            print(f"[警告] 无效的结束日期 '{end}'，忽略")

    return df


def load_daily(start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """快捷加载日线数据"""
    return load_klines("1d", start=start, end=end)


def load_hourly(start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """快捷加载小时数据"""
    return load_klines("1h", start=start, end=end)


def validate_data(df: pd.DataFrame, interval: str = "1d") -> dict:
    """数据完整性校验"""
    if len(df) == 0:
        return {"rows": 0, "date_range": "N/A", "null_counts": {}, "duplicate_index": 0,
                "price_range": "N/A", "negative_volume": 0}

    report = {
        "rows": len(df),
        "date_range": f"{df.index.min()} ~ {df.index.max()}",
        "null_counts": df.isnull().sum().to_dict(),
        "duplicate_index": df.index.duplicated().sum(),
    }

    # 检查价格合理性
    report["price_range"] = f"{df['close'].min():.2f} ~ {df['close'].max():.2f}"
    report["negative_volume"] = (df["volume"] < 0).sum()

    # 检查缺失天数(仅日线)
    if interval == "1d":
        expected_days = (df.index.max() - df.index.min()).days + 1
        report["expected_days"] = expected_days
        report["missing_days"] = expected_days - len(df)

    return report


# 数据切分常量
TRAIN_END = "2022-09-30"
VAL_END = "2024-06-30"

def split_data(df: pd.DataFrame):
    """按时间顺序切分 训练/验证/测试 集"""
    train = df[df.index <= TRAIN_END]
    val = df[(df.index > TRAIN_END) & (df.index <= VAL_END)]
    test = df[df.index > VAL_END]
    return train, val, test
