"""数据预处理模块 - 收益率、去趋势、标准化、衍生指标"""

import pandas as pd
import numpy as np
from typing import Optional


def log_returns(prices: pd.Series) -> pd.Series:
    """对数收益率"""
    return np.log(prices / prices.shift(1)).dropna()


def simple_returns(prices: pd.Series) -> pd.Series:
    """简单收益率"""
    return prices.pct_change().dropna()


def detrend_log_diff(prices: pd.Series) -> pd.Series:
    """对数差分去趋势"""
    return np.log(prices).diff().dropna()


def detrend_linear(series: pd.Series) -> pd.Series:
    """线性去趋势"""
    x = np.arange(len(series))
    coeffs = np.polyfit(x, series.values, 1)
    trend = np.polyval(coeffs, x)
    return pd.Series(series.values - trend, index=series.index)


def hp_filter(series: pd.Series, lamb: float = 1600) -> tuple:
    """Hodrick-Prescott 滤波器"""
    from statsmodels.tsa.filters.hp_filter import hpfilter
    cycle, trend = hpfilter(series.dropna(), lamb=lamb)
    return cycle, trend


def rolling_volatility(returns: pd.Series, window: int = 30) -> pd.Series:
    """滚动波动率（年化）"""
    return returns.rolling(window=window).std() * np.sqrt(365)


def realized_volatility(returns: pd.Series, window: int = 30) -> pd.Series:
    """已实现波动率"""
    return np.sqrt((returns ** 2).rolling(window=window).sum())


def taker_buy_ratio(df: pd.DataFrame) -> pd.Series:
    """Taker买入比例"""
    return df["taker_buy_volume"] / df["volume"].replace(0, np.nan)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加常用衍生特征列"""
    out = df.copy()
    out["log_return"] = log_returns(df["close"])
    out["simple_return"] = simple_returns(df["close"])
    out["log_price"] = np.log(df["close"])
    out["range_pct"] = (df["high"] - df["low"]) / df["close"]
    out["body_pct"] = (df["close"] - df["open"]) / df["open"]
    out["taker_buy_ratio"] = taker_buy_ratio(df)
    out["vol_30d"] = rolling_volatility(out["log_return"], 30)
    out["vol_7d"] = rolling_volatility(out["log_return"], 7)
    out["volume_ma20"] = df["volume"].rolling(20).mean()
    out["volume_ratio"] = df["volume"] / out["volume_ma20"]
    out["abs_return"] = out["log_return"].abs()
    out["squared_return"] = out["log_return"] ** 2
    return out


def standardize(series: pd.Series) -> pd.Series:
    """Z-score标准化"""
    return (series - series.mean()) / series.std()


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Winsorize处理极端值"""
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)
