# BTC/USDT Price Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

A comprehensive quantitative analysis framework for BTC/USDT price dynamics, covering 25 analytical dimensions from statistical distributions to fractal geometry. The framework processes multi-timeframe Binance kline data (1-minute to monthly) spanning 2017-08 to 2026-02, producing reproducible research-grade visualizations and statistical reports.

## Features

- **Multi-timeframe data pipeline** — 15 granularities from 1m to 1M, unified loader with validation
- **25 analysis modules** — each module runs independently; single-module failure does not block others
- **Statistical rigor** — train/validation splits, multiple hypothesis testing corrections, bootstrap confidence intervals
- **Publication-ready output** — 53 charts with Chinese font support, plus a 1300-line Markdown research report
- **Modular architecture** — run all modules or cherry-pick via CLI flags

## Project Structure

```
btc_price_anany/
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
├── data/                   # 15 BTC/USDT kline CSVs (1m ~ 1M)
├── src/                    # 30 analysis & utility modules
│   ├── data_loader.py      # Data loading & validation
│   ├── preprocessing.py    # Derived feature engineering
│   ├── font_config.py      # Chinese font rendering
│   ├── visualization.py    # Summary dashboard generation
│   └── ...                 # 26 analysis modules
├── output/                 # Generated charts (53 PNGs)
├── docs/
│   └── REPORT.md           # Full research report with findings
└── tests/
    └── test_hurst_15scales.py  # Hurst exponent multi-scale test
```

## Quick Start

### Requirements

- Python 3.10+
- ~1 GB disk for kline data

### Installation

```bash
git clone https://github.com/riba2534/btc_price_anany.git
cd btc_price_anany
pip install -r requirements.txt
```

### Usage

```bash
# Run all 25 analysis modules
python main.py

# List available modules
python main.py --list

# Run specific modules
python main.py --modules fft wavelet hurst

# Limit date range
python main.py --start 2020-01-01 --end 2025-12-31
```

## Data

| File | Timeframe | Rows (approx.) |
|------|-----------|-----------------|
| `btcusdt_1m.csv` | 1 minute | ~4,500,000 |
| `btcusdt_3m.csv` | 3 minutes | ~1,500,000 |
| `btcusdt_5m.csv` | 5 minutes | ~900,000 |
| `btcusdt_15m.csv` | 15 minutes | ~300,000 |
| `btcusdt_30m.csv` | 30 minutes | ~150,000 |
| `btcusdt_1h.csv` | 1 hour | ~75,000 |
| `btcusdt_2h.csv` | 2 hours | ~37,000 |
| `btcusdt_4h.csv` | 4 hours | ~19,000 |
| `btcusdt_6h.csv` | 6 hours | ~12,500 |
| `btcusdt_8h.csv` | 8 hours | ~9,500 |
| `btcusdt_12h.csv` | 12 hours | ~6,300 |
| `btcusdt_1d.csv` | 1 day | ~3,100 |
| `btcusdt_3d.csv` | 3 days | ~1,000 |
| `btcusdt_1w.csv` | 1 week | ~450 |
| `btcusdt_1mo.csv` | 1 month | ~100 |

All data sourced from Binance public API, covering 2017-08 to 2026-02.

## Analysis Modules

| Module | Description |
|--------|-------------|
| `fft` | FFT power spectrum, multi-timeframe spectral analysis, bandpass filtering |
| `wavelet` | Continuous wavelet transform scalogram, global spectrum, key period tracking |
| `acf` | ACF/PACF grid analysis for autocorrelation structure |
| `returns` | Return distribution fitting, QQ plots, multi-scale moment analysis |
| `volatility` | Volatility clustering, GARCH modeling, leverage effect quantification |
| `hurst` | R/S and DFA Hurst exponent estimation, rolling window analysis |
| `fractal` | Box-counting dimension, Monte Carlo benchmarking, self-similarity tests |
| `power_law` | Log-log regression, power-law growth corridor, model comparison |
| `volume_price` | Volume-return scatter analysis, OBV divergence detection |
| `calendar` | Weekday, month, hour, and quarter-boundary effects |
| `halving` | Halving cycle analysis with normalized trajectory comparison |
| `indicators` | Technical indicator IC testing with train/validation split |
| `patterns` | K-line pattern recognition with forward-return validation |
| `clustering` | Market regime clustering (K-Means, GMM) with transition matrices |
| `time_series` | ARIMA, Prophet, LSTM forecasting with direction accuracy |
| `causality` | Granger causality testing across volume and price features |
| `anomaly` | Anomaly detection with precursor feature analysis |
| `microstructure` | Market microstructure: spreads, Kyle's lambda, VPIN |
| `intraday` | Intraday session patterns and volume heatmaps |
| `scaling` | Statistical scaling laws and kurtosis decay |
| `multiscale_vol` | HAR volatility, jump detection, higher moment analysis |
| `entropy` | Sample entropy and permutation entropy across scales |
| `extreme` | Extreme value theory: Hill estimator, VaR backtesting |
| `cross_tf` | Cross-timeframe correlation and lead-lag analysis |
| `momentum_rev` | Momentum vs mean-reversion: variance ratios, OU half-life |

## Key Findings

The full analysis report is available at [`docs/REPORT.md`](docs/REPORT.md). Major conclusions include:

- **Non-Gaussian returns**: BTC daily returns exhibit significant fat tails (kurtosis ~10) and are best fit by Student-t distributions, not Gaussian
- **Volatility clustering**: Strong GARCH effects with long memory (d ≈ 0.4), confirming volatility persistence across time scales
- **Hurst exponent H ≈ 0.55**: Weak but statistically significant long-range dependence, transitioning from trending (short-term) to mean-reverting (long-term)
- **Fractal dimension D ≈ 1.4**: Price series is rougher than Brownian motion, exhibiting multi-fractal characteristics
- **Halving cycle impact**: Statistically significant post-halving bull runs with diminishing returns per cycle
- **Calendar effects**: Weak but detectable weekday and monthly seasonality; no exploitable intraday patterns survive transaction costs

## License

This project is licensed under the [MIT License](LICENSE).
