#!/usr/bin/env python3
"""
BTC/USDT K线数据下载脚本

从 Binance 公开数据下载站 (data.binance.vision) 批量下载全部 15 个时间粒度的历史 K 线数据。
相比 API 逐页拉取，zip 批量下载速度快数十倍。

数据范围：2017-08（BTCUSDT 上线月）至今。
支持断点续传：已下载的 zip 文件不会重复拉取。

用法：
    python download_data.py                  # 下载全部 15 个粒度
    python download_data.py 1d 1h 4h         # 只下载指定粒度
    python download_data.py --list           # 查看可用粒度
"""

import csv
import io
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

# ============================================================
# 配置
# ============================================================

SYMBOL = "BTCUSDT"
BASE_URL = "https://data.binance.vision/data/spot"
MAX_WORKERS = 10  # 并发下载线程数

# BTCUSDT 上线月份
START_YEAR = 2017
START_MONTH = 8

# 全部 15 个粒度（与 data.binance.vision 路径一致）
ALL_INTERVALS = [
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1mo",
]

# CSV 表头，与 src/data_loader.py 期望的列名一致
CSV_HEADER = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]


# ============================================================
# URL 生成
# ============================================================

def generate_monthly_urls(interval: str) -> list[tuple[str, str]]:
    """生成所有月度 zip 文件的 (URL, 文件名) 列表。"""
    now = datetime.now(timezone.utc)
    urls = []
    year, month = START_YEAR, START_MONTH
    # 月度文件覆盖到上个月（当月数据用日度文件补全）
    while (year, month) < (now.year, now.month):
        filename = f"{SYMBOL}-{interval}-{year}-{month:02d}.zip"
        url = f"{BASE_URL}/monthly/klines/{SYMBOL}/{interval}/{filename}"
        urls.append((url, filename))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return urls


def generate_daily_urls(interval: str) -> list[tuple[str, str]]:
    """生成当月每日 zip 文件的 (URL, 文件名) 列表。"""
    now = datetime.now(timezone.utc)
    urls = []
    day = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    # 日度数据最多到昨天
    yesterday = now - timedelta(days=1)
    while day.date() <= yesterday.date():
        date_str = day.strftime("%Y-%m-%d")
        filename = f"{SYMBOL}-{interval}-{date_str}.zip"
        url = f"{BASE_URL}/daily/klines/{SYMBOL}/{interval}/{filename}"
        urls.append((url, filename))
        day += timedelta(days=1)
    return urls


# ============================================================
# 下载与解压
# ============================================================

def download_zip(url: str, cache_dir: Path, filename: str) -> Path | None:
    """下载单个 zip 文件，已存在则跳过。"""
    filepath = cache_dir / filename
    if filepath.exists() and filepath.stat().st_size > 0:
        return filepath
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        filepath.write_bytes(resp.content)
        return filepath
    except requests.exceptions.RequestException:
        return None


def extract_csv_rows(zip_path: Path) -> list[list[str]]:
    """从 zip 中提取 CSV 数据行（跳过表头行）。"""
    rows = []
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if name.endswith(".csv"):
                    with zf.open(name) as f:
                        reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
                        for row in reader:
                            # 跳过表头或空行
                            if row and not row[0].startswith("open"):
                                rows.append(row)
    except (zipfile.BadZipFile, Exception) as e:
        print(f"\n  [解压失败] {zip_path.name}: {e}")
    return rows


# ============================================================
# 核心流程
# ============================================================

def download_interval(interval: str, output_dir: Path, cache_dir: Path) -> int:
    """下载并合并单个粒度的全量数据，返回总行数。"""
    filepath = output_dir / f"btcusdt_{interval}.csv"
    interval_cache = cache_dir / interval
    interval_cache.mkdir(parents=True, exist_ok=True)

    # 生成 URL 列表
    monthly_urls = generate_monthly_urls(interval)
    daily_urls = generate_daily_urls(interval)
    all_urls = monthly_urls + daily_urls

    if not all_urls:
        print("  无可下载文件")
        return 0

    print(f"  共 {len(all_urls)} 个文件 "
          f"({len(monthly_urls)} 月度 + {len(daily_urls)} 日度)")

    # 并发下载
    downloaded = []
    failed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(download_zip, url, interval_cache, fname): fname
            for url, fname in all_urls
        }
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                downloaded.append(result)
            else:
                failed += 1
            if i % 20 == 0 or i == len(futures):
                print(f"\r  下载: {i}/{len(futures)} "
                      f"(成功 {len(downloaded)}, 跳过 {failed})",
                      end="", flush=True)

    print()

    if not downloaded:
        print("  无有效数据")
        return 0

    # 按文件名排序保证时间顺序
    downloaded.sort(key=lambda p: p.name)

    # 解压并合并写入 CSV
    print("  合并数据...", end="", flush=True)
    total_rows = 0
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

        for zip_path in downloaded:
            rows = extract_csv_rows(zip_path)
            for row in rows:
                writer.writerow(row)
            total_rows += len(rows)

    print(f"\r  完成: {total_rows:,} 行 -> {filepath.name}")
    return total_rows


# ============================================================
# 入口
# ============================================================

def parse_interval(arg: str) -> str:
    """将用户输入映射为下载站的 interval 标识。"""
    s = arg.strip().lower()
    # 兼容旧的 '1M' 写法
    if s == "1m" and arg.strip() == "1M":
        return "1mo"
    for iv in ALL_INTERVALS:
        if iv.lower() == s:
            return iv
    return ""


def main():
    output_dir = Path(__file__).resolve().parent / "data"
    cache_dir = output_dir / ".cache"
    output_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)

    # --list 模式
    if "--list" in sys.argv:
        print("可用粒度:")
        for iv in ALL_INTERVALS:
            print(f"  {iv}")
        return

    # 解析参数
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        intervals = []
        for arg in args:
            iv = parse_interval(arg)
            if not iv:
                print(f"未知粒度: {arg}")
                print(f"可选: {', '.join(ALL_INTERVALS)}")
                sys.exit(1)
            intervals.append(iv)
    else:
        intervals = list(ALL_INTERVALS)

    print("=" * 60)
    print("BTC/USDT K 线数据下载 (data.binance.vision)")
    print("=" * 60)
    print(f"交易对:  {SYMBOL}")
    print(f"粒度:    {', '.join(intervals)}")
    print(f"起始月:  {START_YEAR}-{START_MONTH:02d}")
    print(f"并发数:  {MAX_WORKERS}")
    print(f"缓存目录: {cache_dir}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    results = {}
    t0 = time.time()

    for i, interval in enumerate(intervals, 1):
        print(f"\n[{i}/{len(intervals)}] {interval}")
        rows = download_interval(interval, output_dir, cache_dir)
        results[interval] = rows

    elapsed = time.time() - t0
    m, s = divmod(int(elapsed), 60)

    print(f"\n{'=' * 60}")
    print(f"全部完成（耗时 {m}m{s}s）：")
    print("=" * 60)
    for tag, rows in results.items():
        print(f"  {tag:5s} -> {rows:>12,} 行")
    print(f"\n数据目录: {output_dir}")


if __name__ == "__main__":
    main()
