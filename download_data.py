#!/usr/bin/env python3
"""
BTC/USDT K线数据下载脚本

从 Binance 公开 API 下载全部 15 个时间粒度的历史 K 线数据。
数据范围：2017-08-17（BTCUSDT 上线日）至今。
支持断点续传：已下载的数据不会重复拉取。

用法：
    python download_data.py                  # 下载全部 15 个粒度
    python download_data.py 1d 1h 4h         # 只下载指定粒度
    python download_data.py --list           # 查看可用粒度
"""

import csv
import sys
import time
import requests
from datetime import datetime, timezone
from pathlib import Path

# ============================================================
# 配置
# ============================================================

SYMBOL = "BTCUSDT"
BASE_URL = "https://api.binance.com/api/v3/klines"
LIMIT = 1000  # 每次请求最大行数

# BTCUSDT 上线时间
START_MS = int(datetime(2017, 8, 17, tzinfo=timezone.utc).timestamp() * 1000)

# 全部 15 个粒度（API 参数值）
ALL_INTERVALS = [
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M",
]

# API interval → 本地文件名中的粒度标识
INTERVAL_TO_FILENAME = {i: i for i in ALL_INTERVALS}
INTERVAL_TO_FILENAME["1M"] = "1mo"  # Binance API 用 '1M'，项目文件用 '1mo'

# CSV 表头，与 src/data_loader.py 期望的列名一致
CSV_HEADER = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]


# ============================================================
# 下载逻辑
# ============================================================

def get_last_timestamp(filepath: Path) -> int | None:
    """读取已有 CSV 最后一行的 close_time，用于断点续传。"""
    if not filepath.exists() or filepath.stat().st_size == 0:
        return None
    last_line = ""
    with open(filepath, "rb") as f:
        # 从文件末尾向前查找最后一行
        f.seek(0, 2)
        pos = f.tell()
        while pos > 0:
            pos -= 1
            f.seek(pos)
            ch = f.read(1)
            if ch == b"\n" and pos < f.tell() - 1:
                last_line = f.readline().decode().strip()
                break
        if not last_line:
            f.seek(0)
            for line in f:
                last_line = line.decode().strip()
    if not last_line or last_line.startswith("open_time"):
        return None
    try:
        close_time = int(last_line.split(",")[6])
        return close_time
    except (IndexError, ValueError):
        return None


def count_lines(filepath: Path) -> int:
    """快速统计 CSV 数据行数（不含表头）。"""
    if not filepath.exists():
        return 0
    with open(filepath, "rb") as f:
        count = sum(1 for _ in f) - 1  # 减去表头
    return max(0, count)


def download_interval(interval: str, output_dir: Path) -> int:
    """下载单个粒度的全量 K 线数据，返回最终行数。"""
    tag = INTERVAL_TO_FILENAME[interval]
    filepath = output_dir / f"btcusdt_{tag}.csv"

    existing_rows = count_lines(filepath)
    last_ts = get_last_timestamp(filepath)

    if last_ts is not None:
        start_time = last_ts + 1
        print(f"  断点续传: 已有 {existing_rows:,} 行，"
              f"从 {ms_to_date(start_time)} 继续")
    else:
        start_time = START_MS

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    if start_time >= now_ms:
        print(f"  已是最新数据，跳过")
        return existing_rows

    # 写入模式：续传用 append，否则新建
    mode = "a" if existing_rows > 0 else "w"
    new_rows = 0
    retries = 0
    max_retries = 10

    with open(filepath, mode, newline="") as f:
        writer = csv.writer(f)
        if existing_rows == 0:
            writer.writerow(CSV_HEADER)

        current = start_time
        while current < now_ms:
            params = {
                "symbol": SYMBOL,
                "interval": interval,
                "startTime": current,
                "limit": LIMIT,
            }
            try:
                resp = requests.get(BASE_URL, params=params, timeout=30)

                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 60))
                    print(f"\n  [限频] 等待 {wait}s...")
                    time.sleep(wait)
                    continue
                if resp.status_code == 418:
                    print(f"\n  [IP 封禁] 等待 120s...")
                    time.sleep(120)
                    continue

                resp.raise_for_status()
                data = resp.json()

                if not data:
                    break

                for row in data:
                    writer.writerow(row)
                new_rows += len(data)

                # 下一批起始点
                current = data[-1][6] + 1  # last close_time + 1

                # 进度
                total = existing_rows + new_rows
                pct = min(100, (current - START_MS) / max(1, now_ms - START_MS) * 100)
                print(f"\r  {ms_to_date(current)} | "
                      f"{total:>10,} 行 | {pct:5.1f}%", end="", flush=True)

                retries = 0
                time.sleep(0.05)

            except KeyboardInterrupt:
                print(f"\n  [中断] 已保存 {existing_rows + new_rows:,} 行")
                return existing_rows + new_rows
            except requests.exceptions.RequestException as e:
                retries += 1
                if retries > max_retries:
                    print(f"\n  [失败] 连续 {max_retries} 次错误，中止: {e}")
                    break
                wait = min(2 ** retries, 60)
                print(f"\n  [重试 {retries}/{max_retries}] {wait}s 后: {e}")
                time.sleep(wait)

    total = existing_rows + new_rows
    print(f"\n  完成: +{new_rows:,} 行，共 {total:,} 行 → {filepath.name}")
    return total


def ms_to_date(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


# ============================================================
# 入口
# ============================================================

def parse_interval(arg: str) -> str:
    """将用户输入的粒度标识映射为 Binance API interval。"""
    s = arg.strip().lower()
    # 处理 '1mo' → '1M'
    if s == "1mo":
        return "1M"
    for iv in ALL_INTERVALS:
        if iv.lower() == s:
            return iv
    return ""


def main():
    output_dir = Path(__file__).resolve().parent / "data"
    output_dir.mkdir(exist_ok=True)

    # --list 模式
    if "--list" in sys.argv:
        print("可用粒度:")
        for iv in ALL_INTERVALS:
            tag = INTERVAL_TO_FILENAME[iv]
            print(f"  {tag:5s}  (API: {iv})")
        return

    # 解析参数
    if len(sys.argv) > 1:
        intervals = []
        for arg in sys.argv[1:]:
            iv = parse_interval(arg)
            if not iv:
                print(f"未知粒度: {arg}")
                tags = [INTERVAL_TO_FILENAME[i] for i in ALL_INTERVALS]
                print(f"可选: {', '.join(tags)}")
                sys.exit(1)
            intervals.append(iv)
    else:
        intervals = list(ALL_INTERVALS)

    tags = [INTERVAL_TO_FILENAME[i] for i in intervals]
    print("=" * 60)
    print(f"BTC/USDT K 线数据下载")
    print(f"=" * 60)
    print(f"交易对:  {SYMBOL}")
    print(f"粒度:    {', '.join(tags)}")
    print(f"起始日:  {ms_to_date(START_MS)}")
    print(f"输出目录: {output_dir}")
    print(f"依赖:    pip install requests")
    print("=" * 60)

    results = {}
    t0 = time.time()

    for i, interval in enumerate(intervals, 1):
        tag = INTERVAL_TO_FILENAME[interval]
        print(f"\n[{i}/{len(intervals)}] {tag}")
        rows = download_interval(interval, output_dir)
        results[tag] = rows

    elapsed = time.time() - t0
    m, s = divmod(int(elapsed), 60)

    print(f"\n{'=' * 60}")
    print(f"全部完成（耗时 {m}m{s}s）：")
    print(f"{'=' * 60}")
    for tag, rows in results.items():
        print(f"  {tag:5s} → {rows:>10,} 行")
    print(f"\n数据目录: {output_dir}")


if __name__ == "__main__":
    main()
