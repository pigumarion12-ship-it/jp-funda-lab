"""Data cache: fetch incrementally from J-Quants V2, persist as parquet.

jp-stock-screener の data.py を土台にしたファンダ用版。
- prices: 日足 550日 (時価総額計算・将来のチャート連携用)
- stmts: 財務サマリー 5.5年分 (FY実績4-5期 + 今期予想)
- dividends: 配当情報 5.5年分 (減配チェック・利回り)
"""
import os
import re
import time
import datetime as dt
import numpy as np
import pandas as pd

REQ_INTERVAL = float(os.environ.get("JQ_REQ_INTERVAL", "1.2"))  # 秒/リクエスト

CACHE_DIR = os.environ.get("CACHE_DIR", "cache")
PRICES_PQ = os.path.join(CACHE_DIR, "prices.parquet")
STMTS_PQ = os.path.join(CACHE_DIR, "statements.parquet")
LISTED_PQ = os.path.join(CACHE_DIR, "listed.parquet")
DIV_PQ = os.path.join(CACHE_DIR, "dividends.parquet")

PRICE_LOOKBACK_DAYS = 550
STMT_LOOKBACK_DAYS = int(365.25 * 5.5)
DIV_LOOKBACK_DAYS = int(365.25 * 5.5)


class RateLimited(Exception):
    pass


def _fetch_retry(fn, **kw):
    """429に当たったら段階的に待って再試行。改善しなければRateLimited。"""
    for wait in (0, 15, 30, 60, 120):
        if wait:
            print(f"rate limited, wait {wait}s...", flush=True)
            time.sleep(wait)
        try:
            time.sleep(REQ_INTERVAL)
            return fn(**kw)
        except Exception as e:
            if "429" not in str(e):
                raise
    raise RateLimited()


def _subscription_window(err_text: str):
    m = re.search(r"(\d{4}-\d{2}-\d{2}) ~ (\d{4}-\d{2}-\d{2})", err_text)
    if m:
        return dt.date.fromisoformat(m.group(1)), dt.date.fromisoformat(m.group(2))
    return None


def _biz_days(cli, start: dt.date, end: dt.date) -> list[str]:
    """取引カレンダー取得。プラン範囲外なら自動でクランプして再試行。"""
    try:
        cal = cli.get_mkt_calendar(
            from_yyyymmdd=start.strftime("%Y%m%d"), to_yyyymmdd=end.strftime("%Y%m%d")
        )
    except Exception as e:
        win = _subscription_window(str(e))
        if not win:
            raise
        s2, e2 = max(start, win[0]), min(end, win[1])
        print(f"plan window: {win[0]} ~ {win[1]} -> clamp {s2} ~ {e2}", flush=True)
        if s2 > e2:
            return []
        cal = cli.get_mkt_calendar(
            from_yyyymmdd=s2.strftime("%Y%m%d"), to_yyyymmdd=e2.strftime("%Y%m%d")
        )
    if cal.empty:
        return []
    days = cal[cal["HolDiv"].astype(str).isin(["1", "2"])]["Date"]
    return sorted(pd.to_datetime(days).dt.strftime("%Y-%m-%d").tolist())


def update_listed(cli) -> pd.DataFrame:
    df = cli.get_list()
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_parquet(LISTED_PQ)
    return df


def _incremental_by_day(cli, fetch_fn_name, pq_path, lookback_days, date_col,
                        dedup_keys, label, day_filter=None):
    """日付イテレーションの増分取得の共通形。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    today = dt.date.today()
    start = today - dt.timedelta(days=lookback_days)

    old, fetched = None, set()
    if os.path.exists(pq_path):
        old = pd.read_parquet(pq_path)
        if date_col in old.columns:
            fetched = set(old[date_col].unique())

    frames = [old] if old is not None else []
    days = _biz_days(cli, start, today)
    if day_filter:
        days = [d for d in days if day_filter(d)]
    new_days = [d for d in days if d not in fetched]
    print(f"{label}: {len(new_days)} days to fetch", flush=True)

    def _save():
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        keys = [k for k in (dedup_keys or []) if k in df.columns]
        if keys:
            df = df.drop_duplicates(subset=keys, keep="last")
        else:
            df = df.drop_duplicates()
        if "Code" in df.columns:
            df["Code"] = df["Code"].astype(str)
        df.to_parquet(pq_path)
        return df

    fn = getattr(cli, fetch_fn_name)
    for i, d in enumerate(new_days):
        try:
            df = _fetch_retry(fn, date_yyyymmdd=d.replace("-", ""))
        except RateLimited:
            print(f"{label}: 持続的なレート制限。{i}/{len(new_days)}日分まで保存して次回再開", flush=True)
            return _save()
        except Exception as e:
            if "400" in str(e) or "404" in str(e):
                continue
            raise
        if not df.empty:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
            frames.append(df)
        if i % 50 == 0:
            print(f"{label} {i}/{len(new_days)} {d} rows={len(df)}", flush=True)
            _save()
    return _save()


def update_prices(cli) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    today = dt.date.today()
    start = today - dt.timedelta(days=PRICE_LOOKBACK_DAYS)

    old, fetched = None, set()
    # 株式分割の遡及調整を取り込むため、PRICE_FULL_REFRESH=1 なら全量取り直し
    full_refresh = os.environ.get("PRICE_FULL_REFRESH", "") not in ("", "0", "false")
    if os.path.exists(PRICES_PQ) and not full_refresh:
        old = pd.read_parquet(PRICES_PQ)
        old = old[old["Date"] >= start.isoformat()]
        fetched = set(old["Date"].unique())
    elif full_refresh:
        print("prices: 全量再取得モード(分割調整を反映)", flush=True)

    frames = [old] if old is not None else []
    new_days = [d for d in _biz_days(cli, start, today) if d not in fetched]
    print(f"prices: {len(new_days)} days to fetch", flush=True)

    def _save():
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        df = df.drop_duplicates(subset=["Code", "Date"], keep="last")
        keep = ["Code", "Date", "AdjO", "AdjH", "AdjL", "AdjC", "AdjVo", "Va"]
        df = df[[c for c in keep if c in df.columns]].copy()
        for c in df.columns:
            if c not in ("Code", "Date"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["Code"] = df["Code"].astype(str)
        df.to_parquet(PRICES_PQ)
        return df

    for i, d in enumerate(new_days):
        try:
            df = _fetch_retry(cli.get_eq_bars_daily, date_yyyymmdd=d.replace("-", ""))
        except RateLimited:
            print(f"prices: 持続的なレート制限。{i}/{len(new_days)}日分まで保存して次回再開", flush=True)
            return _save()
        if not df.empty:
            frames.append(df)
        if i % 25 == 0:
            print(f"prices {i}/{len(new_days)} {d} rows={len(df)}", flush=True)
            _save()
    return _save()


def update_statements(cli) -> pd.DataFrame:
    return _incremental_by_day(
        cli, "get_fin_summary", STMTS_PQ, STMT_LOOKBACK_DAYS,
        "DiscDate", ["Code", "DiscDate", "DocType", "CurPerType"], "stmts",
    )


def update_dividends(cli) -> pd.DataFrame:
    """配当API(get_fin_dividend)はStandardプラン対象外(403実測)。
    年間配当は fin_summary の DivAnn/FDivAnn で代替できるため取得しない。"""
    return pd.DataFrame()





def adjust_splits(prices: pd.DataFrame, tol: float = 0.035) -> pd.DataFrame:
    """データ源の遡及調整前でも整合するよう、株式分割/併合を自前で検知して過去分を調整。
    前日比が 1/2,1/3,1/4,1/5,1/6,1/8,1/10 (併合は2〜10倍) に±3.5%で一致する日を
    分割日とみなし、それ以前の株価(始値/高値/安値/終値)に比率を掛け、出来高を割る。"""
    if not len(prices):
        return prices
    prices = prices.sort_values(["Code", "Date"]).reset_index(drop=True)
    cols = [c for c in ("AdjO", "AdjH", "AdjL", "AdjC") if c in prices.columns]
    n_fix = 0
    out = []
    for code, g in prices.groupby("Code", sort=False):
        c = g["AdjC"].to_numpy(dtype=float)
        if len(c) < 3:
            out.append(g)
            continue
        ratio = c[1:] / np.where(c[:-1] == 0, np.nan, c[:-1])
        g = g.copy()
        for i, r in enumerate(ratio):
            if not np.isfinite(r):
                continue
            # 東証の値幅制限では1日で-35%超/+60%超は起きないため、
            # それを超える前日比は分割(または併合)とみなし整数比に丸めて調整する
            factor = None
            if r <= 0.67:
                n = int(round(1 / r))
                if 2 <= n <= 20:
                    factor = 1 / n
            elif r >= 1.6:
                n = int(round(r))
                if 2 <= n <= 20:
                    factor = float(n)
            if factor is not None:
                idx = i + 1  # 分割後の最初の行
                for col in cols:
                    g.iloc[:idx, g.columns.get_loc(col)] = g[col].iloc[:idx] * factor
                if "AdjVo" in g.columns:
                    g.iloc[:idx, g.columns.get_loc("AdjVo")] = g["AdjVo"].iloc[:idx] / factor
                n_fix += 1
        out.append(g)
    if n_fix:
        print(f"adjust_splits: {n_fix} 件の分割/併合を自前調整", flush=True)
    return pd.concat(out, ignore_index=True)


def load_all():
    prices = pd.read_parquet(PRICES_PQ) if os.path.exists(PRICES_PQ) else pd.DataFrame()
    stmts = pd.read_parquet(STMTS_PQ) if os.path.exists(STMTS_PQ) else pd.DataFrame()
    listed = pd.read_parquet(LISTED_PQ) if os.path.exists(LISTED_PQ) else pd.DataFrame()
    divs = pd.read_parquet(DIV_PQ) if os.path.exists(DIV_PQ) else pd.DataFrame()
    return prices, stmts, listed, divs
