"""ウォッチリスト用データ:
- universe.json: 全銘柄の基本指標 (サイトのウォッチ画面が任意の銘柄を表示するため)
- alerts.json: 全銘柄のうち「当日イベント」がある銘柄のフラグ
  (当日開示 / 前日比±5% / 52週高値 / 25日線が75日線をゴールデンクロス)
サイト側でユーザーのウォッチ銘柄(端末保存)と突き合わせて表示する。
"""
import json
import os

import numpy as np
import pandas as pd


def build_universe(df: pd.DataFrame, path="docs/data/universe.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    def _clean(v):
        if v is None:
            return None
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v
    out = {}
    for _, r in df.iterrows():
        out[r["code4"]] = {
            "n": r["name"], "s": r["sector"], "p": _clean(r["price"]),
            "m": _clean(r["mcap_oku"]), "per": _clean(r["per"]),
            "pbr": _clean(r["pbr"]),
            "y": _clean(r["yield"]), "roe": _clean(r["roe"]),
        }
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, allow_nan=False)
    print(f"universe: {len(out)} codes", flush=True)


def build_alerts(prices: pd.DataFrame, stmts: pd.DataFrame, df: pd.DataFrame,
                 path="docs/data/alerts.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not len(prices):
        return
    last_date = str(prices["Date"].max())
    name_map = dict(zip(df["code"], zip(df["code4"], df["name"])))

    disc_today = set()
    if len(stmts):
        disc_today = set(stmts[stmts["DiscDate"].astype(str) == last_date]["Code"].astype(str))

    items = []
    for code, g in prices.groupby("Code"):
        code = str(code)
        if code not in name_map:
            continue
        g = g.sort_values("Date")
        c = g["AdjC"].dropna()
        if len(c) < 2 or str(g["Date"].iloc[-1]) != last_date:
            continue
        c = c.tail(260).to_numpy(dtype=float)
        flags = []
        chg = (c[-1] / c[-2] - 1) * 100 if c[-2] else 0.0
        if chg >= 5:
            flags.append(f"急騰 +{chg:.1f}%")
        elif chg <= -5:
            flags.append(f"急落 {chg:.1f}%")
        if len(c) >= 100 and c[-1] >= np.max(c[:-1]):
            flags.append("52週高値")
        if len(c) >= 80:
            ma25 = pd.Series(c).rolling(25).mean()
            ma75 = pd.Series(c).rolling(75).mean()
            if (pd.notna(ma75.iloc[-2]) and ma25.iloc[-2] <= ma75.iloc[-2]
                    and ma25.iloc[-1] > ma75.iloc[-1]):
                flags.append("ゴールデンクロス(25/75)")
        if code in disc_today:
            flags.append("当日開示あり")
        if flags:
            code4, name = name_map[code]
            items.append({"code4": code4, "name": name, "flags": flags,
                          "chg": round(chg, 1)})
    out = {"as_of": last_date, "items": items}
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"alerts: {len(items)} codes with events", flush=True)
