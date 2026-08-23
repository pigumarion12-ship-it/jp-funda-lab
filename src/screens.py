"""3つのスクリーニング + 銘柄別指標計算.

タブ1: 弐億貯男式 (割安グロース)
タブ2: 清原式 (ネットキャッシュ系) ※フェーズ1はEqAR/PBRによる暫定版
タブ3: 配当バリュー成長 (辻さん基準)

V2カラム名が未確定の項目は候補リストで防御的に解決し、
build時に reports/schema_dump.txt へ実カラムを出力して確認する。
"""
import json
import os
import numpy as np
import pandas as pd

OKU = 1e8  # 億円

# ---- カラム名の防御的解決 ---------------------------------------------------

CAND = {
    "sales": ["Sales"],
    "op": ["OP"],
    "np": ["NP"],
    "eps": ["EPS"],
    "eq": ["Eq"],
    "ta": ["TA"],
    "eqar": ["EqAR"],
    "cfo": ["CFO"],
    # 予想系 (実データで要確認)
    "f_sales": ["FcstSales", "ForecastSales", "NextYrFcstSales"],
    "f_op": ["FcstOP", "ForecastOP", "NextYrFcstOP"],
    "f_np": ["FcstNP", "ForecastNP", "NextYrFcstNP"],
    "f_eps": ["FcstEPS", "ForecastEPS", "NextYrFcstEPS"],
    # 配当系 (実データで要確認)
    "dps": ["DPSAnn", "ResultDPSAnn", "DPS", "ResultDividendPerShareAnnual"],
    "f_dps": ["FcstDPSAnn", "ForecastDPSAnn", "FcstDPS",
              "ForecastDividendPerShareAnnual"],
    # 発行済株式数 (実データで要確認)
    "shares": ["IssuedShares", "NumShares", "SharesOutstandingFY",
               "NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock"],
}


def col(df: pd.DataFrame, key: str):
    for c in CAND.get(key, []):
        if c in df.columns:
            return c
    return None


def num(row, c):
    if c is None:
        return None
    v = row.get(c)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def dump_schema(stmts, listed, divs, prices, path="reports/schema_dump.txt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for name, df in [("stmts", stmts), ("listed", listed),
                         ("dividends", divs), ("prices", prices)]:
            f.write(f"== {name} ({len(df)} rows) ==\n")
            f.write(", ".join(map(str, df.columns)) + "\n")
            if len(df):
                sample = df[df["Code"] == "72030"] if "Code" in df.columns else df
                if not len(sample):
                    sample = df
                f.write(sample.tail(3).to_string() + "\n")
            f.write("\n")
        if len(stmts):
            for c in ("DocType", "CurPerType"):
                if c in stmts.columns:
                    f.write(f"stmts.{c} values: {stmts[c].astype(str).value_counts().head(30).to_dict()}\n")


# ---- 銘柄別メトリクス --------------------------------------------------------

def _fy_rows(stmts: pd.DataFrame) -> pd.DataFrame:
    """通期(FY)実績のみ・銘柄ごと期末昇順。予想/REIT/ETFは除外。"""
    df = stmts[stmts["CurPerType"].astype(str) == "FY"].copy()
    if "DocType" in df.columns:
        df = df[~df["DocType"].astype(str).str.contains("Forecast|REIT|ETF", na=False)]
    df = df.sort_values(["Code", "CurPerEn", "DiscDate"])
    df = df.drop_duplicates(subset=["Code", "CurPerEn"], keep="last")
    return df


def _forecast_row(grp_all: pd.DataFrame):
    """その銘柄の最新開示行から今期予想を拾う。
    1) Fcst系カラムがあれば最新行の値
    2) DocTypeにForecastを含む行があればその実数カラム
    """
    if not len(grp_all):
        return {}
    latest = grp_all.sort_values("DiscDate").iloc[-1]
    out = {}
    for k in ("f_sales", "f_op", "f_np", "f_eps", "f_dps"):
        c = col(grp_all, k)
        if c is not None:
            # 最新行が欠損なら遡って直近の非欠損を拾う
            s = pd.to_numeric(grp_all.sort_values("DiscDate")[c], errors="coerce").dropna()
            if len(s):
                out[k] = float(s.iloc[-1])
    if not out and "DocType" in grp_all.columns:
        fc = grp_all[grp_all["DocType"].astype(str).str.contains("Forecast", na=False)]
        if len(fc):
            r = fc.sort_values("DiscDate").iloc[-1]
            for k, base in [("f_sales", "sales"), ("f_op", "op"),
                            ("f_np", "np"), ("f_eps", "eps")]:
                v = num(r, col(grp_all, base))
                if v is not None:
                    out[k] = v
    return out


def _series(grp: pd.DataFrame, key: str) -> pd.Series:
    c = col(grp, key)
    if c is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(grp[c], errors="coerce")


def _cagr(s: pd.Series) -> float | None:
    s = s.dropna()
    s = s[s > 0]
    if len(s) < 3:
        return None
    n = len(s) - 1
    try:
        return round((float(s.iloc[-1]) / float(s.iloc[0])) ** (1 / n) * 100 - 100, 1)
    except (ValueError, ZeroDivisionError):
        return None


def _div_metrics(div_one: pd.DataFrame, fy_ends: list) -> dict:
    """配当キャッシュから年間DPS系列を推定 (カラム未確定のため防御的)。"""
    out = {"dps_hist": None, "div_no_cut": None, "div_up": None}
    if div_one is None or not len(div_one):
        return out
    # 年間配当らしきカラムを探す
    cand = [c for c in div_one.columns
            if any(k in c.lower() for k in ("ann", "annual"))
            and any(k in c.lower() for k in ("dps", "div"))]
    if not cand:
        return out
    c = cand[0]
    s = pd.to_numeric(div_one[c], errors="coerce").dropna()
    if len(s) < 2:
        return out
    hist = s.tolist()[-6:]
    out["dps_hist"] = [round(x, 2) for x in hist]
    diffs = np.diff(hist)
    out["div_no_cut"] = bool((diffs >= 0).all())
    out["div_up"] = bool(hist[-1] > hist[0])
    return out


def compute_metrics(stmts, listed, prices, divs) -> pd.DataFrame:
    fy = _fy_rows(stmts)
    last_px = (prices.sort_values("Date").groupby("Code")["AdjC"].last()
               if len(prices) else pd.Series(dtype=float))
    as_of = str(prices["Date"].max()) if len(prices) else None

    linfo = {}
    if len(listed):
        for _, r in listed.iterrows():
            linfo[str(r["Code"])] = {
                "name": r.get("CoName"),
                "sector": r.get("S33Nm"),
                "scale": r.get("ScaleCat"),
                "margin": r.get("MrgnNm"),
            }

    div_by_code = dict(tuple(divs.groupby("Code"))) if len(divs) and "Code" in divs.columns else {}
    all_by_code = dict(tuple(stmts.groupby("Code")))

    rows = []
    for code, grp in fy.groupby("Code"):
        code = str(code)
        info = linfo.get(code, {})
        # ETF/REIT等は listed 側の名称で除外しづらいので stmts 由来のみ
        grp = grp.tail(5)
        px = last_px.get(code)
        if px is None or not np.isfinite(px):
            continue
        latest = grp.iloc[-1]

        sales = _series(grp, "sales")
        op = _series(grp, "op")
        npi = _series(grp, "np")
        eps = _series(grp, "eps")

        m = {
            "code": code,
            "code4": code[:-1] if len(code) == 5 and code.endswith("0") else code,
            "name": info.get("name"),
            "sector": info.get("sector"),
            "scale": info.get("scale"),
            "price": round(float(px), 1),
            "as_of": as_of,
            "fy_end": str(latest.get("CurPerEn"))[:10],
            "sales_cagr3": _cagr(sales.tail(4)),
            "op_cagr3": _cagr(op.tail(4)),
            "op_margin": None,
            "roe": None, "eqar": None, "cfo_pos": None,
            "eps_hist": [round(float(x), 1) for x in eps.dropna().tolist()][-5:],
            "sales_hist": [round(float(x) / OKU, 1) for x in sales.dropna().tolist()][-5:],
            "op_hist": [round(float(x) / OKU, 1) for x in op.dropna().tolist()][-5:],
        }
        s_last, o_last = num(latest, col(grp, "sales")), num(latest, col(grp, "op"))
        n_last, e_last = num(latest, col(grp, "np")), num(latest, col(grp, "eps"))
        eq, ta = num(latest, col(grp, "eq")), num(latest, col(grp, "ta"))
        eqar, cfo = num(latest, col(grp, "eqar")), num(latest, col(grp, "cfo"))
        if s_last and o_last is not None:
            m["op_margin"] = round(o_last / s_last * 100, 1)
        if eq and n_last is not None:
            m["roe"] = round(n_last / eq * 100, 1)
        if eqar is not None:
            m["eqar"] = round(eqar * 100 if eqar <= 1 else eqar, 1)
        if cfo is not None:
            m["cfo_pos"] = bool(cfo > 0)

        # EPS上昇傾向: 上昇年の比率>=0.6 かつ 最新>最古
        e = eps.dropna()
        if len(e) >= 3:
            ups = (np.diff(e) > 0).mean()
            m["eps_uptrend"] = bool(ups >= 0.6 and e.iloc[-1] > e.iloc[0])
            m["eps_up_ratio"] = round(float(ups), 2)
        else:
            m["eps_uptrend"] = None
            m["eps_up_ratio"] = None

        # 発行済株式数: カラムがあれば使用、なければ NP/EPS で推定
        shares = num(latest, col(grp, "shares"))
        if not shares and n_last and e_last:
            shares = abs(n_last) / abs(e_last) if e_last else None
        m["mcap_oku"] = round(px * shares / OKU, 1) if shares else None

        # 予想
        fc = _forecast_row(all_by_code.get(code, grp))
        f_eps = fc.get("f_eps")
        f_dps = fc.get("f_dps")
        m["f_sales_oku"] = round(fc["f_sales"] / OKU, 1) if fc.get("f_sales") else None
        m["f_op_oku"] = round(fc["f_op"] / OKU, 1) if fc.get("f_op") else None
        m["f_eps"] = round(f_eps, 1) if f_eps else None
        m["fcst_zoshu"] = (bool(fc["f_sales"] > s_last)
                          if fc.get("f_sales") and s_last else None)
        m["fcst_zoeki"] = (bool(fc["f_op"] > o_last)
                          if fc.get("f_op") and o_last is not None else None)

        # バリュエーション
        eps_for_per = f_eps if f_eps and f_eps > 0 else (e_last if e_last and e_last > 0 else None)
        m["per"] = round(px / eps_for_per, 1) if eps_for_per else None
        m["per_kind"] = "予" if (f_eps and f_eps > 0) else ("実" if eps_for_per else None)
        m["pbr"] = (round(m["mcap_oku"] * OKU / eq, 2)
                    if m["mcap_oku"] and eq and eq > 0 else None)

        # 配当
        dm = _div_metrics(div_by_code.get(code), None)
        m.update(dm)
        dps_now = f_dps
        if dps_now is None and dm.get("dps_hist"):
            dps_now = dm["dps_hist"][-1]
        m["dps"] = round(dps_now, 2) if dps_now else None
        m["yield"] = round(dps_now / px * 100, 2) if dps_now and px else None
        eps_for_payout = f_eps if f_eps and f_eps > 0 else e_last
        m["payout"] = (round(dps_now / eps_for_payout * 100, 1)
                       if dps_now and eps_for_payout and eps_for_payout > 0 else None)

        rows.append(m)
    return pd.DataFrame(rows)


# ---- スクリーニング ----------------------------------------------------------

def _f(v, default=-1e18):
    return v if v is not None and not (isinstance(v, float) and np.isnan(v)) else default


def screen_niokutameo(df: pd.DataFrame) -> pd.DataFrame:
    """弐億貯男式: PER<=15 × 時価総額<=1000億 × 増収増益グロース。"""
    c = df[
        df["per"].apply(lambda v: _f(v, 1e18) <= 15)
        & df["mcap_oku"].apply(lambda v: 0 < _f(v, 1e18) <= 1000)
        & df["sales_cagr3"].apply(lambda v: _f(v) >= 10)
        & df["op_cagr3"].apply(lambda v: _f(v) >= 10)
        & df["roe"].apply(lambda v: _f(v) > 10)
        & (df["cfo_pos"] != False)  # noqa: E712 (None許容)
        & df["eqar"].apply(lambda v: _f(v) >= 40)
        & (df["fcst_zoshu"] != False)
        & (df["fcst_zoeki"] != False)
    ].copy()

    def score(r):
        s = 0.0
        s += min(_f(r["sales_cagr3"], 0), 40)
        s += min(_f(r["op_cagr3"], 0), 40)
        s += max(0, 15 - _f(r["per"], 15)) * 3          # 割安ほど加点
        s += min(_f(r["roe"], 0), 30) * 0.5
        s += 10 if r["fcst_zoshu"] and r["fcst_zoeki"] else 0
        s += min(_f(r["eqar"], 0), 80) * 0.1
        return round(s, 1)

    c["score"] = c.apply(score, axis=1)
    return c.sort_values("score", ascending=False)


def screen_kiyohara(df: pd.DataFrame) -> pd.DataFrame:
    """清原式(暫定): 小型×低PER×低PBR×自己資本厚め。
    本来はネットキャッシュ比率 = (現金+投資有価証券*0.7-有利子負債)/時価総額。
    EDINET接続(フェーズ2)で本計算に置き換える。"""
    c = df[
        df["mcap_oku"].apply(lambda v: 0 < _f(v, 1e18) <= 500)
        & df["per"].apply(lambda v: 0 < _f(v, 1e18) <= 10)
        & df["pbr"].apply(lambda v: 0 < _f(v, 1e18) <= 1.0)
        & df["eqar"].apply(lambda v: _f(v) >= 60)
        & (df["cfo_pos"] != False)
    ].copy()

    def score(r):
        s = 0.0
        s += max(0, 1.0 - _f(r["pbr"], 1.0)) * 60
        s += max(0, 10 - _f(r["per"], 10)) * 4
        s += (_f(r["eqar"], 60) - 60) * 0.8
        s += min(max(_f(r["roe"], 0), 0), 15)
        s += 5 if _f(r["yield"], 0) >= 3 else 0
        return round(s, 1)

    c["score"] = c.apply(score, axis=1)
    return c.sort_values("score", ascending=False)


def screen_dividend_growth(df: pd.DataFrame) -> pd.DataFrame:
    """辻さん基準: 配当バリュー成長。
    利回り2%以上(3%で割安) × 配当性向30-40%理想 × 減配なし ×
    EPSおおむね右肩上がり × PER10-20 × PBR高すぎない。"""
    c = df[
        df["yield"].apply(lambda v: _f(v) >= 2.0)
        & df["per"].apply(lambda v: 5 <= _f(v, 1e18) <= 20)
        & (df["eps_uptrend"] != False)
        & (df["div_no_cut"] != False)
        & df["mcap_oku"].apply(lambda v: 0 < _f(v, 1e18) <= 3000)
    ].copy()

    def score(r):
        s = 0.0
        y = _f(r["yield"], 0)
        s += min(y, 5) * 12                              # 利回り(3%~で大きく)
        s += 10 if y >= 3 else 0
        po = r["payout"]
        if po is not None:
            if 30 <= po <= 40:
                s += 20                                   # 理想レンジ
            elif 20 <= po <= 50:
                s += 10
            elif po > 70:
                s -= 15                                   # 配当過剰
        s += 15 if r["eps_uptrend"] else 0
        s += 10 if r["div_up"] else 0
        s += 10 if r["div_no_cut"] else 0
        per = _f(r["per"], 20)
        if 10 <= per <= 20:
            s += (20 - per)                               # 割安ほど加点
        pbr = r["pbr"]
        if pbr is not None and pbr <= 1.5:
            s += 5
        s += min(max(_f(r["sales_cagr3"], 0), 0), 15) * 0.5
        return round(s, 1)

    c["score"] = c.apply(score, axis=1)
    return c.sort_values("score", ascending=False)


# ---- 出力 --------------------------------------------------------------------

OUT_COLS = ["code", "code4", "name", "sector", "scale", "price", "mcap_oku",
            "per", "per_kind", "pbr", "yield", "payout", "dps",
            "sales_cagr3", "op_cagr3", "op_margin", "roe", "eqar", "cfo_pos",
            "eps_uptrend", "div_no_cut", "div_up", "fcst_zoshu", "fcst_zoeki",
            "eps_hist", "dps_hist", "sales_hist", "op_hist", "fy_end", "score"]


def _records(df: pd.DataFrame, limit=80):
    if not len(df):
        return []
    cols = [c for c in OUT_COLS if c in df.columns]
    recs = df.head(limit)[cols].to_dict(orient="records")
    out = []
    for r in recs:
        out.append({k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                    for k, v in r.items()})
    return out


def build_output(df: pd.DataFrame, path="docs/data/latest.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = screen_niokutameo(df)
    k = screen_kiyohara(df)
    d = screen_dividend_growth(df)
    out = {
        "as_of": (df["as_of"].dropna().iloc[0] if len(df) and df["as_of"].notna().any() else None),
        "generated_at": pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y-%m-%d %H:%M"),
        "universe": int(len(df)),
        "screens": {
            "niokutameo": {"label": "弐億貯男式", "count": int(len(n)), "items": _records(n)},
            "kiyohara": {"label": "清原式(暫定)", "count": int(len(k)), "items": _records(k)},
            "dividend": {"label": "配当バリュー成長", "count": int(len(d)), "items": _records(d)},
        },
        "notes": {
            "kiyohara": "ネットキャッシュ比率はEDINET接続後に本計算へ置換予定(現在はPBR/自己資本比率による暫定)",
            "data": "出所: J-Quants (Standard)。PERの「予」は会社予想EPS、「実」は直近実績EPSベース。",
        },
    }
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"screens: 弐億貯男式={len(n)} 清原式={len(k)} 配当={len(d)} universe={len(df)}",
          flush=True)
    return out
