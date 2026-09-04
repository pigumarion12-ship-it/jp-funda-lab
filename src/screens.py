"""3つのスクリーニング + 銘柄別指標計算.

タブ1: 弐億貯男式 (割安グロース)
タブ2: 清原式 (ネットキャッシュ系) ※フェーズ1はCashEq/PBRによる簡易版、EDINETで本計算へ
タブ3: 配当バリュー成長 (辻さん基準)

V2 fin_summary 実カラム (2026-08 実測):
  実績: Sales OP OdP NP EPS TA Eq EqAR BPS CFO CFI CFF CashEq DivAnn PayoutRatioAnn
  予想: FSales FOP FNP FEPS FDivAnn FPayoutRatioAnn (来期: NxF*)
  株式: ShOutFY(自己株含む発行済) TrShFY(自己株) AvgSh(期中平均)
"""
import json
import os
import numpy as np
import pandas as pd

OKU = 1e8  # 億円


def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _pct(v):
    """比率が0-1スケールなら%へ。"""
    v = _num(v)
    if v is None:
        return None
    return v * 100 if abs(v) <= 1.5 else v


def dump_schema(stmts, listed, divs, prices, path="reports/schema_dump.txt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for name, df in [("stmts", stmts), ("listed", listed), ("prices", prices)]:
            f.write(f"== {name} ({len(df)} rows) ==\n")
            f.write(", ".join(map(str, df.columns)) + "\n\n")


# ---- 銘柄別メトリクス --------------------------------------------------------

def _fy_rows(stmts: pd.DataFrame) -> pd.DataFrame:
    """通期(FY)実績のみ・銘柄ごと期末昇順。REIT除外・連結優先。"""
    df = stmts[stmts["CurPerType"].astype(str) == "FY"].copy()
    df = df[df["DocType"].astype(str).str.contains("FinancialStatements", na=False)]
    df = df[~df["DocType"].astype(str).str.contains("REIT", na=False)]
    df["_cons"] = (~df["DocType"].astype(str).str.contains("NonConsolidated")).astype(int)
    # 同一期末は 開示日→連結優先 で最後を採用
    df = df.sort_values(["Code", "CurPerEn", "DiscDate", "_cons"])
    df = df.drop_duplicates(subset=["Code", "CurPerEn"], keep="last")
    return df


FC_COLS = ["FSales", "FOP", "FOdP", "FNP", "FEPS", "FDivAnn", "FPayoutRatioAnn"]


def _latest_forecast(grp_all: pd.DataFrame) -> dict:
    """最新開示から今期予想を拾う (各項目ごとに直近の非欠損)。"""
    out = {}
    g = grp_all.sort_values(["DiscDate", "DiscTime"] if "DiscTime" in grp_all.columns
                            else "DiscDate")
    for c in FC_COLS:
        if c in g.columns:
            s = pd.to_numeric(g[c], errors="coerce").dropna()
            if len(s):
                out[c] = float(s.iloc[-1])
    return out


def _quarter_check(grp_all: pd.DataFrame, fy_end: str) -> dict:
    """当期の直近四半期(累計)の営業損益・前年同期比・季節性を判定。"""
    out = {"q_type": None, "q_op_oku": None, "q_loss": None,
           "q_seasonal": None, "q_yoy_down": None}
    if grp_all is None or not len(grp_all):
        return out
    q = grp_all[grp_all["CurPerType"].astype(str).isin(["1Q", "2Q", "3Q"])].copy()
    q = q[q["DocType"].astype(str).str.contains("FinancialStatements", na=False)]
    q = q[~q["DocType"].astype(str).str.contains("REIT", na=False)]
    if not len(q):
        return out
    q = q.sort_values(["CurPerEn", "DiscDate"]).drop_duplicates(
        subset=["CurPerEn"], keep="last")
    latest = q.iloc[-1]
    # 直近FY期末より後の四半期のみ(=進行中の期)
    if str(latest["CurPerEn"])[:10] <= fy_end:
        return out
    q_op = _num(latest["OP"])
    q_np = _num(latest["NP"])
    out["q_type"] = str(latest["CurPerType"])
    out["q_op_oku"] = round(q_op / OKU, 1) if q_op is not None else None
    basis = q_op if q_op is not None else q_np
    if basis is not None:
        out["q_loss"] = bool(basis < 0)
    # 前年同期(同じCurPerType・約1年前の期末)
    try:
        end = pd.Timestamp(latest["CurPerEn"])
        prior = q[(q["CurPerType"] == latest["CurPerType"])
                  & (pd.to_datetime(q["CurPerEn"]) < end - pd.Timedelta(days=270))
                  & (pd.to_datetime(q["CurPerEn"]) > end - pd.Timedelta(days=430))]
    except Exception:
        prior = q.iloc[0:0]
    if len(prior):
        p_op = _num(prior.iloc[-1]["OP"])
        if p_op is not None and basis is not None:
            out["q_yoy_down"] = bool(basis < p_op)
            if out["q_loss"]:
                out["q_seasonal"] = bool(p_op < 0)
    return out


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


def compute_metrics(stmts, listed, prices, divs=None) -> pd.DataFrame:
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
                "mkt": r.get("MktNm"),
            }

    all_by_code = dict(tuple(stmts.groupby("Code")))
    rows = []
    for code, grp in fy.groupby("Code"):
        code = str(code)
        info = linfo.get(code, {})
        # 株式市場(プライム/スタンダード/グロース)以外(REIT/ETF/Pro等)は除外
        if info.get("mkt") not in ("プライム", "スタンダード", "グロース"):
            continue
        px = last_px.get(code)
        if px is None or not np.isfinite(px):
            continue
        grp = grp.tail(6)
        latest = grp.iloc[-1]

        sales = pd.to_numeric(grp["Sales"], errors="coerce")
        op = pd.to_numeric(grp["OP"], errors="coerce")
        eps = pd.to_numeric(grp["EPS"], errors="coerce")
        dps = pd.to_numeric(grp["DivAnn"], errors="coerce")

        s_last, o_last = _num(latest["Sales"]), _num(latest["OP"])
        n_last, e_last = _num(latest["NP"]), _num(latest["EPS"])
        eq, cfo = _num(latest["Eq"]), _num(latest["CFO"])
        casheq, ta = _num(latest.get("CashEq")), _num(latest["TA"])
        bps = _num(latest.get("BPS"))

        m = {
            "code": code,
            "code4": code[:-1] if len(code) == 5 and code.endswith("0") else code,
            "name": info.get("name"), "sector": info.get("sector"),
            "scale": info.get("scale"), "mkt": info.get("mkt"),
            "price": round(float(px), 1), "as_of": as_of,
            "fy_end": str(latest["CurPerEn"])[:10],
            "sales_cagr3": _cagr(sales.tail(4)),
            "op_cagr3": _cagr(op.tail(4)),
            "op_margin": round(o_last / s_last * 100, 1) if s_last and o_last is not None else None,
            "roe": round(n_last / eq * 100, 1) if eq and n_last is not None else None,
            "eqar": round(_pct(latest["EqAR"]), 1) if _num(latest["EqAR"]) is not None else None,
            "cfo_pos": bool(cfo > 0) if cfo is not None else None,
            "eps_hist": [round(float(x), 1) for x in eps.dropna().tolist()][-5:],
            "dps_hist": [round(float(x), 2) for x in dps.dropna().tolist()][-5:],
            "sales_hist": [round(float(x) / OKU, 1) for x in sales.dropna().tolist()][-5:],
            "op_hist": [round(float(x) / OKU, 1) for x in op.dropna().tolist()][-5:],
        }

        # 直近四半期チェック (Q赤字・前年同期比・通期進捗)
        m.update(_quarter_check(all_by_code.get(code), str(latest["CurPerEn"])[:10]))

        # 2期連続減益(営業利益)の検出
        o = op.dropna()
        m["op_declining"] = bool(len(o) >= 3 and o.iloc[-1] < o.iloc[-2] < o.iloc[-3])

        # EPS上昇傾向: 上昇年比率>=0.6 かつ 最新>最古
        e = eps.dropna()
        if len(e) >= 3:
            ups = float((np.diff(e) > 0).mean())
            m["eps_uptrend"] = bool(ups >= 0.6 and e.iloc[-1] > e.iloc[0])
        else:
            m["eps_uptrend"] = None

        # 減配チェック(実績DivAnn系列)
        d = dps.dropna()
        if len(d) >= 3:
            m["div_no_cut"] = bool((np.diff(d) >= 0).all())
            m["div_up"] = bool(d.iloc[-1] > d.iloc[0])
        else:
            m["div_no_cut"] = None
            m["div_up"] = None

        # 株式数 → 時価総額
        sh_out, tr_sh = _num(latest.get("ShOutFY")), _num(latest.get("TrShFY"))
        shares = (sh_out - tr_sh) if sh_out and tr_sh is not None else sh_out
        if not shares and n_last and e_last:
            shares = abs(n_last) / abs(e_last)
        m["mcap_oku"] = round(px * shares / OKU, 1) if shares else None

        # 今期予想
        fc = _latest_forecast(all_by_code.get(code, grp))
        f_sales, f_op = fc.get("FSales"), fc.get("FOP")
        f_eps, f_dps = fc.get("FEPS"), fc.get("FDivAnn")
        m["f_sales_oku"] = round(f_sales / OKU, 1) if f_sales else None
        m["f_op_oku"] = round(f_op / OKU, 1) if f_op else None
        m["f_eps"] = round(f_eps, 1) if f_eps else None
        m["fcst_zoshu"] = bool(f_sales > s_last) if f_sales and s_last else None
        m["fcst_zoeki"] = bool(f_op > o_last) if f_op and o_last is not None else None

        # バリュエーション
        eps_for_per = f_eps if f_eps and f_eps > 0 else (e_last if e_last and e_last > 0 else None)
        m["per"] = round(px / eps_for_per, 1) if eps_for_per else None
        m["per_kind"] = "予" if (f_eps and f_eps > 0) else ("実" if eps_for_per else None)
        if bps and bps > 0:
            m["pbr"] = round(px / bps, 2)
        elif m["mcap_oku"] and eq and eq > 0:
            m["pbr"] = round(m["mcap_oku"] * OKU / eq, 2)
        else:
            m["pbr"] = None

        # 現金系 (清原式簡易版の材料)
        mcap = m["mcap_oku"] * OKU if m["mcap_oku"] else None
        if casheq is not None and mcap:
            m["cash_ratio"] = round(casheq / mcap * 100, 1)       # 現金/時価総額 %
            if ta is not None and eq is not None:
                m["netnet_lite"] = round((casheq - (ta - eq)) / mcap * 100, 1)  # (現金-総負債)/時価総額 %
            else:
                m["netnet_lite"] = None
        else:
            m["cash_ratio"] = None
            m["netnet_lite"] = None

        # 採点用の内部値 (億円)
        m["np_oku"] = round(n_last / OKU, 1) if n_last is not None else None
        m["eq_oku"] = round(eq / OKU, 1) if eq is not None else None
        m["ta_oku"] = round(ta / OKU, 1) if ta is not None else None
        m["casheq_oku"] = round(casheq / OKU, 1) if casheq is not None else None
        m["roa"] = round(n_last / ta * 100, 1) if ta and n_last is not None else None
        cfo_s = pd.to_numeric(grp["CFO"], errors="coerce")
        cfi_s = pd.to_numeric(grp["CFI"], errors="coerce")
        fcf_s = (cfo_s + cfi_s).dropna().tail(3)
        m["fcf_oku"] = round(float(fcf_s.mean()) / OKU, 1) if len(fcf_s) else None
        opm_s = (op / sales * 100).dropna().tail(3)
        m["opm_min3"] = round(float(opm_s.min()), 1) if len(opm_s) else None

        # 配当
        dps_now = f_dps if f_dps is not None else (float(d.iloc[-1]) if len(d) else None)
        m["dps"] = round(dps_now, 2) if dps_now else None
        m["yield"] = round(dps_now / px * 100, 2) if dps_now and px else None
        po = fc.get("FPayoutRatioAnn")
        if po is None and _num(latest.get("PayoutRatioAnn")) is not None:
            po = _num(latest["PayoutRatioAnn"])
        if po is not None:
            m["payout"] = round(_pct(po), 1)
        elif dps_now and eps_for_per and eps_for_per > 0:
            m["payout"] = round(dps_now / eps_for_per * 100, 1)
        else:
            m["payout"] = None

        rows.append(m)
    return pd.DataFrame(rows)


# ---- スクリーニング ----------------------------------------------------------

def _f(v, default=-1e18):
    if v is None:
        return default
    if isinstance(v, float) and np.isnan(v):
        return default
    return v


def screen_niokutameo(df: pd.DataFrame) -> pd.DataFrame:
    """弐億貯男式: PER<=15 × 時価総額<=1000億 × 増収増益グロース。"""
    c = df[
        df["per"].apply(lambda v: 0 < _f(v, 1e18) <= 15)
        & df["mcap_oku"].apply(lambda v: 0 < _f(v, 1e18) <= 1000)
        & df["sales_cagr3"].apply(lambda v: _f(v) >= 10)
        & df["op_cagr3"].apply(lambda v: _f(v) >= 10)
        & df["roe"].apply(lambda v: _f(v) > 10)
        & (df["cfo_pos"] != False)  # noqa: E712
        & df["eqar"].apply(lambda v: _f(v) >= 40)
        & (df["fcst_zoshu"] == True)  # noqa: E712 予想確認できるもののみ
        & (df["fcst_zoeki"] == True)  # noqa: E712
    ].copy()

    def score(r):
        s = 0.0
        s += min(_f(r["sales_cagr3"], 0), 40)
        s += min(_f(r["op_cagr3"], 0), 40)
        s += max(0.0, 15 - _f(r["per"], 15)) * 3
        s += min(_f(r["roe"], 0), 30) * 0.5
        s += min(_f(r["eqar"], 0), 80) * 0.1
        return round(s, 1)

    c["score"] = c.apply(score, axis=1)
    return c.sort_values("score", ascending=False)


def screen_kiyohara(df: pd.DataFrame) -> pd.DataFrame:
    """清原式: ネットキャッシュ比率(流動資産+投資有価証券×0.7-総負債)/時価総額。
    EDINETデータがある銘柄は本計算、無い銘柄は現金同等物ベースの簡易判定。"""
    # ランキングは保守的NC比率(掛け目後)を優先、無ければ素のNC比率
    df = df.copy()
    df["_nc"] = df["netcash_cons"].where(df["netcash_cons"].notna(), df["netcash_ratio"]) \
        if "netcash_cons" in df.columns else df.get("netcash_ratio")
    has_nc = df["_nc"].notna()
    real = df[
        has_nc
        & df["_nc"].apply(lambda v: _f(v) >= 30)
        & df["mcap_oku"].apply(lambda v: 0 < _f(v, 1e18) <= 500)
        & df["per"].apply(lambda v: 0 < _f(v, 1e18) <= 12)
    ].copy()

    fallback = df[
        (~has_nc if isinstance(has_nc, pd.Series) else True)
        & df["mcap_oku"].apply(lambda v: 0 < _f(v, 1e18) <= 500)
        & df["per"].apply(lambda v: 0 < _f(v, 1e18) <= 12)
        & df["pbr"].apply(lambda v: 0 < _f(v, 1e18) <= 1.0)
        & df["eqar"].apply(lambda v: _f(v) >= 50)
        & (df["cfo_pos"] != False)  # noqa: E712
    ].copy()

    def score(r):
        s = 0.0
        ncr = r.get("_nc")
        if ncr is not None and not (isinstance(ncr, float) and np.isnan(ncr)):
            s += min(_f(ncr, 0), 200)                        # 保守的NC比率(掛け目後)
            s += 30 if _f(ncr, 0) >= 100 else 0              # 超割安ボーナス
        else:
            s += min(max(_f(r["cash_ratio"], 0), 0), 120) * 0.5
            s += max(0.0, _f(r["netnet_lite"], -100)) * 0.3
            s += max(0.0, 1.0 - _f(r["pbr"], 1.0)) * 50
        s += max(0.0, 12 - _f(r["per"], 12)) * 3
        s += max(0.0, (_f(r["eqar"], 50) - 50)) * 0.4
        s += 5 if _f(r["yield"], 0) >= 3 else 0
        if r.get("op_declining"):
            s -= 40                                          # 2期連続減益ペナルティ
        opm = r.get("op_margin")
        if opm is not None and _f(opm, 99) < 2:
            s -= 20                                          # 薄利ペナルティ
        return round(s, 1)

    c = pd.concat([real, fallback], ignore_index=True)
    if not len(c):
        c["score"] = []
        return c
    c["score"] = c.apply(score, axis=1)
    return c.sort_values("score", ascending=False)


def screen_dividend_growth(df: pd.DataFrame) -> pd.DataFrame:
    """辻さん基準: 配当バリュー成長。"""
    c = df[
        df["yield"].apply(lambda v: _f(v) >= 2.0)
        & df["per"].apply(lambda v: 5 <= _f(v, 1e18) <= 20)
        & (df["eps_uptrend"] == True)   # noqa: E712
        & (df["div_no_cut"] != False)   # noqa: E712
        & df["mcap_oku"].apply(lambda v: 0 < _f(v, 1e18) <= 3000)
        & df["payout"].apply(lambda v: _f(v, 0) <= 70)
    ].copy()

    def score(r):
        s = 0.0
        y = _f(r["yield"], 0)
        s += min(y, 5) * 12
        s += 10 if y >= 3 else 0
        po = r["payout"]
        if po is not None:
            if 30 <= po <= 40:
                s += 20
            elif 20 <= po <= 50:
                s += 10
        s += 10 if r["div_up"] else 0
        s += 10 if r["div_no_cut"] else 0
        per = _f(r["per"], 20)
        if per <= 20:
            s += (20 - per)
        pbr = r["pbr"]
        if pbr is not None and pbr <= 1.5:
            s += 5
        s += min(max(_f(r["sales_cagr3"], 0), 0), 15) * 0.5
        s += 5 if r["fcst_zoshu"] and r["fcst_zoeki"] else 0
        return round(s, 1)

    c["score"] = c.apply(score, axis=1)
    return c.sort_values("score", ascending=False)


def screen_sougou(df: pd.DataFrame) -> pd.DataFrame:
    """総合(三式ミックス): アークランド式A〜E(40%) × 弐億貯男式成長割安(25%)
    × バリュエーション(20%) × 清原式資産価値(15%)。"""
    if "arc_comp" not in df.columns:
        return df.iloc[0:0].copy()
    c = df[
        df["arc_comp"].apply(lambda v: _f(v) >= 2.0)
        & df["mcap_oku"].apply(lambda v: 0 < _f(v, 1e18) <= 1000)
        & df["per"].apply(lambda v: 0 < _f(v, 1e18) <= 18)
        & (df["cfo_pos"] != False)  # noqa: E712
    ].copy()
    if not len(c):
        c["score"] = []
        return c

    def parts(r):
        arc = _f(r["arc_comp"], 0) / 4 * 100
        g = (min(max(_f(r["sales_cagr3"], 0), 0), 25)
             + min(max(_f(r["op_cagr3"], 0), 0), 25)) * 1.6
        g += 10 if (r.get("fcst_zoshu") and r.get("fcst_zoeki")) else 0
        g += min(max(_f(r["roe"], 0), 0), 20)
        g = min(g, 100)
        v = min(max(0.0, 18 - _f(r["per"], 18)) * 5, 60)
        v += min(max(_f(r["yield"], 0), 0), 5) * 6
        v = min(v, 100)
        nc = r.get("netcash_cons")
        if nc is None or (isinstance(nc, float) and np.isnan(nc)):
            nc = r.get("netcash_ratio")
        a = min(max(_f(nc, 0), 0), 150) / 1.5
        total = 0.4 * arc + 0.25 * g + 0.2 * v + 0.15 * a
        if r.get("op_declining"):
            total -= 8
        return round(total, 1), round(arc), round(g), round(v), round(a)

    res = c.apply(parts, axis=1, result_type="expand")
    c["score"], c["sg_arc"], c["sg_growth"], c["sg_value"], c["sg_asset"] = \
        res[0], res[1], res[2], res[3], res[4]
    return c.sort_values("score", ascending=False)


# ---- 出力 --------------------------------------------------------------------

OUT_COLS = ["code", "code4", "name", "sector", "scale", "mkt", "price", "mcap_oku",
            "per", "per_kind", "pbr", "yield", "payout", "dps",
            "sales_cagr3", "op_cagr3", "op_margin", "roe", "eqar", "cfo_pos",
            "cash_ratio", "netnet_lite", "netcash_ratio", "netcash_cons",
            "op_declining", "seisan_oku", "edinet_end",
            "q_type", "q_op_oku", "q_loss", "q_seasonal", "q_yoy_down",
            "arc_comp", "sg_arc", "sg_growth", "sg_value", "sg_asset",
            "eps_uptrend", "div_no_cut", "div_up", "fcst_zoshu", "fcst_zoeki",
            "f_sales_oku", "f_op_oku", "f_eps",
            "eps_hist", "dps_hist", "sales_hist", "op_hist", "fy_end", "score"]


def _records(df: pd.DataFrame, limit=30):
    if not len(df):
        return []
    cols = [c for c in OUT_COLS if c in df.columns]
    recs = df.head(limit)[cols].to_dict(orient="records")
    out = []
    for r in recs:
        out.append({k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                    for k, v in r.items()})
    return out


def build_charts(prices: pd.DataFrame, codes5: set, path="docs/data/charts.json"):
    """サイトのポップアップチャート用に、掲載銘柄の直近1年の終値系列を出力。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    px = prices[prices["Code"].isin(codes5)]
    series = {}
    for code, g in px.groupby("Code"):
        g = g.sort_values("Date")
        g = g[g["AdjC"].notna()].tail(380)   # 200日線を6ヶ月窓で引くため長めに保持
        if len(g) < 20:
            continue
        code4 = code[:-1] if len(code) == 5 and code.endswith("0") else code
        tail = g.tail(130)  # ローソク足表示分
        cc = tail["AdjC"]
        series[code4] = {
            "c": [round(float(x), 1) for x in g["AdjC"]],
            "o": [round(float(x), 1) for x in tail["AdjO"].fillna(cc)],
            "h": [round(float(x), 1) for x in tail["AdjH"].fillna(cc)],
            "l": [round(float(x), 1) for x in tail["AdjL"].fillna(cc)],
            "s": str(g["Date"].iloc[0]), "e": str(g["Date"].iloc[-1]),
        }
    with open(path, "w") as f:
        json.dump({"series": series}, f, ensure_ascii=False)
    print(f"charts: {len(series)} codes", flush=True)


def _mark_new(screens_dict, new_as_of, path):
    """前回のlatest.jsonと比較して新規入り銘柄にis_newを付ける。
    同じ基準日での再ビルド時は前回のフラグを引き継ぐ。"""
    old_screens, old_as_of = {}, None
    if os.path.exists(path):
        try:
            old = json.load(open(path))
            old_as_of = old.get("as_of")
            old_screens = old.get("screens", {})
        except Exception:
            pass
    for key, s in screens_dict.items():
        old_items = (old_screens.get(key) or {}).get("items", [])
        old_codes = {it["code4"] for it in old_items}
        old_flags = {it["code4"]: it.get("is_new", False) for it in old_items}
        for it in s["items"]:
            if old_as_of and old_as_of == new_as_of:
                it["is_new"] = old_flags.get(it["code4"], it["code4"] not in old_codes)
            elif old_codes:
                it["is_new"] = it["code4"] not in old_codes
            else:
                it["is_new"] = False


def build_trade_charts(prices: pd.DataFrame, trades_path="docs/data/trades.json",
                       path="docs/data/trade_charts.json"):
    """トレードノートの銘柄について、日付つきOHLCを出力(売買マーカー描画用)。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    codes4 = set()
    if os.path.exists(trades_path):
        try:
            t = json.load(open(trades_path))
            codes4 = {f["code4"] for f in t.get("fills", [])}
            codes4 |= {r.get("code4") for r in t.get("realized", []) if r.get("code4")}
        except Exception:
            pass
    out = {}
    if codes4 and len(prices):
        codes5 = {c + "0" for c in codes4} | codes4
        px = prices[prices["Code"].isin(codes5)]
        for code, g in px.groupby("Code"):
            g = g.sort_values("Date")
            g = g[g["AdjC"].notna()]
            if len(g) < 5:
                continue
            code4 = code[:-1] if len(code) == 5 and code.endswith("0") else code
            cc = g["AdjC"]
            out[code4] = {
                "d": [str(x) for x in g["Date"]],
                "o": [round(float(x), 1) for x in g["AdjO"].fillna(cc)],
                "h": [round(float(x), 1) for x in g["AdjH"].fillna(cc)],
                "l": [round(float(x), 1) for x in g["AdjL"].fillna(cc)],
                "c": [round(float(x), 1) for x in cc],
            }
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"trade_charts: {len(out)} codes", flush=True)


def build_output(df: pd.DataFrame, path="docs/data/latest.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = screen_niokutameo(df)
    k = screen_kiyohara(df)
    d = screen_dividend_growth(df)
    sg = screen_sougou(df)
    as_of = (df["as_of"].dropna().iloc[0] if len(df) and df["as_of"].notna().any() else None)
    screens_dict = {
        "sougou": {"label": "総合(三式ミックス)", "count": int(len(sg)), "items": _records(sg)},
        "niokutameo": {"label": "弐億貯男式", "count": int(len(n)), "items": _records(n)},
        "kiyohara": {"label": "清原式", "count": int(len(k)), "items": _records(k)},
        "dividend": {"label": "配当バリュー成長", "count": int(len(d)), "items": _records(d)},
    }
    _mark_new(screens_dict, as_of, path)
    out = {
        "as_of": as_of,
        "generated_at": pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y-%m-%d %H:%M"),
        "universe": int(len(df)),
        "screens": screens_dict,
        "notes": {
            "sougou": "総合点 = アークランド式A〜E評価40% + 成長割安(CAGR/ROE/増収増益予想)25% + バリュエーション(PER/利回り)20% + 資産価値(保守NC比率)15%。2期連続減益は-8点。",
            "kiyohara": "ランキングは保守的NC比率=(現金100%+有価証券100%+売掛85%+在庫50%+その他流動50%+投資有価証券70%−総負債)÷時価総額。2期連続減益は−40点、営業利益率2%未満は−20点。",
            "data": "出所: J-Quants (Standard)。PERの「予」は会社予想EPS、「実」は直近実績EPSベース。",
        },
    }
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"screens: 総合={len(sg)} 弐億貯男式={len(n)} 清原式={len(k)} 配当={len(d)} "
          f"universe={len(df)}", flush=True)
    return out
