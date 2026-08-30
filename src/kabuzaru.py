"""株猿式スクリーニング (note記事の手法を機械化).

5軸×20点のうち機械計算できる4軸(成長性/収益性/財務/市場評価の機械部分)=80点満点で採点。
カタリスト軸(四季報コメント・月次等)は定性のため対象外(レポート側で補完)。

10基準(基礎体力): 売上/営利/経常/純利それぞれ10%成長(前々期→前期・前期→今期予想の両方)、
3期連続増収増益、今期減益予想なし、営利率10%+、ROE10%+、ROA5%+、営業CF+

機械的除外: 債務超過 / 直近3期で2期以上赤字 / 営業CF3期連続マイナス
"""
import json
import os
import numpy as np
import pandas as pd

from .screens import _fy_rows, _latest_forecast, _num, _pct

OKU = 1e8


def _grow(prev, cur):
    if prev is None or cur is None or prev <= 0:
        return None
    return (cur / prev - 1) * 100


def compute(stmts, listed, prices, path="reports/kabuzaru.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fy = _fy_rows(stmts)
    all_by_code = dict(tuple(stmts.groupby("Code")))
    last_px = prices.sort_values("Date").groupby("Code")["AdjC"].last()
    as_of = str(prices["Date"].max())

    linfo = {}
    for _, r in listed.iterrows():
        linfo[str(r["Code"])] = (r.get("CoName"), r.get("S33Nm"), r.get("MktNm"))

    px_by_code = dict(tuple(prices.groupby("Code")))
    rows = []
    for code, grp in fy.groupby("Code"):
        code = str(code)
        name, sector, mkt = linfo.get(code, (None, None, None))
        if mkt not in ("プライム", "スタンダード", "グロース"):
            continue
        grp = grp.tail(4)
        if len(grp) < 3:
            continue
        latest = grp.iloc[-1]
        S = lambda c: pd.to_numeric(grp[c], errors="coerce")
        sales, op, odp, npi = S("Sales"), S("OP"), S("OdP"), S("NP")
        cfo_s, cfi_s = S("CFO"), S("CFI")

        # ---- 機械的除外 ----
        eq = _num(latest["Eq"])
        if eq is not None and eq < 0:
            continue
        np3 = npi.dropna().tail(3)
        if len(np3) >= 2 and (np3 < 0).sum() >= 2:
            continue
        cfo3 = cfo_s.dropna().tail(3)
        if len(cfo3) >= 3 and (cfo3 < 0).all():
            continue

        fc = _latest_forecast(all_by_code.get(code, grp))
        px = last_px.get(code)
        if px is None or not np.isfinite(px):
            continue

        def two_leg(series, fkey):
            s = series.dropna()
            if len(s) < 2:
                return None, None, None
            g1 = _grow(float(s.iloc[-2]), float(s.iloc[-1]))
            g2 = _grow(float(s.iloc[-1]), fc.get(fkey))
            ok = g1 is not None and g1 >= 10 and g2 is not None and g2 >= 10
            return g1, g2, ok

        s_g1, s_g2, s_ok = two_leg(sales, "FSales")
        o_g1, o_g2, o_ok = two_leg(op, "FOP")
        d_g1, d_g2, d_ok = two_leg(odp, "FOdP")
        n_g1, n_g2, n_ok = two_leg(npi, "FNP")

        # ① 成長性 20点 (4項目×5点)
        growth_pts = sum(5 for ok in (s_ok, o_ok, d_ok, n_ok) if ok)

        # ② 収益性 20点
        s_last, o_last, n_last = _num(latest["Sales"]), _num(latest["OP"]), _num(latest["NP"])
        ta = _num(latest["TA"])
        opm = o_last / s_last * 100 if s_last and o_last is not None else None
        roe = n_last / eq * 100 if eq and n_last is not None else None
        roa = n_last / ta * 100 if ta and n_last is not None else None
        prof_pts = 0
        if opm is not None and opm >= 10:
            prof_pts += 7
        if roe is not None and roe >= 10:
            prof_pts += 7 if roe < 15 else 8
        if roa is not None and roa >= 5:
            prof_pts += 5
        prof_pts = min(prof_pts, 20)

        # ③ 財務 20点
        eqar = _pct(latest["EqAR"])
        cfo = _num(latest["CFO"])
        cfi = _num(latest["CFI"])
        fin_pts = 0
        if eqar is not None and eqar >= 30:
            fin_pts += 5 + (2 if eqar >= 50 else 0)
        if cfo is not None and cfo > 0:
            fin_pts += 6
        if cfo is not None and cfi is not None and (cfo + cfi) > 0:
            fin_pts += 5
        fcf3 = (cfo_s + cfi_s).dropna().tail(3)
        if len(fcf3) >= 2 and (fcf3 > 0).mean() >= 0.66:
            fin_pts += 2
        fin_pts = min(fin_pts, 20)

        # ④ 市場評価(機械部分) 20点
        mkt_pts = 0
        g = px_by_code.get(code)
        hi_zone = dc_ok = above_ma75 = None
        if g is not None and len(g) >= 80:
            c = g.sort_values("Date")["AdjC"].dropna().to_numpy(dtype=float)
            half = c[-126:] if len(c) >= 126 else c
            hi_zone = c[-1] >= np.max(half) * 0.90        # 半年以内の高値圏
            ma25 = np.mean(c[-25:])
            ma75 = np.mean(c[-75:])
            dc_ok = ma25 >= ma75                            # デッドクロス非発生
            above_ma75 = c[-1] > ma75
            if hi_zone:
                mkt_pts += 7
            if dc_ok:
                mkt_pts += 7
            if above_ma75:
                mkt_pts += 6

        score = growth_pts + prof_pts + fin_pts + mkt_pts   # 80点満点

        # 10基準
        def cont_up(series):
            s = series.dropna().tail(3)
            return len(s) >= 3 and s.iloc[-1] > s.iloc[-2] > s.iloc[-3]
        no_gensoku = (fc.get("FOP") is not None and o_last is not None
                      and fc["FOP"] >= o_last
                      and (fc.get("FNP") is None or n_last is None or fc["FNP"] >= n_last))
        ten = {
            "売上10%成長": bool(s_ok), "営利10%成長": bool(o_ok),
            "経常10%成長": bool(d_ok), "純利10%成長": bool(n_ok),
            "3期連続増収増益": bool(cont_up(sales) and cont_up(op)),
            "減益予想なし": bool(no_gensoku),
            "営利率10%+": bool(opm is not None and opm >= 10),
            "ROE10%+": bool(roe is not None and roe >= 10),
            "ROA5%+": bool(roa is not None and roa >= 5),
            "営業CF+": bool(cfo is not None and cfo > 0),
        }
        ten_ok = sum(ten.values())

        shares = _num(latest.get("ShOutFY"))
        tr = _num(latest.get("TrShFY"))
        if shares and tr is not None:
            shares -= tr
        if not shares and n_last and _num(latest["EPS"]):
            shares = abs(n_last) / abs(_num(latest["EPS"]))
        mcap = round(px * shares / OKU, 1) if shares else None
        f_eps = fc.get("FEPS")
        per = round(px / f_eps, 1) if f_eps and f_eps > 0 else None

        rows.append({
            "code4": code[:-1] if len(code) == 5 and code.endswith("0") else code,
            "name": name, "sector": sector, "mcap_oku": mcap, "per": per,
            "score80": score, "growth": growth_pts, "prof": prof_pts,
            "fin": fin_pts, "mkt": mkt_pts,
            "ten_ok": int(ten_ok), "ten": ten,
            "opm": round(opm, 1) if opm is not None else None,
            "roe": round(roe, 1) if roe is not None else None,
            "roa": round(roa, 1) if roa is not None else None,
            "eqar": round(eqar, 1) if eqar is not None else None,
            "sales_g": [round(x, 1) if x is not None else None for x in (s_g1, s_g2)],
            "op_g": [round(x, 1) if x is not None else None for x in (o_g1, o_g2)],
            "hi_zone": bool(hi_zone) if hi_zone is not None else None,
        })

    rows.sort(key=lambda r: (-r["ten_ok"], -r["score80"]))
    out = {"as_of": as_of, "universe": len(rows),
           "ten_all": sum(1 for r in rows if r["ten_ok"] == 10),
           "items": rows[:60]}
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"kabuzaru: universe={len(rows)} 10基準全クリア={out['ten_all']}", flush=True)
    return out
