"""DUKE式 (新高値ブレイク投資) スクリーニング.

機械判定 8項目:
 1. 新高値: 過去1〜2年の高値を更新(=高値の99.5%以上) or 目前(97%以上)
 2. 時価総額 500億円以下
 3. 過去3〜5年の経常利益 年率5%以上で安定成長(年次で-20%超の落ち込みなし)
 4. 直近1〜2年の経常利益 20%以上増(前期実績 or 今期予想)。30%以上は理想
 5. 最新四半期(単独)の売上高 前年同期比+10%以上
 6. 最新四半期(単独)の経常利益 前年同期比+20%以上
 7. 経常利益率が上昇傾向(通期 or 四半期で前年比改善)
 8. 長い保ち合い(ボックス)からの上抜け
定性(材料・ビッグチェンジ・新高値の理由)は深掘りレポート側で判定。
"""
import json
import os
import numpy as np
import pandas as pd

from .screens import _fy_rows, _latest_forecast, _num

OKU = 1e8
PERIOD_ORDER = {"1Q": 1, "2Q": 2, "3Q": 3, "FY": 4}


def _quarters(grp_all: pd.DataFrame) -> pd.DataFrame:
    q = grp_all[grp_all["CurPerType"].astype(str).isin(["1Q", "2Q", "3Q", "FY"])].copy()
    q = q[q["DocType"].astype(str).str.contains("FinancialStatements", na=False)]
    q = q[~q["DocType"].astype(str).str.contains("REIT|NonConsolidated", na=False)]
    if not len(q):
        q = grp_all[grp_all["CurPerType"].astype(str).isin(["1Q", "2Q", "3Q", "FY"])].copy()
        q = q[q["DocType"].astype(str).str.contains("FinancialStatements", na=False)]
    q = q.sort_values(["CurPerEn", "DiscDate"]).drop_duplicates(subset=["CurPerEn"], keep="last")
    q["_ord"] = q["CurPerType"].astype(str).map(PERIOD_ORDER)
    for c in ("Sales", "OdP"):
        q[c] = pd.to_numeric(q[c], errors="coerce")
    q["_end"] = pd.to_datetime(q["CurPerEn"], errors="coerce")
    q["_fyst"] = pd.to_datetime(q.get("CurFYSt"), errors="coerce")
    return q


def _standalone(q: pd.DataFrame, row) -> tuple:
    """累計→単独四半期 (同一会計年度の直前累計を引く)。1Qはそのまま。"""
    s, o = row["Sales"], row["OdP"]
    if row["_ord"] == 1:
        return s, o
    same_fy = q[(q["_fyst"] == row["_fyst"]) & (q["_ord"] == row["_ord"] - 1)]
    if not len(same_fy):
        return None, None
    p = same_fy.iloc[-1]
    if pd.isna(s) or pd.isna(p["Sales"]):
        s2 = None
    else:
        s2 = float(s - p["Sales"])
    if pd.isna(o) or pd.isna(p["OdP"]):
        o2 = None
    else:
        o2 = float(o - p["OdP"])
    return s2, o2


def _yoy(cur, prev):
    if cur is None or prev is None or pd.isna(cur) or pd.isna(prev) or prev <= 0:
        return None
    return (cur / prev - 1) * 100


def compute(stmts, listed, prices, path="docs/data/duke.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fy = _fy_rows(stmts)
    all_by_code = dict(tuple(stmts.groupby("Code")))
    px_by_code = dict(tuple(prices.groupby("Code")))
    as_of = str(prices["Date"].max())
    linfo = {str(r["Code"]): (r.get("CoName"), r.get("S33Nm"), r.get("MktNm"))
             for _, r in listed.iterrows()}

    rows = []
    for code, grp in fy.groupby("Code"):
        code = str(code)
        name, sector, mkt = linfo.get(code, (None, None, None))
        if mkt not in ("プライム", "スタンダード", "グロース"):
            continue
        g = px_by_code.get(code)
        if g is None or len(g) < 120:
            continue
        c = g.sort_values("Date")["AdjC"].dropna().to_numpy(dtype=float)
        px = c[-1]
        latest = grp.iloc[-1]

        # ---- 1. 新高値 ----
        hi = float(np.max(c[-500:]))
        new_high = px >= hi * 0.995
        near_high = px >= hi * 0.97
        chk_high = bool(near_high)

        # ---- 2. 時価総額 ----
        shares = _num(latest.get("ShOutFY"))
        tr = _num(latest.get("TrShFY"))
        if shares and tr is not None:
            shares -= tr
        n_last, eps = _num(latest["NP"]), _num(latest["EPS"])
        if not shares and n_last and eps:
            shares = abs(n_last) / abs(eps)
        mcap = round(px * shares / OKU, 1) if shares else None
        chk_mcap = bool(mcap is not None and mcap <= 500)

        # ---- 3. 3〜5年の経常成長 ----
        odp = pd.to_numeric(grp["OdP"], errors="coerce").dropna()
        odp5 = odp.tail(5)
        cagr = None
        stable = None
        if len(odp5) >= 3 and (odp5 > 0).all():
            n = len(odp5) - 1
            cagr = (float(odp5.iloc[-1]) / float(odp5.iloc[0])) ** (1 / n) * 100 - 100
            yoys = odp5.pct_change().dropna() * 100
            stable = bool((yoys > -20).all())
        chk_long = bool(cagr is not None and cagr >= 5 and stable)

        # ---- 4. 直近1〜2年の経常 20%+ ----
        fc = _latest_forecast(all_by_code.get(code, grp))
        last_yoy = _yoy(float(odp.iloc[-1]), float(odp.iloc[-2])) if len(odp) >= 2 else None
        fc_yoy = _yoy(fc.get("FOdP"), float(odp.iloc[-1])) if len(odp) >= 1 else None
        best_recent = max([v for v in (last_yoy, fc_yoy) if v is not None], default=None)
        chk_recent = bool(best_recent is not None and best_recent >= 20)
        ideal_recent = bool(best_recent is not None and best_recent >= 30)

        # ---- 5,6. 最新四半期(単独) 前年同期比 ----
        q = _quarters(all_by_code.get(code, grp))
        q_sales_yoy = q_odp_yoy = None
        q_label = None
        q_margin_up = None
        if len(q):
            lat = q.iloc[-1]
            s1, o1 = _standalone(q, lat)
            prev = q[(q["_ord"] == lat["_ord"])
                     & (q["_end"] < lat["_end"] - pd.Timedelta(days=270))
                     & (q["_end"] > lat["_end"] - pd.Timedelta(days=430))]
            if len(prev):
                s0, o0 = _standalone(q, prev.iloc[-1])
                q_sales_yoy = _yoy(s1, s0)
                q_odp_yoy = _yoy(o1, o0)
                if s1 and s0 and o1 is not None and o0 is not None:
                    q_margin_up = bool((o1 / s1) > (o0 / s0))
            q_label = f"{lat['CurPerType']}" + ("(単独)" if lat["_ord"] != 1 else "")
        chk_q_sales = bool(q_sales_yoy is not None and q_sales_yoy >= 10)
        chk_q_odp = bool(q_odp_yoy is not None and q_odp_yoy >= 20)

        # ---- 7. 利益率上昇傾向 ----
        sales = pd.to_numeric(grp["Sales"], errors="coerce")
        fy_margin_up = None
        if len(grp) >= 2:
            def _margin(row):
                o, s = _num(row["OdP"]), _num(row["Sales"])
                return o / s if (o is not None and s) else None
            m1, m0 = _margin(latest), _margin(grp.iloc[-2])
            if m1 is not None and m0 is not None:
                fy_margin_up = bool(m1 > m0)
        chk_margin = bool(fy_margin_up or q_margin_up)

        # ---- 8. 保ち合い上抜け ----
        chk_box = False
        box_info = None
        if len(c) >= 140:
            box = c[-140:-20]
            bh, bl = float(np.max(box)), float(np.min(box))
            tight = bh / bl <= 1.35 if bl > 0 else False
            broke = float(np.max(c[-20:])) > bh and px >= bh * 0.97
            chk_box = bool(tight and broke)
            box_info = {"box_hi": round(bh, 1), "box_lo": round(bl, 1),
                        "width_pct": round((bh / bl - 1) * 100, 1) if bl > 0 else None}

        checks = {
            "新高値(更新/目前)": chk_high, "時価総額500億以下": chk_mcap,
            "3〜5年経常成長5%+": chk_long, "直近経常+20%": chk_recent,
            "最新Q売上+10%": chk_q_sales, "最新Q経常+20%": chk_q_odp,
            "利益率改善": chk_margin, "保ち合い上抜け": chk_box,
        }
        n_ok = sum(checks.values())
        if not chk_high or n_ok < 5:
            continue

        f_eps = fc.get("FEPS")
        rows.append({
            "code4": code[:-1] if len(code) == 5 and code.endswith("0") else code,
            "name": name, "sector": sector, "price": round(px, 1), "mcap_oku": mcap,
            "per": round(px / f_eps, 1) if f_eps and f_eps > 0 else None,
            "n_ok": int(n_ok), "checks": checks,
            "new_high": bool(new_high), "hi_ratio": round(px / hi * 100, 1),
            "odp_cagr": round(cagr, 1) if cagr is not None else None,
            "odp_last_yoy": round(last_yoy, 1) if last_yoy is not None else None,
            "odp_fc_yoy": round(fc_yoy, 1) if fc_yoy is not None else None,
            "ideal_recent": ideal_recent,
            "q_label": q_label,
            "q_sales_yoy": round(q_sales_yoy, 1) if q_sales_yoy is not None else None,
            "q_odp_yoy": round(q_odp_yoy, 1) if q_odp_yoy is not None else None,
            "box": box_info,
        })

    rows.sort(key=lambda r: (-r["n_ok"], -(r["q_odp_yoy"] or -999)))
    out = {"as_of": as_of, "count": len(rows),
           "full": sum(1 for r in rows if r["n_ok"] == 8), "items": rows[:60]}
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"duke: candidates={len(rows)} 8/8={out['full']}", flush=True)
    return out
