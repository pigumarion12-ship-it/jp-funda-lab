"""アークランド式 (企業分析レポート手順) の機械化パート + 清原式本計算.

- 清原式ネットキャッシュ比率 = (流動資産 + 投資有価証券×0.7 − 総負債) / 時価総額
- 清算価値 = 現金100% + 有価証券100% + 売掛85% + 棚卸50% + 投資有価証券50%
             + 有形固定50% + 無形0% − 総負債
- DCF弱気 = ネットキャッシュ(現金+有価証券-有利子負債) + FCF/10% (成長なし)
  DCF強気 = 清算価値 + 純利益成長PV (5年 g=min(営利CAGR,20%), 以降成長なし, R=10%)
- 危険シグナル + 6項目A〜E評価 (⑥事業素質は定性のため対象外)

※スクリーニング(弐億貯男式など)とは独立した機能。混ぜない。
"""
import json
import os
import numpy as np
import pandas as pd

OKU = 1e8
R = 0.10


def _n(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def merge_edinet(df: pd.DataFrame, ed: pd.DataFrame) -> pd.DataFrame:
    """EDINET最新BSをJ-Quantsメトリクスにマージし、清原式・清算価値を計算。"""
    df = df.copy()
    for c in ("netcash_ratio", "netcash_cons", "seisan_oku", "dcf_weak_oku",
              "dcf_strong_oku", "edinet_end", "cur_ratio", "borrowings_oku"):
        df[c] = None
    if not len(ed):
        return df
    ed = ed.set_index("Code")
    for i, r in df.iterrows():
        e = ed.loc[r["code"]] if r["code"] in ed.index else None
        if e is None:
            continue
        mcap = _n(r["mcap_oku"])
        mcap_yen = mcap * OKU if mcap else None
        cash, sec_s = _n(e.get("cash")), _n(e.get("sec_short")) or 0.0
        inv = _n(e.get("inv_sec")) or 0.0
        ca, cl = _n(e.get("cur_assets")), _n(e.get("cur_liab"))
        liab = _n(e.get("liabilities"))
        recv = _n(e.get("receivables")) or 0.0
        inven = _n(e.get("inventories")) or 0.0
        ppe = _n(e.get("ppe")) or 0.0
        borr = _n(e.get("borrowings"))

        df.at[i, "edinet_end"] = e.get("periodEnd")
        if borr is not None:
            df.at[i, "borrowings_oku"] = round(borr / OKU, 1)
        if ca is not None and cl:
            df.at[i, "cur_ratio"] = round(ca / cl * 100, 1)
        # 清原式ネットキャッシュ比率
        if ca is not None and liab is not None and mcap_yen:
            nc = ca + inv * 0.7 - liab
            df.at[i, "netcash_ratio"] = round(nc / mcap_yen * 100, 1)
            # 保守的NC比率: 流動資産に掛け目(現金100/有価証券100/売掛85/在庫50/その他50)
            if cash is not None:
                other_ca = max(0.0, ca - cash - sec_s - recv - inven)
                adj_ca = cash + sec_s + recv * 0.85 + inven * 0.5 + other_ca * 0.5
                nc_cons = adj_ca + inv * 0.7 - liab
                df.at[i, "netcash_cons"] = round(nc_cons / mcap_yen * 100, 1)
        # 清算価値
        if cash is not None and liab is not None:
            sv = (cash + sec_s + recv * 0.85 + inven * 0.5 + inv * 0.5
                  + ppe * 0.5 - liab)
            df.at[i, "seisan_oku"] = round(sv / OKU, 1)
        # DCF
        fcf = _n(r.get("fcf_oku"))
        npr = _n(r.get("np_oku"))
        if cash is not None and borr is not None:
            nc2 = (cash + sec_s + inv - borr) / OKU
            weak = nc2 + (max(fcf, 0.0) / R if fcf is not None else 0.0)
            df.at[i, "dcf_weak_oku"] = round(weak, 1)
            sv = df.at[i, "seisan_oku"]
            if npr is not None and npr > 0 and sv is not None:
                g = min(max((_n(r.get("op_cagr3")) or 0.0) / 100, 0.0), 0.20)
                pv = sum(npr * ((1 + g) / (1 + R)) ** t for t in range(1, 6))
                pv += npr * (1 + g) ** 5 / R / (1 + R) ** 5
                df.at[i, "dcf_strong_oku"] = round(sv + pv, 1)
    return df


# ---- 危険シグナル / A〜E ------------------------------------------------------

def _danger(r) -> list[str]:
    d = []
    cr = _n(r.get("cur_ratio"))
    if cr is not None and cr < 100:
        d.append("流動負債>流動資産")
    ta, cash = _n(r.get("ta_oku")), _n(r.get("casheq_oku"))
    if ta and cash is not None and cash < ta * 0.02:
        d.append("現金極少")
    eqar = _n(r.get("eqar"))
    if eqar is not None and eqar <= 20:
        d.append("自己資本比率20%以下")
    eq = _n(r.get("eq_oku"))
    if eq is not None and eq < 0:
        d.append("債務超過")
    if r.get("cfo_pos") is False:
        d.append("営業CF赤字")
    borr, fcfo = _n(r.get("borrowings_oku")), _n(r.get("fcf_oku"))
    if borr and fcfo is not None and fcfo <= 0 and borr > 0:
        pass  # FCF赤字+借入はhealthで減点済み
    return d


def _g_asset(r):
    mcap, sv = _n(r.get("mcap_oku")), _n(r.get("seisan_oku"))
    pbr, ncr = _n(r.get("pbr")), _n(r.get("netcash_ratio"))
    if sv is not None and mcap:
        if mcap < sv and (ncr or 0) > 0:
            return "A"
        if mcap < sv * 1.2 or (pbr is not None and pbr < 1):
            return "B"
    if pbr is not None:
        if pbr < 1:
            return "B"
        if pbr <= 1.5:
            return "C"
        return "D"
    return None


def _g_earnings(r):
    mcap = _n(r.get("mcap_oku"))
    weak, strong = _n(r.get("dcf_weak_oku")), _n(r.get("dcf_strong_oku"))
    npr = _n(r.get("np_oku"))
    if npr is not None and npr <= 0:
        return "E"
    if mcap and weak is not None:
        if weak >= 2 * mcap:
            return "A"
        if weak >= mcap:
            return "B"
        if strong is not None and strong >= mcap:
            return "C"
        return "D"
    return None


def _g_health(r, danger):
    eq = _n(r.get("eq_oku"))
    if eq is not None and eq < 0:
        return "E"
    if danger:
        return "D"
    eqar, cr = _n(r.get("eqar")), _n(r.get("cur_ratio"))
    borr, ta = _n(r.get("borrowings_oku")), _n(r.get("ta_oku"))
    borr_ratio = borr / ta * 100 if borr is not None and ta else None
    if eqar is None:
        return None
    strong = (eqar >= 60 and (cr is None or cr >= 200)
              and (borr_ratio is None or borr_ratio < 5))
    good = (eqar >= 40 and (cr is None or cr >= 150)
            and (borr_ratio is None or borr_ratio <= 30))
    if strong:
        return "A"
    if good:
        return "B"
    if eqar >= 30:
        return "C"
    return "D"


def _g_profit(r):
    opm, roe, roa = _n(r.get("op_margin")), _n(r.get("roe")), _n(r.get("roa"))
    if r.get("cfo_pos") is False:
        return "E"
    if opm is None:
        return None
    if opm <= 0:
        return "E"
    opm3 = _n(r.get("opm_min3"))
    if opm >= 10 and (roe or 0) >= 10 and (roa or 0) >= 5 and (opm3 or 0) >= 8:
        return "A"
    if opm >= 5 and (roe or 0) >= 10 and (roa or 0) >= 5:
        return "B"
    if opm >= 3:
        return "C"
    return "D"


def _g_growth(r):
    sc, oc = _n(r.get("sales_cagr3")), _n(r.get("op_cagr3"))
    if sc is None or oc is None:
        return None
    if sc < 0 and oc < 0:
        return "E"
    if sc >= 10 and oc > 0 and r.get("fcst_zoshu") and r.get("fcst_zoeki"):
        return "A"
    if sc > 0 and oc > 0:
        return "B"
    if -2 <= sc <= 2 and oc > 0:
        return "C"
    return "D"


def _g_payout(r):
    dps, po = _n(r.get("dps")), _n(r.get("payout"))
    if not dps:
        return "D"
    if r.get("div_up") and r.get("div_no_cut") and po is not None and 20 <= po <= 50:
        return "A"
    if r.get("div_no_cut") and (po is None or po <= 60):
        return "B"
    return "C"


GRADE_PT = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0}


def build_analysis(df: pd.DataFrame, path="docs/data/analysis.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    items, by_code = [], {}
    for _, r in df.iterrows():
        danger = _danger(r)
        g = {
            "asset": _g_asset(r), "earnings": _g_earnings(r),
            "health": _g_health(r, danger), "profit": _g_profit(r),
            "growth": _g_growth(r), "payout": _g_payout(r),
        }
        pts = [GRADE_PT[v] for v in g.values() if v]
        if len(pts) < 4:
            continue
        comp = round(sum(pts) / len(pts), 2)
        rec = {
            "code4": r["code4"], "name": r["name"], "sector": r["sector"],
            "price": r["price"], "mcap_oku": r["mcap_oku"],
            "per": r["per"], "pbr": r["pbr"], "yield": r["yield"],
            "grades": g, "composite": comp, "danger": danger,
            "netcash_ratio": r.get("netcash_ratio"),
            "seisan_oku": r.get("seisan_oku"),
            "dcf_weak_oku": r.get("dcf_weak_oku"),
            "dcf_strong_oku": r.get("dcf_strong_oku"),
            "edinet_end": r.get("edinet_end"),
            "n_a": sum(1 for v in g.values() if v == "A"),
        }
        items.append(rec)
        by_code[r["code4"]] = {
            "g": "".join((g[k] or "-") for k in
                         ("asset", "earnings", "health", "profit", "growth", "payout")),
            "cp": comp,
            "nc": r.get("netcash_ratio"), "sv": r.get("seisan_oku"),
            "dw": r.get("dcf_weak_oku"), "ds": r.get("dcf_strong_oku"),
            "dg": len(danger),
        }
    items.sort(key=lambda x: (-x["n_a"], -x["composite"]))
    # NEW判定: 前回のanalysis.jsonと比較(同じ基準日なら前回のフラグを引き継ぐ)
    data_as_of = (df["as_of"].dropna().iloc[0]
                  if len(df) and df["as_of"].notna().any() else None)
    old_codes, old_flags, old_as_of = set(), {}, None
    if os.path.exists(path):
        try:
            old = json.load(open(path))
            old_as_of = old.get("data_as_of")
            for it in old.get("items", []):
                old_codes.add(it["code4"])
                old_flags[it["code4"]] = it.get("is_new", False)
        except Exception:
            pass
    for it in items[:50]:
        if old_as_of and old_as_of == data_as_of:
            it["is_new"] = old_flags.get(it["code4"], it["code4"] not in old_codes)
        elif old_codes:
            it["is_new"] = it["code4"] not in old_codes
        else:
            it["is_new"] = False
    out = {
        "data_as_of": data_as_of,
        "generated_at": pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y-%m-%d %H:%M"),
        "note": "使い方: 全銘柄を6項目(①資産割安/②収益割安/③財務/④収益性/⑤成長/⑦還元)でA〜E自動採点し、平均点順に上位を表示。Aが1つでもあれば買材料(手順§5)。カードをタップすると清算価値・DCFレンジ・危険シグナルの詳細が開きます。⑥事業素質は定性のため📝解説で補完。",
        "legend": ["①資産割安", "②収益割安", "③財務健全", "④収益性", "⑤成長性", "⑦株主還元"],
        "items": items[:50],
        "byCode": by_code,
    }
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    n_edinet = int(df["netcash_ratio"].notna().sum())
    print(f"analysis: graded={len(items)} edinet_matched={n_edinet}", flush=True)
    return out
