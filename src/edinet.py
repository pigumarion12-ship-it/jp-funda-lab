"""EDINET API v2: 有報・半期報告書のCSV(XBRL_TO_CSV)からBS主要項目を増分取得.

- 書類一覧 (date指定) → docTypeCode 120(有報)/140(四半期・過去分)/160(半期) を抽出
- 書類CSV zip (type=5) をダウンロードし、主要勘定科目をパースして parquet に蓄積
- 清原式ネットキャッシュ・清算価値・危険シグナルの材料になる
"""
import io
import json
import os
import time
import zipfile
import datetime as dt

import pandas as pd
import requests

BASE = "https://api.edinet-fsa.go.jp/api/v2"
CACHE_DIR = os.environ.get("CACHE_DIR", "cache")
EDINET_PQ = os.path.join(CACHE_DIR, "edinet_fin.parquet")
DAYS_JSON = os.path.join(CACHE_DIR, "edinet_days.json")

DOC_TYPES = {"120", "140", "160"}  # 有報・四半期(過去)・半期
REQ_SLEEP = float(os.environ.get("EDINET_REQ_INTERVAL", "0.3"))

# 勘定科目候補 (要素IDのローカル名)。J-GAAP優先、IFRSはbest-effort。
ITEMS = {
    "cash": ["CashAndDeposits", "CashAndCashEquivalentsIFRS", "CashAndCashEquivalents"],
    "sec_short": ["ShortTermInvestmentSecurities", "Securities"],
    "inv_sec": ["InvestmentSecurities", "OtherFinancialAssetsNCAIFRS"],
    "cur_assets": ["CurrentAssets", "TotalCurrentAssetsIFRS", "CurrentAssetsIFRS"],
    "assets": ["Assets", "TotalAssetsIFRS", "AssetsIFRS"],
    "cur_liab": ["CurrentLiabilities", "TotalCurrentLiabilitiesIFRS", "CurrentLiabilitiesIFRS"],
    "liabilities": ["Liabilities", "TotalLiabilitiesIFRS", "LiabilitiesIFRS"],
    "net_assets": ["NetAssets", "TotalEquityIFRS", "EquityIFRS", "EquityAttributableToOwnersOfParentIFRS"],
    "receivables": ["NotesAndAccountsReceivableTradeAndContractAssets",
                    "NotesAndAccountsReceivableTrade", "AccountsReceivableTrade",
                    "TradeAndOtherReceivablesCAIFRS"],
    "inventories": ["Inventories"],
    "ppe": ["PropertyPlantAndEquipment", "PropertyPlantAndEquipmentIFRS"],
    "intangible": ["IntangibleAssets", "IntangibleAssetsIFRS", "GoodwillIFRS"],
}
INVENTORY_PARTS = ["MerchandiseAndFinishedGoods", "WorkInProcess",
                   "RawMaterialsAndSupplies", "Merchandise", "FinishedGoods",
                   "RawMaterials", "Supplies"]
BORROWING_PARTS = ["ShortTermLoansPayable", "ShortTermBorrowings",
                   "CommercialPapersLiabilities",
                   "CurrentPortionOfLongTermLoansPayable", "CurrentPortionOfBonds",
                   "BondsPayable", "LongTermLoansPayable", "LongTermBorrowings",
                   "LeaseObligationsCL", "LeaseObligationsNCL",
                   "BondsAndBorrowingsCLIFRS", "BondsAndBorrowingsNCLIFRS"]


def _key():
    k = os.environ.get("EDINET_API_KEY")
    if not k:
        raise RuntimeError("EDINET_API_KEY が設定されていません")
    return k


def _get(url, **params):
    params["Subscription-Key"] = _key()
    time.sleep(REQ_SLEEP)
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r


def list_day(date: str) -> list[dict]:
    try:
        r = _get(f"{BASE}/documents.json", date=date, type=2)
        js = r.json()
    except Exception as e:
        print(f"edinet list {date} error: {str(e)[:120]}", flush=True)
        return []
    res = js.get("results") or []
    out = []
    for d in res:
        if (str(d.get("docTypeCode")) in DOC_TYPES and d.get("secCode")
                and str(d.get("csvFlag")) == "1"):
            out.append(d)
    return out


def _parse_csv_zip(content: bytes) -> pd.DataFrame | None:
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile:
        return None
    frames = []
    for name in zf.namelist():
        base = os.path.basename(name)
        if not base.endswith(".csv") or base.startswith("jpaud"):
            continue
        try:
            df = pd.read_csv(io.BytesIO(zf.read(name)), sep="\t", encoding="utf-16",
                             dtype=str, on_bad_lines="skip")
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _extract(df: pd.DataFrame) -> dict:
    need = {"要素ID", "値"}
    if not need.issubset(set(df.columns)):
        return {}
    d = df.copy()
    d["_local"] = d["要素ID"].astype(str).str.split(":").str[-1]
    # 当期のみ
    if "相対年度" in d.columns:
        d = d[d["相対年度"].astype(str).str.startswith("当期")]
    # 連結優先
    cons_col = "連結・個別" if "連結・個別" in d.columns else None
    d["_val"] = pd.to_numeric(d["値"], errors="coerce")
    d = d[d["_val"].notna()]

    def pick(local_names) -> float | None:
        sub = d[d["_local"].isin(local_names)]
        if not len(sub):
            return None
        if cons_col:
            c = sub[sub[cons_col].astype(str) == "連結"]
            if len(c):
                sub = c
        # 候補順を優先
        for n in local_names:
            s = sub[sub["_local"] == n]
            if len(s):
                return float(s["_val"].iloc[0])
        return float(sub["_val"].iloc[0])

    out = {}
    for k, names in ITEMS.items():
        out[k] = pick(names)
    if out.get("inventories") is None:
        parts = [pick([p]) for p in INVENTORY_PARTS[:3]]
        vals = [v for v in parts if v is not None]
        out["inventories"] = sum(vals) if vals else None
    bvals = [pick([p]) for p in BORROWING_PARTS]
    bvals = [v for v in bvals if v is not None]
    out["borrowings"] = sum(bvals) if bvals else (0.0 if out.get("liabilities") is not None else None)
    return out


def fetch_doc(doc: dict) -> dict | None:
    doc_id = doc["docID"]
    try:
        r = _get(f"{BASE}/documents/{doc_id}", type=5)
    except Exception as e:
        print(f"edinet doc {doc_id} error: {str(e)[:120]}", flush=True)
        return None
    df = _parse_csv_zip(r.content)
    if df is None:
        return None
    vals = _extract(df)
    if not any(v is not None for v in vals.values()):
        return None
    code = str(doc.get("secCode"))
    return {
        "Code": code, "docID": doc_id,
        "docType": str(doc.get("docTypeCode")),
        "periodEnd": str(doc.get("periodEnd") or "")[:10],
        "submitDate": str(doc.get("submitDateTime") or "")[:10],
        "coName": doc.get("filerName"),
        **vals,
    }


def update(days_back: int = 10) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    done_days = set()
    if os.path.exists(DAYS_JSON):
        done_days = set(json.load(open(DAYS_JSON)))
    old = pd.read_parquet(EDINET_PQ) if os.path.exists(EDINET_PQ) else pd.DataFrame()
    have_docs = set(old["docID"]) if len(old) else set()

    today = dt.date.today()
    targets = [str(today - dt.timedelta(days=i)) for i in range(1, days_back + 1)]
    targets = [d for d in targets if d not in done_days
               and dt.date.fromisoformat(d).weekday() < 5]
    targets.sort()
    print(f"edinet: {len(targets)} days to fetch", flush=True)

    rows, processed = [], []
    for i, day in enumerate(targets):
        docs = list_day(day)
        for doc in docs:
            if doc["docID"] in have_docs:
                continue
            row = fetch_doc(doc)
            if row:
                rows.append(row)
                have_docs.add(row["docID"])
        processed.append(day)
        if i % 10 == 0 or i == len(targets) - 1:
            print(f"edinet {i + 1}/{len(targets)} {day} docs={len(docs)} total_new={len(rows)}",
                  flush=True)
            # 途中保存
            _save(old, rows, done_days | set(processed))
    return _save(old, rows, done_days | set(processed))


def _save(old, rows, done_days):
    frames = [x for x in (old, pd.DataFrame(rows)) if len(x)]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(df):
        df = df.drop_duplicates(subset=["docID"], keep="last")
        df.to_parquet(EDINET_PQ)
    json.dump(sorted(done_days), open(DAYS_JSON, "w"))
    return df


def latest_by_code() -> pd.DataFrame:
    """銘柄ごとに最新期末のBSデータ1行 (有報優先→期末日優先)。"""
    if not os.path.exists(EDINET_PQ):
        return pd.DataFrame()
    df = pd.read_parquet(EDINET_PQ)
    if not len(df):
        return df
    df["_pri"] = (df["docType"] == "120").astype(int)
    df = df.sort_values(["Code", "periodEnd", "_pri", "submitDate"])
    df = df.drop_duplicates(subset=["Code"], keep="last").drop(columns=["_pri"])
    cov = {k: int(df[k].notna().sum()) for k in
           ("cash", "cur_assets", "liabilities", "inv_sec", "borrowings", "inventories")}
    print(f"edinet coverage: codes={len(df)} {cov}", flush=True)
    return df
