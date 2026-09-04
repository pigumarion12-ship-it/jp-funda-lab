"""証券会社の約定履歴CSVを正規化して docs/data/trades.json に取り込む.

usage: python scripts/import_trades.py <broker> <csv...>
  broker: sbi / esmart / rakuten / monex (不明なら auto)

- 文字コードは cp932 → utf-8-sig → utf-8 の順で自動判定
- ヘッダ行は「約定日/銘柄/数量」系の語を含む行を探して特定
- 重複は (broker,日付,コード,売買,数量,単価) のハッシュで排除
"""
import csv
import hashlib
import io
import json
import re
import sys
import unicodedata
from pathlib import Path

OUT = Path("docs/data/trades.json")

CAND = {
    "date": ["約定日", "約定日時", "国内約定日", "取引日", "受渡日"],
    "name": ["銘柄名", "銘柄", "ファンド名"],
    "code": ["銘柄コード", "証券コード", "コード"],
    "side": ["取引", "売買", "取引区分", "売買区分", "取引種類"],
    "qty": ["約定数量", "数量", "出来数量", "約定株数", "数量[株]", "株数"],
    "price": ["約定単価", "単価", "約定価格", "約定単価[円]", "平均約定単価"],
    "fee": ["手数料", "手数料/諸経費等", "手数料等", "手数料（税込）", "手数料(税込)"],
    "tax": ["税額", "税金等", "課税額"],
    "account": ["口座", "預り区分", "口座区分", "預り"],
}


def norm(s):
    return unicodedata.normalize("NFKC", str(s or "")).strip()


def to_num(s):
    s = norm(s).replace(",", "").replace("円", "").replace("株", "")
    s = re.sub(r"[^\d.\-]", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def read_csv_rows(path):
    raw = Path(path).read_bytes()
    for enc in ("cp932", "utf-8-sig", "utf-8"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise SystemExit(f"文字コード判定失敗: {path}")
    return list(csv.reader(io.StringIO(text)))


def find_header(rows):
    for i, row in enumerate(rows[:30]):
        cells = [norm(c) for c in row]
        joined = "".join(cells)
        if any(k in joined for k in ("約定日", "取引日")) and \
           any(k in joined for k in ("銘柄", "ファンド")) and \
           any(k in joined for k in ("数量", "株数", "口数")):
            return i, cells
    raise SystemExit("ヘッダ行が見つかりません(対応外のCSV形式)")


def col_idx(header, keys):
    for k in keys:
        for i, h in enumerate(header):
            if k in h:
                return i
    return None


def parse_side(v):
    v = norm(v)
    if "買" in v:
        return "buy"
    if "売" in v:
        return "sell"
    return None


def parse_date(v):
    v = norm(v)
    m = re.search(r"(\d{4})[/\-年](\d{1,2})[/\-月](\d{1,2})", v)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.search(r"(\d{2})/(\d{1,2})/(\d{1,2})", v)
    if m:
        return f"20{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return None


def extract_code(code_cell, name_cell):
    # 4桁数字 or 新型コード(数字3+英数字1: 350A等)
    m = re.fullmatch(r"(\d{3}[0-9A-Z])", norm(code_cell))
    if m:
        return m.group(1)
    for v in (code_cell, name_cell):
        m = re.search(r"\b(\d{3}[0-9A-Z])\b", norm(v))
        if m:
            return m.group(1)
    return None


def parse_file(path, broker):
    rows = read_csv_rows(path)
    hi, header = find_header(rows)
    idx = {k: col_idx(header, v) for k, v in CAND.items()}
    if idx["date"] is None or idx["qty"] is None:
        raise SystemExit(f"必須列が見つかりません: {path}")
    fills = []
    for row in rows[hi + 1:]:
        if not row or len(row) < 3:
            continue
        cells = [norm(c) for c in row]
        get = lambda k: cells[idx[k]] if idx[k] is not None and idx[k] < len(cells) else ""
        date = parse_date(get("date"))
        side = parse_side(get("side"))
        qty = to_num(get("qty"))
        price = to_num(get("price"))
        code = extract_code(get("code"), get("name"))
        if not (date and side and qty and price and code):
            continue
        name = re.sub(r"\b\d{4}\b", "", get("name")).strip() or None
        fee = to_num(get("fee")) or 0.0
        tax = to_num(get("tax")) or 0.0
        rec = {
            "broker": broker, "date": date, "code4": code, "name": name,
            "side": side, "qty": qty, "price": price,
            "fee": round(fee + tax, 1), "account": get("account") or None,
        }
        rec["id"] = hashlib.md5(
            f"{broker}|{date}|{code}|{side}|{qty}|{price}".encode()).hexdigest()[:12]
        fills.append(rec)
    return fills


SELL_KINDS = ("現物売", "信用返済", "外国株式売", "国内投信解約")
DIV_KINDS = ("配当", "分配金")


def is_gains_file(rows):
    for r in rows[:40]:
        if any("取得/新規金額" in norm(c) for c in r):
            return True
    return False


def parse_gains(path, broker):
    """特定口座損益明細(譲渡益税明細)CSV → realized/dividends."""
    rows = read_csv_rows(path)
    hi = None
    for i, r in enumerate(rows[:40]):
        if any("取得/新規金額" in norm(c) for c in r):
            hi = i
            header = [norm(c) for c in r]
            break
    if hi is None:
        raise SystemExit("損益明細のヘッダが見つかりません")
    ix = {h: i for i, h in enumerate(header)}
    def g(row, key):
        i = ix.get(key)
        return norm(row[i]) if i is not None and i < len(row) else ""
    groups = {}
    dividends = []
    for row in rows[hi + 1:]:
        if not row or norm(row[0]) == "譲渡益税徴収額":
            continue
        kind = g(row, "取引")
        if not kind:
            continue
        date = parse_date(g(row, "約定日"))
        name = g(row, "銘柄") or None
        code = extract_code(g(row, "銘柄コード"), "")
        pnl = to_num(g(row, "損益金額/徴収額"))
        if any(k in kind for k in DIV_KINDS):
            if date and pnl is not None:
                d = {"broker": broker, "date": date, "code4": code, "name": name,
                     "amount": pnl, "kind": kind}
                d["id"] = hashlib.md5(
                    f"div|{date}|{code or name}|{pnl}".encode()).hexdigest()[:12]
                dividends.append(d)
            continue
        if not any(k in kind for k in SELL_KINDS):
            continue
        qty = to_num(g(row, "数量"))
        proceeds = to_num(g(row, "売却/決済金額"))
        cost = to_num(g(row, "取得/新規金額"))
        acq = parse_date(g(row, "取得/新規年月日"))
        fee = to_num(g(row, "費用")) or 0.0
        if not (date and pnl is not None):
            continue
        key = (code or name, date, kind)
        gr = groups.setdefault(key, {"broker": broker, "date": date, "code4": code,
                                     "name": name, "kind": kind, "qty": 0.0,
                                     "proceeds": 0.0, "cost": 0.0, "fee": 0.0,
                                     "pnl": 0.0, "acq_date": acq})
        gr["qty"] += qty or 0
        gr["proceeds"] += proceeds or 0
        gr["cost"] += cost or 0
        gr["fee"] += fee
        gr["pnl"] += pnl
        if acq and (gr["acq_date"] is None or acq < gr["acq_date"]):
            gr["acq_date"] = acq
    realized = []
    for gr in groups.values():
        gr["id"] = hashlib.md5(
            f"gain|{gr['date']}|{gr['code4'] or gr['name']}|{gr['kind']}|{gr['qty']}|{gr['pnl']}".encode()
        ).hexdigest()[:12]
        realized.append(gr)
    return realized, dividends


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    broker = sys.argv[1]
    data = {"updated_at": None, "fills": []}
    if OUT.exists():
        data = json.loads(OUT.read_text())
    data.setdefault("realized", [])
    data.setdefault("dividends", [])
    known = {f["id"] for f in data["fills"]}
    known_r = {r["id"] for r in data["realized"]}
    known_d = {d["id"] for d in data["dividends"]}
    added = 0
    for p in sys.argv[2:]:
        rows0 = read_csv_rows(p)
        if is_gains_file(rows0):
            rz, dv = parse_gains(p, broker)
            for r in rz:
                if r["id"] not in known_r:
                    data["realized"].append(r); known_r.add(r["id"]); added += 1
            for d in dv:
                if d["id"] not in known_d:
                    data["dividends"].append(d); known_d.add(d["id"]); added += 1
            continue
        for f in parse_file(p, broker):
            if f["id"] not in known:
                data["fills"].append(f)
                known.add(f["id"])
                added += 1
    data["fills"].sort(key=lambda f: (f["date"], f["code4"]))
    data["realized"].sort(key=lambda r: r["date"])
    data["dividends"].sort(key=lambda d: d["date"])
    import datetime as dt
    data["updated_at"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False))
    print(f"取込 {added}件 追加 (合計 {len(data['fills'])}件)")


if __name__ == "__main__":
    main()
