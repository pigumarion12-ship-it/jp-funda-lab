"""jp-funda-lab runner.

usage:
  python run.py update   # J-Quantsからキャッシュ更新
  python run.py edinet   # EDINETから有報/半期を増分取得 (BACKFILL_DAYS 環境変数, 既定10)
  python run.py build    # スクリーニング + 採点 + docs/data/*.json
  python run.py all      # update + edinet + build
"""
import os
import sys

from src import data, screens, edinet, grade, articles, alerts


def update():
    from src.jq import get_client
    cli = get_client()
    listed = data.update_listed(cli)
    print(f"listed: {len(listed)}", flush=True)
    prices = data.update_prices(cli)
    print(f"prices: {len(prices)}", flush=True)
    stmts = data.update_statements(cli)
    print(f"stmts: {len(stmts)}", flush=True)


def edinet_update():
    days = int(os.environ.get("BACKFILL_DAYS") or 10)
    df = edinet.update(days_back=days)
    print(f"edinet_fin: {len(df)} docs", flush=True)


def report():
    """1銘柄の全計算指標+四半期履歴+株価統計を docs/data/deep/metrics_<code>.json に出力."""
    import json
    import numpy as np
    import pandas as pd
    code4 = (os.environ.get("REPORT_CODE") or "").strip()
    if not code4:
        raise SystemExit("REPORT_CODE を指定してください")
    prices, stmts, listed, _ = data.load_all()
    prices = data.adjust_splits(prices)
    df = screens.compute_metrics(stmts, listed, prices)
    ed = edinet.latest_by_code()
    df = grade.merge_edinet(df, ed)

    def _j(v):
        if v is None:
            return None
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, (np.floating, float)):
            return None if not np.isfinite(v) else float(v)
        if isinstance(v, np.bool_):
            return bool(v)
        if isinstance(v, pd.Timestamp):
            return str(v.date())
        if isinstance(v, set):
            return sorted(v)
        return v if isinstance(v, (str, int, bool, list, dict)) else str(v)

    out = {"code4": code4,
           "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
    row = df[df["code4"].astype(str) == code4]
    if len(row):
        out["metrics"] = {k: _j(v) for k, v in row.iloc[0].items()}
    codes = {code4, code4 + "0"}
    st = stmts[stmts["Code"].astype(str).isin(codes)].copy()
    if len(st):
        st = st.sort_values("DiscDate")
        out["statements"] = [{k: _j(v) for k, v in r.items()}
                             for r in st.to_dict("records")]
    px = prices[prices["Code"].astype(str).isin(codes)].sort_values("Date")
    px = px[px["AdjC"].notna()]
    if len(px):
        c = px["AdjC"]
        last = float(c.iloc[-1])

        def ma(n):
            return round(float(c.tail(n).mean()), 1) if len(c) >= n else None

        def ret(n):
            return round((last / float(c.iloc[-n - 1]) - 1) * 100, 1) if len(c) > n else None

        hi52 = float(px.tail(250)["AdjH"].max())
        out["price"] = {
            "date": str(px["Date"].iloc[-1]), "close": last,
            "ma5": ma(5), "ma25": ma(25), "ma50": ma(50), "ma75": ma(75),
            "ma100": ma(100), "ma200": ma(200),
            "hi52": hi52, "lo52": float(px.tail(250)["AdjL"].min()),
            "hi52_ratio": round(last / hi52 * 100, 1),
            "ret_1m": ret(21), "ret_3m": ret(63),
            "ret_6m": ret(126), "ret_12m": ret(250),
        }
    if len(px):
        out["ohlc"] = {
            "d": [str(x) for x in px["Date"]],
            "o": [_j(x) for x in px["AdjO"]],
            "h": [_j(x) for x in px["AdjH"]],
            "l": [_j(x) for x in px["AdjL"]],
            "c": [_j(x) for x in px["AdjC"]],
            "v": [_j(x) for x in px["AdjVo"]] if "AdjVo" in px.columns else None,
        }
    if isinstance(ed, dict) and ed.get(code4):
        out["edinet"] = {k: _j(v) for k, v in dict(ed[code4]).items()}
    path = f"docs/data/deep/metrics_{code4}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"report: {path} metrics={bool(len(row))} stmts={len(st)}", flush=True)


def build():
    prices, stmts, listed, _ = data.load_all()
    prices = data.adjust_splits(prices)
    screens.dump_schema(stmts, listed, None, prices)
    if not len(stmts) or not len(prices):
        print("キャッシュ不足のためbuildスキップ", flush=True)
        return
    df = screens.compute_metrics(stmts, listed, prices)
    print(f"metrics: {len(df)} codes", flush=True)
    ed = edinet.latest_by_code()
    df = grade.merge_edinet(df, ed)
    out2 = grade.build_analysis(df)
    arc = {k: v.get("cp") for k, v in out2["byCode"].items()}
    df["arc_comp"] = df["code4"].map(arc)
    out1 = screens.build_output(df)
    from src import kabuzaru, duke
    out3 = kabuzaru.compute(stmts, listed, prices, path="docs/data/kabuzaru.json")
    out4 = duke.compute(stmts, listed, prices, path="docs/data/duke.json")
    shown4 = set()
    for s in out1["screens"].values():
        shown4 |= {it["code4"] for it in s["items"]}
    shown4 |= {it["code4"] for it in out2["items"]}
    shown4 |= {it["code4"] for it in out3["items"]}
    shown4 |= {it["code4"] for it in out4["items"]}
    code_map = dict(zip(df["code4"], df["code"]))
    codes5 = {code_map[c] for c in shown4 if c in code_map}
    screens.build_charts(prices, codes5)
    alerts.build_universe(df)
    alerts.build_alerts(prices, stmts, df)
    screens.build_trade_charts(prices)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "update":
        update()
    elif cmd == "edinet":
        edinet_update()
    elif cmd == "articles":
        articles.update()
    elif cmd == "kabuzaru":
        from src import kabuzaru
        prices, stmts, listed, _ = data.load_all()
        prices = data.adjust_splits(prices)
        kabuzaru.compute(stmts, listed, prices, path="docs/data/kabuzaru.json")
    elif cmd == "report":
        report()
    elif cmd == "build":
        build()
    elif cmd == "all":
        update()
        edinet_update()
        try:
            articles.update()
        except Exception as e:
            print(f"articles更新失敗(継続): {str(e)[:200]}", flush=True)
        build()
    else:
        raise SystemExit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
