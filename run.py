"""jp-funda-lab runner.

usage:
  python run.py update   # J-Quantsからキャッシュ更新
  python run.py edinet   # EDINETから有報/半期を増分取得 (BACKFILL_DAYS 環境変数, 既定10)
  python run.py build    # スクリーニング + 採点 + docs/data/*.json
  python run.py all      # update + edinet + build
"""
import os
import sys

from src import data, screens, edinet, grade
from src.jq import get_client


def update():
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


def build():
    prices, stmts, listed, _ = data.load_all()
    screens.dump_schema(stmts, listed, None, prices)
    if not len(stmts) or not len(prices):
        print("キャッシュ不足のためbuildスキップ", flush=True)
        return
    df = screens.compute_metrics(stmts, listed, prices)
    print(f"metrics: {len(df)} codes", flush=True)
    ed = edinet.latest_by_code()
    df = grade.merge_edinet(df, ed)
    screens.build_output(df)
    grade.build_analysis(df)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "update":
        update()
    elif cmd == "edinet":
        edinet_update()
    elif cmd == "build":
        build()
    elif cmd == "all":
        update()
        edinet_update()
        build()
    else:
        raise SystemExit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
