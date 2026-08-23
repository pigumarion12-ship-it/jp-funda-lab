"""jp-funda-lab runner.

usage:
  python run.py update   # J-Quantsからキャッシュ更新
  python run.py build    # キャッシュからスクリーニング計算 + docs/data/latest.json
  python run.py all      # update + build
"""
import sys

from src import data, screens
from src.jq import get_client


def update():
    cli = get_client()
    listed = data.update_listed(cli)
    print(f"listed: {len(listed)}", flush=True)
    prices = data.update_prices(cli)
    print(f"prices: {len(prices)}", flush=True)
    stmts = data.update_statements(cli)
    print(f"stmts: {len(stmts)}", flush=True)
    divs = data.update_dividends(cli)
    print(f"dividends: {len(divs)}", flush=True)


def build():
    prices, stmts, listed, divs = data.load_all()
    screens.dump_schema(stmts, listed, divs, prices)
    if not len(stmts) or not len(prices):
        print("キャッシュ不足のためbuildスキップ", flush=True)
        return
    df = screens.compute_metrics(stmts, listed, prices, divs)
    print(f"metrics: {len(df)} codes", flush=True)
    screens.build_output(df)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "update":
        update()
    elif cmd == "build":
        build()
    elif cmd == "all":
        update()
        build()
    else:
        raise SystemExit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
