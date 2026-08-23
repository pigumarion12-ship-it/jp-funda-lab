# jp-funda-lab

辻さん個人用の日本株ファンダメンタルズ・スクリーニングサイト。

- 毎営業日 20:30 JST 台に GitHub Actions が J-Quants からデータを取得し、
  `docs/data/latest.json` を再生成 → GitHub Pages で配信。
- スクリーニング: 弐億貯男式(割安グロース) / 清原式(ネットキャッシュ・暫定) / 配当バリュー成長(辻さん基準)
- フェーズ2で EDINET (XBRL) を接続し、清算価値・ネットキャッシュ比率・危険シグナル・A〜E自動採点を追加予定。

## 構成

- `.github/workflows/update.yml` — cron + workflow_dispatch
- `run.py` — update / build / all
- `src/jq.py` — J-Quants ClientV2
- `src/data.py` — 増分キャッシュ (parquet, actions/cache)
- `src/screens.py` — 指標計算 + 3スクリーニング
- `docs/` — GitHub Pages (index.html + data/latest.json)
- `reports/` — 実行ログ・スキーマダンプ (Claudeが読む)

Secrets: `JQUANTS_API_KEY`, `EDINET_API_KEY` (Actions)
