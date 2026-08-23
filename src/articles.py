"""記事ウォッチ: feeds.json の連載・特集の更新をリスト化して docs/data/articles.json へ.

戦略:
- gnews: GoogleニュースRSS検索 (robots的に安全・安定)
- static: 掲載ページを直接取得してリンク抽出 (robots.txtを尊重、失敗時はスキップ)
初出日(first_seen)を記録し、8日以内はNEW扱い。
"""
import json
import os
import re
import urllib.robotparser
import xml.etree.ElementTree as ET
import datetime as dt

import pandas as pd
import requests

SEEN_PATH = "docs/data/articles_seen.json"  # コミットされる履歴(初出日記録)
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
MAX_PER_FEED = 25


def _gnews(q: str) -> list[dict]:
    url = "https://news.google.com/rss/search"
    try:
        r = requests.get(url, params={"q": q, "hl": "ja", "gl": "JP", "ceid": "JP:ja"},
                         headers=UA, timeout=30)
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception as e:
        print(f"gnews error ({q[:40]}): {str(e)[:100]}", flush=True)
        return []
    out = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        try:
            date = str(pd.to_datetime(pub).date())
        except Exception:
            date = None
        if title and link:
            # Googleニュース経由のタイトルは「タイトル - 媒体名」形式
            if " - " in title:
                base, pub = title.rsplit(" - ", 1)
                if len(pub) <= 25 and len(base.strip()) >= 4:
                    title = base.strip()
            if len(title.strip()) < 4:
                continue
            out.append({"title": title.strip(), "url": link, "date": date})
    return out


def _date_from_url(url: str) -> str | None:
    m = re.search(r"_(\d{2})(\d{2})(\d{2})\.html", url)
    if m:
        y, mo, d = m.groups()
        return f"20{y}-{mo}-{d}"
    m = re.search(r"/(20\d{2})[/-]?(\d{2})[/-]?(\d{2})(?:/|_|\.)", url)
    if m:
        return "-".join(m.groups())
    return None


def _robots_ok(url: str) -> bool:
    from urllib.parse import urlparse
    p = urlparse(url)
    rp = urllib.robotparser.RobotFileParser()
    try:
        r = requests.get(f"{p.scheme}://{p.netloc}/robots.txt", headers=UA, timeout=15)
        if r.status_code >= 400:
            return True
        rp.parse(r.text.splitlines())
        return rp.can_fetch(UA["User-Agent"], url)
    except Exception:
        return False


def _static(url: str, path_prefix: str) -> list[dict]:
    if not _robots_ok(url):
        print(f"static: robots不許可のためスキップ {url}", flush=True)
        return []
    try:
        r = requests.get(url, headers=UA, timeout=30)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or r.encoding
        html = r.text
    except Exception as e:
        print(f"static error {url}: {str(e)[:100]}", flush=True)
        return []
    from urllib.parse import urljoin
    out, seen = [], set()
    for m in re.finditer(r'<a[^>]+href="([^"#?]+)[^"]*"[^>]*>(.*?)</a>', html,
                         re.DOTALL | re.IGNORECASE):
        href, inner = m.group(1), m.group(2)
        full = urljoin(url, href)
        if path_prefix not in full or full.rstrip("/") == url.rstrip("/"):
            continue
        text = re.sub(r"<[^>]+>", " ", inner)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 8 or full in seen:
            continue
        seen.add(full)
        out.append({"title": text[:120], "url": full, "date": _date_from_url(full)})
    return out


def update(feeds_path="feeds.json", out_path="docs/data/articles.json"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    conf = json.load(open(feeds_path))
    seen = {}
    if os.path.exists(SEEN_PATH):
        try:
            seen = json.load(open(SEEN_PATH))
        except Exception:
            seen = {}
    today = str(dt.date.today())

    feeds_out = []
    for f in conf["feeds"]:
        items = []
        for s in f.get("strategies", []):
            got = []
            if s["type"] == "gnews":
                got = _gnews(s["q"])
            elif s["type"] == "static":
                got = _static(s["url"], s.get("path_prefix", "/"))
            for g in got:
                if not any(g["url"] == x["url"] for x in items):
                    items.append(g)
        auto = bool(items)
        fs = seen.setdefault(f["id"], {})
        for g in items:
            if g["url"] not in fs:
                fs[g["url"]] = {"title": g["title"], "date": g.get("date"),
                                "first_seen": today}
            else:  # タイトル・日付は最新の取得結果で上書き(文字化け修正等)
                fs[g["url"]]["title"] = g["title"]
                if g.get("date"):
                    fs[g["url"]]["date"] = g["date"]
        feeds_out.append({**{k: f[k] for k in ("id", "kind", "label", "desc", "home")},
                          "short": f.get("short", f["label"][:8]), "auto": auto})
        print(f"articles {f['id']}: fetched={len(items)} known={len(fs)} (auto={auto})",
              flush=True)

    cutoff = str(dt.date.today() - dt.timedelta(days=8))
    for fo in feeds_out:
        fs = seen.get(fo["id"], {})
        rows = [{"url": u, **v} for u, v in fs.items()]
        rows.sort(key=lambda r: (r.get("date") or r["first_seen"]), reverse=True)
        fo["items"] = [
            {"title": r["title"], "url": r["url"],
             "date": r.get("date") or r["first_seen"],
             "new": bool(r["first_seen"] >= cutoff)}
            for r in rows[:MAX_PER_FEED]
        ]

    with open(SEEN_PATH, "w") as fp:
        json.dump(seen, fp, ensure_ascii=False)

    out = {"generated_at": pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y-%m-%d %H:%M"),
           "feeds": feeds_out}
    with open(out_path, "w") as fp:
        json.dump(out, fp, ensure_ascii=False)
    print(f"articles.json: {sum(len(f['items']) for f in feeds_out)} items total", flush=True)
    return out
