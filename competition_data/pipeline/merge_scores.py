"""Merge official (and hand-filled scores_to_fill.csv) per-run scores into long_clips/long_labels.csv.

official_scores.csv: competition,athlete,attempt,score,scale,source  (one row per run)
Writes score (as published), score_100 (rescaled to 0-100), score_source into long_labels.csv,
and prints clips with no official score plus fall/score sanity checks.
"""
import csv, re, unicodedata
from pathlib import Path
HERE = Path(__file__).resolve().parent; LAB = HERE.parent / "long_clips" / "long_labels.csv"
ALIAS = {"alexander hall": "alex hall", "colby stevensen": "colby stevenson", "oliver magnusson": "oliwer magnusson"}
def norm(n):
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode().lower()
    n = " ".join(n.replace("ö", "o").split()); n = n.replace("hofflin", "hoefflin")
    return ALIAS.get(n, n)
off = {(r["competition"], norm(r["athlete"]), r["attempt"]): r for r in csv.DictReader(open(HERE / "official_scores.csv"))}
FILL = HERE / "scores_to_fill.csv"          # filled in by hand: rows with an empty score are ignored
if FILL.exists():
    for r in csv.DictReader(open(FILL, encoding="utf-8-sig")):
        if r["score"].strip():
            off.setdefault((r["competition"], norm(r["athlete"]), r["attempt"]), r)
rows = list(csv.DictReader(open(LAB))); fields = list(rows[0].keys())
for f in ("score_100", "score_source"):
    if f not in fields: fields.append(f)
miss, used = [], set()
for r in rows:
    k = (r["session_group"], norm(r["athlete"]), r["attempt"].strip())
    o = off.get(k)
    if o:
        used.add(k); r["score"] = o["score"]; r["score_100"] = f"{float(o['score']) * 100 / float(o['scale']):.2f}"; r["score_source"] = o["source"]
    else:
        miss.append(r)
with open(LAB, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
print(f"{len(rows) - len(miss)}/{len(rows)} clips have an official score")
from collections import Counter
print("missing by competition:", dict(Counter(r["session_group"] for r in miss)))
unused = [k for k in off if k not in used]
if unused: print("official runs with no clip:", unused)
print("\nsanity - falls with high score / non-falls with very low score:")
for r in rows:
    if r.get("score_100"):
        s = float(r["score_100"])
        if (r["fall"] == "1" and s > 60) or (r["fall"] == "0" and s < 30):
            print(f"  {r['clip_id']:28s} {r['athlete']:20s} att {r['attempt']} fall={r['fall']} score={s}")
