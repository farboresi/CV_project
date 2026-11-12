"""Check that every athlete has all their runs, per competition.

Reads review_labels.csv (exported from review/index.html), keeps the live jumps (including runs
recovered from 'LIVE MISSING' notes), and for each competition lists, per athlete, the attempts
found and the attempts missing. For a missing attempt it gives a search window in the broadcast:
between the athlete's previous and next labelled run - in a round-by-round broadcast the missing
run sits in that window.
Output: printed report + long_cuts/missing_runs.csv

EXPECTED_RUNS = runs per athlete in each final (check against the official results).
Usage: python3 check_runs.py
"""
import csv, re, unicodedata
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_RUNS = {          # required attempts per athlete (finals format, checked against official results)
    "xg_aspen_2026_men": 3, "xg_aspen_2026_women": 3,
    "xg_aspen_2025_men": 3, "xg_aspen_2025_women": 3,   # 3 playoff runs; the 4 finalists also have attempts 4-5
    "xg_aspen_2023_women": 5,
    "oly_beijing_2022_men": 3, "oly_beijing_2022_women": 3,
}
FINALISTS_EXTRA = {"xg_aspen_2025_men": 2, "xg_aspen_2025_women": 2}   # finals runs numbered 4-5

def norm(name):            # 'TESS  ledeux' / 'Tess Ledeux ' / 'Tèss Ledeux' -> 'tess ledeux'
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", n).strip().lower()

def hms(t):
    t = int(float(t)); return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}"

def load():
    import importlib.util
    spec = importlib.util.spec_from_file_location("cc", HERE / "cut_clips.py")
    src = (HERE / "cut_clips.py").read_text().split("rows = [apply_note")[0].replace("OUT.mkdir(exist_ok=True)", "")
    ns = {"__file__": str(HERE / "cut_clips.py"), "print": lambda *a, **k: None, "__name__": "x"}
    exec(src, ns)                                   # reuse the exact note parsing used for cutting
    rows = [ns["apply_note"](r) for r in csv.DictReader(open(HERE / "review_labels.csv", encoding="utf-8-sig"))]
    rows = [r for r in rows if r["status"] == "keep"]
    rows = [r for r in rows if not (r["clip_id"].endswith("_live") and any(
        o is not r and o["video"] == r["video"] and min(float(o["end"]), float(r["end"])) - max(float(o["start"]), float(r["start"])) > 3
        for o in rows))]
    extra = HERE / "review_xg2025men_labels.csv"
    if extra.exists():
        rows += [r for r in (ns["apply_note"](r) for r in csv.DictReader(open(extra, encoding="utf-8-sig"))) if r["status"] == "keep"]
    return rows

rows = load()
by = defaultdict(lambda: defaultdict(list))
display, noname = {}, defaultdict(int)
for r in rows:
    if not r["athlete"].strip():
        noname[r["competition"]] += 1; continue
    k = norm(r["athlete"]); display.setdefault(k, r["athlete"].strip())
    by[r["competition"]][k].append(r)

out = []
for comp in sorted(by):
    exp = EXPECTED_RUNS.get(comp, 3)
    print(f"\n=== {comp}  ({len(by[comp])} athletes, expecting {exp} runs each)")
    if noname[comp]: print(f"  ! {noname[comp]} live jumps without an athlete name")
    for k in sorted(by[comp], key=lambda k: display[k]):
        runs = by[comp][k]
        att = defaultdict(list)
        for r in runs: att[r["attempt"].strip() or "?"].append(r)
        have = sorted(int(a) for a in att if a.isdigit())
        top = exp + (FINALISTS_EXTRA.get(comp, 0) if any(a > exp for a in have) else 0)
        missing = [a for a in range(1, top + 1) if a not in have]
        dupes = [a for a, rs in att.items() if a != "?" and len(rs) > 1]
        flags = []
        if missing: flags.append("MISSING " + ",".join(map(str, missing)))
        if dupes: flags.append("DUPLICATE attempt " + ",".join(dupes))
        if "?" in att: flags.append(f"{len(att['?'])} without attempt no.")
        print(f"  {'!!' if flags else 'ok'} {display[k]:24s} attempts {have}  {'; '.join(flags)}")
        video = runs[0]["video"]
        for a in missing:
            prev = [float(r["end"]) for r in runs if r["attempt"].isdigit() and int(r["attempt"]) < a]
            nxt = [float(r["start"]) for r in runs if r["attempt"].isdigit() and int(r["attempt"]) > a]
            lo, hi = (max(prev) if prev else 0), (min(nxt) if nxt else None)
            win = f"{hms(lo)} - {hms(hi) if hi else 'end of video'}"
            print(f"       -> look for attempt {a} between {win}")
            out.append(dict(competition=comp, athlete=display[k], missing_attempt=a, video=video,
                            search_from=hms(lo), search_to=hms(hi) if hi else "end"))

with open(HERE / "missing_runs.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["competition", "athlete", "missing_attempt", "video", "search_from", "search_to"])
    w.writeheader(); w.writerows(out)
print(f"\n{len(out)} missing runs -> long_cuts/missing_runs.csv")
