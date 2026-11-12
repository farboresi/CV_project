"""Download full-length Big Air competition videos (to be cut into one clip per jump).

Input:  long_videos.csv   (youtube_url;competition;notes)  - one row per broadcast
Output: long_videos/<id>.mp4          the video (720p H.264, no audio)
        long_videos/<id>.info.json    title, duration, chapters - used to name/cut clips
        long_videos_local.csv         id;competition;title;duration_s;youtube_url

'competition' becomes the session_group of every clip cut from that video, so a whole
competition stays on one side of the train/test split (same jumps also appear in our Shorts).

Already-downloaded videos are skipped, so it is safe to re-run.

Usage (from the Project folder, with the cv2026 env active):
    python download_long.py
"""
import json
from pathlib import Path

import pandas as pd
import yt_dlp  # type: ignore

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "long_videos"

# 720p is plenty for pose (the AQA-7 clips are 320x240) and keeps a 1-2 h broadcast
# around 1-2 GB instead of 4+. Prefer H.264 so OpenCV/decord can read it.
FORMAT = "bv*/b"
FORMAT_SORT = ["res:720", "vcodec:h264"]


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(ROOT / "long_videos.csv", sep=";", encoding="utf-8-sig").drop_duplicates("youtube_url")
    opts = {
        "format": FORMAT,
        "format_sort": FORMAT_SORT,
        "outtmpl": str(OUT_DIR / "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "remux_video": "mp4",
        "writeinfojson": True,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    rows, failed = [], []
    with yt_dlp.YoutubeDL(opts) as ydl:
        for url, comp in zip(df["youtube_url"], df["competition"]):
            try:
                info = ydl.extract_info(url, download=False)
                vid = info["id"]
                if (OUT_DIR / f"{vid}.mp4").exists():
                    print(f"skip  {vid} (already downloaded)")
                else:
                    print(f"get   {vid}  {info.get('title')}  ({info.get('duration', 0) / 60:.0f} min)")
                    ydl.download([url])
                rows.append({"id": vid, "competition": comp, "title": info.get("title"),
                             "duration_s": info.get("duration"), "youtube_url": url})
            except Exception as e:
                failed.append(url)
                print(f"FAIL  {url}: {e}")
    pd.DataFrame(rows).to_csv(ROOT / "long_videos_local.csv", sep=";", index=False)
    print(f"\n{len(rows)} videos in long_videos/, listed in long_videos_local.csv")
    if failed:
        print(f"{len(failed)} failed:", *failed, sep="\n  ")


if __name__ == "__main__":
    main()
