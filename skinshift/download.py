"""Download a class-balanced image subset from the ISIC Archive API.

Uses only the standard library (urllib), so it can run before the ML environment is installed.
For each (collection, diagnosis) pair, pages through /images/search/, keeps every image's
license, attribution and lesion_id, and downloads the "full" file (already a small, resized
copy on the Archive — see config.py for the sizes we measured).

    python -m skinshift.download            # both collections, default caps
    python -m skinshift.download --dry-run   # counts only, no downloads
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterator, List

from .config import (
    API_BASE,
    CLASS_SHORT,
    CLASSES,
    OOD_COLLECTION,
    OOD_PER_CLASS_CAP,
    TRAIN_COLLECTION,
    TRAIN_PER_CLASS_CAP,
    Paths,
)

USER_AGENT = "skinshift-student-project/0.1 (educational, research use)"


def _get_json(url: str, tries: int = 3) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except (urllib.error.URLError, TimeoutError):
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def search(collection: int, diagnosis: str, limit: int) -> Iterator[dict]:
    """Yield up to `limit` image records for one (collection, diagnosis_3) pair, paginated."""
    query = urllib.parse.quote(f'diagnosis_3:"{diagnosis}"')
    url = f"{API_BASE}/images/search/?collections={collection}&query={query}&limit=100"
    n = 0
    while url and n < limit:
        page = _get_json(url)
        for r in page["results"]:
            yield r
            n += 1
            if n >= limit:
                return
        url = page.get("next")


def _download_file(url: str, dest: Path, tries: int = 3) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            dest.write_bytes(data)
            return len(data)
        except (urllib.error.URLError, TimeoutError):
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch_collection(collection: int, per_class_cap: int, out_dir: Path, dry_run: bool = False) -> List[Dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    for diagnosis in CLASSES:
        short = CLASS_SHORT[diagnosis]
        records = list(search(collection, diagnosis, per_class_cap))
        print(f"  {diagnosis:30s} -> {len(records)} images (cap {per_class_cap})")
        if dry_run:
            continue
        class_dir = out_dir / short
        class_dir.mkdir(exist_ok=True)
        for i, r in enumerate(records):
            fname = f"{r['isic_id']}.jpg"
            dest = class_dir / fname
            if not dest.exists():
                _download_file(r["files"]["full"]["url"], dest)
            m = r["metadata"]
            rows.append({
                "isic_id": r["isic_id"],
                "collection": collection,
                "diagnosis_3": diagnosis,
                "class_short": short,
                "lesion_id": m["clinical"].get("lesion_id", ""),
                "image_type": m.get("acquisition", {}).get("image_type", ""),
                "copyright_license": r.get("copyright_license", ""),
                "attribution": r.get("attribution", ""),
                "path": str(dest.relative_to(out_dir.parent.parent)),
            })
            if (i + 1) % 50 == 0:
                print(f"    ...{i + 1}/{len(records)} downloaded")
    return rows


def write_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print counts only, download nothing")
    args = ap.parse_args()

    paths = Paths()
    paths.ensure()

    print(f"Training source: HAM10000 (collection {TRAIN_COLLECTION}), cap {TRAIN_PER_CLASS_CAP}/class")
    train_rows = fetch_collection(TRAIN_COLLECTION, TRAIN_PER_CLASS_CAP, paths.train_images, args.dry_run)
    if not args.dry_run:
        write_csv(train_rows, paths.train_metadata)
        print(f"  wrote {len(train_rows)} rows -> {paths.train_metadata}")

    print(f"\nOOD source: PAD-UFES-20 (collection {OOD_COLLECTION}), cap {OOD_PER_CLASS_CAP}/class")
    ood_rows = fetch_collection(OOD_COLLECTION, OOD_PER_CLASS_CAP, paths.ood_images, args.dry_run)
    if not args.dry_run:
        write_csv(ood_rows, paths.ood_metadata)
        print(f"  wrote {len(ood_rows)} rows -> {paths.ood_metadata}")


if __name__ == "__main__":
    main()
