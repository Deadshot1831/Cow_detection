"""Pivot summary.csv into one row per cow_id with all the images it appears in."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", help="path to summary.csv")
    ap.add_argument("-o", "--out", default=None,
                    help="output CSV path (default: <summary_dir>/cow_index.csv)")
    args = ap.parse_args()

    src = Path(args.summary)
    dst = Path(args.out) if args.out else src.parent / "cow_index.csv"

    # cow_id -> list of unique image names (preserving first-seen order)
    images_by_cow: dict[int, list[str]] = defaultdict(list)
    counts_by_cow: dict[int, int] = defaultdict(int)
    seen: set[tuple[int, str]] = set()

    with src.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row["cow_id"])
            img = row["image"]
            counts_by_cow[cid] += 1
            key = (cid, img)
            if key not in seen:
                seen.add(key)
                images_by_cow[cid].append(img)

    rows = []
    for cid in sorted(images_by_cow.keys()):
        imgs = images_by_cow[cid]
        if len(imgs) <= 1:
            continue   # skip cows seen in only one image
        rows.append({
            "cow_id": cid,
            "num_images": len(imgs),
            "images": " | ".join(imgs),
        })

    # most-photographed cows first
    rows.sort(key=lambda r: -r["num_images"])

    with dst.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cow_id", "num_images", "images"])
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {dst}")
    print(f"  cows kept (>1 image): {len(rows)}")


if __name__ == "__main__":
    main()
