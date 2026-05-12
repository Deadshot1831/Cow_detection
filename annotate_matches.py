"""
Draw bounding boxes with cow_id labels on every image that contains a
re-identified cow (a cow whose cow_id appears in >1 image).

Reads:  <out>/summary.csv
Writes: <out>/annotated/<image>.jpg  with boxes drawn on matched cows
"""

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import cv2


def color_for(cid: int) -> tuple[int, int, int]:
    """Deterministic distinct BGR color per cow_id."""
    h = hashlib.md5(str(cid).encode()).digest()
    return int(h[0]), int(h[1]) | 64, int(h[2]) | 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", help="path to summary.csv")
    ap.add_argument("images", help="folder of source images")
    ap.add_argument("-o", "--out", default=None,
                    help="output folder (default: <summary_dir>/annotated)")
    args = ap.parse_args()

    src = Path(args.summary)
    img_dir = Path(args.images)
    out_dir = Path(args.out) if args.out else src.parent / "annotated"
    out_dir.mkdir(exist_ok=True)

    # cow_id -> set of images; per_image[image] = [(cow_id, x1, y1, x2, y2, conf)]
    images_by_cow: dict[int, set[str]] = defaultdict(set)
    per_image: dict[str, list[tuple]] = defaultdict(list)

    with src.open() as f:
        for row in csv.DictReader(f):
            cid = int(row["cow_id"])
            img = row["image"]
            x1, y1, x2, y2 = map(int, row["box"].split())
            conf = float(row["conf"])
            images_by_cow[cid].add(img)
            per_image[img].append((cid, x1, y1, x2, y2, conf))

    # only keep cows that appear in >1 image
    multi_cows = {cid for cid, imgs in images_by_cow.items() if len(imgs) > 1}
    print(f"re-identified cows (>1 image): {len(multi_cows)}")

    # which images need annotation
    target_images = sorted({img for img, dets in per_image.items()
                            if any(cid in multi_cows for cid, *_ in dets)})
    print(f"images to annotate: {len(target_images)}")

    for img_name in target_images:
        path = img_dir / img_name
        img = cv2.imread(str(path))
        if img is None:
            print(f"  skip unreadable {img_name}")
            continue

        for cid, x1, y1, x2, y2, conf in per_image[img_name]:
            if cid not in multi_cows:
                continue
            col = color_for(cid)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
            label = f"cow_{cid:03d}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.9, 2)
            ty = max(y1 - 8, th + 4)
            cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 8, ty + 2),
                          col, -1)
            cv2.putText(img, label, (x1 + 4, ty - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        out_path = out_dir / img_name
        cv2.imwrite(str(out_path), img)

    print(f"\nwrote annotated images to {out_dir}/")
    print(f"  same color + same label across images = same cow")


if __name__ == "__main__":
    main()
