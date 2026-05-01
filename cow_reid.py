"""
Cow Re-Identification across an image folder.

Pipeline:
  1. Detect cows in every image (YOLOv8 COCO, class=cow).
  2. For each detection crop, build a multi-feature signature:
       - Deep CNN embedding (ResNet50 ImageNet, avg-pool 2048-d)
       - HSV colour histogram (16x16 H-S, lighting-invariant)
       - Body aspect ratio + relative size
  3. Combine features into one normalised vector (weighted L2).
  4. Cluster across all crops with cosine similarity + agglomerative
     linkage; cows above threshold are declared the same individual.
  5. Save:
       - cow_reid_output/cow_<id>/ folders with all crops of that cow
       - cow_reid_output/summary.csv  (image -> cow_id mappings)
       - cow_reid_output/montage_<id>.jpg  (visual proof per cow)
"""

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from torchvision import models, transforms
from ultralytics import YOLO


# ---- weights for combining the three feature blocks --------------------------
W_CNN = 0.70   # deep features dominate (most discriminative)
W_HSV = 0.25   # colour pattern (Holstein spots, Salers red, etc.)
W_GEO = 0.05   # body proportions (weak signal but breaks ties)

SIM_THRESHOLD = 0.78   # cosine similarity above this -> same cow
COW_CLASS = 19         # COCO class index for "cow"
DET_CONF = 0.35
MIN_CROP_PX = 64       # ignore tiny detections

# ---- "fully visible" filter --------------------------------------------------
EDGE_MARGIN_PX = 4         # box must be at least this far from any image edge
MIN_AREA_RATIO = 0.01      # box must cover >=1% of image area (no tiny specks)
MIN_ASPECT = 0.6           # cow body has w/h >= ~0.6 (filters thin slivers)
MAX_ASPECT = 4.0           # and w/h <= ~4 (filters extreme crops)


def load_resnet(device: str) -> tuple[torch.nn.Module, transforms.Compose]:
    """ResNet50 with the classifier head chopped off -> 2048-d embedding."""
    net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    net.fc = torch.nn.Identity()
    net.eval().to(device)
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return net, tf


def cnn_embed(crop: np.ndarray, net, tf, device: str) -> np.ndarray:
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    x = tf(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        v = net(x).cpu().numpy().ravel()
    return v.astype(np.float32)


def hsv_hist(crop: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(h, h)
    return h.flatten().astype(np.float32)


def geom_feat(box, img_shape) -> np.ndarray:
    x1, y1, x2, y2 = box
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    H, W = img_shape[:2]
    return np.array([w / h, (w * h) / (W * H)], dtype=np.float32)


def is_fully_visible(box, img_shape) -> tuple[bool, str]:
    """Reject boxes that touch the frame edge, are too small, or have a
    non-cow-like aspect ratio. Returns (ok, reason)."""
    x1, y1, x2, y2 = box
    H, W = img_shape[:2]
    if x1 <= EDGE_MARGIN_PX or y1 <= EDGE_MARGIN_PX:
        return False, "edge"
    if x2 >= W - EDGE_MARGIN_PX or y2 >= H - EDGE_MARGIN_PX:
        return False, "edge"
    bw, bh = x2 - x1, y2 - y1
    if (bw * bh) / (W * H) < MIN_AREA_RATIO:
        return False, "tiny"
    ar = bw / max(1, bh)
    if ar < MIN_ASPECT or ar > MAX_ASPECT:
        return False, "aspect"
    return True, ""


def detect_cows(model: YOLO, img_path: str, device: str):
    res = model.predict(img_path, conf=DET_CONF, classes=[COW_CLASS],
                        device=device, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []
    out = []
    for b in res.boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, _ = b
        out.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
    return out


def build_signature(crop, box, img_shape, net, tf, device) -> np.ndarray:
    cnn = cnn_embed(crop, net, tf, device)
    cnn = cnn / (np.linalg.norm(cnn) + 1e-9)
    hsv = hsv_hist(crop)
    hsv = hsv / (np.linalg.norm(hsv) + 1e-9)
    geo = geom_feat(box, img_shape)
    geo = geo / (np.linalg.norm(geo) + 1e-9)
    return np.concatenate([W_CNN * cnn, W_HSV * hsv, W_GEO * geo])


def make_montage(crops: list[np.ndarray], thumb=180, cols=6) -> np.ndarray:
    if not crops:
        return np.zeros((thumb, thumb, 3), dtype=np.uint8)
    rows = (len(crops) + cols - 1) // cols
    canvas = np.full((rows * thumb, cols * thumb, 3), 32, dtype=np.uint8)
    for i, c in enumerate(crops):
        r, k = divmod(i, cols)
        resized = cv2.resize(c, (thumb, thumb))
        canvas[r * thumb:(r + 1) * thumb, k * thumb:(k + 1) * thumb] = resized
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="folder of cow images")
    ap.add_argument("-o", "--out", default="cow_reid_output")
    ap.add_argument("--model", default="yolov8m.pt")
    ap.add_argument("--threshold", type=float, default=SIM_THRESHOLD,
                    help="cosine similarity threshold (higher = stricter)")
    ap.add_argument("--device", default=None,
                    help="cuda / mps / cpu (auto if None)")
    args = ap.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    img_folder = Path(args.folder)
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    print("loading models...")
    yolo = YOLO(args.model)
    net, tf = load_resnet(device)

    # ---- pass 1: detect + embed ----------------------------------------------
    records = []   # (image_name, det_idx, box, crop, signature)
    img_files = sorted([p for p in img_folder.iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])
    print(f"images: {len(img_files)}")

    for i, p in enumerate(img_files, 1):
        img = cv2.imread(str(p))
        if img is None:
            print(f"  skip unreadable {p.name}")
            continue
        boxes = detect_cows(yolo, str(p), device)
        kept = 0
        skipped = {"edge": 0, "tiny": 0, "aspect": 0, "small_crop": 0}
        for k, (x1, y1, x2, y2, conf) in enumerate(boxes):
            if (x2 - x1) < MIN_CROP_PX or (y2 - y1) < MIN_CROP_PX:
                skipped["small_crop"] += 1
                continue
            ok, reason = is_fully_visible((x1, y1, x2, y2), img.shape)
            if not ok:
                skipped[reason] += 1
                continue
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            sig = build_signature(crop, (x1, y1, x2, y2), img.shape,
                                  net, tf, device)
            records.append({
                "image": p.name,
                "det": k,
                "box": (x1, y1, x2, y2),
                "conf": conf,
                "crop": crop,
                "sig": sig,
            })
            kept += 1
        print(f"  [{i}/{len(img_files)}] {p.name}: {len(boxes)} detected, "
              f"{kept} fully visible "
              f"(dropped edge={skipped['edge']} tiny={skipped['tiny']} "
              f"aspect={skipped['aspect']} small={skipped['small_crop']})")

    if not records:
        print("no cow detections in any image.")
        return

    # ---- pass 2: cluster signatures ------------------------------------------
    sigs = np.stack([r["sig"] for r in records])
    sigs = normalize(sigs)
    # cosine distance = 1 - cosine similarity
    dist_thresh = 1.0 - args.threshold
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=dist_thresh,
    )
    labels = clusterer.fit_predict(sigs)
    n_unique = len(set(labels))
    print(f"\nunique cows identified: {n_unique}")

    # ---- pass 3: save groupings, montages, and CSV ---------------------------
    summary_rows = []
    by_cow: dict[int, list[dict]] = {}
    for r, lbl in zip(records, labels):
        r["cow_id"] = int(lbl)
        by_cow.setdefault(int(lbl), []).append(r)
        summary_rows.append({
            "image": r["image"],
            "det_index": r["det"],
            "box": " ".join(map(str, r["box"])),
            "conf": f"{r['conf']:.3f}",
            "cow_id": int(lbl),
        })

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        w.writeheader()
        w.writerows(summary_rows)
    print(f"wrote {csv_path}")

    # multi-image cows are the interesting ones
    multi = {cid: rs for cid, rs in by_cow.items()
             if len({r["image"] for r in rs}) > 1}
    print(f"cows appearing in >1 image: {len(multi)}")

    for cid, rs in by_cow.items():
        cow_dir = out_dir / f"cow_{cid:03d}"
        cow_dir.mkdir(exist_ok=True)
        for r in rs:
            stem = Path(r["image"]).stem
            fn = f"{stem}_det{r['det']}.jpg"
            cv2.imwrite(str(cow_dir / fn), r["crop"])
        montage = make_montage([r["crop"] for r in rs])
        cv2.imwrite(str(out_dir / f"montage_{cid:03d}.jpg"), montage)

    print(f"\noutput dir: {out_dir}/")
    print("  summary.csv             - image -> cow_id table")
    print("  cow_<id>/               - all crops grouped by identity")
    print("  montage_<id>.jpg        - visual sheet per cow")
    print("\ntop matches (cows in multiple images):")
    for cid, rs in sorted(multi.items(),
                          key=lambda kv: -len({r['image'] for r in kv[1]}))[:10]:
        imgs = sorted({r["image"] for r in rs})
        print(f"  cow_{cid:03d}  ({len(imgs)} images): {', '.join(imgs[:4])}"
              + (" ..." if len(imgs) > 4 else ""))


if __name__ == "__main__":
    main()
