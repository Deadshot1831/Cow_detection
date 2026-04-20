"""
Two-pass cow tracking + dedup:

  Pass 1 -- run COCO YOLOv8m + BoT-SORT (IoU + Kalman + GMC) on first 2 min.
            Log every detection: (frame_idx, track_id, xyxy, conf, hsv_hist).

  Merge  -- post-hoc: fuse two tracks A,B into one if B starts after A ends,
            the temporal gap is short, their bboxes are near each other, and
            their HSV color signatures match (cows have distinctive coats).
            This catches the main remaining dup source: a cow that leaves
            view / hides behind a pillar longer than the tracker buffer and
            comes back as a "new" track.

  Pass 2 -- re-read the video, apply the merged-ID map, render final MP4.

Note: cow_weight_v2_12.pt is trained on UAV nadir drone imagery and yields
zero detections on this ground-level CCTV footage (domain mismatch), so we
fall back to COCO YOLOv8m (class 19 == cow).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


class FFmpegWriter:
    """Pipe raw BGR frames to ffmpeg for H.264 encoding.

    Output is ~5-10x smaller than cv2.VideoWriter('mp4v') at similar quality.
    Falls back to cv2.VideoWriter if ffmpeg is not on PATH.
    """

    def __init__(self, path: Path, fps: float, w: int, h: int,
                 crf: int = 23, preset: str = "medium") -> None:
        self.path = path
        self._cv_writer: cv2.VideoWriter | None = None
        self._proc: subprocess.Popen | None = None
        if shutil.which("ffmpeg") is None:
            self._cv_writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            return
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "pipe:0",
            "-an",
            "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if self._proc is not None:
            self._proc.stdin.write(frame.tobytes())
        else:
            self._cv_writer.write(frame)

    def release(self) -> None:
        if self._proc is not None:
            self._proc.stdin.close()
            self._proc.wait()
        elif self._cv_writer is not None:
            self._cv_writer.release()

HERE = Path(__file__).parent
MODEL_PATH = HERE / "yolov8m.pt"
TRACKER_YAML = HERE / "botsort_cows.yaml"

COW_CLASS_ID = 19
CONF_THRES = 0.30
IOU_THRES = 0.5
DEVICE = "mps"
IMGSZ = 1280

# --- merge-pass thresholds -------------------------------------------------
MAX_TEMPORAL_GAP_S = 45.0   # do not merge tracks separated by more than this
MIN_TEMPORAL_GAP_S = 0.5    # minimum gap; reject near-adjacent tracks fused on weak evidence
MAX_CENTER_DIST_PX = 250    # A-last-center to B-first-center (full 1920x1080 frame)
HIST_SIM_THRESHOLD = 0.72   # 0..1 HSV-histogram correlation (higher = stricter)
MIN_TRACK_FRAMES = 3        # drop tracks this short as noise before merging


@dataclass
class Detection:
    frame: int
    tid: int
    box: tuple[int, int, int, int]
    conf: float


@dataclass
class Track:
    tid: int
    first_frame: int = 10**9
    last_frame: int = -1
    first_box: tuple[int, int, int, int] = (0, 0, 0, 0)
    last_box: tuple[int, int, int, int] = (0, 0, 0, 0)
    hist_sum: np.ndarray | None = None
    hist_n: int = 0
    n_frames: int = 0
    best_conf: float = 0.0

    def add(self, frame_idx: int, box, conf: float, hist: np.ndarray) -> None:
        if frame_idx < self.first_frame:
            self.first_frame = frame_idx
            self.first_box = box
        if frame_idx > self.last_frame:
            self.last_frame = frame_idx
            self.last_box = box
        if self.hist_sum is None:
            self.hist_sum = hist.copy()
        else:
            self.hist_sum += hist
        self.hist_n += 1
        self.n_frames += 1
        self.best_conf = max(self.best_conf, conf)

    @property
    def hist(self) -> np.ndarray:
        h = self.hist_sum / max(1, self.hist_n)
        cv2.normalize(h, h)
        return h


def hsv_hist(crop: np.ndarray) -> np.ndarray:
    """Normalised HSV histogram signature for a bbox crop."""
    if crop.size == 0:
        return np.zeros((16, 16), dtype=np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # H,S histogram -- cow coat colour pattern; ignore V (lighting)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(h, h)
    return h.astype(np.float32)


def center(box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def merge_tracks(tracks: dict[int, Track], fps: float) -> dict[int, int]:
    """Return {old_tid: canonical_tid}. Greedy pair merge by temporal order.

    Rejects a merge if B's lifetime overlaps any existing member of the
    canonical group A belongs to (prevents duplicate IDs on simultaneously
    visible cows when transitive chains would otherwise collapse distinct
    animals into one).
    """
    ordered = sorted(
        (t for t in tracks.values() if t.n_frames >= MIN_TRACK_FRAMES),
        key=lambda t: t.first_frame,
    )
    parent = {t.tid: t.tid for t in ordered}
    by_tid = {t.tid: t for t in ordered}
    # canonical_tid -> list of (first_frame, last_frame) for every member
    group_intervals: dict[int, list[tuple[int, int]]] = {
        t.tid: [(t.first_frame, t.last_frame)] for t in ordered
    }

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    max_gap_frames = MAX_TEMPORAL_GAP_S * fps
    min_gap_frames = MIN_TEMPORAL_GAP_S * fps
    merges = 0
    rejected_overlap = 0
    for i, a in enumerate(ordered):
        best_j: int | None = None
        best_score = -1.0
        for j in range(i + 1, len(ordered)):
            b = ordered[j]
            gap = b.first_frame - a.last_frame
            if gap < min_gap_frames or gap > max_gap_frames:
                continue
            ax, ay = center(a.last_box)
            bx, by = center(b.first_box)
            dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            if dist > MAX_CENTER_DIST_PX:
                continue
            sim = max(0.0, float(cv2.compareHist(a.hist, b.hist, cv2.HISTCMP_CORREL)))
            if sim < HIST_SIM_THRESHOLD:
                continue
            score = sim * (1 - dist / MAX_CENTER_DIST_PX) * (1 - gap / max_gap_frames)
            if score > best_score:
                best_score = score
                best_j = j
        if best_j is None:
            continue
        b = ordered[best_j]
        ra, rb = find(a.tid), find(b.tid)
        if ra == rb:
            continue
        # Transitive temporal-conflict check: B (and every member of B's group)
        # must not overlap any existing member of A's canonical group.
        a_intervals = group_intervals[ra]
        b_intervals = group_intervals[rb]
        conflict = False
        for bf, bl in b_intervals:
            for af, al in a_intervals:
                if bf <= al and af <= bl:
                    conflict = True
                    break
            if conflict:
                break
        if conflict:
            rejected_overlap += 1
            continue
        parent[rb] = ra
        group_intervals[ra] = a_intervals + b_intervals
        del group_intervals[rb]
        merges += 1

    print(f"  merge pass: fused {merges} track pairs "
          f"(rejected {rejected_overlap} overlap conflicts)")
    return {tid: find(tid) for tid in parent}


def pass1_track(model: YOLO, cap: cv2.VideoCapture, fps: float, max_frames: int
                ) -> tuple[list[Detection], dict[int, Track]]:
    tracks: dict[int, Track] = {}
    dets: list[Detection] = []
    frame_idx = 0
    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.track(
            source=frame, persist=True,
            tracker=str(TRACKER_YAML), classes=[COW_CLASS_ID],
            conf=CONF_THRES, iou=IOU_THRES, imgsz=IMGSZ,
            device=DEVICE, verbose=False,
        )[0]
        if r.boxes is not None and r.boxes.id is not None:
            for (x1, y1, x2, y2), tid, conf in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.id.int().cpu().tolist(),
                r.boxes.conf.cpu().tolist(),
            ):
                x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
                box = (x1i, y1i, x2i, y2i)
                dets.append(Detection(frame_idx, tid, box, conf))
                crop = frame[max(0, y1i):y2i, max(0, x1i):x2i]
                h = hsv_hist(crop)
                tracks.setdefault(tid, Track(tid)).add(frame_idx, box, conf, h)
        frame_idx += 1
        if frame_idx % 250 == 0:
            print(f"  pass1 {frame_idx}/{max_frames}  tracks={len(tracks)}")
    return dets, tracks


def pass2_render(cap: cv2.VideoCapture, writer: cv2.VideoWriter,
                 dets: list[Detection], remap: dict[int, int],
                 dropped: set[int], fps: float, max_frames: int) -> int:
    by_frame: dict[int, list[Detection]] = {}
    for d in dets:
        if d.tid in dropped:
            continue
        by_frame.setdefault(d.frame, []).append(d)

    unique_canonical: set[int] = set()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for frame_idx in range(max_frames):
        ok, frame = cap.read()
        if not ok:
            break
        current: list[int] = []
        for d in by_frame.get(frame_idx, []):
            canon = remap.get(d.tid, d.tid)
            unique_canonical.add(canon)
            current.append(canon)
            x1, y1, x2, y2 = d.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"cow #{canon} {d.conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        hud = [
            f"frame {frame_idx + 1}/{max_frames}  t={frame_idx / fps:5.1f}s",
            f"in-frame cows: {len(current)}",
            f"unique cows  : {len(unique_canonical)}",
        ]
        for i, line in enumerate(hud):
            y = 30 + i * 28
            cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        if (frame_idx + 1) % 500 == 0:
            print(f"  pass2 {frame_idx + 1}/{max_frames}  unique={len(unique_canonical)}")

    return len(unique_canonical)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Track + dedupe cows in a video and render an annotated MP4.",
    )
    p.add_argument("video", type=Path, help="input video (mp4, mov, avi, ...)")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="output MP4 path (default: <video>_annotated.mp4)")
    p.add_argument("-d", "--duration", type=float, default=120.0,
                   help="seconds to process from the start (default 120; 0 = whole video)")
    p.add_argument("--conf", type=float, default=CONF_THRES,
                   help=f"detector conf threshold (default {CONF_THRES})")
    p.add_argument("--imgsz", type=int, default=IMGSZ,
                   help=f"inference image size (default {IMGSZ})")
    p.add_argument("--device", type=str, default=DEVICE,
                   help=f"torch device: mps | cpu | 0 (cuda) (default {DEVICE})")
    p.add_argument("--model", type=Path, default=MODEL_PATH,
                   help=f"YOLO weights (default {MODEL_PATH.name})")
    p.add_argument("--crf", type=int, default=23,
                   help="H.264 quality: lower = better/larger (18 visually lossless, "
                        "23 default, 28 very small). Default 23")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    video_path: Path = args.video
    if not video_path.exists():
        raise SystemExit(f"video not found: {video_path}")
    out_path: Path = args.out or video_path.with_name(video_path.stem + "_annotated.mp4")

    # let CLI overrides propagate to the inference helpers
    global CONF_THRES, IMGSZ, DEVICE, MODEL_PATH
    CONF_THRES, IMGSZ, DEVICE, MODEL_PATH = args.conf, args.imgsz, args.device, args.model

    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = total if args.duration == 0 else min(total, int(args.duration * fps))

    print(f"input : {video_path}  ({w}x{h} @ {fps:.1f}fps, {total} frames)")
    print(f"output: {out_path}")
    print(f"window: first {max_frames} frames ({max_frames / fps:.1f}s)")

    print("--- pass 1: tracking ---")
    dets, tracks = pass1_track(model, cap, fps, max_frames)
    print(f"  raw tracks: {len(tracks)}  detections: {len(dets)}")

    dropped = {tid for tid, t in tracks.items() if t.n_frames < MIN_TRACK_FRAMES}
    print(f"  dropped <{MIN_TRACK_FRAMES}-frame noise tracks: {len(dropped)}")

    print("--- merge: HSV + spatial + temporal ---")
    remap = merge_tracks(tracks, fps)
    canonical = {remap[t] for t in remap if t not in dropped}
    print(f"  canonical tracks after merge: {len(canonical)}")

    print("--- pass 2: rendering (H.264 via ffmpeg) ---")
    writer = FFmpegWriter(out_path, fps, w, h, crf=args.crf)
    unique = pass2_render(cap, writer, dets, remap, dropped, fps, max_frames)
    cap.release()
    writer.release()

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nDone. Unique cows: {unique}")
    print(f"Output: {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
