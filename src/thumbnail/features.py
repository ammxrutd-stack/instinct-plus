"""Thumbnail features: an image path -> dict of named floats.

Measured with OpenCV only (no deep models), so extraction runs in a few
milliseconds per image on any laptop and every feature has a plain-language
meaning a creator can act on. See docs/feature_dictionary_thumbnail.md.

The input is assumed to be YouTube's hqdefault (480x360, 4:3) which letterboxes
16:9 uploads with black bars. The bars are detected and removed first so that
"dark fraction" measures the picture, not the padding. Uploads that are not
letterboxed are used as they are.

Face detection uses the Haar cascade that ships with OpenCV. It misses profile
and heavily stylised faces, which the thumbnail pair can improve on later with
MediaPipe without changing the feature names.
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.schema import CONTENT_TYPES

FEATURE_VERSION = "thumb-v3"
WORK_WIDTH = 480

_FACE_CASCADE = None


def _cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(str(path))
    return _FACE_CASCADE


def load_image(path) -> Optional[np.ndarray]:
    data = np.fromfile(str(path), dtype=np.uint8) if Path(path).exists() else None
    if data is None or data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def strip_letterbox(img: np.ndarray, threshold: int = 18, min_band: int = 8) -> Tuple[np.ndarray, float]:
    """Remove near-black bands at top and bottom. Returns (cropped, band_fraction)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row_mean = gray.mean(axis=1)
    h = gray.shape[0]
    top = 0
    while top < h // 3 and row_mean[top] < threshold:
        top += 1
    bottom = h
    while bottom > 2 * h // 3 and row_mean[bottom - 1] < threshold:
        bottom -= 1
    if top < min_band and h - bottom < min_band:
        return img, 0.0
    # only strip when both bands are present or one is clearly a bar
    cropped = img[top:bottom]
    if cropped.shape[0] < h * 0.4:
        return img, 0.0
    return cropped, float((top + (h - bottom)) / h)


def _resize(img: np.ndarray, width: int = WORK_WIDTH) -> np.ndarray:
    h, w = img.shape[:2]
    if w == width:
        return img
    return cv2.resize(img, (width, max(1, int(round(h * width / w)))), interpolation=cv2.INTER_AREA)


def colorfulness(img: np.ndarray) -> float:
    """Hasler & Süsstrunk (2003) colourfulness metric."""
    b, g, r = cv2.split(img.astype(np.float32))
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def detect_faces(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = _cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(28, 28))
    return [tuple(int(v) for v in f) for f in faces] if len(faces) else []


def detect_text_regions(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Heuristic text-block detector: strong local gradients joined horizontally
    into wide, shallow components. Catches overlaid captions, which is what
    thumbnails have; it is not OCR and does not read the words."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    _, binary = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    joined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, w // 40), 3)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bh < 0.035 * h or bh > 0.45 * h:
            continue
        if bw < 1.3 * bh or bw < 0.08 * w:
            continue
        fill = area / float(bw * bh)
        if fill < 0.3:
            continue
        boxes.append((int(x), int(y), int(bw), int(bh)))
    return boxes


def dominant_colour_count(img: np.ndarray, k: int = 6, min_share: float = 0.06) -> int:
    """Number of colour clusters covering at least min_share of the frame."""
    small = cv2.resize(img, (48, 27), interpolation=cv2.INTER_AREA).reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, _ = cv2.kmeans(small, k, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    shares = np.bincount(labels.ravel(), minlength=k) / float(len(labels))
    return int((shares >= min_share).sum())


def extract_image_features(img: np.ndarray) -> Dict[str, float]:
    """Features for an already-loaded BGR image."""
    img, band_fraction = strip_letterbox(img)
    img = _resize(img)
    h, w = img.shape[:2]
    area = float(h * w)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0].astype(np.float32), hsv[..., 1].astype(np.float32) / 255.0, hsv[..., 2].astype(np.float32) / 255.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # hue buckets, weighted by saturation so grey pixels do not vote
    weights = sat * (val > 0.15)
    total_w = float(weights.sum()) or 1.0

    def hue_share(lo: float, hi: float) -> float:
        if lo <= hi:
            m = (hue >= lo) & (hue < hi)
        else:
            m = (hue >= lo) | (hue < hi)
        return float((weights * m).sum() / total_w)

    red = hue_share(170, 10)
    orange_yellow = hue_share(10, 35)
    green = hue_share(35, 85)
    blue = hue_share(85, 135)
    purple_pink = hue_share(135, 170)

    edges = cv2.Canny(gray, 100, 200)
    ch, cw = h // 3, w // 3
    centre_edges = edges[ch:2 * ch, cw:2 * cw]
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).ravel()
    p = hist / hist.sum()
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())

    faces = detect_faces(img)
    face_areas = [fw * fh for (_, _, fw, fh) in faces]
    largest = max(face_areas) if face_areas else 0
    if faces:
        x, y, fw, fh = faces[int(np.argmax(face_areas))]
        face_cx, face_cy = (x + fw / 2) / w, (y + fh / 2) / h
    else:
        face_cx = face_cy = 0.5

    text_boxes = detect_text_regions(img)
    text_area = sum(bw * bh for (_, _, bw, bh) in text_boxes) / area

    left = gray[:, : w // 2].astype(np.float32)
    right = cv2.flip(gray[:, w - w // 2:], 1).astype(np.float32)
    symmetry = 1.0 - float(np.abs(left - right).mean() / 255.0)

    f: Dict[str, float] = {
        "brightness_mean": float(val.mean()),
        "brightness_std": float(val.std()),
        "contrast": float(gray.std() / 255.0),
        "dark_fraction": float((val < 0.2).mean()),
        "bright_fraction": float((val > 0.85).mean()),
        "saturation_mean": float(sat.mean()),
        "saturation_std": float(sat.std()),
        "colorfulness": colorfulness(img),
        "red_share": red,
        "orange_yellow_share": orange_yellow,
        "green_share": green,
        "blue_share": blue,
        "purple_pink_share": purple_pink,
        "warm_share": red + orange_yellow,
        "dominant_colours": float(dominant_colour_count(img)),
        "edge_density": float(edges.mean() / 255.0),
        "centre_edge_density": float(centre_edges.mean() / 255.0),
        "sharpness": float(min(cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0, 10.0)),
        "entropy": entropy,
        "symmetry": symmetry,
        "face_count": float(len(faces)),
        "face_present": float(bool(faces)),
        "largest_face_area": float(largest / area),
        "face_centre_x": float(face_cx),
        "face_centre_y": float(face_cy),
        "text_area": float(text_area),
        "text_block_count": float(len(text_boxes)),
        "letterbox_fraction": float(band_fraction),
        "aspect_ratio": float(w / h),
    }
    return f


def extract_thumbnail_features(image_path, content_type: str, subscriber_count: Optional[float] = None) -> Dict[str, float]:
    """Features for an image on disk. A missing or unreadable image yields a
    frame of zeros plus `image_missing=1`, so the caller can decide what to
    do; training drops such rows, the app refuses them."""
    img = load_image(image_path) if image_path else None
    if img is None:
        f = {name: 0.0 for name in IMAGE_FEATURE_NAMES}
        f["image_missing"] = 1.0
    else:
        f = extract_image_features(img)
        f["image_missing"] = 0.0
    f["log_subscribers"] = math.log10(subscriber_count) if subscriber_count and subscriber_count > 0 else float("nan")
    for ct in CONTENT_TYPES:
        f["ct_" + ct] = float(content_type == ct)
    return f


def annotate(img: np.ndarray) -> np.ndarray:
    """Draw detected faces (green) and text regions (orange) for the app."""
    img, _ = strip_letterbox(img)
    img = _resize(img).copy()
    for (x, y, w, h) in detect_text_regions(img):
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 140, 255), 2)
    for (x, y, w, h) in detect_faces(img):
        cv2.rectangle(img, (x, y), (x + w, y + h), (80, 220, 80), 2)
    return img


IMAGE_FEATURE_NAMES: List[str] = list(extract_image_features(np.full((36, 48, 3), 128, dtype=np.uint8)).keys())

# Measured and reported, but not given to the model. YouTube's hqdefault pads
# 16:9 uploads with black bars, so in training "has bars" means "was 16:9",
# while a creator's own 1280x720 upload has none: the feature would mean the
# opposite thing at training and at serving time. aspect_ratio, measured on
# the picture after the bars are removed, carries the real information.
NON_MODEL_FEATURES = {"letterbox_fraction"}
THUMBNAIL_FEATURES: List[str] = (
    [n for n in IMAGE_FEATURE_NAMES if n not in NON_MODEL_FEATURES]
    + ["image_missing", "log_subscribers"]
    + ["ct_" + ct for ct in CONTENT_TYPES]
)
