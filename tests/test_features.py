import math

import numpy as np
import pytest

from config.schema import CONTENT_TYPES
from src.title.features import TITLE_FEATURES, extract_title_features


def test_title_feature_names_are_stable():
    f = extract_title_features("anything", "gaming")
    assert list(f.keys()) == TITLE_FEATURES


@pytest.mark.parametrize("ct", CONTENT_TYPES)
def test_content_type_onehot(ct):
    f = extract_title_features("x", ct)
    assert f["ct_" + ct] == 1.0
    assert sum(f["ct_" + c] for c in CONTENT_TYPES) == 1.0


def test_unknown_content_type_has_no_onehot():
    f = extract_title_features("x", "podcast")
    assert sum(f["ct_" + c] for c in CONTENT_TYPES) == 0.0


def test_title_features_measure_what_they_say():
    f = extract_title_features("Why I QUIT Minecraft After 10 Years (honest)", "gaming")
    assert f["char_count"] == 44
    assert f["word_count"] == 8
    assert f["has_number"] == 1 and f["number_count"] == 1
    assert f["caps_word_count"] == 1 and 0 < f["caps_word_ratio"] < 0.2
    assert f["has_parentheses"] == 1
    assert f["first_person"] == 1
    assert f["starts_with_question_word"] == 1
    assert f["is_question"] == 0
    assert f["power_word_count"] >= 1  # honest
    assert math.isnan(f["log_subscribers"])


def test_subscriber_context():
    f = extract_title_features("x", "gaming", 1_000_000)
    assert abs(f["log_subscribers"] - 6.0) < 1e-9
    assert math.isnan(extract_title_features("x", "gaming", 0)["log_subscribers"])


def test_empty_and_emoji_titles_do_not_crash():
    f = extract_title_features("", "gaming")
    assert f["char_count"] == 0 and f["word_count"] == 0
    f = extract_title_features("🔥🔥 WOW 🔥", "gaming")
    assert f["emoji_count"] == 3


def test_question_and_listicle():
    assert extract_title_features("Is this real?", "finance")["is_question"] == 1
    assert extract_title_features("7 ways to save money", "finance")["is_listicle"] == 1
    assert extract_title_features("Xbox vs PS5", "gaming")["is_versus"] == 1
    assert extract_title_features("How to cook rice", "cooking")["how_to"] == 1


# --------------------------------------------------------------- thumbnail --

from src.thumbnail.features import (  # noqa: E402
    THUMBNAIL_FEATURES,
    detect_text_regions,
    extract_image_features,
    extract_thumbnail_features,
    strip_letterbox,
)


def _frame(h=360, w=480, colour=(30, 120, 220)):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = colour
    return img


def test_letterbox_is_stripped():
    img = np.zeros((360, 480, 3), np.uint8)
    img[45:315] = (100, 100, 100)
    cropped, frac = strip_letterbox(img)
    assert cropped.shape[0] == 270
    assert abs(frac - 0.25) < 0.01
    full = _frame()
    same, frac = strip_letterbox(full)
    assert same.shape == full.shape and frac == 0.0


def test_text_regions_found_on_caption():
    import cv2

    img = _frame()
    cv2.putText(img, "BEST LAPTOPS", (40, 180), cv2.FONT_HERSHEY_DUPLEX, 2.2, (255, 255, 255), 5)
    assert len(detect_text_regions(img)) >= 1
    assert len(detect_text_regions(_frame())) == 0


def test_image_features_ranges():
    f = extract_image_features(_frame())
    assert 0 <= f["brightness_mean"] <= 1
    assert f["dark_fraction"] == 0.0
    assert f["face_present"] == 0.0
    assert f["text_area"] == 0.0
    assert abs(f["aspect_ratio"] - 4 / 3) < 0.01
    dark = extract_image_features(_frame(colour=(5, 5, 5)))
    assert dark["dark_fraction"] == 1.0


def test_missing_image_is_flagged(tmp_path):
    f = extract_thumbnail_features(tmp_path / "nope.jpg", "gaming")
    assert f["image_missing"] == 1.0
    assert set(THUMBNAIL_FEATURES) <= set(f.keys())
    assert "letterbox_fraction" in f and "letterbox_fraction" not in THUMBNAIL_FEATURES  # measured, not modelled


def test_thumbnail_feature_names_are_stable(tiny_data):
    p = next((tiny_data / "thumbnails").glob("*.jpg"))
    f = extract_thumbnail_features(p, "gaming", 50_000)
    assert set(THUMBNAIL_FEATURES) <= set(f.keys())
    missing = extract_thumbnail_features(tiny_data / "nope.jpg", "gaming", 50_000)
    assert list(missing.keys()) == list(f.keys())  # same shape whether or not the image loads
    assert f["image_missing"] == 0.0


def test_drawn_faces_are_mostly_detected(tiny_data):
    """Synthetic faces are detected often enough to carry signal."""
    import random

    from src.data.synthetic import make_thumbnail, sample_image_props

    rng = random.Random(5)
    hit = tot = 0
    for i in range(30):
        props = sample_image_props(rng, "finance")
        props["face"] = 1.0
        p = tiny_data / f"facecheck{i}.jpg"
        make_thumbnail(rng, props, p)
        tot += 1
        hit += extract_thumbnail_features(p, "finance")["face_present"]
        p.unlink()
    assert hit / tot >= 0.4
