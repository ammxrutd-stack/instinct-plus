"""Shared dataset schema for Instinct+.

Every other module imports its column names, file patterns and collection
filters from here instead of hardcoding strings, so the title pair, the
thumbnail pair and the interface pair all read the same tables.

Changes to this file require agreement from all three pairs.
"""

from pathlib import Path

# --- paths ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
THUMBNAILS_DIR = DATA_DIR / "thumbnails"
MODELS_DIR = PROJECT_ROOT / "models"

# the processed table every model trains against
VIDEOS_TABLE = PROCESSED_DIR / "videos.parquet"
VIDEOS_TABLE_CSV = PROCESSED_DIR / "videos.csv"

# --- column names -----------------------------------------------------------

VIDEO_ID = "video_id"
TITLE = "title"
CHANNEL_ID = "channel_id"
CHANNEL_TITLE = "channel_title"
SUBSCRIBER_COUNT = "subscriber_count"
VIEW_COUNT = "view_count"
PUBLISHED_AT = "published_at"
DURATION_SECONDS = "duration_seconds"
CONTENT_TYPE = "content_type"
THUMBNAIL_FILE = "thumbnail_file"

COLUMNS = [
    VIDEO_ID,
    TITLE,
    CHANNEL_ID,
    CHANNEL_TITLE,
    SUBSCRIBER_COUNT,
    VIEW_COUNT,
    PUBLISHED_AT,
    DURATION_SECONDS,
    CONTENT_TYPE,
    THUMBNAIL_FILE,
]

# derived by src/scoring/target.py, never by collection
REACH = "reach"                    # age-adjusted log10(views / subscribers)
REACH_SCORE = "reach_score"        # percentile of REACH within content type, 0-100

# --- files ------------------------------------------------------------------

# naming for images in data/thumbnails/
THUMBNAIL_PATTERN = "{video_id}.jpg"

# hqdefault is 480x360 with letterboxing on 16:9 uploads. The thumbnail
# feature extractor crops to this frame before measuring anything.
THUMBNAIL_SIZE = (480, 360)

# --- content types ----------------------------------------------------------

CONTENT_TYPES = ["gaming", "cooking", "finance", "education"]

# --- collection filters -----------------------------------------------------

# the API rounds subscriber counts to three significant figures, so below this
# the channel size feature is too coarse to be useful
MIN_SUBSCRIBERS = 10_000

# view counts need time to settle before they represent the video's reach
MIN_DAYS_SINCE_PUBLISH = 30

# older videos accumulate views for longer, which confounds the target variable
MAX_DAYS_SINCE_PUBLISH = 365

# Shorts are served through a different surface with different view counting
MIN_DURATION_SECONDS = 60

# --- scoring ----------------------------------------------------------------

SCORE_MIN = 0.0
SCORE_MAX = 100.0
