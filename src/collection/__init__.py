"""YouTube Data API v3 collection pipeline.

Five file-to-file stages (discover, fetch, build, thumbnails) behind one
network boundary (client). Run with ``python -m src.collection``; see
docs/superpowers/specs/2026-09-16-collection-pipeline-design.md.
"""

from .client import Client, CollectionError, QuotaExceeded, load_api_key  # noqa: F401
