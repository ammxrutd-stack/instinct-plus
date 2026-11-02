# Schema notes

`config/schema.py` is the single source of truth. This page explains the
choices behind it; the file has the values.

## The processed table

One row per public video, written by `python -m src.collection build` (or by
`src/data/synthetic.py` as a stand-in). Columns, in order:

| Column | Type | Source | Notes |
|---|---|---|---|
| `video_id` | str | `videos.list` `id` | primary key |
| `title` | str | `snippet.title` | as published, untrimmed |
| `channel_id` | str | `snippet.channelId` | the cross-validation group |
| `channel_title` | str | channel `snippet.title` | for reading, never a feature |
| `subscriber_count` | int | channel `statistics.subscriberCount` | at collection time, rounded by the API to three significant figures |
| `view_count` | int | `statistics.viewCount` | a snapshot at collection time |
| `published_at` | str | `snippet.publishedAt` | ISO 8601 UTC |
| `duration_seconds` | int | parsed from `contentDetails.duration` | Shorts (< 60 s) are excluded upstream |
| `content_type` | str | the seed list | one of `CONTENT_TYPES` |
| `thumbnail_file` | str | `{video_id}.jpg` under `data/thumbnails/` | empty when the download failed; the row stays because the title model can still use it |

Stored as `data/processed/videos.parquet` (dtypes preserved) and optionally
`videos.csv`. Thumbnails are YouTube's `hqdefault`: 480x360 with black bars
when the upload was 16:9, which is nearly always.

## Derived columns

`src/scoring/target.py` adds two columns at training time; collection never
writes them.

**`reach`** = log10((views + 1) / subscribers), adjusted for age. Views per
subscriber is the thing a title and thumbnail influence: a million views is
routine at 30M subscribers and remarkable at 100K. The log makes the ratio
roughly normal. A pooled linear slope of reach on log10(age in days) is
removed, so a video that has had 300 days to collect views is not credited for
its age; the slope is stored in the model card. The slope on the synthetic
data is 0.32; on collected data it will be whatever the data says.

**`reach_score`** = percentile rank of `reach` within its content type, 0 to
100. Reporting only. The models train on `reach` and the 0-100 score a user
sees is produced differently (see "Score" below).

## Filters

Applied by `build`, all read from the schema:

- `subscriber_count >= 10,000` (`MIN_SUBSCRIBERS`): below this the API's
  rounding makes channel size too coarse.
- `30 <= age_days <= 365`: views need a month to settle; after a year age
  dominates.
- `duration_seconds >= 60`: Shorts are a different surface with different
  view counting.
- `view_count` present and `liveBroadcastContent == "none"`.

## Score

A score is the percentile of the model's prediction among the model's own
out-of-fold predictions for training videos of the same content type, after
the contribution of channel size has been removed. "72" means "the model
rates this higher than 72% of the category". Two things follow:

- Scores are relative to the category. 72 in finance and 72 in gaming are
  both "better than 72% of that category", not the same expected views.
- Scores are relative to the model, not to the world. A model with low skill
  spreads scores across 0-100 just as much as a good one; the model card's
  Spearman is what says how much a score means.

Drivers are SHAP contributions in prediction units multiplied by the local
slope of the score map (points per unit of prediction between the 20th and
80th percentiles), so they are in points and roughly additive.

## Changing the schema

Adding a column: append to `COLUMNS`, populate it in `src/collection/build.py`
and `src/data/synthetic.py`, rerun the collection tests. Renaming or removing
one: expect every pair's code and tests to change; agree it first.
