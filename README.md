# Instinct+

Instinct+ predicts how far a YouTube video will reach from its title and
thumbnail, before it is published. A creator enters a proposed title, uploads
a thumbnail and picks a category; the app returns a 0-100 reach score for
each, an overall score, the features that drove it, plain-language
suggestions, and alternative titles written by Claude and re-scored by the
same model.

Two gradient boosted tree models are trained separately and combined: a
title model on text features and a thumbnail model on image features, both
predicting the same target (age-adjusted views per subscriber). They train on
public data collected through the YouTube Data API v3, or on a synthetic
stand-in so everything runs without keys.

## Quick start (no API keys)

```bash
git clone <repo-url> && cd Instinct+
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
brew install libomp                                     # macOS only; XGBoost needs OpenMP

python -m src.data.synthetic        # ~3,400 synthetic videos + thumbnails into data/
python -m src.modelling.train       # trains both models into models/, prints held-out metrics
pytest                              # 115 tests, no network
streamlit run app/main.py           # the app, at http://localhost:8501
```

Or `make setup demo test app`. Models trained on the synthetic set show a
banner in the app: they demonstrate the pipeline, not YouTube.

## The real dataset

```bash
cp .env.example .env                # paste YOUTUBE_API_KEY (and ANTHROPIC_API_KEY for suggestions)
python -m src.collection verify     # every seed handle resolves? subscriber counts?
python -m src.collection all --smoke   # 2 channels, cheap: proves the key works
python -m src.collection all        # discover -> fetch -> build -> thumbnails, ~1,300 quota units
python -m src.modelling.train
python -m src.modelling.evaluate --out docs/model_report.md
```

Collection is resumable: quota exhaustion writes a checkpoint and the next
run continues. Raw API responses are kept verbatim under `data/raw/` so the
table can be rebuilt without spending quota. Thumbnails are plain HTTPS
downloads and cost nothing. The design and its rationale are in
`docs/superpowers/specs/2026-09-16-collection-pipeline-design.md`.

The dataset is shared through Google Drive, not git (thumbnails run to
hundreds of megabytes and the API terms restrict redistribution). Unpack it
into `data/` so raw responses land in `data/raw/`, images in
`data/thumbnails/` and the table in `data/processed/`.

## What the app does

| Tab | What you get |
|---|---|
| **Score** | Title, thumbnail and overall gauges; a bar chart of drivers in points; the thumbnail annotated with detected faces and text regions; suggestions; every measured feature in an expander. **Suggest titles** asks Claude for six alternatives that address the diagnosis, scores each, and ranks them; **Use** swaps one in. |
| **Compare** | Two drafts side by side (title + thumbnail each), the winner, and which features separate the titles. |
| **Batch** | Paste titles one per line or upload a CSV; download the scored table. |
| **Model card** | Held-out metrics per content type, feature importance, provenance (data source, training date, data hash), and how to read a score. |

The sidebar takes your subscriber count. The models use it because views per
subscriber falls as channels grow, but its contribution is held out of the
score and shown separately, so a 50K channel and a 5M channel comparing the
same title get the same answer.

Command line equivalent: `python -m src.scoring.cli "7 Ramen Mistakes" --type cooking --thumb t.jpg --subs 250000 --suggest`.

## How a score is made

1. **Target.** `reach = log10(views / subscribers)`, with a pooled age slope
   removed (`src/scoring/target.py`, `docs/schema.md`).
2. **Features.** 41 named title features (`src/title/features.py`) and 28
   named image features (`src/thumbnail/features.py`), each with a
   definition in `docs/feature_dictionary_*.md`. Nothing a creator cannot
   act on.
3. **Model.** XGBoost regression (scikit-learn gradient boosting when
   XGBoost is unavailable), cross-validated with channels grouped so no
   channel is in both training and validation (`src/modelling/`).
4. **Score.** The percentile of the prediction among the model's own
   out-of-fold predictions for the same content type, after removing the
   channel-size contribution. 70 means "better than 70% of the category".
5. **Drivers.** SHAP contributions converted to score points, and turned into
   sentences by `src/scoring/suggestions.py`.

`docs/validation_protocol.md` says how models are judged; the latest run is
`docs/model_report.md`.

## Structure

| Path | Contents |
|---|---|
| `config/schema.py` | shared schema: columns, paths, filters, content types. Needs all-pair agreement to change. |
| `config/channels.py` | seed channel handles per content type |
| `src/collection/` | YouTube Data API v3 pipeline: `client`, `discover`, `fetch`, `build`, `thumbnails`, checkpointing, `__main__` |
| `src/data/synthetic.py` | synthetic dataset generator (writes a `SYNTHETIC` marker) |
| `src/scoring/` | the contract (`interface.py`), target definition, suggestions, placeholder, `registry.py` (real models or placeholder, combination), `cli.py` |
| `src/modelling/` | shared training: backend, calibration, SHAP explainer, `ReachModel`, `train.py`, `evaluate.py` |
| `src/title/` | title features, `train.py`, `model.py` (`score_title`) |
| `src/thumbnail/` | thumbnail features, `train.py`, `model.py` (`score_thumbnail`) |
| `src/generation/` | Claude title generation and the generate-score-rank loop |
| `app/` | Streamlit interface and chart components |
| `tests/` | contract, feature, model, generation, synthetic and collection tests with recorded fixtures |
| `docs/` | schema notes, validation protocol, feature dictionaries, model report, collection design spec |
| `data/`, `models/` | collected data and trained artefacts. Not in git. |
| `notebooks/` | exploration only, one per person |

## Rules

**Never commit data, trained models or `.env`.** The `.gitignore` covers this.

**`config/schema.py` and `src/scoring/interface.py` need agreement from all
three pairs before changing.** Everything else, each pair owns its own
directory.

**Notebooks are for exploration only.** When code works, move it into `src/`
as a function with a test.

**`src/scoring/placeholder.py`** returns random scores. The registry only
uses it when a trained model is missing, and the app says so.

## Branching

`main` plus one branch per pair: `title`, `thumbnail`, `app`. Merge into
`main` at milestone weeks; pull `main` into your branch first.

## Extending it

- **Better faces**: swap the Haar cascade in `src/thumbnail/features.py`
  for MediaPipe; the feature names stay the same.
- **Semantic title features**: a few SVD components over sentence
  embeddings, each labelled by its nearest titles so it can still be a
  driver. `requirements-optional.txt` lists the packages.
- **Fresh view counts**: a top-up collection run that re-fetches
  `statistics` for already-collected ids (out of scope in the spec, cheap to
  add to `src/collection/fetch.py`).
- **A fourth input**: description or tags, as a third model behind the same
  `ScoreResult` contract.
