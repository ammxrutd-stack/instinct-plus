# Instinct+ — start here

This folder is the complete Instinct+ project as demoed at the Week 12 roadshow.
It runs on your own machine with no API keys: the trained models and the
dataset they were trained on are included, so the app works straight away.

## Run the app (10 minutes)

Requirements: Python 3.9 or newer. On macOS you also need Homebrew for one step.

```bash
cd "Instinct+ (team share)"
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
brew install libomp                # macOS only. XGBoost needs it. Skip on Windows/Linux.
streamlit run app/main.py
```

The app opens at http://localhost:8501. Try it with a thumbnail from
`data/channel_demo/projectvolta/` and any title, or use the Project Volta
demo panel inside the Score tab, which recognises those thumbnails and fills
in their original titles.

## Optional: Claude title suggestions

Copy `.env.example` to `.env` and paste your own `ANTHROPIC_API_KEY`.
Without a key the app still works; the Suggest titles button falls back to
rule-based rewrites.

## Optional: retrain or rebuild

```bash
python -m src.data.synthetic       # regenerate the synthetic dataset
python -m src.modelling.train      # retrain both models, prints held-out metrics
pytest                             # 115 tests, no network needed
```

For real YouTube data, add `YOUTUBE_API_KEY` to `.env` and run
`python -m src.collection all` (about 1,300 quota units, one day's free quota).
Then retrain. See README.md for the full pipeline.

## What is where

| Path | What it is |
|---|---|
| `app/main.py` | The Streamlit app (Score, Compare, Batch, Model card tabs) |
| `src/title/` | Title feature extraction (41 features) and title scoring |
| `src/thumbnail/` | Thumbnail feature extraction (OpenCV, 28 features) and scoring |
| `src/modelling/` | XGBoost training, channel-grouped cross validation, SHAP drivers, calibration |
| `src/scoring/` | Shared ScoreResult contract, registry, suggestions, CLI |
| `src/generation/titles.py` | Claude title generation with rule-based fallback |
| `src/collection/` | YouTube Data API collection pipeline |
| `src/data/synthetic.py` | Synthetic dataset generator (the written-down rule) |
| `scripts/channel_check.py` | Pull and score any public channel without an API key |
| `models/` | Trained title and thumbnail models plus model cards |
| `data/processed/` | The training table. `SYNTHETIC` marker means synthetic data |
| `data/channel_demo/projectvolta/` | Project Volta thumbnails, titles and scores used in the demo |
| `docs/presenter_notes.md` | Slide-by-slide notes, numbers to have ready, likely questions |
| `docs/Instinct+ slides.pptx` | The roadshow slides |
| `docs/model_report.md` | Validation results |
| `docs/feature_dictionary_*.md` | Every feature, how it is calculated, why it is there |
| `docs/validation_protocol.md` | How the models are evaluated and why grouped CV |
| `tests/` | 115 tests |

## Honest status

The models in `models/` were trained on the synthetic dataset. The numbers
(title Spearman 0.57, thumbnail 0.55, baseline 0.12) show the pipeline
recovers rules we wrote down, not that it predicts YouTube. The first real
collection run is the next step and the whole pipeline is ready for it.

## Troubleshooting

- `XGBoostError ... libomp` on macOS: run `brew install libomp`.
- `cv2 has no attribute CascadeClassifier`: you have OpenCV 5. Run
  `pip install "opencv-python-headless<5"`.
- Port 8501 busy: `streamlit run app/main.py --server.port 8502`.
