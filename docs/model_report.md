# Model evaluation report

Generated 2026-10-21T01:59+00:00 by `python -m src.modelling.evaluate`.
Data source: **synthetic**. Age slope removed from the target: 0.242.

Metrics are out-of-fold with channels grouped, so no channel appears in both the training and the validation fold.

## title model

3,360 videos from 56 channels.

| | Spearman ρ | R² | MAE |
|---|---|---|---|
| Model | 0.565 | 0.308 | 0.326 |
| Baseline: content-type mean | 0.118 | 0.036 | 0.387 |
| Baseline: global mean | – | 0.000 | 0.397 |

| Content type | videos | Spearman ρ | MAE |
|---|---|---|---|
| cooking | 840 | 0.361 | 0.350 |
| education | 840 | 0.498 | 0.304 |
| finance | 840 | 0.521 | 0.336 |
| gaming | 840 | 0.588 | 0.315 |

Calibration (a flat, rising line means the 0-100 score is honest):

| Score bucket | videos | mean true percentile |
|---|---|---|
| 0-20 | 671 | 30 |
| 20-40 | 673 | 41 |
| 40-60 | 672 | 49 |
| 60-80 | 672 | 59 |
| 80-100 | 672 | 71 |

## thumbnail model

3,360 videos from 56 channels.

| | Spearman ρ | R² | MAE |
|---|---|---|---|
| Model | 0.548 | 0.286 | 0.332 |
| Baseline: content-type mean | 0.118 | 0.036 | 0.387 |
| Baseline: global mean | – | 0.000 | 0.397 |

| Content type | videos | Spearman ρ | MAE |
|---|---|---|---|
| cooking | 840 | 0.328 | 0.358 |
| education | 840 | 0.475 | 0.316 |
| finance | 840 | 0.490 | 0.345 |
| gaming | 840 | 0.599 | 0.309 |

Calibration (a flat, rising line means the 0-100 score is honest):

| Score bucket | videos | mean true percentile |
|---|---|---|
| 0-20 | 672 | 31 |
| 20-40 | 672 | 40 |
| 40-60 | 672 | 50 |
| 60-80 | 672 | 59 |
| 80-100 | 672 | 69 |

> These numbers come from synthetic data and describe how well the pipeline recovers the generator's rules, not how well it predicts YouTube.
