# Validation protocol

How we decide whether a model is good, and whether a change made it better.
`python -m src.modelling.evaluate --out docs/model_report.md` runs this
protocol and writes the numbers; `docs/model_report.md` is the latest run.

## Split

**Grouped by channel, five folds.** A channel's house style (its thumbnail
template, its title formula, its audience) predicts reach on its own. If the
same channel appears in training and validation, a model can score well by
recognising the channel rather than reading the title. `GroupKFold` on
`channel_id` makes every validation fold consist of channels the model has
never seen, which is the situation the app is in: the creator using it is
not in the training set.

Every number reported is **out-of-fold**: each video's prediction comes from
the fold in which it was held out. The final model shipped to the app is
refit on all rows, but its metrics are the out-of-fold ones.

## Metrics

| Metric | Why |
|---|---|
| Spearman ρ between predicted and true reach | The product ranks candidates ("is title B better than A?"); rank correlation is the metric that matches the use. This is the headline number. |
| R² | How much variance is explained. Expect it to be low on real data: most of a video's reach is the channel and the video itself, not the packaging. |
| MAE in reach units (log10 views per subscriber) | Interpretable: 0.3 means the typical prediction is off by a factor of two in views. |
| Per content type | A model that works for gaming and not for finance is a gaming model. Each category is reported separately. |

Two baselines are reported alongside: predict the global mean, and predict
the content-type mean (learned on the training folds). A model must beat the
second by a clear margin or it has learned nothing beyond the category.

## Calibration

The 0-100 score is a percentile of predictions. To check it is honest, videos
are bucketed by score (0-20, 20-40, ...) and the mean true within-category
percentile of each bucket is reported. A good model's buckets rise
monotonically; a perfect one puts the 60-80 bucket at a mean true percentile
near 70. Flat buckets mean the score spreads but does not discriminate.

## What counts as better

A change (new feature, new hyperparameters, more data) is kept when:

1. Out-of-fold Spearman improves on the whole table **and** does not fall on
   any content type by more than it rises elsewhere.
2. The calibration buckets stay monotonic.
3. The driver explanations still make sense on a handful of hand-picked
   titles and thumbnails (`python -m src.scoring.cli ... --json`). A model
   that scores well but attributes everything to "average word length" is
   not a product.

Record the before and after in the pull request from
`docs/model_report.md`.

## What the numbers do not tell you

- **Synthetic data.** With `data/processed/SYNTHETIC` present, every metric
  describes how well the pipeline recovers the generator's rules
  (`src/data/synthetic.py: reach_from_properties`). Spearman 0.5 there says
  the code works; it says nothing about YouTube.
- **Subscriber counts are current, not historical** (see the collection
  spec). Older videos on fast-growing channels look worse than they were.
- **The sample is the seed list.** English-language, mostly large channels,
  four categories. Scores for a Portuguese cooking channel are extrapolation.
- **Correlation, not causation.** Faces appear in thumbnails of videos that
  reach further; this does not prove adding a face causes reach. The product
  frames drivers as "what the model rewards", not "what will happen".
- **One snapshot of views.** Reach is measured once per video at 30-365 days.
  Videos that spike late are mismeasured.

## Running it

```bash
python -m src.modelling.train              # refit both models, prints the same metrics
python -m src.modelling.evaluate           # the protocol, with baselines and calibration
python -m src.modelling.evaluate --out docs/model_report.md
```
