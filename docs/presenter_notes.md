# Instinct+ presenter notes

One paragraph per slide: what is on it, what to say, and the one line that
carries the slide. Numbers to have ready are at the end.

## 1. Cover: Instinct+

Instinct+ predicts how far a YouTube video will reach from its title and
thumbnail, before it is published. Two models (one for the title, one for the
thumbnail), one combined score, and a reason for every point. Say that the
talk ends with a live demo on Project Volta's real uploads.

## 2. The problem: every upload is a blind bet

A viewer sees exactly two things before choosing: the title and the
thumbnail. The video itself never gets a vote. Creators pick both on instinct,
publish, and wait. Analytics arrive after the first 48 hours, which is when
YouTube decides how widely to push a video, so by the time you learn the
packaging failed you cannot fix it. The example is real: an 85-minute Project
Volta match got 71 views, and nothing today can say whether that was the match,
the title or the thumbnail. Line: "We built the tool that answers before you
press publish."

## 3. The solution: score it before you publish

Three things the app gives you. One: a 0 to 100 score for the title, for the
thumbnail and for both together. A 70 means the model rates it better than 70%
of videos in your category. Two: drivers. Each feature's contribution in
points, positive or negative, turned into plain advice ("add a number", "cut
the text to three words"). Three: better titles. Claude writes alternatives
that address the weaknesses, the same model scores them, and they come back
ranked; one click swaps one in. Line: "A score, a reason, and a fix."

## 4. How a score is made: five steps

Walk the boxes left to right.

- Data: public videos from the YouTube Data API, discovered through a curated
  list of seed channels in four categories (gaming, cooking, finance,
  education). Seed channels rather than search, because search ranking is
  itself driven by engagement and would bias the sample toward the thing we
  predict. Videos 30 to 365 days old so views have settled but age does not
  dominate. No Shorts, which are a different surface.
- Target: reach = log of views divided by subscribers, with an age adjustment
  removed. Views per subscriber because a million views is routine at 30M
  subscribers and remarkable at 100K. That is what packaging actually moves.
- Features: 41 title features and 28 image features, every one named and
  defined, every one something a creator can change. No embeddings, on
  purpose.
- Model: XGBoost gradient boosted trees, one model per input, cross-validated
  with whole channels held out.
- Output: the prediction mapped to a percentile within the category, and SHAP
  values converted to points and sentences.

Footer line to say out loud: channel size is given to the model as context but
held out of the score, so a 150-subscriber channel and a 5-million one compare
the same title on equal terms.

## 5. Validation: tested on unseen channels

The headline metric is Spearman rank correlation between predicted and true
reach, because the product ranks candidates. Title model 0.57, thumbnail model
0.55, against 0.12 for a baseline that just predicts the category average. Say
how it was measured: five-fold cross-validation grouped by channel, so every
video is scored by a model that never saw its channel. Why that matters: a
channel's house style predicts reach on its own, and if the same channel were
in training and validation the model would learn to recognise the channel, not
read the title. That is leakage, and it would inflate every number. The 3,360
videos come from 56 channels. Calibration was checked too: bucket the scores
into fifths and the true percentile rises from 30 to 71, so the 0-100 scale is
honest. Then the caveat, said plainly: this is synthetic data. It shows the
pipeline recovers rules we wrote down; the real YouTube numbers come with the
first collection run. Line: "Honest numbers on channels the model never saw."

## 6. What the model sees: features a creator can change

Four families. Faces: present, how many, how big, where (OpenCV Haar cascade).
Text on the image: area covered and number of blocks (a gradient-based
heuristic, not OCR). Light and colour: brightness, contrast, saturation, warm
share, how many distinct colours. Title signals: length, numbers, questions,
capitals, power words, curiosity hooks, sentiment. Everything is measured in
milliseconds with no deep model. The reason for this design: a driver has to
be a sentence a creator can act on. "Dimension 212 of an embedding" is not
advice; "add a face, cut the text to three words" is. Line: "Explainability is
the product."

## 7. A real channel: tested on Project Volta

Using only public pages, no API key, the tool pulled and scored 43 long-form
videos and streams from the channel, and set aside 48 Shorts. Shorts are out of
scope by design: they are served by an autoplay feed, so title and thumbnail
rarely decide the view, and they were excluded from training for the same
reason. The Claude example on the slide: the template title "Regenerate
Football vs 4Ever Relentless | SUPER LEAGUE FULL MATCH" became "Every Minute:
Regenerate Football vs 4Ever Relentless, Super League", scored 3.5 points
higher by the model. What the model keeps saying about the channel: the match
thumbnails are strong, the template titles are interchangeable, and the
streams with a number and an event in the title ("$10,000 tournament, WEEK 2")
score highest. If asked for agreement with real views: a modest +0.23 rank
correlation on the long-form set, which is what a synthetic-trained model
should give. The point is the workflow that real data plugs into.

## 8. The process: six weeks, three pairs

Three pairs (title, thumbnail, app) working on one contract: a shared schema
and a ScoreResult interface that need all-pair agreement to change; everything
else is owned by a pair. Week 1: scope and that contract. Week 2: collection
design, with the decisions on slide 4 written down and the seed list verified
against the API. Week 3: first pipeline, a placeholder scorer and contract
tests so the app pair could start. Week 4: feature dictionaries and the
synthetic dataset. Week 5: models, grouped cross-validation, SHAP drivers and
the validation protocol. Week 6: the Streamlit app, Claude suggestions, A/B
and batch modes, and the Project Volta check. The synthetic set is the thing to
explain: 3,360 drawn thumbnails and template titles produced by a rule we
wrote down, which let three pairs build and test for weeks before real data,
and is flagged on every artefact it touched.

## 9. What we learned: three lessons

Leakage hides in the channel: grouped cross-validation was the single most
important decision; without it the numbers look great and mean nothing.
Explainability is the product: named features cost some accuracy and bought
every sentence of advice the app gives. Synthetic data proves the pipeline,
not the claim: it unblocked everyone, and it had to be labelled everywhere so
nobody mistakes a demo score for a prediction. Optional anecdote: the first
synthetic run gave a title correlation of 0.10 because channel noise swamped
the title signal; adding channel size as context and holding it out of the
score fixed the model and the fairness problem at once.

## 10. What's next: real data, then the real test

Next week: run the collector with an API key, about 32,000 videos for roughly
1,300 quota units, retrain, re-run the validation protocol and publish the
model card. Then: sharper features (MediaPipe faces, a few named semantic
title axes, more categories) under the same feature names and the same
contract. Now: the live demo on Project Volta: a template title, a strong
stream, and what Claude suggests instead.

## Numbers to have ready

| | |
|---|---|
| Title / thumbnail / baseline Spearman | 0.57 / 0.55 / 0.12 |
| Training set | 3,360 videos, 56 channels, 4 categories (synthetic) |
| Features | 41 title, 28 image |
| Calibration | true percentile 30 → 71 across score fifths |
| Collection window | 30 to 365 days, ≥10,000 subscribers, no Shorts |
| Full collection cost | ~32,000 videos, ~1,300 API units, one day's quota |
| Project Volta | 43 long-form (incl. 13 streams), 48 Shorts, long-form ρ +0.23 vs real views |
| Claude example | +3.5 points |

## Likely questions

- Does it work on real YouTube? Not proven yet. The synthetic numbers prove the
  pipeline recovers known rules; the protocol is written and runs the same way
  on collected data.
- Why not a deep model? A driver has to be a sentence. Accuracy was traded for
  advice on purpose.
- How do you stop it learning the channel? Grouped cross-validation, and
  channel size as context that is removed from the score.
- Isn't this correlation, not causation? Yes. The app says "what the model
  rewards", never "what will happen".
- Why exclude Shorts? Different surface, autoplay-driven, different view
  counting. Mixing them puts two distributions under one target.
