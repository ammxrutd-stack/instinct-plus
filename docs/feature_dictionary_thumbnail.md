# Thumbnail feature dictionary

Computed by `src/thumbnail/features.py: extract_thumbnail_features(path, content_type, subscriber_count=None)`
with OpenCV only. Feature version `thumb-v3`.

## Preprocessing

1. Decode; an unreadable or missing file yields all zeros and `image_missing = 1`.
2. **Strip letterbox bars**: rows at the top and bottom whose mean grey is
   below 18 are removed (hqdefault pads 16:9 uploads to 4:3 with black bars).
   `letterbox_fraction` records how much was removed. It is reported in the
   app but **not** given to the model: in training it means "the upload was
   16:9", while a creator's own 1280x720 upload never has bars.
3. Resize to 480 px wide. All measurements below are on this frame.

## Light

| Feature | Definition |
|---|---|
| `brightness_mean`, `brightness_std` | HSV value channel, 0-1 |
| `contrast` | standard deviation of grey ÷ 255 |
| `dark_fraction` | share of pixels with value < 0.2 |
| `bright_fraction` | share with value > 0.85 |
| `entropy` | Shannon entropy of a 64-bin grey histogram (tonal range) |
| `sharpness` | variance of the Laplacian ÷ 1000, capped at 10 |

## Colour

| Feature | Definition |
|---|---|
| `saturation_mean`, `saturation_std` | HSV saturation, 0-1 |
| `colorfulness` | Hasler & Süsstrunk (2003) metric on BGR |
| `red_share`, `orange_yellow_share`, `green_share`, `blue_share`, `purple_pink_share` | saturation-weighted share of hue in each band (OpenCV hue 0-180: red 170-10, orange/yellow 10-35, green 35-85, blue 85-135, purple/pink 135-170); pixels with value < 0.15 do not vote |
| `warm_share` | red + orange/yellow |
| `dominant_colours` | number of k-means (k=6) colour clusters covering at least 6% of the frame |

## Composition

| Feature | Definition |
|---|---|
| `edge_density` | share of Canny edge pixels (100/200) |
| `centre_edge_density` | the same in the centre third of both axes |
| `symmetry` | 1 − mean absolute difference between the left half and the mirrored right half ÷ 255 |
| `aspect_ratio` | width ÷ height of the picture after bars are removed (1.78 for 16:9) |

## Faces

Haar cascade `haarcascade_frontalface_default` on the equalised grey frame,
`scaleFactor 1.1`, `minNeighbors 5`, `minSize 28 px`. Frontal faces only;
profiles and stylised faces are missed. Replacing this with MediaPipe keeps
the feature names.

| Feature | Definition |
|---|---|
| `face_present` | at least one detection |
| `face_count` | number of detections |
| `largest_face_area` | largest detection's box area ÷ frame area |
| `face_centre_x`, `face_centre_y` | centre of the largest face, 0-1 (0.5 when none) |

## Text

A heuristic, not OCR: morphological gradient → Otsu threshold → close
horizontally (kernel width ≈ frame/40) → connected components. A component
is "text" when its height is 3.5-45% of the frame, its width at least 1.3×
its height and 8% of the frame, and at least 30% of its box is filled. It
catches overlaid captions, which is what thumbnails have.

| Feature | Definition |
|---|---|
| `text_area` | summed box area ÷ frame area |
| `text_block_count` | number of such components |

## Context

| Feature | Definition |
|---|---|
| `image_missing` | 1 when the image could not be read; training drops such rows, the app refuses them |
| `log_subscribers` | as for titles: context, removed from the score |
| `ct_*` | one-hot content type |

## Not included, and why

- CLIP or other deep embeddings (`torch`, `transformers`): large dependency,
  slow on CPU, and the embedding dimensions are not explainable as drivers.
  The right extension is a small number of named concept scores (e.g. CLIP
  similarity to "a surprised face", "food close-up") so each stays a
  nameable driver.
- OCR of the caption text: `text_area` captures the amount; the words
  themselves belong to the title model's vocabulary features if anything.
