"""Shared training, calibration and explanation code used by both models.

The title pair and the thumbnail pair each own a feature extractor. Everything
after features (fitting trees, grouped cross-validation, mapping predictions
to a 0-100 score, turning SHAP values into drivers) is identical, so it lives
here once.
"""
