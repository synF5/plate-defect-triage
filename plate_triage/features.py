"""Extra features made from the raw plate measurements."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# A400 is just 1 - A300, and Y_Maximum is almost the same as Y_Minimum
REDUNDANT = ["TypeOfSteel_A400", "Y_Maximum"]


class PlateGeometry(BaseEstimator, TransformerMixin):
    # derive=False only drops the redundant columns.
    # Kept as a parameter so grid search can check if the new features help.

    def __init__(self, derive: bool = True):
        self.derive = derive

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.drop(columns=REDUNDANT)
        if not self.derive:
            return out

        pixels = X["Pixels_Areas"].clip(lower=1)
        span_x = (X["X_Maximum"] - X["X_Minimum"]).clip(lower=1)
        span_y = (X["Y_Maximum"] - X["Y_Minimum"]).clip(lower=1)

        return out.assign(
            span_x=span_x,
            span_y=span_y,
            # scratches are long and thin, blobs are closer to square
            log_elongation=np.log10(span_y / span_x),
            edge_per_pixel=(X["X_Perimeter"] + X["Y_Perimeter"]) / pixels,
            mean_brightness=X["Sum_of_Luminosity"] / pixels,
            brightness_range=X["Maximum_of_Luminosity"] - X["Minimum_of_Luminosity"],
            conveyor_position=X["X_Minimum"] / X["Length_of_Conveyer"],
        )
