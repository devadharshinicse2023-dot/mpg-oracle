"""Shared cleaning + feature engineering (used by training AND the app)."""
import numpy as np
import pandas as pd

BRAND_FIX = {"chevy": "chevrolet", "chevroelt": "chevrolet", "vw": "volkswagen",
             "vokswagen": "volkswagen", "toyouta": "toyota", "maxda": "mazda",
             "mercedes-benz": "mercedes", "capri": "ford"}
ORIGIN_MAP = {1: "USA", 2: "Europe", 3: "Japan"}
NUM = ["cylinders", "displacement", "horsepower", "weight", "acceleration",
       "model_year", "power_to_weight", "disp_per_cyl", "car_age"]
CAT = ["origin", "brand"]


def clean(df):
    """Normalise column names, turn '?' into NaN, extract car brand."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    df = df.replace("?", np.nan)
    for c in ["mpg", "cylinders", "displacement", "horsepower", "weight",
              "acceleration", "model_year", "origin"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "car_name" in df:
        b = df["car_name"].astype(str).str.lower().str.split().str[0]
        df["brand"] = b.replace(BRAND_FIX)
    return df


def add_features(df):
    """Creative engineered features -> first step of the sklearn pipeline."""
    d = df.copy()
    d["power_to_weight"] = d["horsepower"] / d["weight"] * 1000
    d["disp_per_cyl"] = d["displacement"] / d["cylinders"]
    d["car_age"] = 82 - d["model_year"]
    d["origin"] = d["origin"].map(ORIGIN_MAP).fillna("USA")
    d["brand"] = d["brand"].fillna("unknown")
    return d[NUM + CAT]
