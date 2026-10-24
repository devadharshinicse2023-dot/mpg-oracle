"""Train 5 algorithms, compare them, save the best end-to-end pipeline."""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from features import CAT, NUM, add_features, clean

df = clean(pd.read_csv("auto-mpg.csv"))
df = df.dropna(subset=["mpg"])
X = df[["cylinders", "displacement", "horsepower", "weight", "acceleration",
        "model_year", "origin", "brand"]]
y = df["mpg"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

preprocess = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), NUM),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),
])

models = {
    "Ridge Regression": Ridge(alpha=1.0),
    "Support Vector Regressor": SVR(C=10, epsilon=0.2),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                   max_depth=3, random_state=42),
}

rows, fitted = [], {}
for name, m in models.items():
    pipe = Pipeline([("fe", FunctionTransformer(add_features)),
                     ("prep", preprocess), ("model", m)])
    cv = cross_val_score(pipe, X_tr, y_tr, cv=KFold(5, shuffle=True, random_state=42), scoring="r2")
    pipe.fit(X_tr, y_tr)
    p = pipe.predict(X_te)
    rows.append({"Model": name, "CV R2": cv.mean(), "Test R2": r2_score(y_te, p),
                 "RMSE": np.sqrt(mean_squared_error(y_te, p)),
                 "MAE": mean_absolute_error(y_te, p)})
    fitted[name] = pipe
    print(f"{name:26s} CV R2={cv.mean():.3f}  Test R2={rows[-1]['Test R2']:.3f}  RMSE={rows[-1]['RMSE']:.2f}")

results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
best_name = results.loc[0, "Model"]
print("\nBEST MODEL ->", best_name)

joblib.dump({"pipeline": fitted[best_name], "best_name": best_name, "results": results,
             "brands": sorted(df["brand"].dropna().unique())}, "mpg_model.joblib")
print("Saved mpg_model.joblib")
