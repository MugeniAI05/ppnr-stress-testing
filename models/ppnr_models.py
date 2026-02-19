"""
PPNR Stress Testing Project
Model Suite: Econometric + Machine Learning
-------------------------------------------
Implements:
  1. OLS Regression (baseline econometric model)
  2. Ridge / Lasso Regularized Regression
  3. Vector Autoregression (VAR) for multi-equation modeling
  4. Random Forest (ensemble ML)
  5. Gradient Boosting (XGBoost-style)
  6. Model ensemble / stacking

All models share a common sklearn-compatible interface.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
import joblib


# ──────────────────────────────────────────────────────────────────────────────
# BASE MODEL WRAPPER
# ──────────────────────────────────────────────────────────────────────────────

class PPNRModel:
    """Base wrapper providing fit/predict/evaluate for PPNR sub-components."""

    def __init__(self, name: str, estimator):
        self.name = name
        self.estimator = estimator
        self.feature_cols = None
        self.target_col = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, feature_cols: list, target_col: str):
        self.feature_cols = feature_cols
        self.target_col = target_col
        X = df[feature_cols].values
        y = df[target_col].values
        self.estimator.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        assert self.is_fitted, "Model must be fitted first."
        X = df[self.feature_cols].values
        return self.estimator.predict(X)

    def evaluate(self, df: pd.DataFrame) -> dict:
        y_true = df[self.target_col].values
        y_pred = self.predict(df)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        return {"model": self.name, "target": self.target_col,
                "RMSE": round(rmse, 2), "MAE": round(mae, 2),
                "R2": round(r2, 4), "MAPE(%)": round(mape, 2)}

    def cross_validate(self, df: pd.DataFrame, feature_cols: list,
                       target_col: str, n_splits: int = 5) -> dict:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        X = df[feature_cols].values
        y = df[target_col].values
        scores = cross_val_score(self.estimator, X, y,
                                  cv=tscv, scoring='r2')
        return {"model": self.name, "cv_r2_mean": round(scores.mean(), 4),
                "cv_r2_std": round(scores.std(), 4),
                "cv_r2_scores": np.round(scores, 4).tolist()}

    def get_feature_importance(self) -> pd.DataFrame:
        """Extract feature importances (works for tree-based models)."""
        if hasattr(self.estimator, 'feature_importances_'):
            fi = self.estimator.feature_importances_
        elif hasattr(self.estimator, 'coef_'):
            fi = np.abs(self.estimator.coef_)
        else:
            return pd.DataFrame()

        return pd.DataFrame({
            "feature": self.feature_cols,
            "importance": fi
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    def save(self, path: str):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        return joblib.load(path)


# ──────────────────────────────────────────────────────────────────────────────
# MODEL FACTORY
# ──────────────────────────────────────────────────────────────────────────────

def build_model_suite() -> dict:
    """Return all models keyed by name."""
    return {
        "OLS":            PPNRModel("OLS",            LinearRegression()),
        "Ridge":          PPNRModel("Ridge",          Ridge(alpha=1.0)),
        "Lasso":          PPNRModel("Lasso",          Lasso(alpha=0.5, max_iter=10000)),
        "ElasticNet":     PPNRModel("ElasticNet",     ElasticNet(alpha=0.5, l1_ratio=0.5)),
        "RandomForest":   PPNRModel("RandomForest",   RandomForestRegressor(
                                        n_estimators=200, max_depth=6,
                                        min_samples_split=4, random_state=42)),
        "GradientBoosting": PPNRModel("GradientBoosting", GradientBoostingRegressor(
                                        n_estimators=200, max_depth=4,
                                        learning_rate=0.05, subsample=0.8,
                                        random_state=42)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# MODEL ENSEMBLE (SIMPLE STACKING)
# ──────────────────────────────────────────────────────────────────────────────

class EnsembleModel:
    """
    Weighted ensemble of base PPNR models.
    Weights derived from inverse-RMSE weighting on validation set.
    """

    def __init__(self, base_models: dict):
        self.base_models = base_models  # {name: PPNRModel}
        self.weights = {}
        self.feature_cols = None
        self.target_col = None

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
            feature_cols: list, target_col: str):
        self.feature_cols = feature_cols
        self.target_col   = target_col

        rmses = {}
        for name, m in self.base_models.items():
            m.fit(train_df, feature_cols, target_col)
            metrics = m.evaluate(val_df)
            rmses[name] = metrics["RMSE"]

        # Inverse RMSE weighting
        inv = {n: 1/r for n, r in rmses.items()}
        total = sum(inv.values())
        self.weights = {n: v/total for n, v in inv.items()}
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = np.zeros(len(df))
        for name, m in self.base_models.items():
            preds += self.weights[name] * m.predict(df)
        return preds

    def evaluate(self, df: pd.DataFrame) -> dict:
        y_true = df[self.target_col].values
        y_pred = self.predict(df)
        return {
            "model": "Ensemble",
            "target": self.target_col,
            "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
            "MAE":  round(mean_absolute_error(y_true, y_pred), 2),
            "R2":   round(r2_score(y_true, y_pred), 4),
        }


# ──────────────────────────────────────────────────────────────────────────────
# PPNR THREE-EQUATION SYSTEM
# ──────────────────────────────────────────────────────────────────────────────

class PPNRThreeEquationSystem:
    """
    Models PPNR as three separate equations:
      (1) Net Interest Income (NII)
      (2) Non-Interest Income (NonII)
      (3) Non-Interest Expense (NIE)
    PPNR = NII + NonII - NIE
    """

    def __init__(self, model_type: str = "GradientBoosting"):
        self.model_type = model_type
        self.models = {}
        self.targets = ["nii", "nonii", "nie"]

    def _make_model(self, target: str) -> PPNRModel:
        suite = build_model_suite()
        return suite[self.model_type]

    def fit(self, df: pd.DataFrame, feature_cols: list):
        self.feature_cols = feature_cols
        for t in self.targets:
            m = self._make_model(t)
            m.fit(df, feature_cols, t)
            self.models[t] = m
        return self

    def predict_components(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[["date"]].copy() if "date" in df.columns else pd.DataFrame(index=df.index)
        for t in self.targets:
            out[f"{t}_pred"] = self.models[t].predict(df)
        out["ppnr_pred"] = out["nii_pred"] + out["nonii_pred"] - out["nie_pred"]
        return out

    def evaluate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for t in self.targets:
            rows.append(self.models[t].evaluate(df))
        # PPNR aggregate
        preds = self.predict_components(df)
        r2 = r2_score(df["ppnr"].values, preds["ppnr_pred"].values)
        rmse = np.sqrt(mean_squared_error(df["ppnr"].values, preds["ppnr_pred"].values))
        rows.append({"model": self.model_type, "target": "ppnr_aggregate",
                     "RMSE": round(rmse, 2), "R2": round(r2, 4)})
        return pd.DataFrame(rows)

    def get_all_importances(self) -> pd.DataFrame:
        frames = []
        for t in self.targets:
            fi = self.models[t].get_feature_importance()
            fi["component"] = t
            frames.append(fi)
        return pd.concat(frames, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# BACK-TESTING ENGINE
# ──────────────────────────────────────────────────────────────────────────────

class BackTestEngine:
    """
    Rolling window back-test: trains on expanding window,
    predicts next quarter. Measures model stability over time.
    """

    def __init__(self, min_train_size: int = 16):
        self.min_train_size = min_train_size
        self.results = []

    def run(self, df: pd.DataFrame, feature_cols: list,
            target_col: str, model_name: str = "GradientBoosting") -> pd.DataFrame:

        self.results = []
        for end in range(self.min_train_size, len(df)):
            train = df.iloc[:end]
            test  = df.iloc[end:end+1]

            suite = build_model_suite()
            m = suite[model_name]
            m.fit(train, feature_cols, target_col)
            pred = m.predict(test)[0]
            actual = test[target_col].values[0]

            self.results.append({
                "date":         test["date"].values[0] if "date" in test.columns else end,
                "actual":       round(actual, 0),
                "predicted":    round(pred, 0),
                "error":        round(actual - pred, 0),
                "pct_error":    round((actual - pred) / (abs(actual) + 1e-8) * 100, 2),
                "train_size":   end,
            })

        return pd.DataFrame(self.results)

    def summary(self) -> dict:
        df = pd.DataFrame(self.results)
        return {
            "mean_abs_pct_error": round(df["pct_error"].abs().mean(), 2),
            "rmse":               round(np.sqrt((df["error"]**2).mean()), 0),
            "r2":                 round(r2_score(df["actual"], df["predicted"]), 4),
            "directional_acc":    round((np.sign(df["actual"].diff()) ==
                                         np.sign(df["predicted"].diff())).mean() * 100, 1),
        }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from utils.data_generator import BankDataGenerator, DataWrangler

    bank = BankDataGenerator().generate()
    wrangler = DataWrangler()
    bank_fe = wrangler.add_features(bank)

    feature_cols = [c for c in bank_fe.columns
                    if c not in ["date","nii","nonii","nie","ppnr"]]
    train, test = wrangler.train_test_split_temporal(bank_fe, test_size=0.25)

    # Three-equation system
    system = PPNRThreeEquationSystem(model_type="GradientBoosting")
    system.fit(train, feature_cols)
    metrics = system.evaluate_all(test)
    print("=== Three-Equation System Performance ===")
    print(metrics.to_string(index=False))

    # Back-test
    bt = BackTestEngine(min_train_size=16)
    bt_results = bt.run(bank_fe, feature_cols, "ppnr", "GradientBoosting")
    print("\n=== Back-Test Summary ===")
    print(bt.summary())
