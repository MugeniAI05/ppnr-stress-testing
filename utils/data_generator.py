"""
PPNR Stress Testing Project
Data Generation & Preprocessing Utilities
------------------------------------------
Generates synthetic macroeconomic and bank financial data
mimicking real-world DFAST/CCAR stress testing datasets.
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# 1. MACROECONOMIC SCENARIO GENERATION
# ──────────────────────────────────────────────────────────────────────────────

class MacroScenarioGenerator:
    """
    Generates correlated macroeconomic time series for:
    - Baseline, Adverse, and Severely Adverse scenarios
    (consistent with Federal Reserve DFAST/CCAR framework)
    """

    SCENARIOS = {
        "baseline":          {"gdp_shock": 0.0,  "unemp_shock": 0.0,  "hpi_shock": 0.0,  "rate_shock": 0.0 },
        "adverse":           {"gdp_shock": -2.5, "unemp_shock": 3.5,  "hpi_shock": -15.0,"rate_shock": -1.0},
        "severely_adverse":  {"gdp_shock": -6.0, "unemp_shock": 6.0,  "hpi_shock": -25.0,"rate_shock": -2.5},
    }

    def __init__(self, n_quarters: int = 13, start: str = "2020Q1"):
        self.n_quarters = n_quarters
        self.start = start
        self.dates = pd.period_range(start=start, periods=n_quarters, freq="Q")

    def _correlated_shocks(self, n: int) -> np.ndarray:
        """Return correlated macro variable innovations via Cholesky decomposition."""
        # Correlation matrix: [GDP_growth, Unemployment, HPI_growth, Fed_Funds, CRE_spread, VIX]
        corr = np.array([
            [ 1.00, -0.75,  0.65, -0.20, -0.50, -0.60],
            [-0.75,  1.00, -0.70,  0.10,  0.60,  0.55],
            [ 0.65, -0.70,  1.00, -0.15, -0.45, -0.50],
            [-0.20,  0.10, -0.15,  1.00,  0.25,  0.15],
            [-0.50,  0.60, -0.45,  0.25,  1.00,  0.70],
            [-0.60,  0.55, -0.50,  0.15,  0.70,  1.00],
        ])
        L = np.linalg.cholesky(corr)
        z = np.random.randn(n, 6)
        return (L @ z.T).T

    def generate(self, scenario: str) -> pd.DataFrame:
        shocks = self.SCENARIOS[scenario]
        innovations = self._correlated_shocks(self.n_quarters)

        gdp    = 2.0  + shocks["gdp_shock"]   /self.n_quarters + 0.8 * innovations[:, 0]
        unemp  = 4.5  + np.cumsum(np.abs(shocks["unemp_shock"])/self.n_quarters
                                  * np.ones(self.n_quarters)) + 0.5 * innovations[:, 1]
        hpi    = 3.0  + shocks["hpi_shock"]   /self.n_quarters + 1.2 * innovations[:, 2]
        rate   = 5.25 + shocks["rate_shock"]  /self.n_quarters + 0.3 * innovations[:, 3]
        cre    = 2.50 + (abs(shocks["gdp_shock"])/20)           + 0.4 * innovations[:, 4]
        vix    = 18   + abs(shocks["gdp_shock"]) * 2            + 2.0 * np.abs(innovations[:, 5])

        df = pd.DataFrame({
            "date":         self.dates.astype(str),
            "scenario":     scenario,
            "gdp_growth":   np.round(gdp, 2),
            "unemployment": np.round(np.clip(unemp, 3, 15), 2),
            "hpi_growth":   np.round(hpi, 2),
            "fed_funds_rate": np.round(np.clip(rate, 0, 8), 2),
            "cre_spread":   np.round(np.clip(cre, 0.5, 8), 2),
            "vix":          np.round(np.clip(vix, 10, 80), 1),
        })
        return df

    def generate_all(self) -> pd.DataFrame:
        frames = [self.generate(s) for s in self.SCENARIOS]
        return pd.concat(frames, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# 2. HISTORICAL BANK DATA GENERATION (Training Data)
# ──────────────────────────────────────────────────────────────────────────────

class BankDataGenerator:
    """
    Generates synthetic bank P&L data with realistic relationships
    to macroeconomic drivers (for model training/backtesting).
    """

    def __init__(self, n_quarters: int = 40, start: str = "2015Q1"):
        self.n_quarters = n_quarters
        self.dates = pd.period_range(start=start, periods=n_quarters, freq="Q")

    def generate(self) -> pd.DataFrame:
        t = np.arange(self.n_quarters)

        # Macro variables (historical)
        gdp       = 2.5 + 0.3*np.sin(2*np.pi*t/20) + np.random.normal(0, 0.5, self.n_quarters)
        unemp     = 5.0 - 0.05*t + 0.8*np.sin(2*np.pi*t/16) + np.random.normal(0, 0.3, self.n_quarters)
        hpi       = 4.0 + 0.1*t + 1.0*np.random.normal(0, 1, self.n_quarters)
        rate      = np.clip(0.25 + 0.15*t + 0.2*np.random.normal(0, 1, self.n_quarters), 0, 6)
        vix       = 18  + 5*np.sin(2*np.pi*t/12) + 3*np.abs(np.random.normal(0, 1, self.n_quarters))

        # PPNR components — modeled as functions of macro drivers
        # Net Interest Income (NII): rises with rates, hurt by flat/inverted curve
        nii = (8500 + 300*rate - 40*unemp + 20*gdp
               + np.random.normal(0, 150, self.n_quarters))

        # Non-Interest Income (NonII): fee income, market-sensitive
        nonii = (3200 - 50*vix + 80*gdp + 15*hpi
                 + np.random.normal(0, 120, self.n_quarters))

        # Non-Interest Expense (NIE): partially fixed, partially variable
        nie = (5500 + 30*unemp + 0.4*nii + 0.3*nonii
               + np.random.normal(0, 100, self.n_quarters))

        ppnr = nii + nonii - nie

        df = pd.DataFrame({
            "date":            self.dates.astype(str),
            "gdp_growth":      np.round(gdp, 2),
            "unemployment":    np.round(np.clip(unemp, 3, 10), 2),
            "hpi_growth":      np.round(hpi, 2),
            "fed_funds_rate":  np.round(rate, 2),
            "vix":             np.round(vix, 1),
            "nii":             np.round(nii, 0),
            "nonii":           np.round(nonii, 0),
            "nie":             np.round(nie, 0),
            "ppnr":            np.round(ppnr, 0),
        })
        return df


# ──────────────────────────────────────────────────────────────────────────────
# 3. DATA WRANGLING / PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

class DataWrangler:
    """
    Handles data cleaning, feature engineering, and transformation
    for model-ready datasets.
    """

    @staticmethod
    def add_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features, rolling stats, and interaction terms."""
        df = df.copy().reset_index(drop=True)

        macro_cols = ["gdp_growth", "unemployment", "hpi_growth", "fed_funds_rate", "vix"]
        for col in macro_cols:
            if col in df.columns:
                df[f"{col}_lag1"] = df[col].shift(1)
                df[f"{col}_lag2"] = df[col].shift(2)
                df[f"{col}_chg"]  = df[col].diff(1)
                df[f"{col}_ma4"]  = df[col].rolling(4).mean()

        # Interaction terms
        if "fed_funds_rate" in df.columns and "unemployment" in df.columns:
            df["rate_x_unemp"] = df["fed_funds_rate"] * df["unemployment"]

        if "gdp_growth" in df.columns and "hpi_growth" in df.columns:
            df["gdp_x_hpi"] = df["gdp_growth"] * df["hpi_growth"]

        return df.dropna()

    @staticmethod
    def winsorize(df: pd.DataFrame, cols: list, limits=(0.01, 0.01)) -> pd.DataFrame:
        """Winsorize outliers at specified percentile limits."""
        df = df.copy()
        for col in cols:
            df[col] = stats.mstats.winsorize(df[col], limits=limits)
        return df

    @staticmethod
    def standardize(df: pd.DataFrame, feature_cols: list) -> tuple:
        """Standardize features; return scaled df and parameters."""
        means = df[feature_cols].mean()
        stds  = df[feature_cols].std().replace(0, 1)
        df_scaled = df.copy()
        df_scaled[feature_cols] = (df[feature_cols] - means) / stds
        return df_scaled, means, stds

    @staticmethod
    def train_test_split_temporal(df: pd.DataFrame,
                                   test_size: float = 0.25) -> tuple:
        """Time-series aware train/test split (no data leakage)."""
        split_idx = int(len(df) * (1 - test_size))
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


if __name__ == "__main__":
    # Demo
    macro_gen = MacroScenarioGenerator()
    scenarios = macro_gen.generate_all()
    print("Macro Scenarios Shape:", scenarios.shape)
    print(scenarios.groupby("scenario")[["gdp_growth", "unemployment"]].mean())

    bank_gen = BankDataGenerator()
    hist = bank_gen.generate()
    wrangler = DataWrangler()
    hist_fe = wrangler.add_features(hist)
    print("\nHistorical Bank Data with Features Shape:", hist_fe.shape)
    print(hist_fe[["date", "ppnr", "nii", "nonii", "nie"]].tail())
