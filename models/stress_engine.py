"""
PPNR Stress Testing Project
Scenario Analysis & Stress Testing Engine
------------------------------------------
Applies macro scenarios (Baseline / Adverse / Severely Adverse)
to project PPNR across a 9-quarter stress horizon.

Outputs:
  - Scenario projections (PPNR components)
  - Capital impact analysis
  - Sensitivity / elasticity analysis
  - Confidence intervals via bootstrap
  - Model risk quantification
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────────────────────────────────────
# STRESS PROJECTION ENGINE
# ──────────────────────────────────────────────────────────────────────────────

class StressProjectionEngine:
    """
    Given a fitted PPNRThreeEquationSystem and macro scenario data,
    projects NII, NonII, NIE, and PPNR over the stress horizon.
    Also computes confidence intervals via parametric bootstrap.
    """

    def __init__(self, system, n_bootstrap: int = 500):
        self.system = system       # PPNRThreeEquationSystem
        self.n_bootstrap = n_bootstrap

    def _prepare_scenario(self, scenario_df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged/derived features to scenario data."""
        # We use the last quarter of training as the seed for lags
        df = scenario_df.copy()
        feature_cols = self.system.feature_cols

        # Fill lag columns with scenario path approximations
        for col in ["gdp_growth", "unemployment", "hpi_growth",
                    "fed_funds_rate", "vix"]:
            if col in df.columns:
                df[f"{col}_lag1"] = df[col].shift(1).fillna(df[col].mean())
                df[f"{col}_lag2"] = df[col].shift(2).fillna(df[col].mean())
                df[f"{col}_chg"]  = df[col].diff(1).fillna(0)
                df[f"{col}_ma4"]  = df[col].rolling(4, min_periods=1).mean()

        if "fed_funds_rate" in df.columns and "unemployment" in df.columns:
            df["rate_x_unemp"] = df["fed_funds_rate"] * df["unemployment"]
        if "gdp_growth" in df.columns and "hpi_growth" in df.columns:
            df["gdp_x_hpi"] = df["gdp_growth"] * df["hpi_growth"]

        # Keep only needed columns
        avail = [c for c in feature_cols if c in df.columns]
        return df[avail + ["date", "scenario"] if "scenario" in df.columns else avail + ["date"]]

    def project(self, scenario_df: pd.DataFrame) -> pd.DataFrame:
        """Point estimate projections for a single scenario."""
        df = self._prepare_scenario(scenario_df)
        return self.system.predict_components(df)

    def project_with_ci(self, scenario_df: pd.DataFrame,
                         residuals_df: pd.DataFrame = None,
                         alpha: float = 0.10) -> pd.DataFrame:
        """
        Bootstrap confidence intervals for PPNR projections.
        If residuals_df provided, resamples training residuals.
        """
        point = self.project(scenario_df)
        n = len(point)

        # Estimate residual std from training (or use 3% of NII as proxy)
        if residuals_df is not None:
            nii_resid_std   = residuals_df["nii_resid"].std()
            nonii_resid_std = residuals_df["nonii_resid"].std()
            nie_resid_std   = residuals_df["nie_resid"].std()
        else:
            nii_resid_std   = abs(point["nii_pred"].mean()) * 0.04
            nonii_resid_std = abs(point["nonii_pred"].mean()) * 0.05
            nie_resid_std   = abs(point["nie_pred"].mean()) * 0.03

        boot_ppnr = []
        for _ in range(self.n_bootstrap):
            nii_boot   = point["nii_pred"].values   + np.random.normal(0, nii_resid_std, n)
            nonii_boot = point["nonii_pred"].values  + np.random.normal(0, nonii_resid_std, n)
            nie_boot   = point["nie_pred"].values    + np.random.normal(0, nie_resid_std, n)
            boot_ppnr.append(nii_boot + nonii_boot - nie_boot)

        boot_array = np.array(boot_ppnr)  # (n_bootstrap, n_quarters)
        lo = np.percentile(boot_array, alpha/2 * 100, axis=0)
        hi = np.percentile(boot_array, (1 - alpha/2) * 100, axis=0)

        point["ppnr_ci_lo"] = np.round(lo, 0)
        point["ppnr_ci_hi"] = np.round(hi, 0)
        point["ci_width"]   = np.round(hi - lo, 0)

        return point


# ──────────────────────────────────────────────────────────────────────────────
# CAPITAL IMPACT ANALYZER
# ──────────────────────────────────────────────────────────────────────────────

class CapitalImpactAnalyzer:
    """
    Translates PPNR projections into capital adequacy metrics:
    - Pre-tax income contribution
    - CET1 ratio sensitivity
    - Buffer vs. regulatory minimums
    """

    def __init__(self, rwa: float = 350_000, cet1_ratio_start: float = 12.5,
                 tax_rate: float = 0.21, reg_minimum: float = 7.0):
        self.rwa = rwa                         # Risk-Weighted Assets ($MM)
        self.cet1_ratio_start = cet1_ratio_start  # Starting CET1 ratio (%)
        self.tax_rate = tax_rate
        self.reg_minimum = reg_minimum          # Regulatory minimum CET1 (%)

    def analyze(self, projections: dict) -> pd.DataFrame:
        """
        projections: {scenario_name: DataFrame with ppnr_pred column}
        Returns capital impact summary by scenario.
        """
        rows = []
        for scenario, df in projections.items():
            cumulative_ppnr = df["ppnr_pred"].sum()
            after_tax_income = cumulative_ppnr * (1 - self.tax_rate)

            # CET1 impact: simplified, assuming PPNR accretes capital
            cet1_change_bps = (after_tax_income / self.rwa) * 100 * 100  # in bps
            cet1_terminal = self.cet1_ratio_start + cet1_change_bps / 100

            rows.append({
                "scenario": scenario,
                "cumulative_ppnr_$MM": round(cumulative_ppnr, 0),
                "after_tax_income_$MM": round(after_tax_income, 0),
                "cet1_change_bps": round(cet1_change_bps, 1),
                "cet1_terminal_%": round(cet1_terminal, 2),
                "buffer_above_min_bps": round((cet1_terminal - self.reg_minimum) * 100, 1),
                "passes_stress": cet1_terminal >= self.reg_minimum,
            })

        return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# SENSITIVITY / ELASTICITY ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

class SensitivityAnalyzer:
    """
    Measures PPNR sensitivity to individual macro variable shocks.
    Uses partial derivatives (finite differences) around the baseline.
    """

    def __init__(self, system, baseline_scenario: pd.DataFrame):
        self.system = system
        self.baseline = baseline_scenario.copy()

    def _shock_variable(self, var: str, delta: float) -> pd.DataFrame:
        """Return scenario with var shocked by delta."""
        shocked = self.baseline.copy()
        if var in shocked.columns:
            shocked[var] += delta
        return shocked

    def _project(self, scenario_df: pd.DataFrame) -> float:
        """Get total stress-period PPNR for a scenario."""
        engine = StressProjectionEngine(self.system)
        proj = engine.project(scenario_df)
        return proj["ppnr_pred"].sum()

    def run(self, shock_size: float = 1.0) -> pd.DataFrame:
        """
        Compute PPNR elasticity w.r.t. 1-unit shock in each macro variable.
        """
        baseline_ppnr = self._project(self.baseline)

        variables = ["gdp_growth", "unemployment", "hpi_growth",
                     "fed_funds_rate", "vix"]
        rows = []
        for var in variables:
            shocked_ppnr = self._project(self._shock_variable(var, shock_size))
            delta_ppnr = shocked_ppnr - baseline_ppnr
            rows.append({
                "variable":         var,
                "shock_size":       shock_size,
                "ppnr_baseline_$MM": round(baseline_ppnr, 0),
                "ppnr_shocked_$MM":  round(shocked_ppnr, 0),
                "delta_ppnr_$MM":    round(delta_ppnr, 0),
                "direction":         "positive" if delta_ppnr > 0 else "negative",
            })

        return pd.DataFrame(rows).sort_values("delta_ppnr_$MM")


# ──────────────────────────────────────────────────────────────────────────────
# MODEL VALIDATION METRICS
# ──────────────────────────────────────────────────────────────────────────────

class ModelValidationSuite:
    """
    OCC/Fed Model Risk Management (SR 11-7) inspired validation tests:
    1. Goodness-of-fit tests
    2. Residual diagnostics (normality, autocorrelation, heteroskedasticity)
    3. Stability tests (Chow test for structural break)
    4. Predictive accuracy assessment
    """

    @staticmethod
    def durbin_watson(residuals: np.ndarray) -> float:
        """DW statistic for first-order autocorrelation in residuals."""
        diff = np.diff(residuals)
        return round(np.dot(diff, diff) / np.dot(residuals, residuals), 3)

    @staticmethod
    def ljung_box_q(residuals: np.ndarray, lags: int = 4) -> dict:
        """Ljung-Box Q test for residual autocorrelation."""
        n = len(residuals)
        r_k = [pd.Series(residuals).autocorr(lag=k) for k in range(1, lags+1)]
        Q = n * (n + 2) * sum(r**2 / (n - k) for k, r in enumerate(r_k, 1))
        p_val = 1 - stats.chi2.cdf(Q, df=lags)
        return {"Q_stat": round(Q, 3), "p_value": round(p_val, 4),
                "reject_H0_no_autocorr": p_val < 0.05}

    @staticmethod
    def jarque_bera(residuals: np.ndarray) -> dict:
        """Jarque-Bera normality test on residuals."""
        jb_stat, p_val = stats.jarque_bera(residuals)
        return {"JB_stat": round(jb_stat, 3), "p_value": round(p_val, 4),
                "reject_normality": p_val < 0.05}

    @staticmethod
    def breusch_pagan_approx(X: np.ndarray, residuals: np.ndarray) -> dict:
        """Approximate Breusch-Pagan test for heteroskedasticity."""
        from sklearn.linear_model import LinearRegression
        sq_resid = residuals ** 2
        lr = LinearRegression().fit(X, sq_resid)
        fitted = lr.predict(X)
        ss_res = np.sum((sq_resid - fitted) ** 2)
        ss_tot = np.sum((sq_resid - sq_resid.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        bp_stat = len(residuals) * r2
        p_val = 1 - stats.chi2.cdf(bp_stat, df=X.shape[1])
        return {"BP_stat": round(bp_stat, 3), "p_value": round(p_val, 4),
                "reject_homoskedasticity": p_val < 0.05}

    @staticmethod
    def chow_test(df: pd.DataFrame, break_idx: int,
                  feature_cols: list, target_col: str) -> dict:
        """
        Simplified Chow test for structural break at break_idx.
        Tests if model parameters are stable across two sub-periods.
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error

        df1 = df.iloc[:break_idx]
        df2 = df.iloc[break_idx:]

        def sse(d):
            lr = LinearRegression().fit(d[feature_cols], d[target_col])
            pred = lr.predict(d[feature_cols])
            return np.sum((d[target_col].values - pred) ** 2), len(d)

        sse_full, n_full = sse(df)
        sse1, n1 = sse(df1)
        sse2, n2 = sse(df2)

        k = len(feature_cols) + 1
        chow_f = ((sse_full - sse1 - sse2) / k) / ((sse1 + sse2) / (n_full - 2*k))
        p_val  = 1 - stats.f.cdf(chow_f, k, n_full - 2*k)

        return {"F_stat": round(chow_f, 3), "p_value": round(p_val, 4),
                "structural_break_detected": p_val < 0.05,
                "break_quarter": df.iloc[break_idx]["date"] if "date" in df.columns else break_idx}

    def full_diagnostic(self, df: pd.DataFrame, model,
                         feature_cols: list, target_col: str) -> dict:
        """Run all diagnostic tests and return comprehensive report."""
        from sklearn.metrics import r2_score, mean_squared_error
        y = df[target_col].values
        y_hat = model.predict(df)
        resid = y - y_hat
        X = df[feature_cols].values

        report = {
            "model":         model.name,
            "target":        target_col,
            "n_obs":         len(df),
            "r2":            round(r2_score(y, y_hat), 4),
            "rmse":          round(np.sqrt(mean_squared_error(y, y_hat)), 2),
            "durbin_watson": self.durbin_watson(resid),
            "ljung_box":     self.ljung_box_q(resid, lags=4),
            "jarque_bera":   self.jarque_bera(resid),
            "breusch_pagan": self.breusch_pagan_approx(X, resid),
            "chow_test":     self.chow_test(df, len(df)//2, feature_cols, target_col),
        }
        return report


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from utils.data_generator import BankDataGenerator, DataWrangler, MacroScenarioGenerator
    from models.ppnr_models import PPNRThreeEquationSystem

    # Setup
    bank = BankDataGenerator().generate()
    wrangler = DataWrangler()
    bank_fe = wrangler.add_features(bank)
    feature_cols = [c for c in bank_fe.columns if c not in ["date","nii","nonii","nie","ppnr"]]
    train, test = wrangler.train_test_split_temporal(bank_fe, test_size=0.25)

    system = PPNRThreeEquationSystem("GradientBoosting")
    system.fit(train, feature_cols)

    # Generate scenarios
    macro_gen = MacroScenarioGenerator(n_quarters=9, start="2024Q1")
    all_scenarios = macro_gen.generate_all()

    # Project each scenario
    projections = {}
    engine = StressProjectionEngine(system)
    for scen in ["baseline", "adverse", "severely_adverse"]:
        scen_df = all_scenarios[all_scenarios["scenario"] == scen].copy()
        proj = engine.project_with_ci(scen_df)
        proj["scenario"] = scen
        projections[scen] = proj
        print(f"\n=== {scen.upper()} ===")
        print(proj[["date", "ppnr_pred", "ppnr_ci_lo", "ppnr_ci_hi"]].to_string(index=False))

    # Capital impact
    cap = CapitalImpactAnalyzer()
    cap_results = cap.analyze(projections)
    print("\n=== Capital Impact ===")
    print(cap_results.to_string(index=False))

    # Sensitivity
    baseline_df = all_scenarios[all_scenarios["scenario"] == "baseline"].copy()
    sens = SensitivityAnalyzer(system, baseline_df)
    sens_results = sens.run()
    print("\n=== Sensitivity Analysis ===")
    print(sens_results.to_string(index=False))
