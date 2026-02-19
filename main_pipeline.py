"""
PPNR Stress Testing Project
Main Pipeline Orchestrator
----------------------------
End-to-end pipeline:
  1. Data generation & feature engineering
  2. Model training (all 6 models + ensemble)
  3. Back-testing
  4. Stress scenario projections (Baseline/Adverse/Severely Adverse)
  5. Confidence intervals
  6. Capital impact analysis
  7. Sensitivity analysis
  8. Model validation diagnostics
  9. Export all results to CSV

Run: python main_pipeline.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from utils.data_generator import BankDataGenerator, DataWrangler, MacroScenarioGenerator
from models.ppnr_models   import (PPNRThreeEquationSystem, BackTestEngine,
                                   build_model_suite)
from models.stress_engine  import (StressProjectionEngine, CapitalImpactAnalyzer,
                                    SensitivityAnalyzer, ModelValidationSuite)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def banner(text: str):
    width = 60
    print("\n" + "═"*width)
    print(f"  {text}")
    print("═"*width)


def save(df: pd.DataFrame, name: str):
    path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"  ✓ Saved → {path}")
    return df


# ══════════════════════════════════════════════════════════════════
# STEP 1: DATA GENERATION
# ══════════════════════════════════════════════════════════════════
banner("STEP 1: Generating Historical Bank Data")

bank_gen  = BankDataGenerator(n_quarters=40, start="2015Q1")
raw_hist  = bank_gen.generate()
wrangler  = DataWrangler()
hist_fe   = wrangler.add_features(raw_hist)
hist_fe   = wrangler.winsorize(hist_fe, ["ppnr","nii","nonii","nie"])

feature_cols = [c for c in hist_fe.columns
                if c not in ["date","nii","nonii","nie","ppnr"]]

train_df, test_df = wrangler.train_test_split_temporal(hist_fe, test_size=0.25)
print(f"  Train: {len(train_df)} quarters | Test: {len(test_df)} quarters")
print(f"  Features: {len(feature_cols)}")
save(hist_fe, "01_historical_data_with_features")


# ══════════════════════════════════════════════════════════════════
# STEP 2: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════
banner("STEP 2: Training & Comparing All Models on PPNR")

model_suite = build_model_suite()
comparison_rows = []

for name, model in model_suite.items():
    model.fit(train_df, feature_cols, "ppnr")
    train_metrics = model.evaluate(train_df)
    test_metrics  = model.evaluate(test_df)
    cv_results    = model.cross_validate(hist_fe, feature_cols, "ppnr")

    train_metrics["split"] = "train"
    test_metrics["split"]  = "test"
    comparison_rows.extend([train_metrics, test_metrics])

    print(f"  [{name:20s}] Test R²={test_metrics['R2']:.4f}  "
          f"RMSE={test_metrics['RMSE']:,.0f}  "
          f"CV-R²={cv_results['cv_r2_mean']:.4f}±{cv_results['cv_r2_std']:.4f}")

model_comparison = pd.DataFrame(comparison_rows)
save(model_comparison, "02_model_comparison")


# ══════════════════════════════════════════════════════════════════
# STEP 3: THREE-EQUATION SYSTEM (Production Model)
# ══════════════════════════════════════════════════════════════════
banner("STEP 3: Three-Equation PPNR System (NII + NonII - NIE)")

system = PPNRThreeEquationSystem(model_type="GradientBoosting")
system.fit(train_df, feature_cols)

system_metrics = system.evaluate_all(test_df)
print(system_metrics.to_string(index=False))
save(system_metrics, "03_three_equation_metrics")

# Feature importances
fi_all = system.get_all_importances()
save(fi_all, "03b_feature_importances")
print("\n  Top features driving NII:")
print(fi_all[fi_all["component"]=="nii"].head(5)[["feature","importance"]].to_string(index=False))


# ══════════════════════════════════════════════════════════════════
# STEP 4: BACK-TESTING
# ══════════════════════════════════════════════════════════════════
banner("STEP 4: Rolling-Window Back-Test")

bt_engine = BackTestEngine(min_train_size=16)
bt_results = bt_engine.run(hist_fe, feature_cols, "ppnr", "GradientBoosting")
bt_summary = bt_engine.summary()

print(f"  MAPE:              {bt_summary['mean_abs_pct_error']:.2f}%")
print(f"  RMSE:              {bt_summary['rmse']:,.0f}")
print(f"  R²:                {bt_summary['r2']:.4f}")
print(f"  Directional Acc.:  {bt_summary['directional_acc']:.1f}%")

save(bt_results, "04_backtest_results")
pd.DataFrame([bt_summary]).to_csv(f"{OUTPUT_DIR}/04_backtest_summary.csv", index=False)


# ══════════════════════════════════════════════════════════════════
# STEP 5: STRESS SCENARIO PROJECTIONS
# ══════════════════════════════════════════════════════════════════
banner("STEP 5: Stress Scenario Projections (9-Quarter Horizon)")

macro_gen = MacroScenarioGenerator(n_quarters=9, start="2024Q1")
all_macro  = macro_gen.generate_all()
save(all_macro, "05a_macro_scenarios")

projection_engine = StressProjectionEngine(system, n_bootstrap=1000)
all_projections   = {}
all_proj_frames   = []

for scen in ["baseline", "adverse", "severely_adverse"]:
    scen_macro = all_macro[all_macro["scenario"] == scen].copy()
    proj = projection_engine.project_with_ci(scen_macro, alpha=0.10)
    proj["scenario"] = scen
    all_projections[scen] = proj
    all_proj_frames.append(proj)

    cum_ppnr = proj["ppnr_pred"].sum()
    print(f"  {scen:20s}  Cumulative PPNR: ${cum_ppnr:>10,.0f}MM  "
          f"  Avg CI width: ${proj['ci_width'].mean():>8,.0f}MM")

projections_df = pd.concat(all_proj_frames, ignore_index=True)
save(projections_df, "05b_stress_projections")


# ══════════════════════════════════════════════════════════════════
# STEP 6: CAPITAL IMPACT ANALYSIS
# ══════════════════════════════════════════════════════════════════
banner("STEP 6: Capital Impact Analysis")

cap_analyzer = CapitalImpactAnalyzer(
    rwa=350_000, cet1_ratio_start=12.5, tax_rate=0.21, reg_minimum=7.0
)
cap_results = cap_analyzer.analyze(all_projections)
print(cap_results.to_string(index=False))
save(cap_results, "06_capital_impact")


# ══════════════════════════════════════════════════════════════════
# STEP 7: SENSITIVITY ANALYSIS
# ══════════════════════════════════════════════════════════════════
banner("STEP 7: PPNR Sensitivity to Macro Shocks")

baseline_macro = all_macro[all_macro["scenario"] == "baseline"].copy()
sens_analyzer  = SensitivityAnalyzer(system, baseline_macro)
sens_results   = sens_analyzer.run(shock_size=1.0)

print(sens_results[["variable","delta_ppnr_$MM","direction"]].to_string(index=False))
save(sens_results, "07_sensitivity_analysis")


# ══════════════════════════════════════════════════════════════════
# STEP 8: MODEL VALIDATION DIAGNOSTICS (SR 11-7 Framework)
# ══════════════════════════════════════════════════════════════════
banner("STEP 8: Model Validation Diagnostics (SR 11-7)")

validation_suite = ModelValidationSuite()
val_reports = []

best_model = model_suite["GradientBoosting"]
best_model.fit(train_df, feature_cols, "ppnr")

for component in ["ppnr", "nii", "nonii", "nie"]:
    # Fit a fresh model on this component
    m = build_model_suite()["GradientBoosting"]
    m.feature_cols = feature_cols
    m.target_col = component
    m.fit(train_df, feature_cols, component)

    diag = validation_suite.full_diagnostic(test_df, m, feature_cols, component)

    print(f"\n  [{component.upper():6s}]  R²={diag['r2']:.4f}  "
          f"DW={diag['durbin_watson']:.3f}  "
          f"JB p={diag['jarque_bera']['p_value']:.3f}  "
          f"BreakDetected={diag['chow_test']['structural_break_detected']}")

    val_reports.append({
        "target":                  component,
        "r2":                      diag["r2"],
        "rmse":                    diag["rmse"],
        "durbin_watson":           diag["durbin_watson"],
        "ljung_box_p":             diag["ljung_box"]["p_value"],
        "autocorrelation_issue":   diag["ljung_box"]["reject_H0_no_autocorr"],
        "jarque_bera_p":           diag["jarque_bera"]["p_value"],
        "non_normal_residuals":    diag["jarque_bera"]["reject_normality"],
        "breusch_pagan_p":         diag["breusch_pagan"]["p_value"],
        "heteroskedasticity":      diag["breusch_pagan"]["reject_homoskedasticity"],
        "structural_break":        diag["chow_test"]["structural_break_detected"],
        "structural_break_qtr":    diag["chow_test"]["break_quarter"],
    })

val_df = pd.DataFrame(val_reports)
save(val_df, "08_model_validation_diagnostics")


# ══════════════════════════════════════════════════════════════════
# STEP 9: EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════
banner("STEP 9: Executive Summary")

summary = {
    "Project":               "PPNR Stress Testing & Model Validation",
    "Framework":             "DFAST / CCAR-aligned",
    "Model":                 "GradientBoosting Three-Equation System",
    "Training Quarters":     len(train_df),
    "Test Quarters":         len(test_df),
    "Feature Count":         len(feature_cols),
    "Test R²":               system_metrics[system_metrics["target"]=="ppnr_aggregate"]["R2"].values[0],
    "Backtest MAPE (%)":     bt_summary["mean_abs_pct_error"],
    "Backtest Directional":  bt_summary["directional_acc"],
    "Baseline CET1 (%)":     cap_results[cap_results["scenario"]=="baseline"]["cet1_terminal_%"].values[0],
    "Adverse CET1 (%)":      cap_results[cap_results["scenario"]=="adverse"]["cet1_terminal_%"].values[0],
    "Sev. Adverse CET1 (%)": cap_results[cap_results["scenario"]=="severely_adverse"]["cet1_terminal_%"].values[0],
    "All Pass Stress?":      cap_results["passes_stress"].all(),
}

for k, v in summary.items():
    print(f"  {k:<28} {v}")

pd.DataFrame([summary]).to_csv(f"{OUTPUT_DIR}/09_executive_summary.csv", index=False)

banner("✅ Pipeline Complete — All outputs saved to /outputs/")
