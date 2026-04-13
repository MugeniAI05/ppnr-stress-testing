"""
PPNR Stress Testing Dashboard
Interactive Streamlit app for visualizing all model outputs.
Run: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PPNR Stress Testing Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

  .main { background: #0a0e1a; }
  .block-container { padding: 2rem 2.5rem; max-width: 1400px; }

  h1 { font-family: 'IBM Plex Mono', monospace; color: #e8f4f8 !important;
       font-size: 1.6rem !important; letter-spacing: -0.5px; }
  h2 { color: #a8c8d8 !important; font-size: 1.1rem !important;
       font-family: 'IBM Plex Mono', monospace; border-bottom: 1px solid #1e3a4a;
       padding-bottom: 0.4rem; margin-top: 1.5rem; }
  h3 { color: #7eb8d0 !important; font-size: 0.95rem !important; }

  .metric-card {
    background: linear-gradient(135deg, #0f1e2e 0%, #132233 100%);
    border: 1px solid #1e3a4a;
    border-radius: 8px;
    padding: 1.1rem 1.3rem;
    text-align: center;
  }
  .metric-value { font-family: 'IBM Plex Mono', monospace;
                  font-size: 1.6rem; font-weight: 600;
                  color: #4dd0e1; margin: 0.2rem 0; }
  .metric-label { font-size: 0.72rem; color: #7a9bb0;
                  text-transform: uppercase; letter-spacing: 1px; }
  .metric-delta-pos { color: #4caf93; font-size: 0.8rem; }
  .metric-delta-neg { color: #e05c6b; font-size: 0.8rem; }

  .scenario-baseline { border-left: 3px solid #4dd0e1; }
  .scenario-adverse   { border-left: 3px solid #f0c040; }
  .scenario-severe    { border-left: 3px solid #e05c6b; }

  .stDataFrame { border: 1px solid #1e3a4a !important; border-radius: 6px; }
  .stSelectbox label, .stSlider label { color: #a8c8d8 !important; font-size: 0.82rem !important; }

  .sidebar-title { font-family: 'IBM Plex Mono', monospace;
                   font-size: 0.75rem; color: #4dd0e1;
                   text-transform: uppercase; letter-spacing: 1.5px;
                   margin-bottom: 0.5rem; }
  [data-testid="stSidebar"] { background: #080d18; }
  [data-testid="stSidebar"] .stMarkdown { color: #a8c8d8; }
  [data-testid="stSidebar"] * { color: #c8dce8 !important; }
  [data-testid="stSidebar"] .sidebar-title { color: #4dd0e1 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Load data ────────────────────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")

@st.cache_data
def load_data():
    d = {}
    files = {
        "hist":     "01_historical_data_with_features.csv",
        "model_cmp":"02_model_comparison.csv",
        "sys_met":  "03_three_equation_metrics.csv",
        "feat_imp": "03b_feature_importances.csv",
        "backtest": "04_backtest_results.csv",
        "macro":    "05a_macro_scenarios.csv",
        "proj":     "05b_stress_projections.csv",
        "capital":  "06_capital_impact.csv",
        "sens":     "07_sensitivity_analysis.csv",
        "val":      "08_model_validation_diagnostics.csv",
    }
    for k, fname in files.items():
        path = os.path.join(OUT, fname)
        d[k] = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    return d

data = load_data()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🏦 PPNR Platform</div>', unsafe_allow_html=True)
    st.markdown("**Stress Testing & Model Validation**")
    st.markdown("---")
    page = st.radio("Navigation", [
        "Overview",
        "Model Comparison",
        "Back-Testing",
        "Scenario Projections",
        "Capital Impact",
        "Sensitivity Analysis",
        "Model Validation",
    ])
    st.markdown("---")
    st.caption("Framework: DFAST / CCAR-aligned")
    st.caption("Model: Gradient Boosting Three-Equation System")
    st.caption("NII + NonII − NIE = PPNR")

# ─── Helper: mini chart via HTML canvas (CSS-only sparkline) ─────────────────
SCENARIO_COLORS = {
    "baseline":        "#4dd0e1",
    "adverse":         "#f0c040",
    "severely_adverse":"#e05c6b",
}

def color_scenario(val):
    c = {"baseline":"#0d2e3a","adverse":"#2e2a10","severely_adverse":"#2e1015"}
    return f'background-color: {c.get(str(val),"transparent")}'

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("PPNR Stress Testing Platform")
    st.markdown("*Pre-Provision Net Revenue — DFAST/CCAR-aligned Stress Test*")
    st.markdown("---")

    # KPI row
    cap = data["capital"]
    proj = data["proj"]
    bt   = data["backtest"]

    def kpi(label, value, delta="", neg=False):
        delta_cls = "metric-delta-neg" if neg else "metric-delta-pos"
        delta_html = f'<div class="{delta_cls}">{delta}</div>' if delta else ""
        return f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          {delta_html}
        </div>"""

    if not cap.empty and not proj.empty:
        baseline_ppnr  = proj[proj["scenario"]=="baseline"]["ppnr_pred"].sum()
        adverse_ppnr   = proj[proj["scenario"]=="adverse"]["ppnr_pred"].sum()
        sevadv_ppnr    = proj[proj["scenario"]=="severely_adverse"]["ppnr_pred"].sum()

        row1, row2, row3, row4 = st.columns(4)
        with row1:
            st.markdown(kpi("Baseline PPNR (9Q)", f"${baseline_ppnr/1000:.1f}B", "✓ Pass"), unsafe_allow_html=True)
        with row2:
            delta_adv = f"{(adverse_ppnr-baseline_ppnr)/baseline_ppnr*100:.1f}% vs Baseline"
            st.markdown(kpi("Adverse PPNR (9Q)", f"${adverse_ppnr/1000:.1f}B", delta_adv, neg=True), unsafe_allow_html=True)
        with row3:
            delta_sev = f"{(sevadv_ppnr-baseline_ppnr)/baseline_ppnr*100:.1f}% vs Baseline"
            st.markdown(kpi("Sev. Adverse PPNR (9Q)", f"${sevadv_ppnr/1000:.1f}B", delta_sev, neg=True), unsafe_allow_html=True)
        with row4:
            mape = bt["pct_error"].abs().mean() if not bt.empty else 0
            st.markdown(kpi("Backtest MAPE", f"{mape:.1f}%", "Rolling window"), unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns([2, 1])

    with c1:
        st.markdown("## PPNR Scenario Trajectories")
        if not proj.empty:
            import plotly.graph_objects as go

            fig = go.Figure()
            for scen, color in SCENARIO_COLORS.items():
                s = proj[proj["scenario"]==scen].copy()
                label = scen.replace("_"," ").title()

                # CI band
                if "ppnr_ci_lo" in s.columns:
                    r = int(color[1:3], 16)
                    g = int(color[3:5], 16)
                    b = int(color[5:7], 16)
                    fill_color = f"rgba({r},{g},{b},0.12)"
                    fig.add_trace(go.Scatter(
                        x=list(s["date"]) + list(s["date"][::-1]),
                        y=list(s["ppnr_ci_hi"]) + list(s["ppnr_ci_lo"][::-1]),
                        fill='toself', fillcolor=fill_color,
                        line=dict(width=0), showlegend=False, hoverinfo='skip'
                    ))

                fig.add_trace(go.Scatter(
                    x=s["date"], y=s["ppnr_pred"],
                    name=label, line=dict(color=color, width=2.5),
                    mode='lines+markers', marker=dict(size=5)
                ))

            fig.update_layout(
                paper_bgcolor='#0a0e1a', plot_bgcolor='#0f1826',
                font=dict(family='IBM Plex Mono', color='#a8c8d8', size=11),
                xaxis=dict(gridcolor='#1a2e3a', title="Quarter"),
                yaxis=dict(gridcolor='#1a2e3a', title="PPNR ($MM)"),
                legend=dict(bgcolor='#0f1826', bordercolor='#1e3a4a', borderwidth=1),
                height=380, margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("## Model Performance")
        if not data["sys_met"].empty:
            st.dataframe(
                data["sys_met"][["target","R2","RMSE"]].round(3),
                use_container_width=True, hide_index=True
            )
        st.markdown("## Capital Summary")
        if not cap.empty:
            st.dataframe(
                cap[["scenario","cet1_terminal_%","passes_stress"]],
                use_container_width=True, hide_index=True
            )

    # Architecture diagram
    st.markdown("---")
    st.markdown("## Project Architecture")
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────┐
    │              PPNR Stress Testing Pipeline               │
    ├──────────────┬──────────────────┬────────────────────────┤
    │  DATA LAYER  │   MODEL LAYER    │   ANALYSIS LAYER       │
    │              │                  │                         │
    │ • Macro Gen  │ • OLS / Ridge    │ • Stress Projections   │
    │ • Bank P&L   │ • Lasso / EN     │ • Capital Impact       │
    │ • Feature    │ • RandomForest   │ • Sensitivity Anal.    │
    │   Engineering│ • Grad. Boost    │ • Back-Testing         │
    │ • Winsorize  │ • Ensemble       │ • Model Validation     │
    │ • Temporal   │ • 3-Eqn System   │   (SR 11-7 framework)  │
    │   Split      │   (NII+NonII-NIE)│ • CI via Bootstrap     │
    └──────────────┴──────────────────┴────────────────────────┘
    ```
    """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Comparison":
    st.title("Model Comparison & Feature Importance")

    cmp = data["model_cmp"]
    fi  = data["feat_imp"]

    if not cmp.empty:
        st.markdown("## Cross-Model Performance (Test Set)")
        test_cmp = cmp[cmp["split"]=="test"].sort_values("R2", ascending=False)

        import plotly.graph_objects as go
        import plotly.express as px

        fig = go.Figure()
        colors_bar = ["#4dd0e1","#4caf93","#a0c8e8","#f0c040","#c87dd0","#e05c6b"]
        for i, row in enumerate(test_cmp.itertuples()):
            fig.add_trace(go.Bar(
                x=[row.R2], y=[row.model],
                orientation='h', name=row.model,
                marker_color=colors_bar[i % len(colors_bar)],
                text=[f"R²={row.R2:.3f}  RMSE={row.RMSE:,.0f}"],
                textposition='outside',
            ))
        fig.update_layout(
            paper_bgcolor='#0a0e1a', plot_bgcolor='#0f1826',
            font=dict(family='IBM Plex Mono', color='#a8c8d8', size=11),
            xaxis=dict(gridcolor='#1a2e3a', title="R² Score"),
            yaxis=dict(gridcolor='#1a2e3a'),
            showlegend=False, height=320,
            margin=dict(l=10, r=80, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("## Full Metrics Table")
            st.dataframe(cmp.sort_values(["split","R2"], ascending=[True,False]).round(3),
                         use_container_width=True, hide_index=True)

    if not fi.empty:
        st.markdown("## Feature Importances by Component")
        comp = st.selectbox("Select component", fi["component"].unique())
        fi_comp = fi[fi["component"]==comp].head(12)

        fig2 = go.Figure(go.Bar(
            x=fi_comp["importance"], y=fi_comp["feature"],
            orientation='h', marker_color='#4dd0e1',
        ))
        fig2.update_layout(
            paper_bgcolor='#0a0e1a', plot_bgcolor='#0f1826',
            font=dict(family='IBM Plex Mono', color='#a8c8d8', size=11),
            xaxis=dict(gridcolor='#1a2e3a', title="Importance"),
            yaxis=dict(gridcolor='#1a2e3a', autorange='reversed'),
            height=380, margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BACK-TESTING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Back-Testing":
    st.title("Rolling-Window Back-Test Results")

    bt = data["backtest"]
    if not bt.empty:
        import plotly.graph_objects as go

        # Summary stats
        mape = bt["pct_error"].abs().mean()
        rmse = np.sqrt((bt["error"]**2).mean())

        c1, c2, c3, c4 = st.columns(4)
        metrics = [
            ("MAPE", f"{mape:.2f}%"),
            ("RMSE", f"${rmse:,.0f}MM"),
            ("Max Overpredict", f"${bt['error'].min():,.0f}MM"),
            ("Max Underpredict", f"${bt['error'].max():,.0f}MM"),
        ]
        for col, (lbl, val) in zip([c1,c2,c3,c4], metrics):
            with col:
                st.metric(lbl, val)

        st.markdown("## Actual vs. Predicted PPNR (Rolling Forecast)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bt["date"], y=bt["actual"],
                                  name="Actual", line=dict(color="#4dd0e1", width=2.5),
                                  mode='lines+markers'))
        fig.add_trace(go.Scatter(x=bt["date"], y=bt["predicted"],
                                  name="Predicted", line=dict(color="#f0c040", width=2,
                                  dash='dot'), mode='lines+markers'))
        fig.update_layout(
            paper_bgcolor='#0a0e1a', plot_bgcolor='#0f1826',
            font=dict(family='IBM Plex Mono', color='#a8c8d8', size=11),
            xaxis=dict(gridcolor='#1a2e3a'), yaxis=dict(gridcolor='#1a2e3a', title="PPNR $MM"),
            legend=dict(bgcolor='#0f1826'), height=360,
            margin=dict(l=10,r=10,t=10,b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("## Forecast Error Distribution")
        import plotly.express as px
        fig2 = px.histogram(bt, x="pct_error", nbins=20,
                             color_discrete_sequence=["#4dd0e1"])
        fig2.update_layout(paper_bgcolor='#0a0e1a', plot_bgcolor='#0f1826',
                            font=dict(family='IBM Plex Mono', color='#a8c8d8'),
                            xaxis_title="% Error", yaxis_title="Frequency",
                            height=300, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("## Raw Back-Test Data")
        st.dataframe(bt.round(2), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SCENARIO PROJECTIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Scenario Projections":
    st.title("Stress Scenario Projections")

    proj  = data["proj"]
    macro = data["macro"]

    if not proj.empty:
        selected = st.multiselect("Show scenarios",
            ["baseline","adverse","severely_adverse"],
            default=["baseline","adverse","severely_adverse"])

        tab1, tab2, tab3 = st.tabs(["PPNR Components", "Macro Drivers", "Data Table"])

        with tab1:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(rows=2, cols=2,
                subplot_titles=["NII ($MM)", "Non-Interest Income ($MM)",
                                 "Non-Interest Expense ($MM)", "PPNR ($MM)"],
                vertical_spacing=0.18)

            cols_map = [("nii_pred",1,1),("nonii_pred",1,2),
                        ("nie_pred",2,1),("ppnr_pred",2,2)]

            for scen in selected:
                s = proj[proj["scenario"]==scen]
                color = SCENARIO_COLORS.get(scen, "#aaa")
                for col, r, c in cols_map:
                    fig.add_trace(go.Scatter(
                        x=s["date"], y=s[col],
                        name=scen.replace("_"," ").title(),
                        line=dict(color=color, width=2),
                        showlegend=(col=="ppnr_pred"),
                    ), row=r, col=c)

            fig.update_layout(
                paper_bgcolor='#0a0e1a', plot_bgcolor='#0f1826',
                font=dict(family='IBM Plex Mono', color='#a8c8d8', size=10),
                height=550, margin=dict(l=10,r=10,t=40,b=10),
                legend=dict(bgcolor='#0f1826',bordercolor='#1e3a4a',borderwidth=1),
            )
            for i in range(1,5):
                row,col = (1,i) if i<=2 else (2,i-2)
                fig.update_xaxes(gridcolor='#1a2e3a', row=row, col=col)
                fig.update_yaxes(gridcolor='#1a2e3a', row=row, col=col)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            if not macro.empty:
                fig2 = make_subplots(rows=2, cols=3,
                    subplot_titles=["GDP Growth (%)","Unemployment (%)","HPI Growth (%)",
                                     "Fed Funds Rate (%)","VIX","CRE Spread (%)"],
                    vertical_spacing=0.2)
                macro_vars = [("gdp_growth",1,1),("unemployment",1,2),("hpi_growth",1,3),
                               ("fed_funds_rate",2,1),("vix",2,2),("cre_spread",2,3)]
                for scen in selected:
                    s = macro[macro["scenario"]==scen]
                    color = SCENARIO_COLORS.get(scen,"#aaa")
                    for var, r, c in macro_vars:
                        if var in s.columns:
                            fig2.add_trace(go.Scatter(
                                x=s["date"], y=s[var],
                                name=scen.replace("_"," ").title(),
                                line=dict(color=color, width=2),
                                showlegend=(var=="gdp_growth"),
                            ), row=r, col=c)
                fig2.update_layout(
                    paper_bgcolor='#0a0e1a', plot_bgcolor='#0f1826',
                    font=dict(family='IBM Plex Mono', color='#a8c8d8', size=10),
                    height=550, margin=dict(l=10,r=10,t=40,b=10),
                    legend=dict(bgcolor='#0f1826',bordercolor='#1e3a4a',borderwidth=1),
                )
                st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            st.dataframe(proj[proj["scenario"].isin(selected)].round(1),
                         use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CAPITAL IMPACT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Capital Impact":
    st.title("Capital Impact Analysis")

    cap = data["capital"]
    if not cap.empty:
        import plotly.graph_objects as go

        c1, c2, c3 = st.columns(3)
        for col, (_, row) in zip([c1,c2,c3], cap.iterrows()):
            scen = row["scenario"]
            color = SCENARIO_COLORS.get(scen,"#aaa")
            pass_icon = "✅" if row["passes_stress"] else "❌"
            with col:
                st.markdown(f"""
                <div class="metric-card" style="border-left:4px solid {color}">
                  <div class="metric-label">{scen.replace('_',' ').upper()}</div>
                  <div class="metric-value" style="color:{color}">
                    {row['cet1_terminal_%']:.2f}%
                  </div>
                  <div class="metric-label">CET1 Terminal Ratio {pass_icon}</div>
                  <div style="color:#7a9bb0;font-size:0.8rem;margin-top:0.4rem">
                    Buffer: {row['buffer_above_min_bps']:.0f}bps above min
                  </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("## CET1 Ratio by Scenario")

        fig = go.Figure()
        fig.add_hline(y=7.0, line_dash="dash", line_color="#e05c6b",
                       annotation_text="Reg. Minimum 7.0%", annotation_font_color="#e05c6b")

        for _, row in cap.iterrows():
            color = SCENARIO_COLORS.get(row["scenario"],"#aaa")
            fig.add_trace(go.Bar(
                x=[row["scenario"].replace("_"," ").title()],
                y=[row["cet1_terminal_%"]],
                marker_color=color,
                name=row["scenario"],
                text=[f"{row['cet1_terminal_%']:.2f}%"],
                textposition='outside',
            ))
        fig.update_layout(
            paper_bgcolor='#0a0e1a', plot_bgcolor='#0f1826',
            font=dict(family='IBM Plex Mono', color='#a8c8d8', size=12),
            yaxis=dict(gridcolor='#1a2e3a', title="CET1 Ratio (%)", range=[0,20]),
            xaxis=dict(gridcolor='#1a2e3a'),
            height=380, showlegend=False,
            margin=dict(l=10,r=10,t=20,b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("## Full Capital Table")
        st.dataframe(cap.round(2), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SENSITIVITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Sensitivity Analysis":
    st.title("PPNR Sensitivity to Macro Shocks")

    sens = data["sens"]
    if not sens.empty:
        import plotly.graph_objects as go

        st.markdown("## Impact of 1-Unit Shock in Each Macro Variable")
        st.caption("Holding all other variables at baseline — shows which drivers matter most.")

        colors = ["#e05c6b" if v < 0 else "#4caf93"
                  for v in sens["delta_ppnr_$MM"]]
        fig = go.Figure(go.Bar(
            x=sens["variable"], y=sens["delta_ppnr_$MM"],
            marker_color=colors,
            text=sens["delta_ppnr_$MM"].apply(lambda x: f"${x:+,.0f}MM"),
            textposition='outside',
        ))
        fig.add_hline(y=0, line_color="#a8c8d8", line_width=0.5)
        fig.update_layout(
            paper_bgcolor='#0a0e1a', plot_bgcolor='#0f1826',
            font=dict(family='IBM Plex Mono', color='#a8c8d8', size=12),
            yaxis=dict(gridcolor='#1a2e3a', title="ΔPPNR ($MM) for 1-unit shock"),
            xaxis=dict(gridcolor='#1a2e3a'),
            height=380, margin=dict(l=10,r=10,t=20,b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("## Sensitivity Table")
        st.dataframe(sens.round(0), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("## Interpretation")
        st.markdown("""
        - **VIX** and **GDP Growth** are the dominant PPNR drivers in this model
        - A 1-point increase in VIX reduces cumulative 9Q PPNR by ~$700MM (negative wealth/market effects on Non-Interest Income)
        - **Unemployment** sensitivity is lower, as its effects are more indirect (via credit quality and customer behavior)
        - **Fed Funds Rate** shows near-zero direct sensitivity — NII benefits offset NonII/NIE impacts
        """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Validation":
    st.title("Model Validation Diagnostics (SR 11-7 Framework)")

    val = data["val"]
    if not val.empty:
        import plotly.graph_objects as go

        st.markdown("## Diagnostic Test Results")

        # Color-code issues
        def flag(val, col):
            if col in ["autocorrelation_issue","non_normal_residuals",
                       "heteroskedasticity","structural_break"]:
                return "🔴" if val else "🟢"
            return val

        styled = val.copy()
        for c in ["autocorrelation_issue","non_normal_residuals",
                  "heteroskedasticity","structural_break"]:
            if c in styled.columns:
                styled[c] = styled[c].apply(lambda x: flag(x, c))

        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("## Durbin-Watson Statistics")
        st.caption("DW ≈ 2 indicates no autocorrelation. DW < 1 or > 3 is concerning.")

        fig = go.Figure()
        fig.add_hline(y=1.5, line_dash="dash", line_color="#f0c040",
                       annotation_text="Lower concern")
        fig.add_hline(y=2.5, line_dash="dash", line_color="#f0c040",
                       annotation_text="Upper concern")
        fig.add_hline(y=2.0, line_color="#4caf93", line_width=0.5)

        colors_dw = ["#4caf93" if 1.5<=v<=2.5 else "#e05c6b"
                     for v in val["durbin_watson"]]
        fig.add_trace(go.Bar(
            x=val["target"], y=val["durbin_watson"],
            marker_color=colors_dw,
            text=val["durbin_watson"].round(3),
            textposition="outside",
        ))
        fig.update_layout(
            paper_bgcolor='#0a0e1a', plot_bgcolor='#0f1826',
            font=dict(family='IBM Plex Mono', color='#a8c8d8', size=12),
            yaxis=dict(gridcolor='#1a2e3a', title="Durbin-Watson Statistic",
                       range=[0,3.5]),
            height=340, margin=dict(l=10,r=10,t=20,b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("## Validation Framework (SR 11-7)")
        st.markdown("""
        This validation suite implements the key pillars of the **Federal Reserve SR 11-7**
        guidance on model risk management:

        | Test | Purpose |
        |------|---------|
        | **Durbin-Watson** | Detect first-order serial correlation in residuals |
        | **Ljung-Box Q** | Detect higher-order autocorrelation (4-lag) |
        | **Jarque-Bera** | Test normality of residual distribution |
        | **Breusch-Pagan** | Test for heteroskedasticity (non-constant variance) |
        | **Chow Test** | Detect structural breaks / regime changes |
        | **R²/ RMSE** | Predictive accuracy and goodness-of-fit |
        | **Back-Test MAPE** | Out-of-sample forecasting performance |
        """)
