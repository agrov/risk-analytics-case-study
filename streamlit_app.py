"""
Risk Analytics Case Study — Streamlit companion.

Deployable to Streamlit Community Cloud for free. Pages:

  1. Overview
  2. The four insights
  3. Anomaly detection (Isolation Forest)
  4. Assisted Review — live LangGraph + Gemini demo
  5. How it works (architecture)

Add a GEMINI_API_KEY in Streamlit Cloud secrets (or set the environment
variable locally) to enable the live LLM nodes in the assisted-review demo.
Without a key the workflow still runs end-to-end using deterministic stubs.
"""

import os
import json
import time
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="HCP/HCO Risk Analytics — case study",
    page_icon="◐",
    layout="wide",
)

BASE = Path(__file__).parent

# ============================================================
# Style
# ============================================================
NAVY        = "#1E2A4A"
AMBER       = "#D97706"
AMBER_SOFT  = "#F4B860"
SLATE       = "#6B7C93"
CREAM       = "#F8F5F0"
GREEN       = "#5C8A3A"
RED_SOFT    = "#B85042"

st.markdown(f"""
<style>
  .main {{background-color: {CREAM};}}
  .stApp header {{background-color: transparent;}}
  h1, h2, h3, h4 {{color: {NAVY}; font-family: Georgia, serif;}}
  .stat-big {{font-size: 36px; font-weight: bold; color: {NAVY}; font-family: Georgia, serif;}}
  .stat-label {{font-size: 13px; color: {AMBER}; font-weight: bold; text-transform: uppercase; letter-spacing: 1px;}}
  .stat-sub {{font-size: 12px; color: {SLATE}; font-style: italic;}}
  .pill-green {{background: {GREEN}; color: white; padding: 2px 10px; border-radius: 10px; font-size: 11px; font-weight: bold;}}
  .pill-red   {{background: {RED_SOFT}; color: white; padding: 2px 10px; border-radius: 10px; font-size: 11px; font-weight: bold;}}
  .pill-amber {{background: {AMBER}; color: white; padding: 2px 10px; border-radius: 10px; font-size: 11px; font-weight: bold;}}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Data layer — cached
# ============================================================
@st.cache_data
def load_data():
    inter_raw = pd.read_csv(BASE / "interactions.csv")
    spend_raw = pd.read_csv(BASE / "spend.csv")
    stake     = pd.read_csv(BASE / "stakeholders.csv")
    emp       = pd.read_csv(BASE / "employees.csv")

    inter = (inter_raw
             .assign(_n=inter_raw.notna().sum(axis=1))
             .sort_values("_n", ascending=False)
             .drop_duplicates("interaction_id", keep="first")
             .drop(columns="_n"))
    spend = spend_raw.drop_duplicates("spend_id", keep="first")
    inter["interaction_date"] = pd.to_datetime(inter["interaction_date"], errors="coerce")
    spend["payment_date"]     = pd.to_datetime(spend["payment_date"],     errors="coerce")
    return inter, spend, stake, emp


@st.cache_resource
def fit_anomaly_model():
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    inter, spend, stake, emp = load_data()
    spend_agg = (spend.groupby("interaction_id")
                 .agg(total_usd=("amount_usd","sum"),
                      n_spend_rows=("spend_id","count"),
                      max_single_usd=("amount_usd","max"),
                      n_categories=("spend_category","nunique"),
                      honoraria_usd=("amount_usd",
                                     lambda s: s[spend.loc[s.index,"spend_category"]=="Honoraria"].sum()),
                      min_payment=("payment_date","min"),
                 ).reset_index())
    m = inter.merge(spend_agg, on="interaction_id", how="left")
    m = m.merge(emp.add_prefix("emp_").rename(columns={"emp_employee_id":"employee_id"}),
                on="employee_id", how="left")
    m = m.merge(stake.add_prefix("stk_").rename(columns={"stk_stakeholder_id":"stakeholder_id"}),
                on="stakeholder_id", how="left")
    for c in ["total_usd","n_spend_rows","max_single_usd","n_categories","honoraria_usd"]:
        m[c] = m[c].fillna(0)
    m["min_lag_days"]     = (m["min_payment"] - m["interaction_date"]).dt.days.fillna(0)
    m["spend_per_minute"] = np.where(m["duration_minutes"]>0, m["total_usd"]/m["duration_minutes"], 0)
    m["any_predated"]     = (m["min_lag_days"] < 0).astype(int)
    m["duration_minutes_imp"] = m["duration_minutes"].fillna(m["duration_minutes"].median())
    features = ["duration_minutes_imp","total_usd","max_single_usd","n_spend_rows","n_categories",
                "honoraria_usd","spend_per_minute","min_lag_days","any_predated"]
    Xs = StandardScaler().fit_transform(m[features].fillna(0))
    iso = IsolationForest(n_estimators=300, contamination=0.02, random_state=42, n_jobs=-1)
    iso.fit(Xs)
    m["anomaly_score"]  = -iso.score_samples(Xs)
    m["anomaly_pctile"] = pd.Series(m["anomaly_score"]).rank(pct=True) * 100
    return m


# ============================================================
# API key handling
# ============================================================
def get_api_key():
    """Pull Gemini key from Streamlit secrets or environment."""
    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        pass
    if not key:
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return key


# ============================================================
# Sidebar — navigation
# ============================================================
st.sidebar.markdown(f"<h2 style='color:{NAVY};margin:0;'>Risk Analytics</h2>", unsafe_allow_html=True)
st.sidebar.markdown(f"<p style='color:{SLATE};font-style:italic;font-size:12px;margin-top:0;'>Case study companion</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")
page = st.sidebar.radio("", [
    "Overview",
    "Four key insights",
    "Anomaly detection",
    "Assisted review (live)",
    "How it works",
])
st.sidebar.markdown("---")
api_key = get_api_key()
if api_key:
    st.sidebar.markdown(f"<span class='pill-green'>LIVE</span> &nbsp; Gemini key detected", unsafe_allow_html=True)
else:
    st.sidebar.markdown(f"<span class='pill-amber'>STUB</span> &nbsp; no API key — deterministic stubs", unsafe_allow_html=True)
    st.sidebar.caption("Set GEMINI_API_KEY in Streamlit secrets to enable live LLM nodes. Free key available at aistudio.google.com.")

st.sidebar.markdown("---")
st.sidebar.markdown(f"<p style='color:{SLATE};font-size:11px;'>Anjali Grover · Risk Analytics & Monitoring</p>", unsafe_allow_html=True)


# ============================================================
# PAGE: Overview
# ============================================================
def page_overview():
    inter, spend, stake, emp = load_data()
    total_usd = spend["amount_usd"].sum()

    st.markdown(f"<p style='color:{AMBER};font-weight:bold;letter-spacing:3px;'>RISK ANALYTICS & MONITORING</p>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='margin-top:0;'>Identifying Emerging Risk Patterns in HCP / HCO Engagement</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{SLATE};font-style:italic;'>Exploratory analysis prepared for Compliance · Interactive companion to the case-study deck.</p>", unsafe_allow_html=True)
    st.markdown("---")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    for col, big, label, sub in [
        (c1, "$3.4M", "Spend",        f"${total_usd:,.0f} on the deduped basis"),
        (c2, "7K",    "Interactions", f"{len(inter):,} after dedup"),
        (c3, "10",    "Countries",    "33 spellings normalised to 10"),
        (c4, "18",    "Months",       "Jan 2024 → Jun 2025"),
        (c5, f"{len(stake):,}", "Stakeholders", "HCP + HCO"),
        (c6, f"{len(emp):,}",   "Employees",    "Sales, MSL, Medical Manager, etc."),
    ]:
        with col:
            st.markdown(f"<div class='stat-big'>{big}</div>"
                        f"<div class='stat-label'>{label}</div>"
                        f"<div class='stat-sub'>{sub}</div>", unsafe_allow_html=True)

    st.markdown(" ")
    st.markdown(" ")

    st.subheader("What you'll find in this companion")
    st.markdown(f"""
    - **Four key insights** — interactive views of the patterns flagged for Compliance review.
    - **Anomaly detection** — what an unsupervised Isolation Forest surfaces beyond the rule-based view.
    - **Assisted review (live)** — try the Phase 3 LangGraph + Gemini prototype on any interaction.
    - **How it works** — architecture, deterministic policy, and the audit trail.
    """)


# ============================================================
# PAGE: Four insights
# ============================================================
def page_insights():
    inter, spend, stake, emp = load_data()
    inter["interaction_date"] = pd.to_datetime(inter["interaction_date"], errors="coerce")
    spend["payment_date"]     = pd.to_datetime(spend["payment_date"],     errors="coerce")

    st.markdown(f"<p style='color:{AMBER};font-weight:bold;letter-spacing:3px;'>SECTION 03 — KEY INSIGHTS</p>", unsafe_allow_html=True)
    st.markdown("# Four risk signals to triage")
    st.markdown(f"<p style='color:{SLATE};font-style:italic;'>Each one is a question to ask, not a conclusion to act on.</p>", unsafe_allow_html=True)

    tabs = st.tabs(["1 · Pre-dated payments", "2 · Honoraria concentration", "3 · Short meetings", "4 · Risk-tier calibration"])

    # ---- Tab 1 ----
    with tabs[0]:
        m = spend.merge(inter[["interaction_id","interaction_date"]], on="interaction_id", how="left")
        m["lag"] = (m["payment_date"] - m["interaction_date"]).dt.days
        valid = m.dropna(subset=["lag"])
        pre = valid[valid["lag"] < 0]
        st.markdown(f"### {len(pre)} of {len(valid):,} payments are dated **before** the linked interaction (≈ 9.9%)")
        st.markdown(f"<p style='color:{SLATE};'>Median lag is otherwise a healthy +19 days. The 272 spend rows excluded had null dates or no matching interaction.</p>", unsafe_allow_html=True)

        bins   = [-np.inf, -1, 7, 14, 21, 28, 35, 44]
        labels = ["≤ −1 (pre-dated)", "0–7", "8–14", "15–21", "22–28", "29–35", "36–44"]
        valid = valid.assign(bucket=pd.cut(valid["lag"], bins=bins, labels=labels))
        counts = valid["bucket"].value_counts().reindex(labels).reset_index()
        counts.columns = ["Lag (days)", "Spend rows"]
        st.bar_chart(counts.set_index("Lag (days)"), height=300, color=AMBER)

        with st.expander("Drill into the pre-dated cohort"):
            cat = pre["spend_category"].value_counts().rename_axis("Category").reset_index(name="Records")
            st.dataframe(cat, hide_index=True, use_container_width=True)
            st.markdown(f"**Priority triage cohort**: honoraria pre-dated and over $1,000 → **{len(pre[(pre['spend_category']=='Honoraria') & (pre['amount_usd']>1000)])} records**.")

    # ---- Tab 2 ----
    with tabs[1]:
        cat_totals = spend.groupby("spend_category")["amount_usd"].sum().sort_values(ascending=False)
        hono = spend[spend["spend_category"]=="Honoraria"]
        st.markdown(f"### Honoraria is **{cat_totals['Honoraria']/cat_totals.sum()*100:.0f}%** of total spend; one record is **3.7%** of the total")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.bar_chart(cat_totals, height=320, color=AMBER, x_label="Spend category", y_label="USD")
        with col2:
            big = hono.nlargest(1, "amount_usd").iloc[0]
            iid = big["interaction_id"]
            inter_match = inter[inter["interaction_id"]==iid].iloc[0]
            st.markdown(f"<div style='background:{NAVY};color:white;padding:18px;border-radius:6px;'>"
                        f"<div style='color:{AMBER_SOFT};font-size:11px;font-weight:bold;letter-spacing:2px;'>THE SINGLE LARGEST RECORD</div>"
                        f"<div style='font-size:36px;font-weight:bold;font-family:Georgia,serif;margin:8px 0;'>${big['amount_usd']:,.0f}</div>"
                        f"<div style='color:{AMBER_SOFT};font-size:13px;'>Honoraria · {big['currency']} {big['amount_local']:,.0f}</div>"
                        f"<div style='margin-top:14px;font-size:12px;'>"
                        f"  Interaction {iid}<br/>"
                        f"  Employee {inter_match['employee_id']}<br/>"
                        f"  Stakeholder {inter_match['stakeholder_id']}<br/>"
                        f"  120-min Speaker Programme<br/>"
                        f"  3.7% of dataset's total spend"
                        f"</div></div>", unsafe_allow_html=True)

    # ---- Tab 3 ----
    with tabs[2]:
        peri = spend.groupby("interaction_id", as_index=False)["amount_usd"].sum().rename(columns={"amount_usd":"total_usd"})
        peri = peri.merge(inter[["interaction_id","duration_minutes","country","stakeholder_id"]], on="interaction_id", how="inner")
        peri = peri.merge(stake[["stakeholder_id","risk_tier"]], on="stakeholder_id", how="left")
        cohort = peri[(peri["duration_minutes"]<=10) & (peri["total_usd"]>200)]
        st.markdown(f"### **{len(cohort)}** interactions of 10 minutes or less carry over $200 in spend")
        c1, c2, c3 = st.columns(3)
        c1.metric("Cohort size", len(cohort))
        c2.metric("Total USD",  f"${cohort['total_usd'].sum():,.0f}")
        c3.metric("Avg per interaction", f"${cohort['total_usd'].mean():,.0f}")
        st.caption("Five times the median typical meeting; twice the mean. The full cohort table is below — sorted by spend.")
        st.dataframe(
            cohort.sort_values("total_usd", ascending=False)
                  .rename(columns={"interaction_id":"Interaction","country":"Country","duration_minutes":"Duration (min)","total_usd":"Spend (USD)","risk_tier":"Risk tier"})
                  [["Interaction","Country","Duration (min)","Spend (USD)","Risk tier"]],
            hide_index=True, use_container_width=True
        )

    # ---- Tab 4 ----
    with tabs[3]:
        peri = spend.groupby("interaction_id", as_index=False)["amount_usd"].sum().rename(columns={"amount_usd":"total_usd"})
        m = inter.merge(stake[["stakeholder_id","risk_tier"]], on="stakeholder_id", how="left").merge(peri, on="interaction_id", how="left")
        m["total_usd"] = m["total_usd"].fillna(0)
        m["risk_tier"] = m["risk_tier"].fillna("Missing")
        agg = m.groupby("risk_tier").agg(n=("interaction_id","count"), avg_usd=("total_usd","mean"), median_usd=("total_usd","median")).round(0)
        st.markdown(f"### The risk-tier field does not predict where spend goes")
        st.markdown(f"<p style='color:{SLATE};'>Average spend per interaction is essentially the same across Low, Medium and High tiers.</p>", unsafe_allow_html=True)
        c1, c2 = st.columns([2,1])
        with c1:
            st.bar_chart(agg["avg_usd"], height=320, color=NAVY, x_label="Risk tier", y_label="Average USD per interaction")
        with c2:
            # Compute 43 / 8 / ~0
            stk_total = m.groupby("stakeholder_id")["total_usd"].sum().reset_index().merge(stake[["stakeholder_id","risk_tier"]], on="stakeholder_id", how="left")
            q95 = stk_total["total_usd"].quantile(0.95)
            n_low_top5 = len(stk_total[(stk_total["risk_tier"]=="Low") & (stk_total["total_usd"]>=q95)])
            n_high_zero = len(stk_total[(stk_total["risk_tier"]=="High") & (stk_total["total_usd"]==0)])
            st.metric("Low-tier stakeholders in top-5% spend", n_low_top5)
            st.metric("High-tier stakeholders with $0 spend", n_high_zero)


# ============================================================
# PAGE: Anomaly detection
# ============================================================
def page_anomaly():
    st.markdown(f"<p style='color:{AMBER};font-weight:bold;letter-spacing:3px;'>APPENDIX — MODEL-BASED CHECK</p>", unsafe_allow_html=True)
    st.markdown("# Anomaly detection (Isolation Forest)")
    st.markdown(f"<p style='color:{SLATE};font-style:italic;'>Unsupervised model run on the 7,000 deduplicated interactions. Top 2% flagged.</p>", unsafe_allow_html=True)

    with st.spinner("Fitting Isolation Forest..."):
        m = fit_anomaly_model()

    n_anom = (m["anomaly_pctile"] >= 98).sum()
    st.metric("Anomalies flagged", n_anom)

    cohort = m[m["anomaly_pctile"] >= 98].copy().sort_values("anomaly_score", ascending=False)
    cols_show = ["country","duration_minutes","total_usd","max_single_usd","honoraria_usd",
                 "min_lag_days","stk_risk_tier","stk_stakeholder_type","emp_role","anomaly_score","anomaly_pctile"]
    cohort_display = cohort.reset_index()[["interaction_id"] + cols_show].rename(columns={
        "interaction_id":"Interaction","country":"Country","duration_minutes":"Duration",
        "total_usd":"Spend (USD)","max_single_usd":"Max single","honoraria_usd":"Honoraria",
        "min_lag_days":"Min lag","stk_risk_tier":"Risk tier","stk_stakeholder_type":"Type","emp_role":"Employee role",
        "anomaly_score":"Score","anomaly_pctile":"Pctile",
    })
    st.dataframe(cohort_display, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Top features driving the anomaly cohort")
    st.caption("σ-gap between the mean of anomalies and the mean of the baseline.")
    features = [
        ("Spend per minute",       8.7),
        ("Total spend (USD)",      8.4),
        ("Max single payment",     8.2),
        ("Venue / AV spend",       6.6),
        ("Honoraria spend",        6.3),
        ("Number of categories",   1.3),
        ("Pre-dated payment",      0.8),
    ]
    df = pd.DataFrame(features, columns=["Feature","σ-gap"]).set_index("Feature")
    st.bar_chart(df, height=320, color=AMBER)


# ============================================================
# PAGE: Assisted review (live)
# ============================================================
def page_assisted_review():
    st.markdown(f"<p style='color:{AMBER};font-weight:bold;letter-spacing:3px;'>APPENDIX · PHASE 3 PROTOTYPE</p>", unsafe_allow_html=True)
    st.markdown("# Assisted Review — live")
    st.markdown(f"<p style='color:{SLATE};font-style:italic;'>LangGraph workflow. Six deterministic checks plus Gemini for text-based document review. Each case produces a structured dossier and a full audit log.</p>", unsafe_allow_html=True)

    # Pre-populate examples
    examples = {
        "I002316 — the $125K honorarium (escalates)": "I002316",
        "I004586 — pre-dated honoraria, top-anomaly score": "I004586",
        "I002241 — short meeting with $14.7K venue charge": "I002241",
        "(auto-close example) I005587": "I005587",
    }
    col1, col2 = st.columns([2, 1])
    with col1:
        choice = st.selectbox("Pick a case to review, or enter any interaction_id below:", list(examples.keys()))
        iid = st.text_input("interaction_id", value=examples[choice]).strip()
    with col2:
        run = st.button("Run assisted review", type="primary", use_container_width=True)
        st.caption(f"{'LIVE — using Gemini' if get_api_key() else 'STUB — no API key set'}")

    if not run:
        st.info("Pick a case above and click 'Run assisted review' to see the LangGraph workflow process it end-to-end.")
        return

    # Import lazily so the page renders even if langgraph isn't installed
    try:
        from langgraph_assisted_review import review
    except Exception as e:
        st.error(f"Could not import the workflow: {e}")
        return

    with st.spinner(f"Processing {iid} through the LangGraph workflow..."):
        t0 = time.time()
        out = review(iid)
        elapsed = time.time() - t0

    if out.get("disposition") == "ERROR":
        st.error("Interaction not found in the deduplicated dataset.")
        return

    # ---------- Header banner ----------
    h = out["case_header"]
    auto = out["disposition"] == "AUTO-CLOSE"
    color = GREEN if auto else RED_SOFT
    badge = "AUTO-CLOSE" if auto else "ESCALATE"
    st.markdown(f"<div style='background:{color};color:white;padding:14px 20px;border-radius:6px;'>"
                f"<span style='font-size:11px;letter-spacing:3px;'>{badge}</span> &nbsp;·&nbsp; "
                f"<b>{iid}</b> &nbsp;·&nbsp; {h.get('interaction_date')} &nbsp;·&nbsp; "
                f"{h.get('country')} &nbsp;·&nbsp; {h.get('business_unit')} &nbsp;·&nbsp; "
                f"{out.get('severity','')}</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total spend (USD)", f"${h['total_usd']:,.0f}")
    c2.metric("Honoraria",         f"${h['honoraria_usd']:,.0f}")
    c3.metric("Anomaly percentile", f"{out['anomaly']['percentile']:.0f}")
    c4.metric("Failed checks",     f"{out['fails']} / 6")

    st.markdown("### Case header")
    st.json({
        "interaction": {
            "date": h["interaction_date"],
            "country": h["country"],
            "business_unit": h["business_unit"],
            "type": h["interaction_type"],
            "duration_minutes": h["duration_minutes"],
            "purpose": h["declared_purpose"],
        },
        "employee":    {"id": h["employee_id"], "role": h["employee_role"], "country": h["employee_country"]},
        "stakeholder": {"id": h["stakeholder_id"], "type": h["stakeholder_type"], "country": h["stakeholder_country"],
                        "specialty": h["stakeholder_specialty"], "risk_rating": h["risk_rating"]},
        "history_with_stakeholder": out["history"],
    })

    st.markdown("### Deterministic checks")
    for c in out["deterministic_checks"]:
        pill = "pill-green" if c["passed"] else "pill-red"
        mark = "PASS" if c["passed"] else "FAIL"
        st.markdown(f"<span class='{pill}'>{mark}</span> &nbsp; **{c['id']}** &nbsp; {c['name']}", unsafe_allow_html=True)

    if not auto:
        st.markdown("### Documents reviewed by the LLM")
        for doc in out.get("documents", []) or []:
            with st.expander(f"📄 {doc['name']}  ·  source: {doc['source']}"):
                st.code(doc["text"])

        st.markdown("### Per-document findings (Gemini)")
        for f in out.get("llm_findings", []) or []:
            with st.container():
                st.markdown(f"**{f['document']}**")
                st.write(f["finding"])
                st.divider()

        st.markdown("### Reviewer brief")
        st.info(out.get("reviewer_brief") or "(no brief generated)")

    st.markdown("### Audit log")
    st.code("\n".join(out["audit_log"]))
    st.caption(f"Workflow completed in {elapsed:.1f} seconds. The full state is logged for regulator inspection regardless of disposition.")


# ============================================================
# PAGE: How it works
# ============================================================
def page_how_it_works():
    st.markdown(f"<p style='color:{AMBER};font-weight:bold;letter-spacing:3px;'>HOW THIS COMPANION IS BUILT</p>", unsafe_allow_html=True)
    st.markdown("# Architecture")
    st.markdown(f"<p style='color:{SLATE};font-style:italic;'>Three pieces: a rule layer, an unsupervised model layer, and a LangGraph workflow that uses Gemini for text-based document review.</p>", unsafe_allow_html=True)
    st.markdown("---")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("### The LangGraph workflow")
        st.markdown("""
Each case flows through a typed state machine. Deterministic nodes are pure Python; LLM nodes are bounded to reading text-based documents and writing the reviewer brief.

1. **gather_evidence** — pull the interaction, related payments, stakeholder profile, employee profile, and 12-month relationship history.
2. **score_anomaly** — apply the pre-fit Isolation Forest, return a percentile rank.
3. **run_deterministic_checks** — six rules Compliance signs off on. Pass/fail boolean per rule.
4. **Conditional edge** — all pass → auto-close branch; any fail → LLM branch.
5. **llm_review_documents (Gemini)** — for each retrieved document, compare against the structured payment record and surface inconsistencies. Bounded to factual comparison, not disposition.
6. **synthesize_brief (Gemini)** — write a six-sentence reviewer brief from the deterministic results and the document findings.
7. **log_audit** — append a structured audit entry. In production this writes to an immutable store.
        """)

    with c2:
        st.markdown("### The six deterministic checks")
        st.markdown("""
| ID | Check |
|---|---|
| D1 | Total spend under $1,000 |
| D2 | No payments dated before the interaction |
| D3 | Stakeholder has a risk rating on file |
| D4 | Duration at least 5 minutes |
| D5 | No honoraria over $5,000 |
| D6 | Anomaly score below 95th percentile |
        """)
        st.markdown("### What the LLM does (and does not do)")
        st.markdown(f"""
- **Reads**: contract excerpts, FMV benchmarks, approval chains, meeting minutes.
- **Compares**: those documents against the structured payment record.
- **Surfaces**: factual mismatches, missing signatories, scope inconsistencies.
- **Does not**: make the disposition. Disposition is bounded by the six deterministic checks above.
        """)

    st.markdown("---")
    st.markdown("### Files in this repo")
    st.markdown("""
- `analysis.py` — the rule-based exploration that produced the four insights.
- `anomaly_model.py` — the Isolation Forest cross-check.
- `langgraph_assisted_review.py` — the LangGraph workflow (this app calls into it).
- `streamlit_app.py` — this companion.
- `pre_dated_payments.csv`, `anomaly_scored_interactions.csv` — derived outputs.
    """)


# ============================================================
# Router
# ============================================================
if page == "Overview":
    page_overview()
elif page == "Four key insights":
    page_insights()
elif page == "Anomaly detection":
    page_anomaly()
elif page == "Assisted review (live)":
    page_assisted_review()
elif page == "How it works":
    page_how_it_works()
