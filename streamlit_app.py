"""
Risk Analytics Case Study — Assisted Review live demo.

A focused, single-page Streamlit app that demonstrates the Phase 3
assisted-review prototype on the case-study dataset. Built on LangGraph;
the document-review and brief-synthesis nodes call Gemini live.

Add a GEMINI_API_KEY in Streamlit Cloud secrets (or as an environment
variable locally) to enable the live LLM nodes. Without a key the
workflow still runs end-to-end using deterministic stubs so the demo
never hard-crashes.
"""

import os
import time
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Assisted Review · Risk Analytics case study",
    page_icon="◐",
    layout="wide",
)

BASE = Path(__file__).parent

# ============================================================
# Palette
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
  .pill-green {{background: {GREEN}; color: white; padding: 2px 10px; border-radius: 10px; font-size: 11px; font-weight: bold;}}
  .pill-red   {{background: {RED_SOFT}; color: white; padding: 2px 10px; border-radius: 10px; font-size: 11px; font-weight: bold;}}
  .pill-amber {{background: {AMBER}; color: white; padding: 2px 10px; border-radius: 10px; font-size: 11px; font-weight: bold;}}
  .intro-box  {{background: white; border-left: 4px solid {AMBER}; padding: 14px 20px; margin: 0 0 18px 0; border-radius: 4px;}}
</style>
""", unsafe_allow_html=True)


# ============================================================
# API key
# ============================================================
def get_api_key():
    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        pass
    if not key:
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return key


# ============================================================
# Sidebar — credential indicator + credit
# ============================================================
st.sidebar.markdown(f"<h2 style='color:{NAVY};margin:0;'>Assisted Review</h2>", unsafe_allow_html=True)
st.sidebar.markdown(f"<p style='color:{SLATE};font-style:italic;font-size:12px;margin-top:0;'>Phase 3 prototype · live demo</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

api_key = get_api_key()
if api_key:
    st.sidebar.markdown(f"<span class='pill-green'>LIVE</span> &nbsp; Gemini key detected", unsafe_allow_html=True)
    st.sidebar.caption("LLM nodes call Gemini 2.5 Flash Lite directly. Responses are cached per case.")
else:
    st.sidebar.markdown(f"<span class='pill-amber'>STUB</span> &nbsp; no API key", unsafe_allow_html=True)
    st.sidebar.caption("LLM nodes return deterministic stubs. Set GEMINI_API_KEY in Streamlit secrets to enable live calls.")

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<p style='color:{SLATE};font-size:11px;line-height:1.5;'>
Built for the GSK Risk Analytics case study.<br/>
<b>Anjali Grover</b> · May 2026.<br/><br/>
Stack: LangGraph · Google Gemini · scikit-learn · Streamlit.
</p>
""", unsafe_allow_html=True)


# ============================================================
# Cached workflow call — avoids burning quota on repeat demos
# ============================================================
@st.cache_data(show_spinner=False)
def review(iid: str):
    # Import inside the cached function to avoid any module-load timing edge cases.
    from langgraph_assisted_review import review as _review_uncached
    return _review_uncached(iid)


# ============================================================
# Main page
# ============================================================
st.markdown(f"<p style='color:{AMBER};font-weight:bold;letter-spacing:3px;margin-bottom:0;'>RISK ANALYTICS · PHASE 3 PROTOTYPE</p>", unsafe_allow_html=True)
st.markdown("# Assisted review — live workflow demo")
st.markdown(f"<p style='color:{SLATE};font-style:italic;font-size:15px;'>Built on LangGraph. Six deterministic checks Compliance signs off on, plus Gemini for text-based document review. The model never decides disposition — it reads documents, surfaces inconsistencies, and writes the reviewer brief. Disposition is bounded by the deterministic rules.</p>", unsafe_allow_html=True)

# Intro box explaining what this is
st.markdown(f"""
<div class='intro-box'>
<b>What this demo does</b><br/>
Given any <code>interaction_id</code> from the case-study dataset, the workflow gathers
all related evidence (payments, stakeholder profile, employee profile, 12-month history),
applies six deterministic checks, and either auto-closes the case as routine or assembles
a prep dossier for a human reviewer. On escalated cases, Gemini reads the supporting
documents (contract, fair-market-value benchmark, approval chain, meeting minutes) and
flags any inconsistency with the payment record. Every step is logged for audit.
</div>
""", unsafe_allow_html=True)

# ---- Picker ----
examples = {
    "I002316 — the $125K honorarium (escalates)":         "I002316",
    "I004586 — pre-dated honoraria · top anomaly score":  "I004586",
    "I002241 — short meeting with $14.7K venue charge":   "I002241",
    "I005587 — routine case · auto-close":                "I005587",
}

col_pick, col_id, col_btn = st.columns([3, 1.5, 1])
with col_pick:
    choice = st.selectbox("Pick a case to review:", list(examples.keys()))
with col_id:
    iid = st.text_input("…or enter any interaction_id", value=examples[choice]).strip()
with col_btn:
    st.write("")  # vertical alignment
    st.write("")
    run = st.button("Run assisted review", type="primary", use_container_width=True)

if not run:
    st.info("Pick one of the four example cases above (or paste any interaction_id from the dataset) and click **Run assisted review** to watch the LangGraph workflow process it end-to-end. The first run for each case calls the LLM live; subsequent runs are cached.")
    st.stop()

# ---- Run workflow ----
try:
    with st.spinner(f"Processing {iid} through the LangGraph workflow..."):
        t0 = time.time()
        out = review(iid)
        elapsed = time.time() - t0
except Exception as e:
    st.error(f"Workflow failed: {type(e).__name__}: {e}")
    st.stop()

if out.get("disposition") == "ERROR":
    st.error(f"Interaction `{iid}` was not found in the deduplicated dataset. Try one of the example cases above.")
    st.stop()

# ---- Header banner ----
h = out["case_header"]
auto = out["disposition"] == "AUTO-CLOSE"
banner_color = GREEN if auto else RED_SOFT
banner_label = "AUTO-CLOSE · ROUTINE" if auto else f"ESCALATE · {out.get('severity', 'REVIEW')}"
st.markdown(f"""
<div style='background:{banner_color};color:white;padding:14px 22px;border-radius:6px;margin-top:12px;'>
  <span style='font-size:11px;letter-spacing:3px;'>{banner_label}</span>
  &nbsp;·&nbsp; <b>{iid}</b>
  &nbsp;·&nbsp; {h.get('interaction_date')}
  &nbsp;·&nbsp; {h.get('country')}
  &nbsp;·&nbsp; {h.get('business_unit')}
</div>
""", unsafe_allow_html=True)

# ---- Top metrics ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total spend (USD)", f"${h['total_usd']:,.0f}")
c2.metric("Honoraria",         f"${h['honoraria_usd']:,.0f}")
c3.metric("Anomaly percentile", f"{out['anomaly']['percentile']:.0f}")
c4.metric("Failed checks",     f"{out['fails']} / 6")

# ---- Case header ----
st.markdown("### Case details")
st.json({
    "interaction": {
        "date":             h["interaction_date"],
        "country":          h["country"],
        "business_unit":    h["business_unit"],
        "type":             h["interaction_type"],
        "duration_minutes": h["duration_minutes"],
        "purpose":          h["declared_purpose"],
    },
    "employee":    {"id": h["employee_id"], "role": h["employee_role"], "country": h["employee_country"]},
    "stakeholder": {
        "id":          h["stakeholder_id"],
        "type":        h["stakeholder_type"],
        "country":     h["stakeholder_country"],
        "specialty":   h["stakeholder_specialty"],
        "risk_rating": h["risk_rating"],
    },
    "history_with_stakeholder_last_12_months": out["history"],
})

# ---- Deterministic checks ----
st.markdown("### Deterministic checks")
for c in out["deterministic_checks"]:
    pill = "pill-green" if c["passed"] else "pill-red"
    mark = "PASS" if c["passed"] else "FAIL"
    st.markdown(f"<span class='{pill}'>{mark}</span> &nbsp; **{c['id']}** &nbsp; {c['name']}", unsafe_allow_html=True)

# ---- If escalated: documents, findings, brief ----
if not auto:
    st.markdown("### Documents the LLM reviewed")
    st.caption("In a production deployment these would be pulled from the contract repository, FMV rate card, spend authorisation system, and event-management system. Here they are synthesised from the structured case data so the workflow can run end-to-end on the case-study dataset.")
    for doc in out.get("documents", []) or []:
        with st.expander(f"📄 {doc['name']}  ·  source: {doc['source']}"):
            st.code(doc["text"])

    st.markdown("### Per-document findings (Gemini)")
    findings = out.get("llm_findings") or []
    if not findings:
        st.info("No documents needed review for this case.")
    else:
        for f in findings:
            st.markdown(f"**{f['document']}**")
            st.write(f["finding"])
            st.divider()

    st.markdown("### Reviewer brief")
    brief = out.get("reviewer_brief")
    if brief:
        st.info(brief)
    else:
        st.caption("No brief generated.")

# ---- Audit log ----
st.markdown("### Audit log")
st.caption("Every node execution is logged with a UTC timestamp. In production, this writes to an immutable audit store for regulator inspection.")
st.code("\n".join(out["audit_log"]))
st.caption(f"Workflow completed in {elapsed:.1f} seconds.")

# ---- Footer ----
st.markdown("---")
st.markdown(f"""
<p style='color:{SLATE};font-size:11px;line-height:1.6;'>
This is one piece of a wider case study on identifying emerging risk patterns in HCP/HCO engagement.
The deck covers four insights from the rule-based analysis, a cross-check using an unsupervised anomaly model,
recommended actions for Compliance, and a phased path from rule-based monitoring to assisted review.
This page is the working version of the Phase 3 step.
</p>
""", unsafe_allow_html=True)
