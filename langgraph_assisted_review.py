"""
Phase 3 — Assisted Review built on LangGraph with Gemini.

The graph orchestrates the workflow shown on the slide:

    gather_evidence  →  score_anomaly  →  run_deterministic_checks  →
       (if all pass)   →  auto_close  →  log_audit  →  END
       (else)          →  llm_review_documents  →  synthesize_brief  →  log_audit  →  END

Deterministic checks remain deterministic (auditable, no model risk).
The LLM (Gemini) only reads text-based supporting documents and writes the
reviewer brief, so its role is bounded to retrieval, comparison and
summarisation — never to disposition.

In a real deployment, the documents would be fetched from source systems
(contract repository, FMV rate card, approval workflow, attendee CRM, meeting
minutes archive). Here we synthesise them from the structured case data so the
graph can run end-to-end on the case-study dataset.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict, Optional, Annotated
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from langgraph.graph import StateGraph, END

BASE = Path(__file__).parent

# ============================================================
# Lazy data and model load
# ============================================================
_CACHE: dict = {}

def _load():
    if "loaded" in _CACHE:
        return _CACHE["loaded"]
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

    _CACHE["loaded"] = (inter, spend, stake, emp)
    return _CACHE["loaded"]


def _fit_anomaly():
    if "scored" in _CACHE:
        return _CACHE["scored"]
    inter, spend, stake, emp = _load()
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
    _CACHE["scored"] = m.set_index("interaction_id")
    return _CACHE["scored"]


# ============================================================
# GRAPH STATE
# ============================================================
class CaseState(TypedDict, total=False):
    interaction_id: str
    case_header: dict
    payments: list
    history: dict
    anomaly: dict
    deterministic_checks: list
    fails: int
    documents: list                 # synthesised text-based documents
    llm_findings: Optional[list]    # LLM output per document
    reviewer_brief: Optional[str]   # LLM summary
    disposition: str                # AUTO-CLOSE | ESCALATE
    severity: str
    audit_log: list


# ============================================================
# NODE 1 — gather_evidence
# ============================================================
def node_gather_evidence(state: CaseState) -> CaseState:
    iid = state["interaction_id"]
    inter, spend, stake, emp = _load()
    scored = _fit_anomaly()
    if iid not in scored.index:
        return {**state, "audit_log": state.get("audit_log", []) + [
            f"{datetime.utcnow().isoformat()}Z  gather_evidence  FAIL  not_found"
        ], "disposition": "ERROR"}

    case = scored.loc[iid]
    payments = spend[spend["interaction_id"] == iid].to_dict("records")
    # 12-month history with stakeholder
    ref = case["interaction_date"]
    if pd.notna(ref):
        hist = inter[(inter["stakeholder_id"] == case["stakeholder_id"]) &
                     (inter["interaction_id"]  != iid) &
                     (inter["interaction_date"] >= ref - pd.Timedelta(days=365)) &
                     (inter["interaction_date"] <= ref)]
        hist_spend = spend[spend["interaction_id"].isin(hist["interaction_id"])]
        history = {
            "prior_interactions": int(len(hist)),
            "prior_employees":    int(hist["employee_id"].nunique()),
            "prior_total_usd":    round(float(hist_spend["amount_usd"].sum()), 2),
        }
    else:
        history = {"prior_interactions": 0, "prior_employees": 0, "prior_total_usd": 0.0}

    case_header = {
        "interaction_id":  iid,
        "interaction_date": str(case["interaction_date"].date()) if pd.notna(case["interaction_date"]) else None,
        "country":          case["country"],
        "business_unit":    case["business_unit"],
        "interaction_type": case["interaction_type"],
        "declared_purpose": case["declared_purpose"],
        "duration_minutes": float(case["duration_minutes"]) if pd.notna(case["duration_minutes"]) else None,
        "employee_id":      case["employee_id"],
        "employee_role":    case["emp_role"],
        "employee_country": case["emp_country"],
        "stakeholder_id":   case["stakeholder_id"],
        "stakeholder_type": case["stk_stakeholder_type"],
        "stakeholder_country": case["stk_country"],
        "stakeholder_specialty": case["stk_specialty"],
        "risk_rating":      case["stk_risk_tier"] if pd.notna(case["stk_risk_tier"]) else None,
        "total_usd":        float(case["total_usd"]),
        "honoraria_usd":    float(case["honoraria_usd"]),
        "max_single_usd":   float(case["max_single_usd"]),
        "min_lag_days":     int(case["min_lag_days"]),
        "any_predated":     bool(case["any_predated"]),
    }
    return {
        **state,
        "case_header": case_header,
        "payments":    payments,
        "history":     history,
        "audit_log":   state.get("audit_log", []) + [
            f"{datetime.utcnow().isoformat()}Z  gather_evidence  OK  payments={len(payments)} history={history['prior_interactions']}"
        ],
    }


# ============================================================
# NODE 2 — score_anomaly
# ============================================================
def node_score_anomaly(state: CaseState) -> CaseState:
    iid = state["interaction_id"]
    scored = _fit_anomaly()
    case = scored.loc[iid]
    anomaly = {
        "score":      round(float(case["anomaly_score"]), 3),
        "percentile": round(float(case["anomaly_pctile"]), 1),
    }
    return {
        **state,
        "anomaly":   anomaly,
        "audit_log": state["audit_log"] + [
            f"{datetime.utcnow().isoformat()}Z  score_anomaly  OK  pct={anomaly['percentile']}"
        ],
    }


# ============================================================
# NODE 3 — run_deterministic_checks
# ============================================================
def node_run_deterministic_checks(state: CaseState) -> CaseState:
    h = state["case_header"]
    a = state["anomaly"]
    rules = [
        ("D1", "Total spend under $1,000",       h["total_usd"] < 1000),
        ("D2", "No pre-dated payments",           not h["any_predated"]),
        ("D3", "Risk rating on file",             h["risk_rating"] is not None),
        ("D4", "Duration at least 5 minutes",     (h["duration_minutes"] or 0) >= 5),
        ("D5", "No honoraria over $5,000",        h["honoraria_usd"] <= 5000),
        ("D6", "Anomaly score below 95th pct",    a["percentile"] < 95),
    ]
    checks = [{"id": rid, "name": name, "passed": bool(passed)} for rid, name, passed in rules]
    fails  = sum(1 for c in checks if not c["passed"])
    return {
        **state,
        "deterministic_checks": checks,
        "fails": fails,
        "audit_log": state["audit_log"] + [
            f"{datetime.utcnow().isoformat()}Z  run_deterministic_checks  OK  fails={fails}"
        ],
    }


# ============================================================
# DECISION — auto-close vs escalate
# ============================================================
def route_decision(state: CaseState) -> str:
    if state["fails"] == 0:
        return "auto_close"
    return "llm_review_documents"


# ============================================================
# AUTO-CLOSE branch
# ============================================================
def node_auto_close(state: CaseState) -> CaseState:
    return {
        **state,
        "disposition": "AUTO-CLOSE",
        "severity":    "ROUTINE",
        "documents":   [],
        "llm_findings": None,
        "reviewer_brief": None,
        "audit_log":   state["audit_log"] + [
            f"{datetime.utcnow().isoformat()}Z  auto_close  OK  routed_to=sampling_queue"
        ],
    }


# ============================================================
# Document synthesis — what the LLM reads
# ============================================================
def _synthesize_documents(state: CaseState) -> list:
    h = state["case_header"]
    docs = []

    if h["honoraria_usd"] > 0:
        docs.append({
            "name": "Honoraria service agreement (excerpt)",
            "source": "contract repository (CLM)",
            "text": (
                f"SERVICE AGREEMENT\n"
                f"Parties: GSK (engaging) and stakeholder {h['stakeholder_id']} ({h['stakeholder_type']} based in {h['stakeholder_country']}).\n"
                f"Services: {h['declared_purpose']} — engagement under the {h['business_unit']} business unit.\n"
                f"Fee structure: total honoraria not to exceed USD 6,500 per engagement, payable on completion.\n"
                f"Term: effective {h['interaction_date']}, single engagement.\n"
                f"Signatories on file: employee {h['employee_id']} (engagement lead); Legal counter-signature on file.\n"
            ),
        })
        docs.append({
            "name": "Fair-market-value benchmark (excerpt)",
            "source": "annual third-party FMV survey",
            "text": (
                f"FMV RATE CARD ENTRY\n"
                f"Stakeholder profile: {h['stakeholder_type']} · {h['stakeholder_specialty']} · {h['stakeholder_country']}.\n"
                f"Engagement type: {h['interaction_type']}.\n"
                f"Benchmarked ceiling (per engagement hour): USD 950.\n"
                f"Reference benchmark survey: Cutting Edge Information, 2024 edition.\n"
            ),
        })
    if h["total_usd"] > 5000:
        docs.append({
            "name": "Approval chain (excerpt)",
            "source": "spend authorisation workflow",
            "text": (
                f"APPROVAL CHAIN — interaction {h['interaction_id']}\n"
                f"Initiator: employee {h['employee_id']} ({h['employee_role']}, {h['employee_country']}).\n"
                f"Authorised amount: up to USD 5,000 under standing delegation of authority for the role.\n"
                f"Submitted approval level: Medical Affairs lead.\n"
                f"Legal sign-off: pending — not yet retrieved.\n"
                f"Finance sign-off: pending — not yet retrieved.\n"
            ),
        })
    if h["interaction_type"] in ("Speaker Program","Advisory Board","Congress/Conference"):
        docs.append({
            "name": "Meeting minutes (excerpt)",
            "source": "event-management system",
            "text": (
                f"MEETING MINUTES — {h['interaction_date']}\n"
                f"Type: {h['interaction_type']} held in {h['country']}.\n"
                f"Duration recorded: {h['duration_minutes']} minutes.\n"
                f"Stakeholder attendance: confirmed for {h['stakeholder_id']}.\n"
                f"Topics covered: {h['declared_purpose']}.\n"
                f"Materials reviewed: on-label clinical data; off-label discussion: none recorded.\n"
                f"Deliverables agreed: scientific advisory output, to be filed within 14 days.\n"
            ),
        })
    return docs


# ============================================================
# LLM call (Gemini) — bounded to retrieval-style review
# ============================================================
def _gemini_call(prompt: str) -> str:
    """Call Gemini. Returns the text response. Falls back to a deterministic
    stub if no API key or any error occurs (so the demo never hard-crashes)."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return _stub_response(prompt)
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", api_key=api_key, temperature=0)
        resp = llm.invoke(prompt)
        return resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        return _stub_response(prompt) + f"\n\n[NOTE: LLM call failed — {type(e).__name__}: {e}]"


def _stub_response(prompt: str) -> str:
    """Deterministic stub used when no API key is configured. Lets the
    workflow run end-to-end without requiring credentials."""
    if "REVIEW DOCUMENT" in prompt:
        return ("FINDING: Document is structurally complete but contains terms inconsistent with the payment "
                "record. Fee ceiling and actual fee differ materially. Counter-signatures referenced but not "
                "verified within the document. Recommendation: human reviewer to retrieve full document and "
                "verify against payment ledger.\n"
                "[stub output — no LLM credentials configured]")
    return ("Reviewer brief: this case carries failures on the deterministic policy and at least one document "
            "shows inconsistencies with the payment record. Recommend a human reviewer confirm contract scope, "
            "FMV alignment, and approval level before disposing of the case.\n"
            "[stub output — no LLM credentials configured]")


# ============================================================
# NODE 4 — llm_review_documents (Gemini)
# ============================================================
def node_llm_review_documents(state: CaseState) -> CaseState:
    docs = _synthesize_documents(state)
    findings = []
    h = state["case_header"]
    for d in docs:
        prompt = (
            "You are reviewing a single piece of supporting evidence as part of an "
            "automated pharma-compliance triage. Your job is to compare the document "
            "text against the structured payment record and surface any factual "
            "inconsistency, missing element, or mismatch in scope. Do not make a "
            "judgement on whether the case is compliant or not. Be concise (3-4 lines).\n\n"
            f"PAYMENT RECORD\n"
            f"  Interaction: {h['interaction_id']} on {h['interaction_date']}\n"
            f"  Total spend: USD {h['total_usd']:.2f}\n"
            f"  Honoraria component: USD {h['honoraria_usd']:.2f}\n"
            f"  Largest single payment: USD {h['max_single_usd']:.2f}\n"
            f"  Stakeholder: {h['stakeholder_id']} ({h['stakeholder_type']}, {h['stakeholder_country']})\n"
            f"  Employee: {h['employee_id']} ({h['employee_role']})\n"
            f"  Declared purpose: {h['declared_purpose']}\n\n"
            "REVIEW DOCUMENT\n"
            f"  Document name: {d['name']}\n"
            f"  Source system:  {d['source']}\n"
            f"  Document text:\n{d['text']}\n\n"
            "Return a single short paragraph beginning with 'FINDING:'."
        )
        findings.append({"document": d["name"], "source": d["source"], "finding": _gemini_call(prompt)})

    return {
        **state,
        "documents":   docs,
        "llm_findings": findings,
        "audit_log":   state["audit_log"] + [
            f"{datetime.utcnow().isoformat()}Z  llm_review_documents  OK  n_docs={len(docs)}"
        ],
    }


# ============================================================
# NODE 5 — synthesize_brief (Gemini)
# ============================================================
def node_synthesize_brief(state: CaseState) -> CaseState:
    h = state["case_header"]
    fails = [c for c in state["deterministic_checks"] if not c["passed"]]
    fail_lines = "\n".join(f"  - {c['id']} {c['name']}" for c in fails)
    findings_lines = "\n\n".join(
        f"  {f['document']}: {f['finding']}" for f in (state.get("llm_findings") or [])
    )
    prompt = (
        "You are preparing a short reviewer brief for a Compliance officer who will "
        "spend roughly five minutes on this case. Summarise the findings so the "
        "reviewer knows what to look at first. Do not recommend a disposition. "
        "Be concrete and bounded to what the evidence supports. 6 sentences maximum.\n\n"
        f"CASE: {h['interaction_id']} on {h['interaction_date']} ({h['country']}, {h['business_unit']}, {h['interaction_type']})\n"
        f"Spend: USD {h['total_usd']:.2f} (honoraria USD {h['honoraria_usd']:.2f}, largest single USD {h['max_single_usd']:.2f})\n"
        f"Stakeholder: {h['stakeholder_id']} ({h['stakeholder_type']}, {h['stakeholder_country']}); risk rating {h['risk_rating']}.\n"
        f"Employee: {h['employee_id']} ({h['employee_role']}).\n\n"
        f"FAILED DETERMINISTIC CHECKS:\n{fail_lines or '  (none)'}\n\n"
        f"DOCUMENT REVIEW FINDINGS:\n{findings_lines or '  (none)'}"
    )
    brief = _gemini_call(prompt)
    fails_n = state["fails"]
    severity = "PRIORITY REVIEW" if fails_n >= 3 else ("STANDARD REVIEW" if fails_n >= 1 else "ROUTINE")
    return {
        **state,
        "disposition":  "ESCALATE",
        "severity":     severity,
        "reviewer_brief": brief,
        "audit_log":    state["audit_log"] + [
            f"{datetime.utcnow().isoformat()}Z  synthesize_brief  OK  severity={severity}"
        ],
    }


# ============================================================
# NODE 6 — log_audit (terminal)
# ============================================================
def node_log_audit(state: CaseState) -> CaseState:
    # In production this would append to an immutable audit store.
    return {
        **state,
        "audit_log": state["audit_log"] + [
            f"{datetime.utcnow().isoformat()}Z  log_audit  OK  disposition={state.get('disposition','?')}"
        ],
    }


# ============================================================
# Build graph
# ============================================================
def build_graph():
    g = StateGraph(CaseState)
    g.add_node("gather_evidence",          node_gather_evidence)
    g.add_node("score_anomaly",            node_score_anomaly)
    g.add_node("run_deterministic_checks", node_run_deterministic_checks)
    g.add_node("auto_close",               node_auto_close)
    g.add_node("llm_review_documents",     node_llm_review_documents)
    g.add_node("synthesize_brief",         node_synthesize_brief)
    g.add_node("log_audit",                node_log_audit)

    g.set_entry_point("gather_evidence")
    g.add_edge("gather_evidence",          "score_anomaly")
    g.add_edge("score_anomaly",            "run_deterministic_checks")
    g.add_conditional_edges(
        "run_deterministic_checks",
        route_decision,
        {"auto_close": "auto_close", "llm_review_documents": "llm_review_documents"},
    )
    g.add_edge("auto_close",               "log_audit")
    g.add_edge("llm_review_documents",     "synthesize_brief")
    g.add_edge("synthesize_brief",         "log_audit")
    g.add_edge("log_audit",                END)
    return g.compile()


_GRAPH = None
def review(interaction_id: str) -> dict:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    initial: CaseState = {"interaction_id": interaction_id, "audit_log": []}
    return _GRAPH.invoke(initial)


# ============================================================
# CLI runner
# ============================================================
if __name__ == "__main__":
    import sys
    test_ids = sys.argv[1:] or ["I002316", "I005587"]
    for iid in test_ids:
        print(f"\n{'#'*78}\n# CASE: {iid}\n{'#'*78}")
        out = review(iid)
        print(json.dumps({
            "disposition":   out.get("disposition"),
            "severity":      out.get("severity"),
            "fails":         out.get("fails"),
            "anomaly_pctile": out.get("anomaly",{}).get("percentile"),
            "n_documents":   len(out.get("documents") or []),
            "n_llm_findings": len(out.get("llm_findings") or []) if out.get("llm_findings") else 0,
            "reviewer_brief": (out.get("reviewer_brief") or "")[:400],
        }, indent=2, default=str))
        print("\n--- AUDIT LOG ---")
        for line in out["audit_log"]:
            print(line)
