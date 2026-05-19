# How to deploy this case study to a public URL

You want a single URL the interviewer can click and explore. This walks
through the cheapest reliable path: **Streamlit Community Cloud + a free
Gemini key from Google AI Studio**. No credit card needed for either.

The whole thing takes about 15 minutes the first time.

---

## Step 1 — get a free Gemini API key

1. Go to **https://aistudio.google.com**
2. Sign in with any Google account
3. Click **"Get API key"** (top-left in the sidebar) → **"Create API key"**
4. Copy the key. It looks like `AIzaSy...` (about 40 characters)

The free tier of `gemini-2.5-flash` gives you ~15 requests per minute and
~1,500 per day, which is more than enough for an interview demo.

---

## Step 2 — put the files in a GitHub repo

You need a public GitHub repo with:

```
gsk_risk_case_dataset/
├── streamlit_app.py
├── langgraph_assisted_review.py
├── requirements.txt
├── interactions.csv
├── spend.csv
├── stakeholders.csv
└── employees.csv
```

A few notes:
- `data_dictionary.md` is optional but worth including.
- You do **not** need to commit the API key. That goes in Streamlit Cloud secrets.
- All the files above are already in this folder.

Quickest way if you don't already have a repo:

```bash
cd "C:/Users/Anjali Grover/Documents/Projects/leet/gsk_risk_case_dataset"
git init
git add streamlit_app.py langgraph_assisted_review.py requirements.txt \
        interactions.csv spend.csv stakeholders.csv employees.csv data_dictionary.md
git commit -m "Initial commit — Risk Analytics case study companion"
# Then on github.com: create a new public repo (e.g. "risk-analytics-case-study")
git remote add origin https://github.com/<your-username>/risk-analytics-case-study.git
git branch -M main
git push -u origin main
```

> ⚠️ Make sure the repo is **public** — Streamlit Cloud's free tier requires it.
> The case data is synthetic and de-identified per the case-study brief, so
> there is no privacy issue.

---

## Step 3 — deploy on Streamlit Community Cloud

1. Go to **https://share.streamlit.io**
2. Sign in with your GitHub account
3. Click **"Create app"** → **"Deploy from existing repo"**
4. Pick the repo you just created
5. Branch: `main`
6. Main file path: `streamlit_app.py`
7. App URL (optional): set something memorable like `anjali-risk-case`
8. Click **"Advanced settings"** → **"Secrets"**, and paste:

   ```toml
   GEMINI_API_KEY = "AIzaSy...your-key-here..."
   ```

9. Click **"Deploy"**

In about 2-3 minutes you will have a live URL such as:

```
https://anjali-risk-case.streamlit.app
```

That URL is what you paste into the deck.

---

## Step 4 — verify it works

Open the URL and check:

- The **sidebar** should show a green `LIVE · Gemini key detected` pill.
- The **Overview** page loads with the six headline metrics.
- The **Four key insights** tabs each render a chart.
- The **Anomaly detection** page shows the 140-record cohort.
- The **Assisted review (live)** page:
  - Pick "I002316 — the $125K honorarium" from the dropdown
  - Click "Run assisted review"
  - You should see the workflow complete in ~5-10 seconds with real Gemini
    findings on each of the four documents and a written reviewer brief.

If the LLM call fails (rate limit, key error, etc.) the app falls back to
deterministic stub output and labels it clearly — so the demo never crashes
in front of an interviewer.

---

## Optional — local testing before deploying

```bash
# From the case-study folder:
pip install -r requirements.txt
$env:GEMINI_API_KEY = "AIzaSy..."   # PowerShell
# or: export GEMINI_API_KEY="AIzaSy..." on bash
python -m streamlit run streamlit_app.py
```

The app will open in your browser on http://localhost:8501.

---

## Cost / quota considerations

- **Streamlit Cloud free tier:** unlimited public apps, 1 GB RAM per app,
  sleeps after a week of inactivity (wakes up in ~30 seconds when accessed).
- **Gemini free tier:** ~15 requests/minute, ~1,500/day on `gemini-2.5-flash`.
  Each "Run assisted review" makes up to 5 LLM calls (one per document plus
  the reviewer brief). A 60-case demo per day stays comfortably inside the
  free tier.

Both are zero cost. You will not be charged.

---

## What to put in the deck

Use a single line on the Phase 3 slide. Something like:

> *Try it live: **anjali-risk-case.streamlit.app**  ·  pick any interaction_id and watch the workflow run end-to-end.*

Format the URL as a hyperlink so the interviewer can click it directly from
the deck. Keep the URL short — the muscle memory for "casenamehere.streamlit.app"
is the cleanest version.
