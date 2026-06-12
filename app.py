# =============================================================================
# app.py - Streamlit Frontend for the ML Resume Screening System
#
# Entry point: run with   streamlit run app.py
#
# Pipeline wired here (in order):
#   UploadedFile -> parser -> preprocess -> extractor
#                                       -> embedder
#   JD text      -> preprocess -> extractor
#                             -> embedder
#   scorer (batch) -> ranker -> display / CSV export
# =============================================================================

import io
import logging
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path fix: ensure project root is on sys.path so all package imports work
# regardless of how Streamlit is launched.
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# Backend imports
# ---------------------------------------------------------------------------
from core.parser      import extract_text
from core.preprocess  import clean_text
from core.extractor   import extract_skills
from core.embedder    import encode
from core.scorer      import compute_scores
from core.ranker      import rank_candidates
from utils.exporter   import to_csv_bytes, get_export_filename
from config.settings  import MIN_SCORE_THRESHOLD

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — professional dark theme with accent colours
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Background */
    .stApp {
        background: #0a0a0a;
        color: #d4d4d4;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #1a1a1a;
        border-right: 1px solid #2a2a2a;
    }

    /* Cards and containers */
    .metric-card {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card h2 { font-size: 1.8rem; font-weight: 600; margin: 0; color: #e4e4e4; }
    .metric-card p  { font-size: 0.8rem; color: #888888; margin: 0.3rem 0 0; }

    /* Score badge */
    .score-chip {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .chip-green  { background: #2a3a2a; color: #8ab28a; border: 1px solid #3a4a3a; }
    .chip-yellow { background: #3a3a2a; color: #b8a878; border: 1px solid #4a4a3a; }
    .chip-red    { background: #3a2a2a; color: #b87878; border: 1px solid #4a3a3a; }

    /* Buttons */
    .stButton > button {
        background: #2a2a2a;
        color: #d4d4d4;
        border: 1px solid #3a3a3a;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        font-size: 0.9rem;
        transition: background 0.2s;
    }
    .stButton > button:hover { background: #3a3a3a; }

    /* Progress bar */
    .stProgress > div > div > div { background: #4a4a4a; }

    /* Expander */
    .streamlit-expanderHeader {
        background: #1a1a1a !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        color: #d4d4d4 !important;
    }

    /* Section headers */
    .section-header {
        font-size: 1rem;
        font-weight: 600;
        color: #d4d4d4;
        letter-spacing: 0.02em;
        text-transform: none;
        margin: 1.2rem 0 0.5rem;
        border-bottom: 1px solid #2a2a2a;
        padding-bottom: 0.4rem;
    }

    /* Table tweaks */
    .dataframe { font-size: 0.85rem !important; }
    thead tr th { background: #1a1a1a !important; border-bottom: 1px solid #2a2a2a !important; }

    /* Skill tag */
    .skill-tag {
        display: inline-block;
        background: #1a1a1a;
        color: #a0a0a0;
        border: 1px solid #2a2a2a;
        border-radius: 4px;
        padding: 0.12rem 0.5rem;
        font-size: 0.75rem;
        margin: 0.12rem 0.15rem;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# Session state initialisation
# ===========================================================================
def _init_state() -> None:
    defaults = {
        "ranked_df":    None,   # Final ranked DataFrame
        "jd_skills":    set(),  # Skills extracted from job description
        "processed":    False,  # Whether analysis has been run
        "error_files":  [],     # PDFs that failed to parse
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()


# ===========================================================================
# Sidebar — configuration & info
# ===========================================================================
with st.sidebar:
    st.markdown("## Configuration")
    st.markdown("---")

    score_threshold = st.slider(
        "Shortlist Threshold (%)",
        min_value=0, max_value=100,
        value=int(MIN_SCORE_THRESHOLD),
        step=5,
        help="Candidates scoring above this are marked Shortlisted.",
    )

    st.markdown("---")
    st.markdown("### Scoring Weights")
    st.markdown(
        """
        | Component | Weight |
        |-----------|--------|
        | Semantic Similarity | **70%** |
        | Skill Match | **30%** |
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### Model")
    st.code("all-MiniLM-L6-v2", language=None)
    st.caption("384-dim - HuggingFace Transformers")

    st.markdown("---")
    st.markdown("### About")
    st.caption(
        "ML-based resume screening system.\n"
        "Ranks candidates using semantic similarity and skill matching."
    )


# ===========================================================================
# Header
# ===========================================================================
st.markdown(
    """
    <div style='text-align:center; padding: 1.5rem 0 0.5rem;'>
        <h1 style='font-size:2.4rem; font-weight:600; color:#d4d4d4; margin-bottom:0.3rem;'>
            AI Resume Screener
        </h1>
        <p style='color:#888888; font-size:0.95rem; margin:0;'>
            Upload resumes - Paste a job description - Rank candidates
        </p>
    </div>
    <hr style='border:none; border-top:1px solid #2a2a2a; margin:1rem 0 1.5rem;'>
    """,
    unsafe_allow_html=True,
)

# ===========================================================================
# Step 1 & 2 — Input section (two columns)
# ===========================================================================
col_upload, col_jd = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown('<p class="section-header">Step 1 - Upload Resumes</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload PDF resumes (multiple allowed)",
        type=["pdf"],
        accept_multiple_files=True,
        key="resume_uploader",
        help="Only PDF files are accepted. Max 10 MB per file.",
    )
    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) ready")
        for f in uploaded_files:
            size_kb = round(f.size / 1024, 1)
            st.markdown(
                f"<small style='color:#888888;'>{f.name} - {size_kb} KB</small>",
                unsafe_allow_html=True,
            )

with col_jd:
    st.markdown('<p class="section-header">Step 2 - Job Description</p>', unsafe_allow_html=True)
    job_description = st.text_area(
        "Paste the full job description here",
        height=230,
        placeholder=(
            "Example:\n\n"
            "We are looking for a Python developer with experience in Machine Learning,\n"
            "NLP, and SQL. Knowledge of TensorFlow and scikit-learn is a plus...\n"
        ),
        key="job_desc_input",
    )
    jd_word_count = len(job_description.split()) if job_description.strip() else 0
    if jd_word_count:
        st.caption(f"{jd_word_count} words entered")

st.markdown("<br>", unsafe_allow_html=True)

# ===========================================================================
# Step 3 — Process button
# ===========================================================================
_, btn_col, _ = st.columns([2, 1, 2])
with btn_col:
    analyze_clicked = st.button("Analyze Resumes", use_container_width=True)


# ===========================================================================
# Input validation helpers
# ===========================================================================
def _validate_inputs() -> bool:
    """Return True if inputs are valid; else show an error and return False."""
    if not uploaded_files:
        st.error("Please upload at least one PDF resume before analyzing.")
        return False
    if not job_description or not job_description.strip():
        st.error("Job description cannot be empty. Please paste a job description.")
        return False
    if len(job_description.strip()) < 20:
        st.warning("Job description seems very short. Results may be inaccurate.")
    return True


# ===========================================================================
# Core pipeline
# ===========================================================================
def _run_pipeline(files, jd_text: str, threshold: float) -> None:
    """
    Execute the full ML pipeline and store results in st.session_state.

    Steps:
        1. Extract text from each PDF (parser)
        2. Clean and preprocess text (preprocess)
        3. Extract skills from each resume and from JD (extractor)
        4. Encode all texts in one batch (embedder)
        5. Compute scores (scorer)
        6. Rank candidates (ranker)
    """
    progress  = st.progress(0, text="Initialising pipeline...")
    status    = st.status("Processing resumes...", expanded=True)

    candidate_names:    list[str]     = []
    raw_texts:          list[str]     = []
    cleaned_texts:      list[str]     = []
    resume_skills_list: list[set]     = []
    error_files:        list[str]     = []

    total = len(files)

    # Step 1, 2, and 3: Per-resume extraction + preprocessing + skills
    status.write("Extracting text from PDFs...")
    for i, uploaded_file in enumerate(files):
        progress.progress(
            int((i / total) * 40),
            text=f"Parsing {uploaded_file.name} ({i+1}/{total})…",
        )

        try:
            raw = extract_text(uploaded_file)

            if not raw.strip():
                st.warning(
                    f"**{uploaded_file.name}** - No text could be extracted. "
                    "This PDF may be scanned/image-based. Skipping."
                )
                error_files.append(uploaded_file.name)
                continue

            cleaned   = clean_text(raw)
            skills    = extract_skills(cleaned)

            candidate_names.append(uploaded_file.name.replace(".pdf", ""))
            raw_texts.append(raw)
            cleaned_texts.append(cleaned)
            resume_skills_list.append(skills)

            status.write(f"   {uploaded_file.name} - {len(skills)} skill(s) found")

        except Exception as exc:
            st.warning(f"**{uploaded_file.name}** - Failed to process: {exc}")
            error_files.append(uploaded_file.name)
            logger.error(f"Pipeline error on {uploaded_file.name}: {exc}", exc_info=True)

    if not candidate_names:
        progress.empty()
        status.update(label="No valid resumes to process.", state="error")
        st.error(
            "No valid text could be extracted from the uploaded PDFs. "
            "Ensure the files are text-based PDFs (not scanned images)."
        )
        return

    # Step 2b: JD preprocessing + skills
    progress.progress(42, text="Processing job description...")
    status.write("Processing job description...")
    cleaned_jd = clean_text(jd_text)
    jd_skills  = extract_skills(cleaned_jd)
    status.write(f"   JD - {len(jd_skills)} skill(s) identified")

    # Step 4: Batch embedding
    progress.progress(55, text="Generating embeddings (this may take a moment)...")
    status.write("Generating semantic embeddings...")

    all_texts      = cleaned_texts + [cleaned_jd]
    all_embeddings = encode(all_texts)

    resume_embeddings = all_embeddings[:-1]          # shape (n, 384)
    jd_embedding      = all_embeddings[-1:]          # shape (1, 384)

    status.write(f"   Embeddings ready - shape {resume_embeddings.shape}")

    # Step 5: Scoring
    progress.progress(80, text="Computing match scores...")
    status.write("Scoring candidates...")

    score_results = compute_scores(
        resume_embeddings  = resume_embeddings,
        jd_embedding       = jd_embedding,
        resume_skills_list = resume_skills_list,
        jd_skills          = jd_skills,
        candidate_names    = candidate_names,
    )

    # Step 6: Ranking
    progress.progress(95, text="Ranking candidates...")
    status.write("Ranking candidates...")

    ranked_df = rank_candidates(score_results)

    # Re-apply sidebar threshold (overrides config default)
    ranked_df["Status"] = ranked_df["Final Score (%)"].apply(
        lambda s: "Shortlisted" if s >= threshold else "Rejected"
    )

    # Done
    progress.progress(100, text="Complete!")
    status.update(label="Analysis complete!", state="complete", expanded=False)

    # Persist to session state
    st.session_state["ranked_df"]   = ranked_df
    st.session_state["jd_skills"]   = jd_skills
    st.session_state["processed"]   = True
    st.session_state["error_files"] = error_files


# ===========================================================================
# Run pipeline on button click
# ===========================================================================
if analyze_clicked:
    if _validate_inputs():
        st.session_state["processed"] = False   # reset before new run
        _run_pipeline(uploaded_files, job_description, float(score_threshold))


# ===========================================================================
# Step 4 — Results display
# ===========================================================================
if st.session_state["processed"] and st.session_state["ranked_df"] is not None:

    ranked_df: pd.DataFrame = st.session_state["ranked_df"]
    jd_skills: set          = st.session_state["jd_skills"]

    st.markdown("---")
    st.markdown('<p class="section-header">🏆 Step 4 — Ranked Candidates</p>', unsafe_allow_html=True)

    # ── Summary metrics ────────────────────────────────────────────────────
    total_cands    = len(ranked_df)
    shortlisted    = int((ranked_df["Status"] == "✅ Shortlisted").sum())
    rejected       = total_cands - shortlisted
    avg_score      = round(ranked_df["Final Score (%)"].mean(), 1)
    top_score      = round(ranked_df["Final Score (%)"].max(), 1)

    m1, m2, m3, m4, m5 = st.columns(5)
    metrics = [
        (m1, "📋 Total",       str(total_cands), "resumes processed"),
        (m2, "✅ Shortlisted", str(shortlisted),  "above threshold"),
        (m3, "❌ Rejected",    str(rejected),      "below threshold"),
        (m4, "📈 Avg Score",   f"{avg_score}%",   "across all resumes"),
        (m5, "🥇 Top Score",   f"{top_score}%",   "best match"),
    ]
    for col, icon_label, value, sub in metrics:
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <p>{icon_label}</p>
                    <h2>{value}</h2>
                    <p>{sub}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── JD Skills identified ───────────────────────────────────────────────
    if jd_skills:
        st.markdown("**🛠️ Skills detected in Job Description:**")
        tags_html = "".join(
            f'<span class="skill-tag">{s}</span>' for s in sorted(jd_skills)
        )
        st.markdown(tags_html, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Search & Filter controls ───────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([2, 1.5, 1.5])

    with ctrl1:
        search_query = st.text_input(
            "🔍 Search Candidate",
            placeholder="Type a name to filter…",
            key="search_input",
        )
    with ctrl2:
        min_filter = st.slider(
            "Minimum Score (%)",
            min_value=0, max_value=100,
            value=0, step=5,
            key="score_filter",
        )
    with ctrl3:
        status_filter = st.selectbox(
            "Status Filter",
            options=["All", "✅ Shortlisted", "❌ Rejected"],
            key="status_filter",
        )

    # Apply filters
    filtered_df = ranked_df.copy()
    if search_query.strip():
        filtered_df = filtered_df[
            filtered_df["Candidate"].str.contains(search_query, case=False, na=False)
        ]
    if min_filter > 0:
        filtered_df = filtered_df[filtered_df["Final Score (%)"] >= min_filter]
    if status_filter != "All":
        filtered_df = filtered_df[filtered_df["Status"] == status_filter]

    st.caption(f"Showing **{len(filtered_df)}** of **{total_cands}** candidate(s)")

    # ── Main results table ─────────────────────────────────────────────────
    display_cols = [
        "Rank", "Candidate", "Final Score (%)",
        "Semantic Similarity", "Skill Match (%)", "Status",
    ]
    st.dataframe(
        filtered_df[display_cols].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Final Score (%)": st.column_config.ProgressColumn(
                "Final Score (%)",
                format="%.1f%%",
                min_value=0, max_value=100,
            ),
            "Semantic Similarity": st.column_config.NumberColumn(
                "Semantic Sim (%)", format="%.1f"
            ),
            "Skill Match (%)": st.column_config.NumberColumn(
                "Skill Match (%)", format="%.1f"
            ),
        },
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Per-candidate detailed breakdown ──────────────────────────────────
    st.markdown('<p class="section-header">🔍 Candidate Details</p>', unsafe_allow_html=True)

    for _, row in filtered_df.iterrows():
        score  = row["Final Score (%)"]
        status = row["Status"]

        # Colour chip for score
        if score >= 75:
            chip_cls, chip_label = "chip-green",  f"🟢 {score}%"
        elif score >= 50:
            chip_cls, chip_label = "chip-yellow", f"🟡 {score}%"
        else:
            chip_cls, chip_label = "chip-red",    f"🔴 {score}%"

        rank_prefix = "🥇" if row["Rank"] == 1 else "#" + str(int(row["Rank"]))
        expander_title = (
            f"{rank_prefix}  {row['Candidate']}  —  {score}%  {status}"
        )

        with st.expander(expander_title):
            d1, d2, d3 = st.columns(3)
            d1.metric("Final Score",        f"{score}%")
            d2.metric("Semantic Similarity", f"{row['Semantic Similarity']}%")
            d3.metric("Skill Match",         f"{row['Skill Match (%)']}%")

            matched = row["Matched Skills"]
            missing = row["Missing Skills"]

            col_m, col_x = st.columns(2)
            with col_m:
                st.markdown("**✅ Matched Skills:**")
                if matched and matched != "None":
                    tags = "".join(
                        f'<span class="skill-tag">{s.strip()}</span>'
                        for s in matched.split(",")
                    )
                    st.markdown(tags, unsafe_allow_html=True)
                else:
                    st.caption("No matched skills found.")

            with col_x:
                st.markdown("**❌ Missing Skills from JD:**")
                if missing and missing != "None":
                    tags = "".join(
                        f'<span class="skill-tag" style="color:#f87171;border-color:rgba(248,113,113,0.35);">'
                        f'{s.strip()}</span>'
                        for s in missing.split(",")
                    )
                    st.markdown(tags, unsafe_allow_html=True)
                else:
                    st.caption("All JD skills matched! 🎉")

    # ── CSV Download ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">📥 Export Results</p>', unsafe_allow_html=True)

    dl_col1, dl_col2 = st.columns([2, 1])
    with dl_col1:
        st.markdown(
            "Download the full ranked results as a CSV file. "
            "The export includes all scores, matched skills, missing skills, and shortlist status."
        )
    with dl_col2:
        csv_bytes = to_csv_bytes(ranked_df)
        st.download_button(
            label="⬇️ Download CSV Report",
            data=csv_bytes,
            file_name=get_export_filename(),
            mime="text/csv",
            use_container_width=True,
            disabled=(len(csv_bytes) == 0),
        )

    # Error file summary
    if st.session_state["error_files"]:
        st.markdown("---")
        st.markdown("**Files that could not be processed:**")
        for ef in st.session_state["error_files"]:
            st.markdown(f"- `{ef}`")

# ===========================================================================
# Empty state - shown before first analysis
# ===========================================================================
elif not st.session_state["processed"]:
    st.markdown(
        """
        <div style='text-align:center; padding:3rem 0; color:#666666;'>
            <div style='font-size:3rem; margin-bottom:1rem;'>R</div>
            <h3 style='color:#888888; font-weight:500;'>
                Upload resumes and paste a job description to get started.
            </h3>
            <p style='color:#666666; font-size:0.9rem;'>
                Results will appear here after you click <strong>Analyze Resumes</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
