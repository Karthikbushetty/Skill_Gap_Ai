# ==========================================
# SkillGapAI - Milestone 2: Skill Extraction using NLP
# Premium Dark Dashboard UI
# ==========================================

import streamlit as st
import spacy
import re
import matplotlib.pyplot as plt

# ------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------
st.set_page_config(
    page_title="SkillGapAI - Milestone 2",
    layout="wide",
    page_icon="🧠"
)

# ------------------------------------------
# GLOBAL STYLES (PREMIUM DARK UI)
# ------------------------------------------
st.markdown(
    """
    <style>
    body {
        background-color: #020617;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    /* Remove top padding */
    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    /* Top navigation bar */
    .top-nav {
        width: 100%;
        padding: 10px 18px;
        border-radius: 999px;
        background: linear-gradient(90deg, #020617, #020617, #020617);
        border: 1px solid rgba(148, 163, 184, 0.4);
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 18px;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.9);
    }

    .top-nav-left {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .top-logo {
        width: 34px;
        height: 34px;
        border-radius: 11px;
        background: linear-gradient(135deg, #06b6d4, #3b82f6);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        color: white;
        font-size: 18px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.8);
    }
    .top-title-main {
        color: #f9fafb;
        font-weight: 700;
        font-size: 18px;
    }
    .top-title-sub {
        color: #9ca3af;
        font-size: 12px;
    }
    .top-tag {
        padding: 6px 12px;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.55);
        color: #e5e7eb;
        font-size: 12px;
        background: rgba(15, 23, 42, 0.85);
    }

    /* Hero header */
    .hero-card {
        margin-top: 6px;
        margin-bottom: 20px;
        padding: 16px 18px;
        border-radius: 18px;
        background: radial-gradient(circle at top left, #0f172a, #020617);
        border: 1px solid #1f2937;
        box-shadow: 0 20px 50px rgba(15, 23, 42, 0.95);
    }
    .hero-title {
        font-size: 26px;
        color: #f9fafb;
        font-weight: 750;
        margin-bottom: 6px;
    }
    .hero-text {
        color: #9ca3af;
        font-size: 14px;
    }

    /* Glass cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.95);
        border-radius: 18px;
        padding: 18px 18px 20px 18px;
        border: 1px solid rgba(30, 64, 175, 0.6);
        box-shadow: 0 20px 45px rgba(15, 23, 42, 0.9);
        backdrop-filter: blur(8px);
        margin-bottom: 18px;
    }

    .glass-title {
        font-size: 20px;
        font-weight: 700;
        color: #f9fafb;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .glass-subtitle {
        font-size: 13px;
        color: #9ca3af;
        margin-bottom: 8px;
    }

    /* Textareas */
    .stTextArea textarea {
        background: #020617 !important;
        border-radius: 12px !important;
        border: 1px solid #1f2937 !important;
        color: #e5e7eb !important;
        font-size: 13px !important;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-bottom: 16px;
    }
    .metric-card {
        flex: 1 1 160px;
        min-width: 150px;
        background: radial-gradient(circle at top left, #0b1220, #020617);
        border-radius: 14px;
        padding: 10px 12px;
        border: 1px solid #1f2937;
        box-shadow: 0 14px 35px rgba(15, 23, 42, 0.95);
    }
    .metric-label {
        font-size: 12px;
        color: #9ca3af;
    }
    .metric-value {
        font-size: 20px;
        font-weight: 700;
        color: #fefce8;
    }
    .metric-extra {
        font-size: 11px;
        color: #6b7280;
    }

    /* Skill chips */
    .skill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    }
    .skill-chip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 5px 12px;
        border-radius: 999px;
        background: #d9f3ec;
        color: #064e3b;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid rgba(34, 197, 94, 0.35);
    }
    .skill-chip-soft {
        background: #ede9fe;
        color: #4c1d95;
        border-color: rgba(129, 140, 248, 0.5);
    }

    /* Highlighted text box */
    .highlight-box {
        background: #020617;
        border-radius: 14px;
        padding: 14px;
        border: 1px solid #1f2937;
        color: #e5e7eb;
        font-size: 13px;
        line-height: 1.45;
        max-height: 260px;
        overflow-y: auto;
    }
    .highlight-skill {
        background: #bef264;
        color: #1f2937;
        padding: 1px 4px;
        border-radius: 4px;
        font-weight: 600;
    }

    /* Gap chips */
    .chip-small {
        display:inline-block;
        padding:3px 9px;
        margin:2px 4px 4px 0;
        border-radius:999px;
        font-size:11px;
        font-weight:500;
        border:1px solid rgba(148,163,184,0.7);
        background:rgba(15,23,42,0.95);
        color:#e5e7eb;
    }
    .chip-match { border-color:rgba(34,197,94,0.9); background:rgba(22,101,52,0.9); }
    .chip-missing { border-color:rgba(248,113,113,0.9); background:rgba(127,29,29,0.95); }
    .chip-extra { border-color:rgba(59,130,246,0.9); background:rgba(30,64,175,0.95); }

    .gap-title {
        font-size: 15px;
        font-weight: 600;
        color: #f9fafb;
        margin-bottom: 4px;
    }
    .gap-sub {
        font-size: 12px;
        color:#9ca3af;
        margin-bottom:4px;
    }

    /* Footer */
    .footer {
        margin-top: 22px;
        text-align:center;
        color:#6b7280;
        font-size:12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------
# TOP NAV + HERO
# ------------------------------------------
st.markdown(
    """
    <div class="top-nav">
      <div class="top-nav-left">
        <div class="top-logo">SG</div>
        <div>
          <div class="top-title-main">SkillGapAI Dashboard</div>
          <div class="top-title-sub">Milestone 2 • Skill Extraction using NLP</div>
        </div>
      </div>
      <div class="top-nav-right">
        <span class="top-tag">AI & DS • Project Interface</span>
      </div>
    </div>

    <div class="hero-card">
      <div class="hero-title">🧠 Skill Extraction & Analysis</div>
      <div class="hero-text">
        This module extracts <b>technical</b> and <b>soft skills</b> from both Resume and Job Description,
        compares them, and visualizes the skill coverage and gaps using interactive charts and tags.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------
# LOAD SPACY MODEL
# ------------------------------------------
@st.cache_resource
def load_model():
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_model()

# ------------------------------------------
# SKILL LISTS
# ------------------------------------------
technical_skills = [
    "python", "java", "c++", "sql", "html", "css", "javascript", "react", "node.js",
    "tensorflow", "pytorch", "machine learning", "data analysis", "data visualization",
    "aws", "azure", "gcp", "power bi", "tableau", "django", "flask", "scikit-learn", "nlp"
]

soft_skills = [
    "communication", "leadership", "teamwork", "problem solving", "time management",
    "adaptability", "critical thinking", "creativity", "collaboration", "decision making"
]

# ------------------------------------------
# HELPERS
# ------------------------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return text.lower().strip()

def extract_skills(text: str):
    text_clean = clean_text(text)
    found_tech = [s.title() for s in technical_skills if s in text_clean]
    found_soft = [s.title() for s in soft_skills if s in text_clean]
    return sorted(set(found_tech)), sorted(set(found_soft))

def highlight_text(raw_text: str, skills):
    if not raw_text:
        return ""
    highlighted = raw_text
    # longest first so substrings don't break
    for s in sorted(skills, key=len, reverse=True):
        pattern = re.compile(re.escape(s), re.IGNORECASE)
        highlighted = pattern.sub(lambda m: f"<span class='highlight-skill'>{m.group(0)}</span>", highlighted)
    return highlighted.replace("\n", "<br>")

def confidence_scores(skills):
    n = len(skills)
    if n == 0:
        return {}
    start = 96
    step = 8 / max(n - 1, 1)
    scores = {}
    for i, s in enumerate(skills):
        scores[s] = max(70, round(start - i * step))
    return scores

def compute_gap(resume_tech, resume_soft, jd_tech, jd_soft):
    r_all = set(resume_tech + resume_soft)
    j_all = set(jd_tech + jd_soft)
    common = sorted(r_all & j_all)
    missing = sorted(j_all - r_all)   # present in JD, not in resume
    extra = sorted(r_all - j_all)     # present in resume, not in JD
    if len(j_all) > 0:
        match_pct = round((len(common) / len(j_all)) * 100)
    else:
        match_pct = 0
    return common, missing, extra, match_pct, len(r_all), len(j_all)

def render_skill_chips(skills, soft=False):
    if not skills:
        return "<span style='font-size:12px;color:#9ca3af;'>None found.</span>"
    cls = "skill-chip-soft" if soft else "skill-chip"
    html = "<div class='skill-row'>"
    for s in skills:
        html += f"<span class='{cls}'>{s}</span>"
    html += "</div>"
    return html

# ------------------------------------------
# INPUT SECTION
# ------------------------------------------
col_r, col_j = st.columns(2)

with col_r:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='glass-title'>👨‍💻 Resume Text</div>", unsafe_allow_html=True)
    st.markdown("<div class='glass-subtitle'>Paste the candidate's resume text here.</div>", unsafe_allow_html=True)
    resume_text = st.text_area(" ", value="", height=210, key="resume_text")
    st.markdown("</div>", unsafe_allow_html=True)

with col_j:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='glass-title'>🏢 Job Description Text</div>", unsafe_allow_html=True)
    st.markdown("<div class='glass-subtitle'>Paste the job description or requirements here.</div>", unsafe_allow_html=True)
    jd_text = st.text_area("  ", value="", height=210, key="jd_text")
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------
# PROCESSING
# ------------------------------------------
if resume_text or jd_text:
    # Extract skills
    tech_resume, soft_resume = extract_skills(resume_text) if resume_text else ([], [])
    tech_jd, soft_jd = extract_skills(jd_text) if jd_text else ([], [])

    common_skills, missing_skills, extra_skills, match_pct, total_r, total_j = compute_gap(
        tech_resume, soft_resume, tech_jd, soft_jd
    )

    # ---------------- METRIC OVERVIEW ----------------
    st.markdown("<div class='glass-title' style='margin-bottom:8px;'>📊 Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='metric-row'>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">Total Resume Skills</div>
          <div class="metric-value">{total_r}</div>
          <div class="metric-extra">Technical + Soft</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">Total JD Skills</div>
          <div class="metric-value">{total_j}</div>
          <div class="metric-extra">Required by Job</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">Resume–JD Match</div>
          <div class="metric-value">{match_pct}%</div>
          <div class="metric-extra">{len(common_skills)} overlapping skills</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">Missing Skills (from JD)</div>
          <div class="metric-value">{len(missing_skills)}</div>
          <div class="metric-extra">Good targets for upskilling</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- SKILL VIEW + DISTRIBUTION ----------------
    left, right = st.columns([1.6, 1.1])

    with left:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='glass-title'>📘 Skill View</div>", unsafe_allow_html=True)

        source = st.radio(
            "View extracted skills from:",
            ["Resume", "Job Description"],
            horizontal=True,
            key="view_source",
        )

        if source == "Resume":
            view_tech, view_soft = tech_resume, soft_resume
            view_text = resume_text
        else:
            view_tech, view_soft = tech_jd, soft_jd
            view_text = jd_text

        conf = confidence_scores(view_tech + view_soft)

        st.markdown("**🏷️ Skill Tags**", unsafe_allow_html=True)
        tag_html = "<div class='skill-row'>"
        for s in view_tech:
            tag_html += f"<span class='skill-chip'>{s} {conf.get(s, 90)}%</span>"
        for s in view_soft:
            tag_html += f"<span class='skill-chip skill-chip-soft'>{s} {conf.get(s, 88)}%</span>"
        tag_html += "</div>"
        if not (view_tech or view_soft):
            tag_html = "<span style='font-size:12px;color:#9ca3af;'>No configured skills detected in the text.</span>"
        st.markdown(tag_html, unsafe_allow_html=True)

        st.markdown("<br><b>🖍 Highlighted Text</b>", unsafe_allow_html=True)
        highlighted_html = highlight_text(view_text, view_tech + view_soft)
        if highlighted_html:
            st.markdown(f"<div class='highlight-box'>{highlighted_html}</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class='highlight-box'>No text available yet. Paste content above to see highlighted skills.</div>",
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='glass-title'>📊 Skill Distribution</div>", unsafe_allow_html=True)

        if source == "Resume":
            t_count, s_count = len(tech_resume), len(soft_resume)
        else:
            t_count, s_count = len(tech_jd), len(soft_jd)

        total_sel = t_count + s_count
        if total_sel > 0:
            fig, ax = plt.subplots(figsize=(3.4, 3.4))
            sizes = [t_count, s_count]
            labels = ["Technical", "Soft"]
            colors = ["#1D4ED8", "#22C55E"]

            wedges, texts, autotexts = ax.pie(
                sizes,
                labels=labels,
                autopct="%1.1f%%",
                startangle=90,
                colors=colors,
                wedgeprops=dict(width=0.4, edgecolor="white"),
            )
            ax.axis("equal")
            st.pyplot(fig)
        else:
            st.info("No skills to display in the chart yet.")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- GAP ANALYSIS ----------------
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='glass-title'>🧩 Resume vs JD Gap Analysis</div>", unsafe_allow_html=True)

    gap_col1, gap_col2, gap_col3 = st.columns(3)

    with gap_col1:
        st.markdown("<div class='gap-title'>✅ Matching Skills</div>", unsafe_allow_html=True)
        st.markdown("<div class='gap-sub'>Skills present in both Resume and JD.</div>", unsafe_allow_html=True)
        if common_skills:
            html = ""
            for s in common_skills:
                html += f"<span class='chip-small chip-match'>{s}</span>"
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.markdown("<span style='font-size:12px;color:#9ca3af;'>No overlapping skills detected.</span>", unsafe_allow_html=True)

    with gap_col2:
        st.markdown("<div class='gap-title'>⚠️ Missing in Resume</div>", unsafe_allow_html=True)
        st.markdown("<div class='gap-sub'>Required by JD but not present in Resume.</div>", unsafe_allow_html=True)
        if missing_skills:
            html = ""
            for s in missing_skills:
                html += f"<span class='chip-small chip-missing'>{s}</span>"
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.markdown("<span style='font-size:12px;color:#9ca3af;'>No missing skills (for current lists).</span>", unsafe_allow_html=True)

    with gap_col3:
        st.markdown("<div class='gap-title'>📎 Extra in Resume</div>", unsafe_allow_html=True)
        st.markdown("<div class='gap-sub'>Skills in Resume that are not explicitly asked in JD.</div>", unsafe_allow_html=True)
        if extra_skills:
            html = ""
            for s in extra_skills:
                html += f"<span class='chip-small chip-extra'>{s}</span>"
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.markdown("<span style='font-size:12px;color:#9ca3af;'>No extra skills beyond JD.</span>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("Paste resume and/or job description text above to start skill extraction and analysis.")

# ------------------------------------------
# FOOTER
# ------------------------------------------
st.markdown(
    """
    <div class="footer">
      Milestone 2 • Skill Extraction using NLP • SkillGapAI Project • Developed by Suriya Varshan
    </div>
    """,
    unsafe_allow_html=True,
)

