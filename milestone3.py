# ==========================================
# SkillGapAI - Milestone 3: Skill Gap Analysis & Similarity Matching
# Weeks 5–6
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer, util

# ------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------
st.set_page_config(
    page_title="SkillGapAI - Milestone 3",
    layout="wide",
    page_icon="🧠"
)

# ------------------------------------------
# CUSTOM STYLING (PREMIUM UI)
# ------------------------------------------
st.markdown(
    """
    <style>
    /* Overall app background */
    .stApp {
        background: radial-gradient(circle at top left, #1f2937, #020617);
        color: #f9fafb;
        font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Header card */
    .header-card {
        background: linear-gradient(90deg, #5B2C6F, #8E44AD);
        padding: 18px 24px;
        border-radius: 14px;
        color: white;
        box-shadow: 0 8px 20px rgba(0,0,0,0.35);
        margin-bottom: 10px;
    }

    .header-title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .header-subtitle {
        font-size: 15px;
        opacity: 0.9;
    }

    /* Section titles */
    .section-title {
        font-size: 22px;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 8px;
        color: #e5e7eb;
    }

    /* Input cards */
    .card {
        background: rgba(15,23,42,0.95);
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.35);
        border: 1px solid rgba(148,163,184,0.25);
    }

    .card h3 {
        font-size: 18px;
        margin-bottom: 10px;
        color: #e5e7eb;
    }

    /* Dataframe tweaks */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 18px rgba(0,0,0,0.4);
    }

    /* Metric styling */
    [data-testid="metric-container"] {
        background: linear-gradient(145deg, #020617, #0b1120);
        padding: 14px;
        border-radius: 14px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.4);
        border: 1px solid rgba(148,163,184,0.3);
    }

    /* Button style */
    .stButton > button {
        background: linear-gradient(90deg, #22c55e, #16a34a);
        color: white;
        border-radius: 999px;
        padding: 10px 26px;
        border: none;
        font-weight: 600;
        box-shadow: 0 6px 15px rgba(22,163,74,0.45);
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #16a34a, #15803d);
        transform: translateY(-1px);
    }

    /* Tabs styling */
    button[role="tab"] {
        border-radius: 999px !important;
        padding: 6px 18px !important;
        margin-right: 4px;
    }

    /* Missing skills pill style */
    .skill-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: rgba(248,113,113,0.15);
        border: 1px solid rgba(248,113,113,0.6);
        color: #fecaca;
        font-size: 13px;
        margin: 2px 4px;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        color: #9ca3af;
        font-size: 13px;
        margin-top: 8px;
        padding-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------
# HEADER
# ------------------------------------------
st.markdown(
    """
    <div class="header-card">
        <div class="header-title">
            🧠 SkillGapAI – Milestone 3: Skill Gap Analysis & Similarity Matching
        </div>
        <div class="header-subtitle">
            Compare candidate and job skills using BERT embeddings to identify <b>matched</b>,
            <b>partially matched</b> and <b>missing</b> skills, with interactive visual insights.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "Tip: Paste the extracted skills from your Smart Resume and Job Description to analyse the alignment."
)

# ------------------------------------------
# LOAD SENTENCE TRANSFORMER MODEL
# ------------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ------------------------------------------
# SKILL INPUT SECTION
# ------------------------------------------
st.markdown("<div class='section-title'>🧾 Input Skill Sets</div>", unsafe_allow_html=True)

input_col1, input_col2 = st.columns(2)

with input_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### 👨‍💻 Resume Skills")
    resume_skills_input = st.text_area(
        "Enter resume skills (comma-separated):",
        "Python, SQL, Machine Learning, Tableau",
        height=120,
        label_visibility="collapsed"
    )
    st.caption("Example: `Python, SQL, Machine Learning, Tableau, Communication`")
    st.markdown("</div>", unsafe_allow_html=True)

with input_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### 🏢 Job Description Skills")
    jd_skills_input = st.text_area(
        "Enter job-required skills (comma-separated):",
        "Python, Data Visualization, Deep Learning, Communication, AWS",
        height=120,
        label_visibility="collapsed"
    )
    st.caption("Example: `Python, Deep Learning, Data Visualization, AWS, Problem Solving`")
    st.markdown("</div>", unsafe_allow_html=True)

# Convert inputs to lists
resume_skills = [s.strip() for s in resume_skills_input.split(",") if s.strip()]
jd_skills = [s.strip() for s in jd_skills_input.split(",") if s.strip()]

# Run button
center_btn_col = st.columns([1, 1, 1])[1]
with center_btn_col:
    run_analysis = st.button("🚀 Run Skill Gap Analysis")

# ------------------------------------------
# SIDEBAR – PARAMETERS / INFO
# ------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Analysis Settings")
    st.write("These thresholds define how skills are classified based on similarity:")
    st.code(
        "Matched  ≥ 0.80\nPartial  ∈ [0.50, 0.80)\nMissing  < 0.50",
        language="text"
    )
    st.markdown("---")
    st.markdown("#### 📌 Notes")
    st.markdown(
        "- Uses `all-MiniLM-L6-v2` sentence embeddings\n"
        "- Cosine similarity for skill matching\n"
        "- Works best when skills are meaningful phrases"
    )

# ------------------------------------------
# SKILL GAP ANALYSIS
# ------------------------------------------
if run_analysis and resume_skills and jd_skills:

    st.markdown("<div class='section-title'>🔎 Skill Gap Analysis Results</div>", unsafe_allow_html=True)

    # Encode skills
    resume_embeddings = model.encode(resume_skills, convert_to_tensor=True)
    jd_embeddings = model.encode(jd_skills, convert_to_tensor=True)

    # Cosine similarity
    similarity_matrix = util.cos_sim(resume_embeddings, jd_embeddings)
    similarity_matrix = similarity_matrix.cpu().numpy()

    sim_df = pd.DataFrame(
        similarity_matrix,
        index=resume_skills,
        columns=jd_skills
    )

    # --------------------------------------
    # Skill Classification
    # --------------------------------------
    matched_skills = []
    partial_skills = []
    missing_skills = []

    for skill in jd_skills:
        max_score = sim_df[skill].max()
        if max_score >= 0.80:
            matched_skills.append(skill)
        elif 0.50 <= max_score < 0.80:
            partial_skills.append(skill)
        else:
            missing_skills.append(skill)

    total_skills = len(jd_skills)
    overall_match = round(
        ((len(matched_skills) + 0.5 * len(partial_skills)) / total_skills) * 100, 2
    )

    # --------------------------------------
    # TOP SUMMARY ROW
    # --------------------------------------
    top_summary_col1, top_summary_col2, top_summary_col3, top_summary_col4 = st.columns([1, 1, 1, 1.2])

    with top_summary_col1:
        st.metric("✅ Matched", len(matched_skills))
    with top_summary_col2:
        st.metric("🟡 Partial", len(partial_skills))
    with top_summary_col3:
        st.metric("❌ Missing", len(missing_skills))
    with top_summary_col4:
        st.metric("📈 Overall Match Score", f"{overall_match} %")

    st.markdown("")

    # --------------------------------------
    # TABS FOR VISUALIZATION & DETAILS
    # --------------------------------------
    tab1, tab2, tab3 = st.tabs(["📊 Overview & Pie", "🧩 Similarity Matrix", "📋 Detailed Table"])

    # --- TAB 1: OVERVIEW + PIE ---
    with tab1:
        ov_col1, ov_col2 = st.columns([1.3, 1])

        with ov_col1:
            st.markdown("#### 🎯 Match Summary Insights")
            if overall_match >= 80:
                st.success("Great alignment! The resume is highly compatible with the job skill requirements.")
            elif overall_match >= 50:
                st.warning("Moderate alignment. Some key skills are matched, but improvement is needed.")
            else:
                st.error("Low alignment. Many important job skills are missing from the resume.")

            with st.expander("🔍 View Matched & Partial Skills"):
                st.markdown("**✅ Matched Job Skills:**")
                if matched_skills:
                    st.write(", ".join(matched_skills))
                else:
                    st.write("No strongly matched job skills.")

                st.markdown("**🟡 Partially Matched Job Skills:**")
                if partial_skills:
                    st.write(", ".join(partial_skills))
                else:
                    st.write("No partial matches.")

        with ov_col2:
            st.markdown("#### 🧮 Distribution of Skill Categories")
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            ax2.pie(
                [len(matched_skills), len(partial_skills), len(missing_skills)],
                labels=["Matched", "Partial", "Missing"],
                autopct="%1.1f%%",
                startangle=90
            )
            ax2.axis("equal")
            st.pyplot(fig2)

    # --- TAB 2: HEATMAP ---
    with tab2:
        st.markdown("#### 📈 BERT-Based Skill Similarity Heatmap")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(
            sim_df,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            linewidths=0.5
        )
        ax.set_title("Cosine Similarity Between Resume & Job Skills")
        plt.yticks(rotation=0)
        st.pyplot(fig)

    # --- TAB 3: DETAILED TABLE ---
    with tab3:
        st.markdown("#### 📋 Skill-Level Similarity Details")

        detailed_data = []
        for jd_skill in jd_skills:
            best_resume_skill = sim_df[jd_skill].idxmax()
            best_score = sim_df[jd_skill].max()
            detailed_data.append({
                "Job Skill (JD)": jd_skill,
                "Closest Resume Skill": best_resume_skill,
                "Similarity Score (%)": round(best_score * 100, 2),
            })

        st.dataframe(pd.DataFrame(detailed_data), use_container_width=True)

    # --------------------------------------
    # MISSING SKILLS SECTION
    # --------------------------------------
    st.markdown("<div class='section-title'>🚨 Missing Skills (Priority to Improve)</div>", unsafe_allow_html=True)

    if missing_skills:
        missing_html = "<div>"
        for s in missing_skills:
            missing_html += f"<span class='skill-pill'>🚫 {s}</span>"
        missing_html += "</div>"
        st.markdown(missing_html, unsafe_allow_html=True)
        st.caption("These job-required skills are not strongly reflected in the resume.")
    else:
        st.success("🎯 Excellent! There are no missing job skills based on the current thresholds.")

elif run_analysis:
    st.warning("Please make sure both Resume and Job Description skills are entered correctly.")

# ------------------------------------------
# FOOTER
# ------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div class="footer-text">
        Milestone 3 • Skill Gap Analysis & Similarity Matching • SkillGapAI Project<br>
        Developed by <b>Suriya Varshan</b>
    </div>
    """,
    unsafe_allow_html=True
)
