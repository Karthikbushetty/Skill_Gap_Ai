# ==========================================
# SkillGapAI - Milestone 1: Resume & JD Parser
# ==========================================

import streamlit as st
import PyPDF2
import docx2txt
import re

# ------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------
st.set_page_config(page_title="SkillGapAI - Milestone 1", layout="wide")

st.markdown(
    """
    <h2 style='color:white; background-color:#117A65; padding:15px; border-radius:10px'>
        📄 SkillGapAI - Milestone 1: Resume & Job Description Parser
    </h2>
    <p><b>Objective:</b> Upload resumes or job descriptions, extract text, clean it,
    and preview the processed output for further NLP stages.</p>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------
# CLEANING FUNCTION
# ------------------------------------------
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\n", " ")
    return text.strip()

# ------------------------------------------
# FILE PARSING FUNCTION
# ------------------------------------------
def parse_file(uploaded_file):
    file_type = uploaded_file.name.split(".")[-1].lower()

    try:
        if file_type == "pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + " "
            return clean_text(text)

        elif file_type == "docx":
            text = docx2txt.process(uploaded_file)
            return clean_text(text)

        elif file_type == "txt":
            return clean_text(uploaded_file.read().decode("utf-8"))

        else:
            st.error("❌ Unsupported file format.")
            return ""
    except:
        st.error("❌ Error reading file.")
        return ""

# ------------------------------------------
# LAYOUT: UPLOAD + PREVIEW
# ------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📤 Upload Resume / JD File")
    uploaded = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])

    if uploaded:
        parsed_text = parse_file(uploaded)
    else:
        parsed_text = ""

with col2:
    st.markdown("### 👀 Extracted Text Preview")

    if parsed_text:
        st.text_area("Parsed Output", parsed_text[:4000], height=350)

        st.write(f"**Characters:** {len(parsed_text)}")
        st.write(f"**Words:** {len(parsed_text.split())}")

        st.download_button(
            label="⬇️ Download Cleaned Text",
            data=parsed_text,
            file_name="cleaned_text.txt",
            mime="text/plain"
        )
    else:
        st.info("Upload a file to view extracted text here.")

# ------------------------------------------
# MANUAL JOB DESCRIPTION INPUT
# ------------------------------------------
st.markdown("---")
st.markdown("### 📝 Enter Job Description Text Manually")

jd_input = st.text_area("Paste Job Description Here:", "", height=200)

if jd_input:
    cleaned_jd = clean_text(jd_input)

    st.markdown("#### 🔍 Cleaned Job Description")
    st.text_area("Cleaned Output", cleaned_jd, height=200)

    st.download_button(
        label="⬇️ Download Cleaned JD",
        data=cleaned_jd,
        file_name="cleaned_jd.txt",
        mime="text/plain"
    )

# ------------------------------------------
# FOOTER
# ------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Milestone 1 • Resume & JD Parser • SkillGapAI Project</p>",
    unsafe_allow_html=True
)
