from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from streamlit.runtime.uploaded_file_manager import UploadedFile


REPORT_PATH = Path(__file__).with_name("blood_work.txt")
load_dotenv()


@st.cache_resource
def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash")


def analyze_blood_work(report: str) -> tuple[str, str]:
    extraction_prompt = f"""
You are a medical data extraction assistant.

From the blood report below, extract ALL test values and classify each one as HIGH,
LOW, or NORMAL based only on the reference ranges provided in the report.

Format each result as:
- Test Name: value | Status: HIGH/LOW/NORMAL | Reference: range

Blood Report:
{report}
"""
    extracted_values = get_llm().invoke(extraction_prompt).text

    diet_prompt = f"""
You are a clinical nutritionist specializing in Indian dietary habits.

Based on the blood work analysis below, write:
1. A short health summary in 4-5 lines explaining the patient's condition in simple language.
2. A short, practical Indian diet plan with exactly two sections: "Foods to avoid" and
   "Foods to eat more of".

Do not diagnose, prescribe medication, or add sections beyond those requested.

Blood Work Analysis:
{extracted_values}
"""
    diet_plan = get_llm().invoke(diet_prompt).text
    return extracted_values, diet_plan


def get_uploaded_report(uploaded_file: UploadedFile) -> str:
    try:
        return uploaded_file.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        st.error("The uploaded report must be a UTF-8 encoded text file.")
        st.stop()


st.set_page_config(page_title="Blood Work Analysis", page_icon="🩺", layout="wide")
st.title("Blood Work Analysis")
st.caption("AI-assisted educational analysis of blood-report values and diet guidance.")
st.warning(
    "This tool is not medical advice. Review all results with a qualified healthcare professional "
    "before making healthcare decisions."
)

sample_report = REPORT_PATH.read_text(encoding="utf-8")
uploaded_file = st.file_uploader("Upload a blood report (.txt)", type=["txt"])
report_text = st.text_area(
    "Blood report",
    value=sample_report,
    height=360,
    help="Paste a report here, or upload a UTF-8 text file. An uploaded file takes precedence.",
)

if st.button("Analyze report", type="primary", use_container_width=True):
    blood_report = get_uploaded_report(uploaded_file) if uploaded_file else report_text

    if not blood_report.strip():
        st.error("Enter or upload a blood report before running the analysis.")
    else:
        with st.spinner("Extracting values and preparing guidance..."):
            extracted_values, diet_plan = analyze_blood_work(blood_report)

        extraction_column, guidance_column = st.columns(2)
        with extraction_column:
            st.subheader("Extracted values")
            st.markdown(extracted_values)
        with guidance_column:
            st.subheader("Health summary and diet plan")
            st.markdown(diet_plan)
