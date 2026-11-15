import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))  # so diet_agent.py next to this file is importable
from diet_agent import build_agent, image_message, run_agent  # noqa: E402


SAMPLE_REPORT_PATH = Path(__file__).with_name("blood_work.png")
load_dotenv()


@st.cache_resource
def get_agent():
    return build_agent()


def analyze_report(image_bytes: bytes) -> tuple[str, list[str]]:
    """Run the agent on a report image; return its answer and the findings it sent to the tool."""
    result = run_agent(get_agent(), image_message(image_bytes))
    return result["answer"], result["findings"]


st.set_page_config(page_title="Diet Recommender", page_icon="🥗", layout="wide")
st.title("Diet Recommender")
st.caption("Upload a blood report image to get flagged results and Indian diet recommendations.")
st.warning(
    "This tool is not medical advice. Review all results with a qualified healthcare professional "
    "before making changes to your diet or treatment."
)

uploaded_file = st.file_uploader(
    "Upload a blood report image",
    type=["png", "jpg", "jpeg", "webp", "avif"],
    help="Large photos are resized automatically before they are sent to the model.",
)
use_sample = st.checkbox("Use the sample blood report", value=uploaded_file is None)

if uploaded_file is not None:
    report_bytes = uploaded_file.getvalue()
elif use_sample and SAMPLE_REPORT_PATH.exists():
    report_bytes = SAMPLE_REPORT_PATH.read_bytes()
else:
    report_bytes = None

image_column, result_column = st.columns([1, 2])

with image_column:
    if report_bytes:
        st.image(report_bytes, caption="Blood report", width="stretch")
    else:
        st.info("Upload a report image or select the sample report.")

with result_column:
    if st.button("Analyze report", type="primary", disabled=report_bytes is None, width="stretch"):
        try:
            with st.spinner("Reading the report and preparing diet recommendations..."):
                answer, findings = analyze_report(report_bytes)
        except Exception as error:
            st.error(f"Analysis failed: {error}")
        else:
            if findings:
                st.subheader("Abnormal findings sent to the diet tool")
                st.write(", ".join(f"`{finding}`" for finding in findings))
            st.markdown(answer)
