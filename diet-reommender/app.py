import base64
import io
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from PIL import Image


SAMPLE_REPORT_PATH = Path(__file__).with_name("blood_work.png")
MAX_IMAGE_SIDE = 1024  # keeps the base64 payload small enough for Groq's request limits
load_dotenv()


DIET_GUIDE = {
    "high cholesterol": {
        "avoid": ["Ghee, butter and vanaspati", "Deep-fried snacks (samosa, pakora, puri)", "Full-fat paneer and cream", "Red meat and organ meat"],
        "eat_more": ["Oats and barley", "Rajma, chana and moong dal", "Almonds and walnuts (a small handful)", "Flaxseed (alsi)"],
    },
    "high ldl": {
        "avoid": ["Ghee, butter and vanaspati", "Bakery items (biscuits, cakes, khari)", "Processed meats"],
        "eat_more": ["Oats and psyllium husk (isabgol)", "Soluble-fibre vegetables (bhindi, brinjal)", "Mustard or rice bran oil in small amounts"],
    },
    "low hdl": {
        "avoid": ["Refined sugar and sweets", "Trans fats in packaged snacks"],
        "eat_more": ["Walnuts, almonds and flaxseed", "Fatty fish (rohu, mackerel) 2x a week", "Daily brisk walking (30 min) alongside diet"],
    },
    "high triglycerides": {
        "avoid": ["Sugar, jaggery and sweets", "White rice and maida in large portions", "Sugary drinks and fruit juices", "Alcohol"],
        "eat_more": ["Whole grains (jowar, bajra, ragi)", "Fatty fish or flaxseed for omega-3", "Non-starchy vegetables"],
    },
    "high glucose": {
        "avoid": ["Sweets, sugar and sugary drinks", "White bread, maida and polished rice", "Potatoes in large portions"],
        "eat_more": ["Millets (ragi, jowar, bajra)", "Methi seeds and bitter gourd (karela)", "Dal and sprouts for protein"],
    },
    "low hemoglobin": {
        "avoid": ["Tea or coffee right after meals (blocks iron absorption)"],
        "eat_more": ["Spinach, methi and other leafy greens", "Jaggery, dates and pomegranate", "Vitamin C foods (amla, lemon) with meals"],
    },
    "low vitamin d": {
        "avoid": ["Staying indoors all day"],
        "eat_more": ["Morning sunlight 15-20 min", "Fortified milk and curd", "Egg yolks and fatty fish"],
    },
    "low vitamin b12": {
        "avoid": ["Skipping dairy entirely if vegetarian"],
        "eat_more": ["Milk, curd and paneer", "Eggs", "Fortified cereals"],
    },
}


@tool
def get_diet_recommendations(abnormal_findings: list[str]) -> str:
    """Get Indian diet recommendations for abnormal blood test findings.

    Args:
        abnormal_findings: Abnormal results from the blood report, using these names:
            "high cholesterol", "high ldl", "low hdl", "high triglycerides", "high glucose",
            "low hemoglobin", "low vitamin d", "low vitamin b12".
    """
    sections, unknown = [], []
    for finding in abnormal_findings:
        guide = DIET_GUIDE.get(finding.strip().lower())
        if guide is None:
            unknown.append(finding)
            continue
        sections.append(
            f"{finding.upper()}\n"
            f"  Avoid: {'; '.join(guide['avoid'])}\n"
            f"  Eat more: {'; '.join(guide['eat_more'])}"
        )
    if unknown:
        sections.append(f"No diet guidance available for: {', '.join(unknown)}")
    return "\n\n".join(sections) if sections else "No abnormal findings provided."


@st.cache_resource
def get_agent():
    # Groq's free tier allows 1000 output tokens per minute, so cap each reply below that.
    llm = ChatGroq(model="qwen/qwen3.8-27b", max_tokens=800)
    return create_agent(
        llm,
        tools=[get_diet_recommendations],
        system_prompt=(
            "You are a health assistant. When given a blood report image: "
            "1) extract every test result and flag values outside the reference range as HIGH or LOW, "
            "2) call get_diet_recommendations with the abnormal findings (e.g. 'high ldl', 'low hdl'), "
            "3) reply with a short summary of the results followed by the diet plan from the tool. "
            "Do not diagnose or prescribe medication, and remind the user to consult a doctor."
        ),
    )


def image_to_base64_jpeg(image_bytes: bytes) -> str:
    """Resize the image and re-encode it as JPEG so any format and size works with the model."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode()


def analyze_report(image_bytes: bytes) -> tuple[str, list[str]]:
    """Run the agent on a report image; return its answer and the findings it sent to the tool."""
    message = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64_jpeg(image_bytes)}"}},
        {"type": "text", "text": "Analyze my blood report and give me diet recommendations."},
    ])
    result = get_agent().invoke({"messages": [message]})

    findings = [
        finding
        for msg in result["messages"]
        for call in getattr(msg, "tool_calls", None) or []
        if call["name"] == "get_diet_recommendations"
        for finding in call["args"].get("abnormal_findings", [])
    ]
    return result["messages"][-1].text, findings


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
