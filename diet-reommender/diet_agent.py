"""Diet recommender agent shared by the Streamlit app (app.py) and the evaluation (evaluate.py)."""

import base64
import io

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from PIL import Image


MODEL = "qwen/qwen3.8-27b"
MAX_IMAGE_SIDE = 1024  # keeps the base64 payload small enough for Groq's request limits
SYSTEM_PROMPT = (
    "You are a health assistant. When given a blood report: "
    "1) extract every test result and flag values outside the reference range as HIGH or LOW, "
    "2) call get_diet_recommendations with the abnormal findings (e.g. 'high ldl', 'low hdl'), "
    "3) reply with a short summary of the results followed by the diet plan from the tool. "
    "Do not diagnose or prescribe medication, and remind the user to consult a doctor. "
    "If the input is not a blood test report (for example a photo of an animal or object, another kind "
    "of document, or a question with no lab values), say clearly that it is not a blood report, briefly "
    "describe what you received, ask the user to upload a blood report, and do not call "
    "get_diet_recommendations or invent any findings."
)
USER_REQUEST = "Analyze my blood report and give me diet recommendations."


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


def build_agent(max_retries: int = 2):
    # Groq's free tier allows 1000 output tokens per minute, so cap each reply below that.
    llm = ChatGroq(model=MODEL, max_tokens=800, max_retries=max_retries)
    return create_agent(llm, tools=[get_diet_recommendations], system_prompt=SYSTEM_PROMPT)


def image_to_base64_jpeg(image_bytes: bytes) -> str:
    """Resize the image and re-encode it as JPEG so any format and size works with the model."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode()


def image_message(image_bytes: bytes) -> HumanMessage:
    return HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64_jpeg(image_bytes)}"}},
        {"type": "text", "text": USER_REQUEST},
    ])


def text_message(report: str) -> HumanMessage:
    return HumanMessage(content=f"{USER_REQUEST}\n\nBlood report:\n{report}")


def run_agent(agent, message: HumanMessage) -> dict:
    """Run the agent; return its final answer and the findings it sent to the diet tool."""
    result = agent.invoke({"messages": [message]})
    findings = [
        finding
        for msg in result["messages"]
        for call in getattr(msg, "tool_calls", None) or []
        if call["name"] == "get_diet_recommendations"
        for finding in call["args"].get("abnormal_findings", [])
    ]
    return {"answer": result["messages"][-1].text, "findings": findings}
