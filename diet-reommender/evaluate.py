"""Evaluate the diet recommender agent with LangSmith.

Run from the repository root:
    uv run python diet-reommender/evaluate.py --version v1    # 16 examples, includes hard cases
    uv run python diet-reommender/evaluate.py --version v0    # 5 easy baseline examples

Needs LANGSMITH_API_KEY and GROQ_API_KEY in .env. Each version is its own LangSmith dataset
(diet-recommender-eval-v0 / -v1) under Datasets & Experiments.

Datasets live in eval_data/dataset_<version>.json. Each example has:
    inputs:  {"report": "<text>"} or {"image": "<path relative to eval_data/>"}
    outputs: {"findings": [...expected tool findings...],
              "must_mention": [...] (optional), "is_blood_report": false (optional)}
"""

import argparse
import json
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import Client
from pydantic import BaseModel, Field

from diet_agent import MODEL, build_agent, image_message, run_agent, text_message


load_dotenv()

EVAL_DIR = Path(__file__).parent / "eval_data"

# Retry generously: Groq's free tier rate-limits by tokens per minute.
agent = build_agent(max_retries=8)


def load_examples(version: str) -> list[dict]:
    return json.loads((EVAL_DIR / f"dataset_{version}.json").read_text(encoding="utf-8"))


def target(inputs: dict) -> dict:
    """What LangSmith runs for each example: the same agent the Streamlit app uses."""
    if "image" in inputs:
        message = image_message((EVAL_DIR / inputs["image"]).read_bytes())
    else:
        message = text_message(inputs["report"])
    return run_agent(agent, message)


def _normalize(findings: list[str]) -> set[str]:
    return {finding.strip().lower() for finding in findings}


def findings_f1(outputs: dict, reference_outputs: dict) -> dict:
    """How well the findings sent to the diet tool match the expected ones (1.0 = perfect)."""
    predicted, expected = _normalize(outputs["findings"]), _normalize(reference_outputs["findings"])
    comment = f"predicted={sorted(predicted)} expected={sorted(expected)}"
    if not predicted and not expected:
        return {"key": "findings_f1", "score": 1.0, "comment": comment}
    true_positives = len(predicted & expected)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"key": "findings_f1", "score": f1, "comment": comment}


def findings_exact_match(outputs: dict, reference_outputs: dict) -> dict:
    """1 if the agent flagged exactly the expected findings, else 0."""
    match = _normalize(outputs["findings"]) == _normalize(reference_outputs["findings"])
    return {"key": "findings_exact_match", "score": int(match)}


def mentions_doctor(outputs: dict, reference_outputs: dict) -> dict:
    """For real blood reports, the answer must tell the user to consult a doctor."""
    if reference_outputs.get("is_blood_report") is False:
        return {"key": "mentions_doctor", "score": None, "comment": "n/a: not a blood report"}
    found = re.search(r"doctor|physician|healthcare (professional|provider)", outputs["answer"], re.IGNORECASE)
    return {"key": "mentions_doctor", "score": int(bool(found))}


def no_medication(outputs: dict) -> dict:
    """The answer must not name prescription drugs (the agent is told not to prescribe)."""
    drugs = r"\b(statins?|atorvastatin|rosuvastatin|metformin|insulin|fenofibrate|ezetimibe|levothyroxine|allopurinol)\b"
    named = re.findall(drugs, outputs["answer"], re.IGNORECASE)
    return {"key": "no_medication", "score": int(not named), "comment": f"named: {named}" if named else None}


def mentions_required(outputs: dict, reference_outputs: dict) -> dict:
    """Abnormal tests the tool can't handle (e.g. uric acid) must still be mentioned in the answer."""
    required = reference_outputs.get("must_mention")
    if not required:
        return {"key": "mentions_required", "score": None, "comment": "n/a"}
    missing = [term for term in required if term.lower() not in outputs["answer"].lower()]
    score = 1 - len(missing) / len(required)
    return {"key": "mentions_required", "score": score, "comment": f"missing: {missing}" if missing else None}


class NonReportVerdict(BaseModel):
    says_not_a_blood_report: bool = Field(description="True if the answer clearly tells the user the input is not a blood report.")
    reason: str = Field(description="One sentence explaining the verdict.")


# A different model family judges the answer, so the agent isn't grading itself.
judge = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0).with_structured_output(NonReportVerdict)


def recognizes_non_report(outputs: dict, reference_outputs: dict) -> dict:
    """For non-reports (e.g. a cat photo): the agent must say it isn't a blood report and not call the diet tool."""
    if reference_outputs.get("is_blood_report") is not False:
        return {"key": "recognizes_non_report", "score": None, "comment": "n/a: real blood report"}
    if outputs["findings"]:
        return {"key": "recognizes_non_report", "score": 0, "comment": f"called the diet tool with {outputs['findings']}"}
    verdict = judge.invoke(
        "The user was supposed to upload a blood test report but sent something else. "
        "Does this assistant answer clearly tell the user that what they sent is not a blood report?\n\n"
        f"Assistant answer:\n{outputs['answer']}"
    )
    return {"key": "recognizes_non_report", "score": int(verdict.says_not_a_blood_report), "comment": verdict.reason}


EVALUATORS = [findings_f1, findings_exact_match, mentions_doctor, no_medication, mentions_required, recognizes_non_report]


def ensure_dataset(client: Client, version: str) -> str:
    """Create the LangSmith dataset for this version from its JSON file, if it doesn't exist yet."""
    name = f"diet-recommender-eval-{version}"
    if not client.has_dataset(dataset_name=name):
        examples = load_examples(version)
        dataset = client.create_dataset(name, description=f"Blood reports with expected findings ({version}).")
        client.create_examples(dataset_id=dataset.id, examples=[
            {"inputs": ex["inputs"], "outputs": ex["outputs"],
             "metadata": {"id": ex["id"], "description": ex["description"]}}
            for ex in examples
        ])
    return name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="v1", help="dataset version in eval_data/ (default: v1)")
    args = parser.parse_args()

    client = Client()
    dataset_name = ensure_dataset(client, args.version)
    results = client.evaluate(
        target,
        data=dataset_name,
        evaluators=EVALUATORS,
        experiment_prefix=f"diet-agent-{args.version}",
        metadata={"model": MODEL, "dataset_version": args.version},
        max_concurrency=1,  # one example at a time to stay within Groq's free-tier limits
    )
    scores = results.to_pandas().filter(like="feedback.")
    print(scores.mean(numeric_only=True).round(3))
