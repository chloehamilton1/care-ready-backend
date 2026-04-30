import os
import json
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from docx import Document
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("ANTHROPIC_API_KEY")
DOCS_FOLDER = "docs"


def load_policy_docs() -> str:
    """Read all .docx files in /docs and combine into one text block."""
    chunks = []

    if not os.path.exists(DOCS_FOLDER):
        return ""

    for filename in sorted(os.listdir(DOCS_FOLDER)):
        if filename.endswith(".docx"):
            path = os.path.join(DOCS_FOLDER, filename)
            try:
                doc = Document(path)
                text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                chunks.append(f"\n--- SOURCE: {filename} ---\n{text}")
            except Exception as e:
                chunks.append(f"\n--- SOURCE: {filename} ---\n[ERROR READING FILE: {e}]")

    return "\n".join(chunks)


POLICY_TEXT = load_policy_docs()


class ChatMessage(BaseModel):
    role: str
    text: str

class QueryRequest(BaseModel):
    question: str
    role: str
    agency_id: str
    state: str
    history: list[ChatMessage] = []

class QueryResponse(BaseModel):
    response_text: str
    escalation_level: str
    policy_reference: str
    resources: list[str]
    confidence: str


@app.post("/query", response_model=QueryResponse)
def query_ai(req: QueryRequest):
    url = "https://api.anthropic.com/v1/messages"

    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    history_text = "\n".join(
    [f"{m.role}: {m.text}" for m in req.history[-6:]]
)
    
    prompt = f"""
You are a caregiving assistant.

You MUST use the policy context below first before relying on general best practices.

POLICY CONTEXT:
{POLICY_TEXT[:120000]}

Role: {req.role}
State: {req.state}
Agency ID: {req.agency_id}

Recent conversation:
{history_text}

Question: {req.question}

Respond ONLY in valid JSON with this exact schema:

{{
  "response_text": "clear, simple guidance in plain text (no markdown)",
  "escalation_level": "handle_yourself | notify_supervisor | urgent",
  "policy_reference": "exact source filename from the policy context above, or 'unknown'",
  "resources": [],
  "confidence": "low | medium | high"
}}

Rules:
- Keep response_text under 100 words.
- Always respond in 3–4 numbered steps.
- Each step must be on its own line starting with "1.", "2.", etc.
- Each step should be 1 sentence max.
- Use simple caregiver-friendly language.
- Prioritize the most important safety action first.
- If multiple concerns are mentioned, address each briefly.
- Avoid long paragraphs.
- Use the policy context above whenever relevant.
- Do not invent policy names or citations.
- If the answer comes from the documents, set policy_reference to the exact filename.
- If no relevant policy context exists, set policy_reference to "unknown".
- Do not include anything outside the JSON.
"""

    data = {
        #"model": "claude-sonnet-4-6", - latency reasons, testing out quicker models
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 250,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=data, timeout=30)
    result = response.json()

    try:
        raw_text = result["content"][0]["text"]
        print("RAW CLAUDE OUTPUT:")
        print(raw_text)

        raw_text = raw_text.strip()

        if raw_text.startswith("```json"):
            raw_text = raw_text.removeprefix("```json").strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.removeprefix("```").strip()
        if raw_text.endswith("```"):
            raw_text = raw_text.removesuffix("```").strip()

        parsed = json.loads(raw_text)

    except Exception as e:
        print("JSON PARSE ERROR:", e)
        print("FULL RESULT:")
        print(result)
        parsed = {
            "response_text": "I’m not fully certain, but here are general safety steps: keep the client safe, stay calm, do not force care, and contact your supervisor if there are concerns or changes in condition.",
            "escalation_level": "notify_supervisor",
            "policy_reference": "unknown",
            "resources": [],
            "confidence": "low"
        }

    # Light guardrails
    if parsed.get("escalation_level") not in ["handle_yourself", "notify_supervisor", "urgent"]:
        parsed["escalation_level"] = "notify_supervisor"

    if parsed.get("confidence") not in ["low", "medium", "high"]:
        parsed["confidence"] = "low"

    if not parsed.get("policy_reference"):
        parsed["policy_reference"] = "unknown"

    if not isinstance(parsed.get("resources"), list):
        parsed["resources"] = []

    return parsed

@app.post("/feedback")
def log_feedback(data: dict):
    print("FEEDBACK RECEIVED:")
    print(data)
    return {"status": "ok"}