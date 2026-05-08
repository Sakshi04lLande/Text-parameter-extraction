# api.py

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import re
import os
from dotenv import load_dotenv

from nlp_engine import analyze
from embedder import embed
from psychology import analyze_sentence
from feature_extractor import extract

# 🔥 Load environment variables
load_dotenv()

app = FastAPI(title="Mental Health NLP Analyzer")

# -------------------------------
# 🔐 API KEY CONFIG
# -------------------------------
MENTAL_HEALTH_ANALYZER_KEY = os.getenv("MENTAL_HEALTH_ANALYZER_KEY")


def verify_api_key(x_api_key: str):
    if x_api_key != MENTAL_HEALTH_ANALYZER_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# -------------------------------
# Request Schema
# -------------------------------
class ConversationRequest(BaseModel):
    text: str


# -------------------------------
# Helper: Extract Client Text
# -------------------------------
def extract_client_text(text: str):
    matches = re.findall(
        r"(?:^|\n)\s*client\s*:\s*(.*?)(?=\n\s*(?:assistant|client)\s*:|$)",
        text,
        flags=re.I | re.S
    )

    if matches:
        return " ".join(m.strip() for m in matches)

    return text.strip()


# -------------------------------
# Main API Endpoint
# -------------------------------
@app.post("/analyze")
def analyze_conversation(
    request: ConversationRequest,
    x_api_key: str = Header(None)
):
    # 🔐 Validate API key
    verify_api_key(x_api_key)

    # Step 1: Extract client-only text
    client_text = extract_client_text(request.text)

    if not client_text.strip():
        raise HTTPException(status_code=400, detail="No client text found")

    # Step 2: NLP
    nlp = analyze(client_text)

    # Step 3: Embeddings
    sentences = nlp.get("sentences", [])
    embeddings = embed(sentences)

    # Step 4: Psychology
    psych = [analyze_sentence(s) for s in sentences]

    # Step 5: Feature Extraction
    result = extract(client_text, nlp, embeddings, psych)

    return {
        "status": "success",
        "features": result
    }