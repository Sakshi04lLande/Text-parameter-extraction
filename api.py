from fastapi import FastAPI, Request, Header, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from main import run_with_latency
import re
import logging
import os
from dotenv import load_dotenv

# 🔄 Load environment variables
load_dotenv()

app = FastAPI(title="Psychological Analysis API")


# 🔐 ================= SECURITY CONFIG =================

API_KEY_value = os.getenv("API_KEY")   # must be set in environment
MAX_LENGTH = 5000


def verify_api_key(x_api_key: str = Header(None)):
    # ❌ If API key not configured on server
    if not API_KEY_value:
        raise HTTPException(
            status_code=500,
            detail="API key not configured on server"
        )

    # ❌ If invalid key provided
    if x_api_key != API_KEY_value:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )


# 📝 ================= REQUEST MODEL =================

class ConversationRequest(BaseModel):
    conversation: Optional[str] = None


# 🏠 ================= HOME =================

@app.get("/")
def home():
    return {"message": "API is running 🚀"}


# 🧠 ================= EXTRACTION =================

def extract_client_text(full_conversation: str) -> str:
    matches = re.findall(
        r'client:\s*(.*?)(?=\n\s*(assistant:|client:)|$)',
        full_conversation,
        re.IGNORECASE | re.DOTALL
    )
    return " ".join([m[0].strip() for m in matches])


# 📊 ================= MAIN API =================

@app.post("/analyze", dependencies=[Depends(verify_api_key)])
async def analyze_conversation(request: Request, req: ConversationRequest = None):
    try:
        # ✅ Get input (JSON or raw)
        if req and req.conversation:
            conversation = req.conversation
        else:
            body = await request.body()
            conversation = body.decode("utf-8")

        # ✅ Normalize input
        conversation = conversation.replace("\\n", "\n")
        conversation = re.sub(r'\r\n|\r', '\n', conversation)
        conversation = conversation.strip()

        # ❌ Empty check
        if not conversation:
            return {
                "status": "error",
                "message": "No conversation provided"
            }

        # ❌ Size limit protection
        if len(conversation) > MAX_LENGTH:
            return {
                "status": "error",
                "message": "Input too large"
            }

        # ✅ Extract client text
        client_text = extract_client_text(conversation)

        if not client_text:
            return {
                "status": "error",
                "message": "No client text found. Use format: Client: ..."
            }

        # ✅ Run pipeline
        result, timings = run_with_latency(client_text)

        return {
            "status": "success",
            "client_text": client_text,
            "analysis": result,
            "latency": timings
        }

    except Exception as e:
        # 🔒 Log internal error (not exposed to user)
        logging.error(f"Error in /analyze: {str(e)}")

        return {
            "status": "error",
            "message": "Something went wrong"
        }