# main.py

import sys
import re
import json
from nlp_engine import analyze
from embedder import embed
from psychology import analyze_sentence
from feature_extractor import extract


def extract_client_text(text: str) -> str:
    """
    Extracts only client utterances from a conversation transcript.
    """
    try:
        matches = re.findall(
            r"(?:^|\n)\s*client:\s*(.*?)(?=\n\s*(?:assistant:|client:)|$)",
            text,
            flags=re.I | re.S,
        )
        return " ".join(m.strip() for m in matches if m.strip())
    except Exception:
        return ""


if __name__ == "__main__":
    conversation = sys.stdin.read()
    
    client_text = extract_client_text(conversation)
    if not client_text.strip():
        client_text = conversation
    sentences = [s.strip() for s in client_text.split("\n") if s.strip()]

    nlp_result = analyze(client_text)
    embeddings = embed(sentences)
    psychology = [analyze_sentence(s) for s in sentences]
    features = extract(client_text, nlp_result, embeddings, psychology)

    output = {
        "nlp": nlp_result,
        "psychology": psychology,
        "features": features,
    }

    print(json.dumps(output, indent=2))