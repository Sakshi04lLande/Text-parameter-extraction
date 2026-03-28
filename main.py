import sys
import time
from nlp_engine import analyze
from embedder import embed
from psychology import analyze_sentence
from feature_extractor import extract
import json


def extract_client_text(full_conversation):
    client_lines = []

    for line in full_conversation.split("\n"):
        line = line.strip()
        if line.lower().startswith("client:"):
            cleaned = line.split("Client:", 1)[1].strip()
            client_lines.append(cleaned)

    return " ".join(client_lines)


def run_with_latency(text):
    timings = {}

    total_start = time.time()

    # ⏱ NLP
    t0 = time.time()
    nlp = analyze(text)
    timings["nlp_time"] = time.time() - t0

    # ⏱ Embeddings
    t0 = time.time()
    embeddings = embed(nlp["sentences"])
    timings["embedding_time"] = time.time() - t0

    # ⏱ LLM (per sentence)
    t0 = time.time()
    llm_infos = []
    for s in nlp["sentences"]:
        start = time.time()
        result = analyze_sentence(s)
        llm_infos.append(result)
    timings["llm_total_time"] = time.time() - t0
    timings["llm_avg_per_sentence"] = timings["llm_total_time"] / max(len(nlp["sentences"]), 1)

    # ⏱ Feature extraction
    t0 = time.time()
    result = extract(text, nlp, embeddings, llm_infos)
    timings["feature_extraction_time"] = time.time() - t0

    # ⏱ Total
    timings["total_time"] = time.time() - total_start

    return result, timings


if __name__ == "__main__":
    print("Paste the full conversation below. End with CTRL+D (Linux/Mac) or CTRL+Z (Windows):")

    full_conversation = sys.stdin.read()

    client_text = extract_client_text(full_conversation)

    print("\n✅ Extracted Client Text:")
    print(client_text)

    print("\n🚀 Running Psychological Pipeline...\n")

    result, timings = run_with_latency(client_text)

    print("📊 RESULT:\n")
    print(json.dumps(result, indent=4, ensure_ascii=False))

    print("\n⏱ LATENCY BREAKDOWN:\n")
    print(json.dumps(timings, indent=4))