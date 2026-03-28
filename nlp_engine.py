# nlp_engine.py

import stanza
from langdetect import detect
import numpy as np
import re

# Supported languages
SUPPORTED = {"en", "hi", "mr"}

# Cache pipelines
PIPELINES = {}

def detect_language(text):
    try:
        lang = detect(text)
        return lang if lang in SUPPORTED else "en"
    except:
        return "en"

def get_pipeline(lang):
    if lang not in PIPELINES:
        PIPELINES[lang] = stanza.Pipeline(
            lang=lang,
            processors="tokenize,pos,lemma,depparse",
            tokenize_no_ssplit=False,
            verbose=False
        )
    return PIPELINES[lang]


# ✅ FIX 2 — IMPROVED SENTENCE SPLITTING
def preprocess_text(text):
    """
    Force proper sentence boundaries before sending to Stanza.
    Fixes cases where multiple sentences merge incorrectly.
    """

    # Replace special/curved punctuation
    text = text.replace("…", ". ")
    text = text.replace("...", ". ")
    text = re.sub(r'[“”]', '"', text)

    # Fix missing space after punctuation
    text = re.sub(r'\.(?! )', '. ', text)
    text = re.sub(r'\?(?! )', '? ', text)
    text = re.sub(r'\!(?! )', '! ', text)

    # Convert punctuation into newline boundaries
    text = text.replace(". ", ".\n")
    text = text.replace("? ", "?\n")
    text = text.replace("! ", "!\n")

    # Remove double newlines
    text = re.sub(r'\n+', '\n', text)

    return text.strip()


def dependency_depth(word, sent):
    """Compute dependency tree depth."""
    depth = 0
    id_map = {w.id: w for w in sent.words}
    current = word

    while current.head != 0:
        depth += 1
        current = id_map[current.head]

    return depth


def analyze(text):
    """Perform full linguistic preprocessing."""
    
    # ✅ FIX 2 — Preprocess BEFORE parsing
    text = preprocess_text(text)

    lang = detect_language(text)
    nlp = get_pipeline(lang)
    doc = nlp(text)

    tokens = []
    sentences = []
    parse_depths = []
    pos_count = {"NOUN": 0, "VERB": 0, "ADJ": 0, "ADV": 0}

    for sent in doc.sentences:
        words = [w.text.lower() for w in sent.words]
        tokens.extend(words)
        sentences.append(" ".join(words))

        # Count POS tags
        for w in sent.words:
            if w.upos in pos_count:
                pos_count[w.upos] += 1

        # Parse depth
        depth = max(dependency_depth(w, sent) for w in sent.words)
        parse_depths.append(depth)

    total = len(tokens)
    avg_sentence_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0

    return {
        "language": lang,
        "tokens": tokens,
        "sentences": sentences,
        "noun_ratio": pos_count["NOUN"]/total if total else 0,
        "verb_ratio": pos_count["VERB"]/total if total else 0,
        "adj_ratio": pos_count["ADJ"]/total if total else 0,
        "adv_ratio": pos_count["ADV"]/total if total else 0,
        "avg_sentence_length": avg_sentence_len,
        "parse_tree_depth": float(np.mean(parse_depths)) if parse_depths else 0
    }