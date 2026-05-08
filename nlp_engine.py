# nlp_engine.py

import stanza
from langdetect import detect
import numpy as np
import re

from psychology import ALL_SELF_WORDS

SUPPORTED = {"en", "hi", "mr"}
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


# ✅ Improved preprocessing
def preprocess_text(text):
    text = text.replace("…", ". ")
    text = text.replace("...", ". ")
    text = re.sub(r'[“”]', '"', text)

    # safer sentence split
    text = re.sub(r'([.!?])\s+', r'\1\n', text)

    text = re.sub(r'\n+', '\n', text)
    return text.strip()


def dependency_depth(word, sent):
    depth = 0
    id_map = {w.id: w for w in sent.words}
    current = word

    while current.head != 0:
        depth += 1
        current = id_map.get(current.head)
        if current is None:
            break

    return depth


# 🔹 Multilingual lexicons
PRONOUNS = {
    "en": ["i","me","my","mine","we","us","our"],
    "hi": ["मैं","मुझे","मेरा","हम","हमें"],
    "mr": ["मी","मला","माझा","आपण","आम्ही"]
}

NEGATIONS = {
    "en": ["not","no","never","nothing"],
    "hi": ["नहीं","मत","कभी नहीं"],
    "mr": ["नाही","कधीच नाही"]
}

MODALS = {
    "en": ["can","could","should","would","may","might"],
    "hi": ["सकता","चाहिए","होगा"],
    "mr": ["शकतो","पाहिजे","होईल"]
}

FILLERS = ["um","uh","hmm","अं","हं"]


def analyze(text):

    text = preprocess_text(text)
    lang = detect_language(text)
    nlp = get_pipeline(lang)
    doc = nlp(text)

    tokens = []
    sentences = []
    parse_depths = []
    pos_count = {"NOUN":0,"VERB":0,"ADJ":0,"ADV":0}
    dep_lengths = []

    pronoun_count = 0
    negation_count = 0
    modal_count = 0
    filler_count = 0
    clause_count = 0
    subordinate_count = 0
    self_ref_count = 0
    self_reference_density = 0

    for sent in doc.sentences:
        words = [w.text.lower() for w in sent.words]
        tokens.extend(words)
        sentences.append(" ".join(words))
        

        clause_count += 1
        

        has_subordinate = False
        for w in sent.words:
            
            # POS
            if w.upos in pos_count:
                pos_count[w.upos] += 1

            # Pronouns
            if w.text.lower() in PRONOUNS.get(lang, []):
                pronoun_count += 1

            # Negation
            if w.text.lower() in NEGATIONS.get(lang, []):
                negation_count += 1

            # Modals
            if w.text.lower() in MODALS.get(lang, []):
                modal_count += 1

            dep_lengths.append(abs(w.id - w.head) if w.head != 0 else 0)
          

            # Filler
            if w.text.lower() in FILLERS:
                filler_count += 1

            clean_word = w.text.strip().lower()
            if clean_word in ALL_SELF_WORDS:
                self_ref_count += 1

            # Subordinate clause (simple heuristic)
            if w.deprel in ["advcl","ccomp","xcomp"]:
                has_subordinate = True
        if has_subordinate:
            subordinate_count += 1

        depth = max(dependency_depth(w, sent) for w in sent.words)
        parse_depths.append(depth)

    total = len(tokens)
    self_reference_density = self_ref_count / total if total else 0

    # 🔥 Fix subordinate clause ratio
    sub_ratio = subordinate_count / max(clause_count, 1)
    sub_ratio = min(sub_ratio, 1.0)

    avg_sentence_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0

    return {
        "language": lang,
        "tokens": tokens,
        "sentences": sentences,
        "self_reference_density": self_reference_density,

        "noun_ratio": pos_count["NOUN"]/total if total else 0,
        "verb_ratio": pos_count["VERB"]/total if total else 0,
        "adj_ratio": pos_count["ADJ"]/total if total else 0,
        "adv_ratio": pos_count["ADV"]/total if total else 0,

        "pronoun_ratio": pronoun_count/total if total else 0,
        "negation_ratio": negation_count/total if total else 0,
        "modal_ratio": modal_count/total if total else 0,

        "filler_ratio": filler_count/total if total else 0,

        "clause_count": clause_count,
        "subordinate_clause_ratio": sub_ratio,
        "avg_sentence_length": avg_sentence_len,
        "parse_tree_depth": float(np.mean(parse_depths)) if parse_depths else 0,

        "avg_dep_length": float(np.mean(dep_lengths)) if dep_lengths else 0,
        # simple cognitive proxy

        "cognitive_load_score": (
            avg_sentence_len + np.mean(parse_depths) if parse_depths else avg_sentence_len
        )
    }