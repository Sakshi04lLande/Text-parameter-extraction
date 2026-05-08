# psychology.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re

from embedder import embed

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


PROTOTYPES = {
    "absolutist": [
        # English
        "nothing ever works for me",
        "everything is always wrong",
        "i never do anything right",
        "this will never change",
        "things are always bad",
        "nothing is good in my life",

    # Hindi
        "मेरे साथ कभी कुछ अच्छा नहीं होता",
        "सब कुछ हमेशा गलत होता है",
        "मैं कभी सही नहीं करता",
        "ये कभी नहीं बदलेगा",
        "मेरी जिंदगी में कुछ भी अच्छा नहीं है",
        "सब हमेशा खराब ही रहता है",

    # Marathi
        "माझ्यासोबत कधीच काही चांगलं होत नाही",
        "सगळं नेहमी चुकीचं होतं",
        "मी कधीच बरोबर करत नाही",
        "हे कधीच बदलणार नाही",
        "माझ्या आयुष्यात काहीच चांगलं नाही",
        "सगळं नेहमी वाईटच असतं"
    ],
    "helplessness": [
           # English
           "i can't do anything",
           "there is no way out",
           "nothing is in my control",
           "i feel powerless",
           "i cannot change this situation",
           "i am stuck and can't move forward",

    # Hindi
           "मैं कुछ नहीं कर सकता",
           "इससे निकलने का कोई रास्ता नहीं है",
           "कुछ भी मेरे नियंत्रण में नहीं है",
           "मैं खुद को असहाय महसूस करता हूँ",
           "मैं इस स्थिति को बदल नहीं सकता",
           "मैं फंसा हुआ हूँ",

    # Marathi
          "मी काहीच करू शकत नाही",
          "यातून बाहेर पडण्याचा काहीच मार्ग नाही",
          "काहीच माझ्या नियंत्रणात नाही",
          "मी स्वतःला असहाय्य वाटतो",
          "मी ही परिस्थिती बदलू शकत नाही",
          "मी अडकलो आहे"
    ],
    "catastrophizing": [
        # English
          "everything will go wrong",
           "this will be a disaster",
           "worst will happen",
           "everything will fall apart",
           "my life will be ruined",
          "this is going to end badly",

    # Hindi
         "सब कुछ खराब हो जाएगा",
         "यह बहुत बुरा होने वाला है",
         "सब कुछ बिखर जाएगा",
         "मेरी जिंदगी बर्बाद हो जाएगी",
         "सब कुछ गलत हो जाएगा",
         "सब खत्म हो जाएगा",

    # Marathi
          "सगळं बिघडेल",
           "हे खूप वाईट होणार आहे",
           "सगळं कोलमडून जाईल",
           "माझं आयुष्य संपेल",
           "सगळं चुकीचं होईल",
           "सगळं नष्ट होईल"
    ],
    "rumination": [
        # English
             "i keep thinking about it again and again",
             "i cannot stop thinking about it",
             "my mind keeps repeating the same thoughts",
             "i overthink everything",
             "these thoughts keep coming back",

    # Hindi
           "मैं बार-बार उसी बात के बारे में सोचता रहता हूँ",
           "मैं सोचना बंद नहीं कर पा रहा हूँ",
           "मेरे दिमाग में वही बातें घूमती रहती हैं",
           "मैं हर चीज़ को ज्यादा सोचता हूँ",
           "वो विचार बार-बार आते रहते हैं",

    # Marathi
          "मी पुन्हा पुन्हा त्याच गोष्टीचा विचार करतो",
          "मी विचार थांबवू शकत नाही",
          "माझ्या डोक्यात तेच विचार फिरत राहतात",
          "मी खूप जास्त विचार करतो",
          "ते विचार सतत परत येतात"
    ],
    "avoidance": [
        # English
          "i try to ignore it",
          "i avoid thinking about it",
          "i distract myself",
          "i stay away from it",
          "i don't want to deal with it",

    # Hindi
         "मैं इसे नजरअंदाज करने की कोशिश करता हूँ",
         "मैं इसके बारे में सोचने से बचता हूँ",
         "मैं खुद को दूसरी चीजों में उलझा लेता हूँ",
         "मैं इससे दूर रहता हूँ",
         "मैं इसका सामना नहीं करना चाहता",

    # Marathi
         "मी ते दुर्लक्ष करण्याचा प्रयत्न करतो",
         "मी त्याचा विचार टाळतो",
         "मी स्वतःला दुसऱ्या गोष्टींमध्ये गुंतवतो",
         "मी त्यापासून दूर राहतो",
         "मला त्याला सामोरे जायचं नाही"
    ],
    "threat": [
        # English
          "something bad will happen",
          "i am afraid of what will happen",
          "i feel something is wrong",
          "i fear the future",
          "danger is coming",

    # Hindi
          "कुछ बुरा होने वाला है",
          "मुझे डर लग रहा है कि क्या होगा",
          "मुझे लगता है कुछ गलत होने वाला है",
           "मुझे भविष्य से डर लगता है",
           "कोई खतरा आने वाला है",

    # Marathi
          "काहीतरी वाईट होणार आहे",
          "मला भीती वाटते काय होईल याची",
          "मला वाटतं काहीतरी चुकीचं होणार आहे",
          "मला भविष्याची भीती वाटते",
          "धोका येणार आहे"
    ],
    "uncertainty": [
       # English
          "maybe something will happen",
          "i am not sure",
          "it might go wrong",
          "i don't know what will happen",
          "things are unclear",

    # Hindi
         "शायद कुछ होगा",
         "मुझे यकीन नहीं है",
         "यह गलत हो सकता है",
         "मुझे नहीं पता क्या होगा",
         "सब कुछ स्पष्ट नहीं है",

    # Marathi
          "कदाचित काहीतरी होईल",
          "मला खात्री नाही",
          "हे चुकीचं होऊ शकतं",
          "मला माहित नाही काय होईल",
          "सगळं स्पष्ट नाही"
    ],
    "external_locus": [
        # English
          "it is not in my control",
          "others decide everything",
          "my situation is controlled by others",
          "circumstances control my life",
          "i have no control over this",

    # Hindi
          "यह मेरे नियंत्रण में नहीं है",
          "सब कुछ दूसरे लोग तय करते हैं",
          "मेरी स्थिति दूसरों के हाथ में है",
          "परिस्थितियाँ मेरी जिंदगी को नियंत्रित करती हैं",
          "मेरा इस पर कोई नियंत्रण नहीं है",

    # Marathi
          "हे माझ्या नियंत्रणात नाही",
          "सगळं दुसरे लोक ठरवतात",
          "माझी परिस्थिती इतरांच्या हातात आहे",
          "परिस्थिती माझं आयुष्य नियंत्रित करते",
          "माझा यावर काहीच ताबा नाही"
    ]
}

# ==============================
# 🔹 Precompute embeddings ONCE
# ==============================

PROTO_EMBS = {
    key: embed(sentences)
    for key, sentences in PROTOTYPES.items()
}


# ==============================
# 🔹 Similarity scoring
# ==============================

def get_score(sentence_emb, proto_embs):
    sims = cosine_similarity([sentence_emb], proto_embs)[0]
    top2 = sorted(sims, reverse=True)[:2]
    return float(sum(top2) / len(top2))


# ==============================
# 🔹 Time Focus Detection
# ==============================
PAST_WORDS = [

    # English
    "was", "were", "had", "did", "before", "earlier", "yesterday",
    "last time", "in the past", "used to", "once", "previously","past", "earlier", "before", "ago",

    # Hindi
    "था", "थे", "थी", "किया था", "पहले", "कल", "बीते हुए", "पहले कभी",
    "पहले ऐसा था", "पहले हुआ था",

    # Marathi
    "होतं", "होते", "होती", "केलं होतं", "पूर्वी", "काल", "आधी",
    "पूर्वी असं होतं", "आधी झालं होतं"
]
PRESENT_WORDS = [

    # English
    "is", "am", "are", "right now", "currently", "today", "now",
    "these days", "at this moment",

    # Hindi
    "है", "हूँ", "हैं", "अभी", "आज", "इस समय", "अभी के समय",
    "आजकल", "इस वक्त",

    # Marathi
    "आहे", "आहेत", "आहेस", "आत्ता", "आज", "सध्या", "या क्षणी",
    "आताच्या काळात"
]
FUTURE_WORDS = [

    # English
    "will", "going to", "tomorrow", "next", "future", "soon",
    "might", "could", "in future", "later",

    # Hindi
    "होगा", "होगी", "होंगे", "करूँगा", "करूँगी", "कल", "आगे",
    "भविष्य", "हो सकता है", "आने वाला",

    # Marathi
    "होईल", "होणार", "करणार", "उद्या", "पुढे", "भविष्यात",
    "होऊ शकतं", "लवकरच"
]

def detect_time_focus(sentence):
    s = sentence.lower()
    words = s.split()
    
    past_score = sum(1 for w in PAST_WORDS if w in words)
    present_score = sum(1 for w in PRESENT_WORDS if w in words)
    future_score = sum(1 for w in FUTURE_WORDS if w in words)

    scores = {
        "past": past_score,
        "present": present_score,
        "future": future_score
    }

    # If all zero → default to present
    if all(v == 0 for v in scores.values()):
        return "present"

    return max(scores, key=scores.get)


# ==============================
# 🔹 Self Reference (basic)
# ==============================
EN_SELF = [
    "i", "me", "my", "mine", "myself"
]
HI_SELF = [
    "मैं", "मुझे", "मुझ", "मेरा", "मेरी", "मेरे","हम",
    "खुद", "स्वयं", "अपने", "अपना", "अपनी"
]
MR_SELF = [
    "मी", "मला", "माझा", "माझी", "माझे",
    "स्वतः", "स्वत:", "माझ्याकडे", "माझ्यासाठी"
]

ALL_SELF_WORDS = EN_SELF + HI_SELF + MR_SELF


def self_reference_score(sentence):
    words = re.findall(r'\w+', sentence.lower())
    if not words:
        return 0.0
 
    count = sum(1 for w in words if w in ALL_SELF_WORDS)

    return count / len(words)


# ==============================
# 🔹 MAIN FUNCTION
# ==============================

def analyze_sentence(sentence):
    embs = embed([sentence])
    if embs is None or len(embs) == 0:
        return {
        "absolutist_score": 0,
        "helplessness_score": 0,
        "catastrophizing_score": 0,
        "external_locus_score": 0,
        "rumination_score": 0,
        "uncertainty_score": 0,
        "avoidance_score": 0,
        "threat_score": 0,
        "time_focus": "present"
    }

    emb = embs[0]
    

    # Compute scores
    absolutist_score = get_score(emb, PROTO_EMBS["absolutist"])
    helpless_score = get_score(emb, PROTO_EMBS["helplessness"])
    cat_score = get_score(emb, PROTO_EMBS["catastrophizing"])
    locus_score = get_score(emb, PROTO_EMBS["external_locus"])
    rumi_score = get_score(emb, PROTO_EMBS["rumination"])
    unct_score = get_score(emb, PROTO_EMBS["uncertainty"])
    avoid_score = get_score(emb, PROTO_EMBS["avoidance"])
    threat_score = get_score(emb, PROTO_EMBS["threat"])
    
  
    # Thresholds (tune if needed)
    return {
        "absolutist_score": absolutist_score,
        "helplessness_score": helpless_score,
        "catastrophizing_score": cat_score,
        "external_locus_score": locus_score,
        "rumination_score": rumi_score,
        "uncertainty_score": unct_score,
        "avoidance_score": avoid_score,
        "threat_score": threat_score,
        "time_focus": detect_time_focus(sentence)
    }