from typing import List
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Load model once
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def embed(sentences: List[str]):
    """
    Generate sentence embeddings safely.
    """
    try:
        if not sentences:
            return []

        embeddings = model.encode(
            sentences,
            batch_size=16,
            convert_to_numpy=True
        )

        return embeddings

    except Exception:
        logger.exception("Embedding generation failed")
        return []