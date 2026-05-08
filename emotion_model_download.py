import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "bhadresh-savani/bert-base-go-emotion"


def download_model():
    try:
        logger.info(f"Downloading tokenizer for {MODEL_NAME}...")
        AutoTokenizer.from_pretrained(MODEL_NAME, revision="main")

        logger.info(f"Downloading model for {MODEL_NAME}...")
        AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            revision="main",
            trust_remote_code=False
        )

        logger.info("✅ Emotion model downloaded successfully!")

    except Exception:
        logger.exception("❌ Failed to download emotion model")
        raise


if __name__ == "__main__":
    download_model()