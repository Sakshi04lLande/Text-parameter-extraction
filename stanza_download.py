import stanza
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LANGUAGES = ["en", "hi", "mr"]
PROCESSORS = "tokenize,pos,lemma,depparse"


def download_stanza_models():
    """
    Download required Stanza models safely.
    """
    for lang in LANGUAGES:
        try:
            logger.info(f"Downloading {lang} model...")
            stanza.download(lang, processors=PROCESSORS)
            logger.info(f"✅ {lang} downloaded")
        except Exception:
            logger.exception(f"❌ Failed to download {lang}")

    logger.info("🎯 Stanza setup complete")


if __name__ == "__main__":
    download_stanza_models()