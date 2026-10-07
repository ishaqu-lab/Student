import re
import string
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

BASE_DIR = Path(__file__).resolve().parent
NLTK_DATA_DIR = BASE_DIR / "nltk_data"
NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
if str(NLTK_DATA_DIR) not in nltk.data.path:
    nltk.data.path.insert(0, str(NLTK_DATA_DIR))


def ensure_nltk_resources():
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }

    for path, package in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, download_dir=str(NLTK_DATA_DIR), quiet=True)


ensure_nltk_resources()

try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError as exc:
    raise RuntimeError(
        "NLTK stopwords are unavailable. Run the application's build step "
        "to download the required NLTK resources."
    ) from exc

# Keep negation words because removing words such as "not" can reverse
# the meaning of student feedback (e.g. "not helpful" -> "helpful").
NEGATION_WORDS = {
    "not", "no", "nor", "never", "neither", "without",
    "cannot", "can't", "isn't", "wasn't", "aren't", "weren't",
    "don't", "doesn't", "didn't", "won't", "wouldn't", "shouldn't",
    "couldn't", "can't"
}
STOP_WORDS = STOP_WORDS - NEGATION_WORDS
LEMMATIZER = WordNetLemmatizer()


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    try:
        tokens = nltk.word_tokenize(text)
    except LookupError as exc:
        raise RuntimeError(
            "NLTK tokenizer resources are unavailable. Run the application's "
            "build step to download the required NLTK resources."
        ) from exc

    tokens = [
        LEMMATIZER.lemmatize(token)
        for token in tokens
        if token.isalpha() and token not in STOP_WORDS
    ]
    return " ".join(tokens)
