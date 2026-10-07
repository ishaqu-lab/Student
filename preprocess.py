import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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
            try:
                nltk.download(package, quiet=True)
            except Exception:
                pass

ensure_nltk_resources()

try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))

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
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        tokens = nltk.word_tokenize(text)

    tokens = [
        LEMMATIZER.lemmatize(token)
        for token in tokens
        if token.isalpha() and token not in STOP_WORDS
    ]
    return " ".join(tokens)
