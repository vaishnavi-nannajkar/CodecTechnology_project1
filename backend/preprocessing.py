"""
Text Preprocessing Module
Handles cleaning and normalization of tweet text.
"""

import re
import string
import logging

logger = logging.getLogger(__name__)

# Graceful imports — these are optional heavy dependencies
try:
    import spacy
    _spacy_available = True
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        _spacy_available = False
        logger.warning("spaCy model 'en_core_web_sm' not found. Lemmatization disabled.")
except ImportError:
    _spacy_available = False
    logger.warning("spaCy not installed. Lemmatization disabled.")

try:
    from nltk.corpus import stopwords
    import nltk
    try:
        STOPWORDS = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        STOPWORDS = set(stopwords.words("english"))
    _nltk_available = True
except ImportError:
    _nltk_available = False
    STOPWORDS = set()
    logger.warning("NLTK not installed. Stopword removal disabled.")

try:
    import emoji
    _emoji_available = True
except ImportError:
    _emoji_available = False


class TextPreprocessor:
    """
    Cleans raw tweet text through a multi-step pipeline:
    1. Lowercase
    2. Remove URLs
    3. Remove mentions (@user)
    4. Remove hashtag symbols (keep word)
    5. Remove emojis
    6. Remove punctuation & numbers
    7. Remove stopwords
    8. Lemmatize (if spaCy available)
    """

    def __init__(self, remove_stopwords: bool = True, lemmatize: bool = True):
        self.remove_stopwords = remove_stopwords and _nltk_available
        self.lemmatize = lemmatize and _spacy_available
        self._url_re = re.compile(r"https?://\S+|www\.\S+")
        self._mention_re = re.compile(r"@\w+")
        self._hashtag_re = re.compile(r"#(\w+)")
        self._num_re = re.compile(r"\d+")
        self._spaces_re = re.compile(r"\s+")

    def _remove_urls(self, text: str) -> str:
        return self._url_re.sub("", text)

    def _remove_mentions(self, text: str) -> str:
        return self._mention_re.sub("", text)

    def _clean_hashtags(self, text: str) -> str:
        """Remove # symbol but keep the word."""
        return self._hashtag_re.sub(r"\1", text)

    def _remove_emojis(self, text: str) -> str:
        if _emoji_available:
            return emoji.replace_emoji(text, replace="")
        # Fallback: strip common emoji Unicode ranges
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002700-\U000027BF"
            "\U0001F900-\U0001F9FF"
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub("", text)

    def _remove_punctuation(self, text: str) -> str:
        return text.translate(str.maketrans("", "", string.punctuation))

    def _remove_numbers(self, text: str) -> str:
        return self._num_re.sub("", text)

    def _remove_stopwords(self, text: str) -> str:
        if not self.remove_stopwords:
            return text
        return " ".join(w for w in text.split() if w not in STOPWORDS)

    def _lemmatize(self, text: str) -> str:
        if not self.lemmatize or not text.strip():
            return text
        try:
            doc = nlp(text)
            return " ".join(token.lemma_ for token in doc if not token.is_space)
        except Exception as e:
            logger.warning(f"Lemmatization error: {e}")
            return text

    def clean(self, text: str) -> str:
        """Full cleaning pipeline."""
        if not text or not isinstance(text, str):
            return ""
        try:
            text = text.lower()
            text = self._remove_urls(text)
            text = self._remove_mentions(text)
            text = self._clean_hashtags(text)
            text = self._remove_emojis(text)
            text = self._remove_punctuation(text)
            text = self._remove_numbers(text)
            text = self._spaces_re.sub(" ", text).strip()
            text = self._remove_stopwords(text)
            text = self._lemmatize(text)
            return text.strip()
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return text.lower().strip()

    def batch_clean(self, texts: list) -> list:
        """Clean a list of texts."""
        return [self.clean(t) for t in texts]