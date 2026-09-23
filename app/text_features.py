"""Text cleaners used by the saved NLP pipelines (importable, so joblib can unpickle them).

Copied from Week 6 with one fix: Week 6's clean_text() replaces links with " URL "
and then deletes every non-lowercase character, which also deletes the "URL"
token, so the model never learns "this SMS contains a link". Here the token is
lowercase ("urltoken") so it survives.
"""
import re

from sklearn.base import BaseEstimator, TransformerMixin


def clean_text(s):
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " urltoken ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def clean_url(s):
    s = str(s).lower()
    s = re.sub(r"https?://", " ", s)
    s = re.sub(r"[/\.\-_?=&]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


class Cleaner(BaseEstimator, TransformerMixin):
    def __init__(self, kind="sms"):
        self.kind = kind

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        f = clean_url if self.kind == "url" else clean_text
        return [f(x) for x in X]
