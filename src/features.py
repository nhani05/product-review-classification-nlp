"""
Feature extraction module for text classification.
Implements BoW (CountVectorizer) and TF-IDF (TfidfVectorizer) with n-gram support
according to README.md Section 7.
Provides seamless scikit-learn integration with a robust fallback for standalone environments.
"""

import os
import re
import math
import pickle
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter


# Detect scikit-learn
try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class FallbackCountVectorizer:
    """Lightweight CountVectorizer fallback when scikit-learn is not installed."""

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 1),
        max_features: Optional[int] = 5000,
        min_df: int = 1,
    ):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.vocabulary_: Dict[str, int] = {}
        self.feature_names_: List[str] = []

    def _extract_ngrams(self, text: str) -> List[str]:
        words = re.findall(r"\b\w+\b", text.lower())
        ngrams = []
        min_n, max_n = self.ngram_range
        for n in range(min_n, max_n + 1):
            for i in range(len(words) - n + 1):
                ngrams.append(" ".join(words[i : i + n]))
        return ngrams

    def fit(self, raw_documents: List[str]):
        doc_freq = Counter()
        for doc in raw_documents:
            ngrams = set(self._extract_ngrams(doc))
            doc_freq.update(ngrams)

        # Filter by min_df
        valid_terms = [t for t, freq in doc_freq.items() if freq >= self.min_df]
        # Sort by frequency descending then alphabetically
        valid_terms.sort(key=lambda t: (-doc_freq[t], t))
        if self.max_features:
            valid_terms = valid_terms[: self.max_features]
        valid_terms.sort()

        self.vocabulary_ = {term: idx for idx, term in enumerate(valid_terms)}
        self.feature_names_ = valid_terms
        return self

    def transform(self, raw_documents: List[str]):
        import numpy as np

        rows = len(raw_documents)
        cols = len(self.vocabulary_)
        matrix = np.zeros((rows, cols), dtype=np.float32)

        for row_idx, doc in enumerate(raw_documents):
            ngrams = self._extract_ngrams(doc)
            counts = Counter(ngrams)
            for term, count in counts.items():
                if term in self.vocabulary_:
                    matrix[row_idx, self.vocabulary_[term]] = count
        return matrix

    def fit_transform(self, raw_documents: List[str]):
        return self.fit(raw_documents).transform(raw_documents)

    def get_feature_names_out(self) -> List[str]:
        return self.feature_names_


class FallbackTfidfVectorizer(FallbackCountVectorizer):
    """Lightweight TfidfVectorizer fallback with sublinear/smooth idf and L2 norm."""

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 1),
        max_features: Optional[int] = 5000,
        min_df: int = 1,
    ):
        super().__init__(ngram_range, max_features, min_df)
        self.idf_: List[float] = []

    def fit(self, raw_documents: List[str]):
        super().fit(raw_documents)
        total_docs = len(raw_documents)
        cols = len(self.feature_names_)
        doc_counts = [0] * cols

        for doc in raw_documents:
            ngrams = set(self._extract_ngrams(doc))
            for term in ngrams:
                if term in self.vocabulary_:
                    doc_counts[self.vocabulary_[term]] += 1

        # Smooth IDF: log((1 + n) / (1 + df)) + 1
        self.idf_ = [
            math.log((1 + total_docs) / (1 + df)) + 1.0 for df in doc_counts
        ]
        return self

    def transform(self, raw_documents: List[str]):
        import numpy as np

        tf_matrix = super().transform(raw_documents)
        if len(self.idf_) == 0:
            return tf_matrix

        idf_arr = np.array(self.idf_, dtype=np.float32)
        tfidf_matrix = tf_matrix * idf_arr

        # L2 normalize rows
        norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return tfidf_matrix / norms

    def fit_transform(self, raw_documents: List[str]):
        return self.fit(raw_documents).transform(raw_documents)


def build_vectorizer(
    method: str = "tfidf",
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 5000,
    min_df: int = 2,
    **kwargs,
):
    """
    Factory function creating CountVectorizer or TfidfVectorizer.
    """
    method = method.lower().strip()
    if method not in ("bow", "tfidf"):
        raise ValueError(f"Method không hợp lệ: '{method}'. Phải là 'bow' hoặc 'tfidf'.")

    if HAS_SKLEARN:
        if method == "bow":
            return CountVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                min_df=min_df,
                **kwargs,
            )
        return TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            **kwargs,
        )

    if method == "bow":
        return FallbackCountVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
        )
    return FallbackTfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
    )


def extract_features(
    train_texts: List[str],
    test_texts: Optional[List[str]] = None,
    method: str = "tfidf",
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 5000,
    min_df: int = 2,
):
    """
    Fits vectorizer on train_texts and transforms train and optional test texts.
    Returns (X_train, X_test, vectorizer).
    """
    vectorizer = build_vectorizer(
        method=method,
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts) if test_texts is not None else None
    return X_train, X_test, vectorizer


def save_vectorizer(vectorizer: Any, filepath: str) -> None:
    """Saves fitted vectorizer to disk."""
    folder = os.path.dirname(filepath)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"[+] Đã lưu vectorizer vào '{filepath}'")


def load_vectorizer(filepath: str) -> Any:
    """Loads vectorizer from disk."""
    with open(filepath, "rb") as f:
        return pickle.load(f)
