"""
Preprocessing module for Vietnamese customer review text.
Implements the 5-step pipeline specified in README.md Section 6:
1. Lowercasing, Vietnamese Unicode normalization (NFC).
2. Feature extraction (has_phone, has_url) & removal of URLs, phones, emails, HTML, emoji, special chars.
3. Shortening repeated characters (e.g. 'ngonnnnn' -> 'ngon').
4. Vietnamese word tokenization (underthesea or regex compound fallback).
5. Stopword filtering using stopwords/vietnamese-stopwords.txt.
"""

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import os
import re
import csv
import unicodedata
import argparse
from typing import Set, List, Optional, Tuple, Dict, Any


# Regex patterns for cleaning
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+|\b[a-zA-Z0-9-]+\.(?:com|vn|net|org|io|xyz|top)\b", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"(?:\+?84|0)(?:\d[\s\.]?){8,10}\d")
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
HTML_TAG_PATTERN = re.compile(r"<.*?>")
SPECIAL_CHARS_PATTERN = re.compile(r"[^\w\s_À-ỹ]")
REPEATED_CHAR_PATTERN = re.compile(r"(\w)\1{2,}")


def load_stopwords(filepath: str) -> Set[str]:
    """Loads Vietnamese stop words from a text file."""
    if not os.path.exists(filepath):
        return set()
    with open(filepath, "r", encoding="utf-8") as f:
        words = {line.strip().lower() for line in f if line.strip() and not line.startswith("#")}
    return words


def normalize_unicode(text: str) -> str:
    """Standardizes text to Unicode NFC form."""
    return unicodedata.normalize("NFC", text)


def shorten_repeated_chars(text: str) -> str:
    """Shortens character repetitions like 'ngonnnnn' to 'ngon'."""
    return REPEATED_CHAR_PATTERN.sub(r"\1", text)


def tokenize_vietnamese(text: str) -> List[str]:
    """
    Tokenizes Vietnamese text into words/compound phrases.
    Prefers underthesea; falls back to clean whitespace splitting.
    """
    try:
        from underthesea import word_tokenize
        tokenized = word_tokenize(text, format="text")
        return tokenized.split()
    except Exception:
        # Fallback tokenizer
        return text.split()


def preprocess_text(
    text: str,
    stopwords: Optional[Set[str]] = None,
    keep_indicators: bool = False,
) -> Tuple[str, Dict[str, int]]:
    """
    Executes end-to-end preprocessing pipeline on a single string.
    Returns: (cleaned_text, indicator_dict)
    """
    indicators = {"has_phone": 0, "has_url": 0}
    if not text:
        return "", indicators

    # 1. Lowercase & Unicode NFC
    text = normalize_unicode(text.lower().strip())

    # 2. Extract indicators before removal
    if PHONE_PATTERN.search(text):
        indicators["has_phone"] = 1
    if URL_PATTERN.search(text):
        indicators["has_url"] = 1

    # Remove URLs, Phones, Emails, HTML
    text = URL_PATTERN.sub(" ", text)
    text = PHONE_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)
    text = HTML_TAG_PATTERN.sub(" ", text)

    # 3. Shorten repeated chars
    text = shorten_repeated_chars(text)

    # Remove special chars, numbers, emojis
    text = SPECIAL_CHARS_PATTERN.sub(" ", text)

    # 4. Vietnamese word tokenization
    tokens = tokenize_vietnamese(text)

    # 5. Remove stop words
    if stopwords:
        tokens = [t for t in tokens if t not in stopwords and len(t) > 1]

    cleaned_text = " ".join(tokens).strip()
    return cleaned_text, indicators


def process_dataset(
    input_file: str,
    output_file: str,
    stopwords_file: str = "stopwords/vietnamese-stopwords.txt",
) -> None:
    """Preprocesses entire dataset and saves to processed/."""
    stopwords = load_stopwords(stopwords_file)
    print(f"[+] Đã tải {len(stopwords)} từ dừng từ '{stopwords_file}'")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Không tìm thấy file: {input_file}")

    folder = os.path.dirname(output_file)
    if folder:
        os.makedirs(folder, exist_ok=True)

    rows = []
    with open(input_file, mode="r", encoding="utf-8-sig", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"[+] Bắt đầu tiền xử lý {len(rows)} bản ghi...")
    processed_rows = []

    for idx, row in enumerate(rows):
        raw_text = row.get("review") or row.get("noi_dung") or ""
        clean_text, indicators = preprocess_text(raw_text, stopwords=stopwords)

        processed_rows.append({
            "product": row.get("product") or row.get("san_pham") or "",
            "review_raw": raw_text,
            "review_clean": clean_text,
            "rating": row.get("rating") or row.get("so_sao") or "",
            "label": row.get("label", "0"),
            "has_phone": indicators["has_phone"],
            "has_url": indicators["has_url"],
        })

    with open(output_file, mode="w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "product",
            "review_raw",
            "review_clean",
            "rating",
            "label",
            "has_phone",
            "has_url",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(processed_rows)

    print(f"[+] Hoàn tất! Đã lưu {len(processed_rows)} bản ghi vào '{output_file}'.")


def main():
    parser = argparse.ArgumentParser(description="Tiền xử lý văn bản đánh giá tiếng Việt.")
    parser.add_argument(
        "--input",
        default="data/labeled/reviews_labeled.csv",
        help="File đánh giá đầu vào (mặc định: data/labeled/reviews_labeled.csv)",
    )
    parser.add_argument(
        "--output",
        default="data/processed/reviews_clean.csv",
        help="File đầu ra sau tiền xử lý (mặc định: data/processed/reviews_clean.csv)",
    )
    parser.add_argument(
        "--stopwords",
        default="stopwords/vietnamese-stopwords.txt",
        help="Đường dẫn file stopwords tiếng Việt",
    )
    args = parser.parse_args()

    process_dataset(args.input, args.output, args.stopwords)


if __name__ == "__main__":
    main()
