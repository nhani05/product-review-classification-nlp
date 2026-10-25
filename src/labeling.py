"""
Labeling module: Assigns binary spam/ham labels to crawled raw reviews
according to criteria defined in README.md Section 5.3:
- 1 (spam): Advertisements, phone numbers, external links, social media handles,
            repeated gibberish, irrelevant content, or ultra-short meaningless text.
- 0 (ham): Valid customer product/experience reviews.
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
import argparse
from typing import Tuple, Dict, Any, List


# Regular expressions for spam detection
PHONE_REGEX = re.compile(r"(?:\+?84|0)(?:\d[\s\.]?){8,10}\d")
URL_REGEX = re.compile(
    r"https?://\S+|www\.\S+|\b[a-zA-Z0-9-]+\.(?:com|vn|net|org|io|xyz|top|me|cc)\b",
    re.IGNORECASE,
)
CONTACT_KEYWORDS = re.compile(
    r"\b(zalo|inbox|liên hệ|lh|kết bạn|call me|telegram|tele|viber|vay tiền|lãi suất|tuyển dụng|tuyendung|shopee|lazada|tiki|sđt|hotline)\b",
    re.IGNORECASE,
)
REPEATED_CHARS = re.compile(r"(\w)\1{4,}")
PUNCTUATION_OR_DIGITS_ONLY = re.compile(r"^[\s\d.,!?;:_\-+*#@%^&()=/\\~`'\"]+$")


def classify_spam(text: str, rating: Any = None) -> Tuple[int, str]:
    """
    Evaluates whether a review is spam (1) or ham (0).
    Returns (label, reason).
    """
    clean = (text or "").strip()

    if not clean:
        return 1, "empty"

    if len(clean) <= 2 or PUNCTUATION_OR_DIGITS_ONLY.match(clean):
        return 1, "too_short_or_meaningless"

    # Repeated meaningless phrases like "ok ok ok ok", "cham cham cham"
    tokens = clean.lower().split()
    if len(tokens) >= 3 and len(set(tokens)) == 1 and len(tokens[0]) <= 3:
        return 1, "repeated_tokens"

    if PHONE_REGEX.search(clean):
        return 1, "phone_number"

    if URL_REGEX.search(clean):
        return 1, "url_link"

    if CONTACT_KEYWORDS.search(clean):
        return 1, "contact_keyword"

    if REPEATED_CHARS.search(clean):
        return 1, "repeated_characters"

    return 0, "ham"


def process_labeling(input_path: str, output_path: str) -> Dict[str, Any]:
    """
    Reads raw reviews, applies labeling criteria, and saves the labeled dataset
    conforming to the schema: product, review, rating, label.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    folder = os.path.dirname(output_path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    with open(input_path, mode="r", encoding="utf-8-sig", errors="ignore") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    spam_count = 0
    ham_count = 0
    reason_stats: Dict[str, int] = {}

    labeled_data = []
    for r in rows:
        text = r.get("noi_dung", "").strip()
        product = r.get("san_pham", "").strip()
        rating = r.get("so_sao", "").strip()

        label, reason = classify_spam(text, rating)
        if label == 1:
            spam_count += 1
            reason_stats[reason] = reason_stats.get(reason, 0) + 1
        else:
            ham_count += 1

        labeled_data.append({
            "product": product,
            "review": text,
            "rating": rating,
            "label": label,
        })

    with open(output_path, mode="w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["product", "review", "rating", "label"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(labeled_data)

    stats = {
        "total": len(labeled_data),
        "spam": spam_count,
        "ham": ham_count,
        "spam_ratio": round(spam_count / max(len(labeled_data), 1), 4),
        "reason_stats": reason_stats,
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Gán nhãn Spam/Ham cho dữ liệu đánh giá.")
    parser.add_argument(
        "--input",
        default="data/raw/reviews_raw.csv",
        help="Đường dẫn file thô (mặc định: data/raw/reviews_raw.csv)",
    )
    parser.add_argument(
        "--output",
        default="data/labeled/reviews_labeled.csv",
        help="Đường dẫn file gán nhãn (mặc định: data/labeled/reviews_labeled.csv)",
    )
    args = parser.parse_args()

    print(f"[*] Đang gán nhãn từ '{args.input}' -> '{args.output}'...")
    stats = process_labeling(args.input, args.output)

    print("\n========== THỐNG KÊ GÁN NHÃN ==========")
    print(f"Tổng số review: {stats['total']}")
    print(f"Số review ham (0): {stats['ham']}")
    print(f"Số review spam (1): {stats['spam']}")
    print(f"Tỷ lệ spam: {stats['spam_ratio'] * 100:.2f}%")
    print("Phân bố lý do spam:", stats["reason_stats"])
    print("========================================")


if __name__ == "__main__":
    main()
