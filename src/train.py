"""
Training module for Naive Bayes and SVM review classification.
Supports BoW and TF-IDF feature representations, hyperparameter tuning,
and evaluation reporting according to README.md Section 8, 9, 10.
"""

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import os
import csv
import pickle
import argparse
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from features import extract_features, save_vectorizer
from evaluate import (
    compute_metrics,
    print_evaluation_report,
    plot_and_save_confusion_matrix,
    update_metrics_csv,
)

# Detect scikit-learn
try:
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class FallbackMultinomialNB:
    """Multinomial Naive Bayes classifier fallback."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.class_log_prior_: np.ndarray = np.array([])
        self.feature_log_prob_: np.ndarray = np.array([])
        self.classes_: np.ndarray = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        class_counts = np.array([np.sum(y == c) for c in self.classes_], dtype=np.float64)
        self.class_log_prior_ = np.log(class_counts / n_samples)

        feature_counts = np.zeros((n_classes, n_features), dtype=np.float64)
        for idx, c in enumerate(self.classes_):
            feature_counts[idx, :] = np.sum(X[y == c], axis=0)

        smoothed_fc = feature_counts + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1, keepdims=True)
        self.feature_log_prob_ = np.log(smoothed_fc / smoothed_cc)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        jll = X @ self.feature_log_prob_.T + self.class_log_prior_
        return self.classes_[np.argmax(jll, axis=1)]


class FallbackLinearSVM:
    """Linear SVM (hinge loss SGD with class weighting) fallback."""

    def __init__(self, C: float = 1.0, lr: float = 0.05, epochs: int = 40):
        self.C = C
        self.lr = lr
        self.epochs = epochs
        self.weights_: np.ndarray = np.array([])
        self.bias_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        # Convert y to {-1, +1}
        y_signed = np.where(y == 1, 1.0, -1.0)

        # Class weights for imbalanced data
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        pos_weight = n_neg / max(n_pos, 1)

        rng = np.random.RandomState(42)
        self.weights_ = rng.normal(0, 0.01, size=n_features).astype(np.float32)
        self.bias_ = 0.0

        for epoch in range(self.epochs):
            indices = rng.permutation(n_samples)
            current_lr = self.lr / (1.0 + 0.005 * epoch)

            for i in indices:
                xi = X[i]
                yi = y_signed[i]
                weight_mult = pos_weight if yi == 1.0 else 1.0
                margin = yi * (np.dot(xi, self.weights_) + self.bias_)

                if margin < 1.0:
                    grad_w = self.weights_ - self.C * weight_mult * yi * xi
                    grad_b = -self.C * weight_mult * yi
                else:
                    grad_w = self.weights_
                    grad_b = 0.0

                self.weights_ -= (current_lr * grad_w).astype(np.float32)
                self.bias_ -= current_lr * grad_b
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = np.dot(X, self.weights_) + self.bias_
        return np.where(scores >= 0, 1, 0)


def stratified_split(
    texts: List[str],
    labels: List[int],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[str], List[str], List[int], List[int]]:
    """Splits texts and labels maintaining class distribution."""
    if HAS_SKLEARN:
        return train_test_split(
            texts, labels, test_size=test_size, stratify=labels, random_state=random_state
        )

    rng = np.random.RandomState(random_state)
    pos_idx = [i for i, y in enumerate(labels) if y == 1]
    neg_idx = [i for i, y in enumerate(labels) if y == 0]

    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    split_pos = int(len(pos_idx) * (1 - test_size))
    split_neg = int(len(neg_idx) * (1 - test_size))

    train_indices = pos_idx[:split_pos] + neg_idx[:split_neg]
    test_indices = pos_idx[split_pos:] + neg_idx[split_neg:]

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    train_texts = [texts[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_texts = [texts[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]

    return train_texts, test_texts, train_labels, test_labels


def train_single_experiment(
    train_texts: List[str],
    test_texts: List[str],
    train_labels: List[int],
    test_labels: List[int],
    feature_type: str,
    model_type: str,
    tune: bool = False,
    models_dir: str = "results/models",
) -> Dict[str, Any]:
    """Trains, predicts, and evaluates a specific (feature, model) combination."""
    print(f"\n>>> Thực nghiệm: Feature={feature_type.upper()}, Model={model_type.upper()}")

    # 1. Trích chọn đặc trưng
    X_train, X_test, vectorizer = extract_features(
        train_texts=train_texts,
        test_texts=test_texts,
        method=feature_type,
        ngram_range=(1, 2),
        max_features=5000,
        min_df=2,
    )
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    # 2. Huấn luyện mô hình
    if HAS_SKLEARN:
        if model_type == "nb":
            model = MultinomialNB()
            if tune:
                print("[+] GridSearchCV tuning alpha cho Naive Bayes...")
                grid = GridSearchCV(model, {"alpha": [0.1, 0.5, 1.0, 2.0]}, cv=3, scoring="f1")
                grid.fit(X_train, y_train)
                model = grid.best_estimator_
                print(f"[+] Tham số tối ưu: {grid.best_params_}")
            else:
                model.fit(X_train, y_train)
        else:
            model = LinearSVC(class_weight="balanced", random_state=42)
            if tune:
                print("[+] GridSearchCV tuning C cho LinearSVC...")
                grid = GridSearchCV(model, {"C": [0.1, 0.5, 1.0, 5.0]}, cv=3, scoring="f1")
                grid.fit(X_train, y_train)
                model = grid.best_estimator_
                print(f"[+] Tham số tối ưu: {grid.best_params_}")
            else:
                model.fit(X_train, y_train)
    else:
        if model_type == "nb":
            model = FallbackMultinomialNB(alpha=0.5 if feature_type == "bow" else 0.1)
            model.fit(X_train, y_train)
        else:
            model = FallbackLinearSVM(C=1.0)
            model.fit(X_train, y_train)

    # 3. Dự đoán trên tập kiểm tra
    y_pred = list(model.predict(X_test))

    # 4. Đánh giá và lưu kết quả
    metrics = compute_metrics(test_labels, y_pred)
    print_evaluation_report(metrics, feature_type, model_type)
    plot_and_save_confusion_matrix(metrics["confusion_matrix"], feature_type, model_type)
    update_metrics_csv(feature_type, model_type, metrics)

    # 5. Lưu mô hình & vectorizer
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"model_{feature_type}_{model_type}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "vectorizer": vectorizer}, f)
    print(f"[+] Đã lưu mô hình đã huấn luyện vào '{model_path}'")

    return metrics


def load_dataset(filepath: str) -> Tuple[List[str], List[int]]:
    """Loads clean text and labels from processed CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {filepath}")

    texts = []
    labels = []
    with open(filepath, mode="r", encoding="utf-8-sig", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            txt = row.get("review_clean") or row.get("review") or row.get("noi_dung") or ""
            lbl_raw = row.get("label", "0")
            try:
                lbl = int(lbl_raw)
            except ValueError:
                lbl = 0
            texts.append(txt)
            labels.append(lbl)

    return texts, labels


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện và đánh giá mô hình phân loại Spam/Ham.")
    parser.add_argument(
        "--input",
        default="data/processed/reviews_clean.csv",
        help="Đường dẫn file dữ liệu sạch (mặc định: data/processed/reviews_clean.csv)",
    )
    parser.add_argument(
        "--features",
        choices=["bow", "tfidf", "all"],
        default="tfidf",
        help="Phương pháp trích chọn đặc trưng: bow, tfidf hoặc all (mặc định: tfidf)",
    )
    parser.add_argument(
        "--model",
        choices=["nb", "svm", "all"],
        default="svm",
        help="Thuật toán phân loại: nb, svm hoặc all (mặc định: svm)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Kích hoạt tìm kiếm siêu tham số bằng GridSearchCV",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Tỷ lệ tập kiểm tra (mặc định: 0.2)",
    )
    args = parser.parse_args()

    print(f"[*] Đang tải dữ liệu từ '{args.input}'...")
    texts, labels = load_dataset(args.input)
    print(f"[+] Tổng số mẫu: {len(texts)} (Ham: {labels.count(0)}, Spam: {labels.count(1)})")

    train_texts, test_texts, train_labels, test_labels = stratified_split(
        texts, labels, test_size=args.test_size, random_state=42
    )
    print(f"[+] Chia tập: Huấn luyện = {len(train_texts)}, Kiểm tra = {len(test_texts)}")

    feature_list = ["bow", "tfidf"] if args.features == "all" else [args.features]
    model_list = ["nb", "svm"] if args.model == "all" else [args.model]

    for feat in feature_list:
        for mdl in model_list:
            train_single_experiment(
                train_texts=train_texts,
                test_texts=test_texts,
                train_labels=train_labels,
                test_labels=test_labels,
                feature_type=feat,
                model_type=mdl,
                tune=args.tune,
            )

    print("\n[V] Đã hoàn thành toàn bộ các thực nghiệm! Bảng so sánh tại 'results/metrics.csv'.")


if __name__ == "__main__":
    main()
