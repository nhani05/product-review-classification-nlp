"""
Evaluation module: Computes metrics, prints classification reports,
plots/saves confusion matrix figures, and records results into results/metrics.csv
according to README.md Section 10.
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
from typing import List, Dict, Any, Tuple, Optional


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    """
    Computes Accuracy, Precision, Recall, F1-Score for macro, weighted,
    and specifically for class 1 (Spam), plus the Confusion Matrix.
    """
    total = len(y_true)
    if total == 0:
        return {}

    # True Negative, False Positive, False Negative, True Positive
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)

    accuracy = (tp + tn) / total

    # Class 1 (Spam)
    spam_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    spam_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spam_f1 = (
        2 * spam_prec * spam_rec / (spam_prec + spam_rec)
        if (spam_prec + spam_rec) > 0
        else 0.0
    )

    # Class 0 (Ham)
    ham_prec = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    ham_rec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ham_f1 = (
        2 * ham_prec * ham_rec / (ham_prec + ham_rec)
        if (ham_prec + ham_rec) > 0
        else 0.0
    )

    # Macro averages
    prec_macro = (ham_prec + spam_prec) / 2.0
    rec_macro = (ham_rec + spam_rec) / 2.0
    f1_macro = (ham_f1 + spam_f1) / 2.0

    # Weighted averages
    n_ham = tn + fp
    n_spam = fn + tp
    prec_weighted = (ham_prec * n_ham + spam_prec * n_spam) / total
    rec_weighted = (ham_rec * n_ham + spam_rec * n_spam) / total
    f1_weighted = (ham_f1 * n_ham + spam_f1 * n_spam) / total

    cm = [[tn, fp], [fn, tp]]

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(prec_weighted, 4),
        "recall": round(rec_weighted, 4),
        "f1_score": round(f1_weighted, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "spam_precision": round(spam_prec, 4),
        "spam_recall": round(spam_rec, 4),
        "spam_f1": round(spam_f1, 4),
        "ham_precision": round(ham_prec, 4),
        "ham_recall": round(ham_rec, 4),
        "ham_f1": round(ham_f1, 4),
        "confusion_matrix": cm,
        "support": {"ham": n_ham, "spam": n_spam, "total": total},
    }


def print_evaluation_report(
    metrics: Dict[str, Any], feature_name: str, model_name: str
) -> None:
    """Prints a styled evaluation report."""
    cm = metrics["confusion_matrix"]
    sup = metrics["support"]

    print("\n" + "=" * 60)
    print(f" BÁO CÁO ĐÁNH GIÁ: {feature_name.upper()} + {model_name.upper()}")
    print("=" * 60)
    print(f"Tổng mẫu kiểm tra: {sup['total']} (Ham: {sup['ham']}, Spam: {sup['spam']})")
    print("-" * 60)
    print(f"{'Class':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 60)
    print(
        f"{'Ham (0)':<12} | {metrics['ham_precision']:<10.4f} | {metrics['ham_recall']:<10.4f} | {metrics['ham_f1']:<10.4f}"
    )
    print(
        f"{'Spam (1)':<12} | {metrics['spam_precision']:<10.4f} | {metrics['spam_recall']:<10.4f} | {metrics['spam_f1']:<10.4f}"
    )
    print("-" * 60)
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Macro F1-Score:  {metrics['f1_macro']:.4f}")
    print(f"Weighted F1:     {metrics['f1_score']:.4f}")
    print("\nConfusion Matrix:")
    print(f"                Dự đoán Ham    Dự đoán Spam")
    print(f"  Thực tế Ham     {cm[0][0]:<14} {cm[0][1]}")
    print(f"  Thực tế Spam    {cm[1][0]:<14} {cm[1][1]}")
    print("=" * 60)


def plot_and_save_confusion_matrix(
    cm: List[List[int]],
    feature_name: str,
    model_name: str,
    output_dir: str = "results/figures",
) -> Optional[str]:
    """
    Saves confusion matrix as a PNG heatmap if matplotlib/seaborn are available,
    and always saves a text summary.
    """
    os.makedirs(output_dir, exist_ok=True)
    slug = f"{feature_name.lower()}_{model_name.lower()}"
    img_path = os.path.join(output_dir, f"cm_{slug}.png")
    txt_path = os.path.join(output_dir, f"cm_{slug}.txt")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Confusion Matrix: {feature_name.upper()} + {model_name.upper()}\n")
        f.write(f"TN: {cm[0][0]}, FP: {cm[0][1]}\n")
        f.write(f"FN: {cm[1][0]}, TP: {cm[1][1]}\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Ham (0)", "Spam (1)"],
            yticklabels=["Ham (0)", "Spam (1)"],
        )
        plt.title(f"Confusion Matrix ({feature_name.upper()} - {model_name.upper()})")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(img_path, dpi=300)
        plt.close()
        print(f"[+] Đã lưu biểu đồ Confusion Matrix vào '{img_path}'")
        return img_path
    except Exception:
        print(f"[+] Đã lưu bản ghi ma trận vào '{txt_path}'")
        return txt_path


def update_metrics_csv(
    feature_name: str,
    model_name: str,
    metrics: Dict[str, Any],
    filepath: str = "results/metrics.csv",
) -> None:
    """
    Updates results/metrics.csv with columns matching README.md Section 10:
    Đặc trưng | Mô hình | Accuracy | Precision | Recall | F1-Score | (kèm Spam Recall, Spam F1)
    """
    folder = os.path.dirname(filepath)
    if folder:
        os.makedirs(folder, exist_ok=True)

    fieldnames = [
        "feature",
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "spam_recall",
        "spam_f1",
    ]

    existing_rows = []
    if os.path.exists(filepath):
        with open(filepath, mode="r", encoding="utf-8-sig", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Deduplicate by (feature, model)
                if not (
                    row.get("feature", "").lower() == feature_name.lower()
                    and row.get("model", "").lower() == model_name.lower()
                ):
                    existing_rows.append(row)

    new_row = {
        "feature": feature_name.upper(),
        "model": model_name.upper(),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "spam_recall": metrics["spam_recall"],
        "spam_f1": metrics["spam_f1"],
    }
    existing_rows.append(new_row)

    with open(filepath, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)

    print(f"[+] Đã cập nhật kết quả vào '{filepath}'")
