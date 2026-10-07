"""Accuracy, precision/recall/F1 and a confusion matrix, with no extra dependencies."""
from __future__ import annotations

from .classifier import UNCLASSIFIED


def evaluate(y_true: list[str], y_pred: list[str], categories: tuple[str, ...]) -> dict:
    labels = list(categories)
    columns = labels + [UNCLASSIFIED]
    matrix = {t: dict.fromkeys(columns, 0) for t in labels}
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred if pred in labels else UNCLASSIFIED] += 1

    per_class = {}
    for c in labels:
        tp = matrix[c][c]
        fp = sum(matrix[t][c] for t in labels if t != c)
        fn = sum(matrix[c][p] for p in columns if p != c)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    total = len(y_true)
    correct = sum(matrix[c][c] for c in labels)
    return {
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(m["f1"] for m in per_class.values()) / len(labels),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def format_report(metrics: dict) -> str:
    lines = [f"Accuracy: {metrics['accuracy']:.1%}   Macro-F1: {metrics['macro_f1']:.3f}   "
             f"(n={metrics['total']})", "",
             f"{'':<12}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}"]
    for name, m in metrics["per_class"].items():
        lines.append(f"{name:<12}{m['precision']:>10.2f}{m['recall']:>10.2f}"
                     f"{m['f1']:>10.2f}{m['support']:>10}")
    matrix = metrics["confusion_matrix"]
    columns = list(next(iter(matrix.values())))
    lines += ["", "Confusion matrix (rows = true label, columns = predicted)",
              f"{'':<12}" + "".join(f"{c:>13}" for c in columns)]
    for true, row in matrix.items():
        lines.append(f"{true:<12}" + "".join(f"{row[c]:>13}" for c in columns))
    return "\n".join(lines)
