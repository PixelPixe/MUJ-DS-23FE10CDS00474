#!/usr/bin/env python3
"""Command-line entry point for the email segmentation project.

Examples:
  python main.py --csv data/sample_emails.csv
  python main.py --eml-dir my_emails/
  python main.py --subject "Lunch tomorrow?" --body "Are we still on for 1 PM?"
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from segmenter.cache import PredictionCache
from segmenter.classifier import Email, EmailClassifier, Prediction
from segmenter.config import Config, load_config
from segmenter.data_loader import load_csv, load_eml_dir
from segmenter.evaluate import evaluate, format_report
from segmenter.llm_client import create_client
from segmenter.prompts import load_prompts

logger = logging.getLogger("segmenter")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Segment emails into Work / Personal / Commercial using an LLM.")
    p.add_argument("--csv", help="CSV file with subject/body columns (optional: id, sender, label)")
    p.add_argument("--eml-dir", help="Folder containing .eml files")
    p.add_argument("--subject", help="Classify one email: subject line")
    p.add_argument("--body", help="Classify one email: body text")
    p.add_argument("--sender", default="", help="Classify one email: sender address")
    p.add_argument("--config", default="config.yaml", help="Path to config file")
    p.add_argument("--output", help="Where to write predictions (default: <output_dir>/predictions.csv)")
    p.add_argument("--limit", type=int, help="Only process the first N emails")
    p.add_argument("--workers", type=int, help="Override number of parallel API calls")
    p.add_argument("--no-cache", action="store_true", help="Ignore and do not update the cache")
    p.add_argument("-v", "--verbose", action="store_true", help="Show debug logs")
    args = p.parse_args(argv)
    if not (args.csv or args.eml_dir or args.subject or args.body):
        p.error("provide --csv, --eml-dir, or --subject/--body")
    return args


def load_emails(args: argparse.Namespace) -> list[Email]:
    if args.csv:
        emails = load_csv(args.csv)
    elif args.eml_dir:
        emails = load_eml_dir(args.eml_dir)
    else:
        emails = [Email(id="1", sender=args.sender, subject=args.subject or "", body=args.body or "")]
    return emails[: args.limit] if args.limit else emails


def build_classifier(cfg: Config) -> EmailClassifier:
    prompts = load_prompts(cfg.runtime.prompt_file)
    for category in cfg.classification.categories:
        if category not in prompts.system:
            logger.warning("Category %r is in config.yaml but not mentioned in the prompt file", category)
    return EmailClassifier(
        client=create_client(cfg.llm),
        prompts=prompts,
        cfg=cfg.classification,
        cache=PredictionCache(cfg.runtime.cache_path, cfg.runtime.cache_enabled),
        model_name=f"{cfg.llm.provider}:{cfg.llm.model}",
    )


def write_results(path: Path, emails: list[Email], predictions: list[Prediction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    has_labels = any(e.label for e in emails)
    header = ["id", "sender", "subject", "predicted_category", "confidence", "needs_review", "reason"]
    if has_labels:
        header += ["true_label", "correct"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for e, p in zip(emails, predictions):
            row = [e.id, e.sender, e.subject, p.category, f"{p.confidence:.2f}", p.needs_review,
                   p.reason or p.error or ""]
            if has_labels:
                row += [e.label or "", (e.label or "").lower() == p.category.lower() if e.label else ""]
            writer.writerow(row)


def print_summary(predictions: list[Prediction]) -> None:
    counts = Counter(p.category for p in predictions)
    total = len(predictions)
    print("\nCategory distribution")
    for category, n in counts.most_common():
        print(f"  {category:<13}{n:>4}  ({n / total:>5.1%})  {'#' * round(30 * n / total)}")
    review = sum(p.needs_review for p in predictions)
    if review:
        print(f"\n{review} email(s) flagged needs_review (low confidence or failed call).")


def run_evaluation(emails: list[Email], predictions: list[Prediction], cfg: Config, out_dir: Path) -> None:
    canonical = {c.lower(): c for c in cfg.classification.categories}
    pairs = [(canonical[e.label.lower()], p.category)
             for e, p in zip(emails, predictions) if e.label and e.label.lower() in canonical]
    if not pairs:
        return
    skipped = sum(1 for e in emails if e.label) - len(pairs)
    if skipped:
        logger.warning("%d labelled email(s) skipped: label not in %s", skipped, list(canonical.values()))
    y_true, y_pred = zip(*pairs)
    metrics = evaluate(list(y_true), list(y_pred), cfg.classification.categories)
    print("\n" + format_report(metrics))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    cfg = load_config(args.config)
    if args.workers:
        cfg = replace(cfg, runtime=replace(cfg.runtime, workers=args.workers))
    if args.no_cache:
        cfg = replace(cfg, runtime=replace(cfg.runtime, cache_enabled=False))

    emails = load_emails(args)
    try:
        classifier = build_classifier(cfg)
    except (RuntimeError, ValueError, ImportError) as exc:
        logger.error("%s", exc)
        return 1

    predictions = classifier.classify_many(emails, workers=cfg.runtime.workers)

    out_dir = Path(cfg.runtime.output_dir)
    out_path = Path(args.output) if args.output else out_dir / "predictions.csv"
    write_results(out_path, emails, predictions)

    if len(emails) == 1:
        p = predictions[0]
        print(json.dumps({"category": p.category, "confidence": p.confidence, "reason": p.reason,
                          "needs_review": p.needs_review}, indent=2))
    else:
        print_summary(predictions)
        run_evaluation(emails, predictions, cfg, out_dir)
    print(f"\nResults saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
