"""Type or paste an email and see its category.   Run:  python live.py"""
import logging

from dotenv import load_dotenv

from main import build_classifier
from segmenter.classifier import Email
from segmenter.config import load_config


def read_body() -> str:
    print("Body (paste the email, then press Enter on an empty line):")
    lines = []
    while True:
        line = input()
        if not line.strip():
            break
        lines.append(line)
    return " ".join(lines)


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.WARNING)  # hides the technical INFO lines
    try:
        classifier = build_classifier(load_config("config.yaml"))
    except (RuntimeError, ValueError, ImportError) as exc:
        print(f"Setup problem: {exc}")
        return

    print("Email classifier ready. Press Enter on an empty subject to quit.")
    while True:
        subject = input("\nSubject: ").strip()
        if not subject:
            break
        sender = input("From (optional): ").strip()
        body = read_body()

        result = classifier.classify(Email(id="live", sender=sender, subject=subject, body=body))
        if result.error:
            print(f"Could not classify: {result.error}")
            continue
        print(f"\n=> {result.category.upper()}  (confidence {result.confidence:.2f})")
        print(f"   Reason: {result.reason}")
        if result.needs_review:
            print("   Note: low confidence, so check this one yourself.")


if __name__ == "__main__":
    main()
    