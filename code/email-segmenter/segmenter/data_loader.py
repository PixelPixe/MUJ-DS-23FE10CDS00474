"""Load emails from a CSV file or a folder of .eml files."""
from __future__ import annotations

import csv
import html
import re
from email import policy
from email.parser import BytesParser
from pathlib import Path

from .classifier import Email

_SCRIPT_STYLE = re.compile(r"<(script|style).*?</\1>", re.DOTALL | re.IGNORECASE)
_TAGS = re.compile(r"<[^>]+>")


def load_csv(path: str | Path) -> list[Email]:
    """CSV columns: subject, body (at least one), plus optional id, sender, label."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path} is empty")
        fields = {name.strip().lower() for name in reader.fieldnames}
        if not fields & {"subject", "body"}:
            raise ValueError(f"{path} needs a 'subject' and/or 'body' column")
        emails = []
        for i, raw_row in enumerate(reader, start=1):
            row = {k.strip().lower(): (v or "").strip() for k, v in raw_row.items() if k}
            emails.append(Email(
                id=row.get("id") or str(i),
                sender=row.get("sender", ""),
                subject=row.get("subject", ""),
                body=row.get("body", ""),
                label=row.get("label") or None,
            ))
    return emails


def _html_to_text(markup: str) -> str:
    return html.unescape(_TAGS.sub(" ", _SCRIPT_STYLE.sub(" ", markup)))


def load_eml_dir(folder: str | Path) -> list[Email]:
    """Parse every *.eml file in a folder (e.g. a Gmail/Outlook export)."""
    paths = sorted(Path(folder).glob("*.eml"))
    if not paths:
        raise ValueError(f"No .eml files found in {folder}")
    emails = []
    for path in paths:
        with open(path, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)
        part = msg.get_body(preferencelist=("plain", "html"))
        body = part.get_content() if part else ""
        if part and part.get_content_type() == "text/html":
            body = _html_to_text(body)
        emails.append(Email(
            id=path.stem,
            sender=str(msg.get("From", "")),
            subject=str(msg.get("Subject", "")),
            body=body,
        ))
    return emails
