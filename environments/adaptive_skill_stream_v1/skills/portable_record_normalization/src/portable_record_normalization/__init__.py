"""Stable record normalization used by the installed-skill curriculum arm."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping


def canonicalize(label: str) -> str:
    """Return the portable canonical form of one label."""
    if not isinstance(label, str):
        raise TypeError("label must be a string")
    return re.sub(r"[^a-z0-9]+", "-", label.strip().lower()).strip("-")


def summarize(records: Iterable[Mapping[str, object]]) -> dict[str, int]:
    """Sum integer `amount` fields by canonicalized `label`."""
    totals: dict[str, int] = {}
    for record in records:
        label = record.get("label")
        amount = record.get("amount")
        if not isinstance(label, str):
            raise TypeError("every record label must be a string")
        if not isinstance(amount, int) or isinstance(amount, bool):
            raise TypeError("every record amount must be an integer")
        key = canonicalize(label)
        totals[key] = totals.get(key, 0) + amount
    return dict(sorted(totals.items()))


__all__ = ["canonicalize", "summarize"]
