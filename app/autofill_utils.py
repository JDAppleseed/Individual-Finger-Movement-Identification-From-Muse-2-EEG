from __future__ import annotations

from typing import Optional, Set


def should_replace_autofilled_text(
    current_text: str,
    previous_auto_text: Optional[str],
    legacy_values: Set[str],
) -> bool:
    current = current_text.strip()
    if not current:
        return True
    if current in legacy_values:
        return True
    if previous_auto_text and current == previous_auto_text:
        return True
    return False
