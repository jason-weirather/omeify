"""Display escaping; serialized source and model values stay unchanged."""
from __future__ import annotations


def escape_terminal(value: str, *, multiline: bool = False) -> str:
    """Show control/bidirectional characters literally, retaining scientific Unicode."""
    return "".join(
        char if char.isprintable() or (multiline and char == "\n")
        else char.encode("unicode_escape").decode("ascii")
        for char in value
    )
