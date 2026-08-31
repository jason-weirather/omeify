from __future__ import annotations


def readable_runtime(seconds: float) -> str:
    """Render elapsed seconds as ``HH:MM:SS.ss``."""

    hours, remainder = divmod(float(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:05.2f}"
