from __future__ import annotations

from pathlib import Path

from config import get_settings


def load_prompt(name: str) -> str:
    """Load a prompt template from the prompts directory."""
    settings = get_settings()
    prompt_path = settings.prompts_dir / f"{name}.txt"
    if not prompt_path.exists():
        return ""
    return prompt_path.read_text(encoding="utf-8").strip()
