"""Connection profiles (manager + worker URLs/keys) and small dataclasses
shared across DevFlock. Profiles are saved under ~/.devflock/profiles.json
with 0600 permissions so tokens aren't world-readable; nothing here ever logs
a raw key.
"""
from __future__ import annotations

import dataclasses
import json
import os
import stat
from pathlib import Path
from typing import Optional

DEVFLOCK_HOME = Path.home() / ".devflock"
PROFILES_PATH = DEVFLOCK_HOME / "profiles.json"


@dataclasses.dataclass
class Endpoint:
    """One llama-server-style Anthropic-compatible endpoint (manager or worker)."""
    name: str
    base_url: str
    api_key: str
    auth_style: str = "x-api-key"  # or "bearer"
    model_name: str = "qwen"

    def masked(self) -> str:
        k = self.api_key
        shown = k[:4] + "…" + k[-2:] if len(k) > 8 else "…"
        return f"{self.name} @ {self.base_url} (key {shown})"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Endpoint":
        return cls(**d)


def ensure_home():
    DEVFLOCK_HOME.mkdir(parents=True, exist_ok=True)
    os.chmod(DEVFLOCK_HOME, stat.S_IRWXU)  # 0700


def load_profiles() -> dict[str, Endpoint]:
    if not PROFILES_PATH.exists():
        return {}
    data = json.loads(PROFILES_PATH.read_text())
    return {k: Endpoint.from_dict(v) for k, v in data.items()}


def save_profiles(profiles: dict[str, Endpoint]):
    ensure_home()
    PROFILES_PATH.write_text(json.dumps({k: v.to_dict() for k, v in profiles.items()}, indent=2))
    os.chmod(PROFILES_PATH, stat.S_IRUSR | stat.S_IWUSR)  # 0600


@dataclasses.dataclass
class RunConfig:
    """Top-level settings for one DevFlock run, collected by the wizard."""
    project_dir: str
    manager: Endpoint
    workers: list[Endpoint]
    mode: str  # "fixed" | "auto" | "idea"
    idea_prompt: Optional[str] = None
    context_recycle_tokens: int = 60_000  # conservative default; Phase-0 probe should tune this
    max_repair_attempts: int = 3
    max_turns_per_task: int = 8  # defense against a model that won't self-terminate
    task_timeout_s: float = 180.0  # hard wall-clock cap per attempt; see worker.run_task
    enable_cross_region_tools: bool = False  # ask_region/request_change; see README "Known gaps"
