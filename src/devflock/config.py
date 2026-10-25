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


PROVIDER_ANTHROPIC = "anthropic"    # llama-server & friends: speaks /v1/messages natively
PROVIDER_CLOUDFLARE = "cloudflare"  # Workers AI: OpenAI-style, translated by the gateway

# Claude Code is given this model name for Cloudflare workers; the gateway swaps in
# the real "@cf/..." id. (Keeps odd characters out of the CLI's model handling.)
CF_CLIENT_MODEL_ALIAS = "devflock-cloudflare"


@dataclasses.dataclass
class Endpoint:
    """One model endpoint (manager or worker).

    provider="anthropic":  llama-server-style Anthropic-compatible URL + key.
    provider="cloudflare": Cloudflare Workers AI. base_url is the account's
        ``.../accounts/<id>/ai/v1`` root, api_key is the Cloudflare API token and
        model_name is the ``@cf/...`` model id picked for THIS worker.
    """
    name: str
    base_url: str
    api_key: str
    auth_style: str = "x-api-key"  # or "bearer"
    model_name: str = "qwen"
    provider: str = PROVIDER_ANTHROPIC
    gateway_id: str = ""  # optional cf-aig-gateway-id header (Cloudflare only)

    @property
    def client_key(self) -> str:
        """Key handed to the Claude Code subprocess. For Cloudflare the real token stays
        in the gateway only -- a worker's Bash tool must not be able to read it."""
        return "devflock-local" if self.provider == PROVIDER_CLOUDFLARE else self.api_key

    @property
    def client_model(self) -> str:
        """Model name Claude Code itself is told to use."""
        return CF_CLIENT_MODEL_ALIAS if self.provider == PROVIDER_CLOUDFLARE else self.model_name

    def masked(self) -> str:
        k = self.api_key
        shown = k[:4] + "…" + k[-2:] if len(k) > 8 else "…"
        if self.provider == PROVIDER_CLOUDFLARE:
            return f"{self.name} @ Cloudflare Workers AI · {self.model_name} (token {shown})"
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
