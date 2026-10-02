"""Runs each task's acceptance command in the region's own worktree. This is
the ground truth DevFlock trusts -- not a worker's claim that it's done."""
from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path


@dataclasses.dataclass
class VerifyResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: int


def verify(region_dir: str | Path, acceptance_cmd: str | None, timeout: int = 300) -> VerifyResult:
    if not acceptance_cmd:
        return VerifyResult(ok=True, stdout="(no acceptance command; skipped)", stderr="", returncode=0)
    try:
        r = subprocess.run(acceptance_cmd, shell=True, cwd=str(region_dir),
                            capture_output=True, text=True, timeout=timeout)
        return VerifyResult(ok=r.returncode == 0, stdout=r.stdout[-4000:], stderr=r.stderr[-4000:],
                             returncode=r.returncode)
    except subprocess.TimeoutExpired as e:
        return VerifyResult(ok=False, stdout=(e.stdout or "")[-2000:] if isinstance(e.stdout, str) else "",
                             stderr=f"timed out after {timeout}s", returncode=-1)
