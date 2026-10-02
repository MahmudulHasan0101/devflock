"""Each region is a git worktree on its own branch, so parallel workers never
collide on the working directory, and merging back is an ordinary git
operation the person can inspect, rebase, or reject.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


class GitError(RuntimeError):
    pass


def _run(args: list[str], cwd: str | Path) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed in {cwd}:\n{r.stderr}")
    return r.stdout.strip()


def _exclude_state_dir(p: Path):
    """Keep DevFlock's own state (.devflock/ at the project root) out of git,
    via .git/info/exclude so we never modify a user's tracked .gitignore."""
    exclude = p / ".git" / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    existing = exclude.read_text() if exclude.exists() else ""
    if "/.devflock/" not in existing:
        exclude.write_text(existing.rstrip("\n") + "\n/.devflock/\n")


def ensure_repo(project_dir: str | Path):
    p = Path(project_dir)
    if not (p / ".git").exists():
        _run(["init"], p)
        _exclude_state_dir(p)  # must precede the first `add -A`, or our own state gets committed
        _run(["config", "user.email", "devflock@localhost"], p)
        _run(["config", "user.name", "DevFlock"], p)
        # need at least one commit for worktrees to branch from
        (p / ".gitkeep").touch(exist_ok=True)
        _run(["add", "-A"], p)
        _run(["commit", "-m", "devflock: initial commit", "--allow-empty"], p)
    _exclude_state_dir(p)


def create_region_worktree(project_dir: str | Path, region: str) -> Path:
    project_dir = Path(project_dir)
    worktrees_root = project_dir / ".devflock" / "worktrees"
    worktrees_root.mkdir(parents=True, exist_ok=True)
    wt_path = worktrees_root / region
    branch = f"devflock/{region}"
    if wt_path.exists():
        return wt_path
    existing_branches = _run(["branch", "--list", branch], project_dir)
    if existing_branches:
        _run(["worktree", "add", str(wt_path), branch], project_dir)
    else:
        _run(["worktree", "add", "-b", branch, str(wt_path)], project_dir)
    return wt_path


def region_subdir(worktree: str | Path, subdir: str) -> Path:
    """The folder a region's worker owns *inside* its worktree. Workers get
    this as their cwd and path-guard root, so regions touch disjoint paths
    and merge without conflicts by construction."""
    d = Path(worktree) / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d


def commit_region(project_dir: str | Path, region: str, message: str) -> bool:
    """Commit whatever changed in a region's worktree. Returns False if
    there was nothing to commit."""
    wt_path = Path(project_dir) / ".devflock" / "worktrees" / region
    _run(["add", "-A"], wt_path)
    status = _run(["status", "--porcelain"], wt_path)
    if not status:
        return False
    _run(["commit", "-m", message], wt_path)
    return True


def current_branch(project_dir: str | Path) -> str:
    return _run(["rev-parse", "--abbrev-ref", "HEAD"], project_dir)


def integrate(project_dir: str | Path, regions: list[str],
              into_branch: str = "devflock/integration") -> list[str]:
    """Merge each region branch into an integration branch, in the given order.
    Returns regions whose merge conflicted (left out, never auto-resolved). The
    person's original branch is always restored afterwards, so their working
    tree is never left on a DevFlock branch."""
    project_dir = Path(project_dir)
    original = current_branch(project_dir)
    if _run(["branch", "--list", into_branch], project_dir):
        _run(["checkout", into_branch], project_dir)
    else:
        _run(["checkout", "-b", into_branch], project_dir)
    conflicted = []
    try:
        for region in regions:
            try:
                _run(["merge", "--no-edit", f"devflock/{region}"], project_dir)
            except GitError:
                if (project_dir / ".git" / "MERGE_HEAD").exists():
                    _run(["merge", "--abort"], project_dir)
                conflicted.append(region)
    finally:
        _run(["checkout", original], project_dir)
    return conflicted


def worktree_diff_stat(project_dir: str | Path, region: str) -> str:
    wt_path = Path(project_dir) / ".devflock" / "worktrees" / region
    return _run(["diff", "--stat", "HEAD"], wt_path)


def is_git_repo(project_dir: str | Path) -> bool:
    return (Path(project_dir) / ".git").exists()


def is_dirty(project_dir: str | Path) -> bool:
    """True if the project has uncommitted changes. Region worktrees branch from
    HEAD, so uncommitted work would be invisible to the workers."""
    if not is_git_repo(project_dir):
        return False
    return bool(_run(["status", "--porcelain"], project_dir))
