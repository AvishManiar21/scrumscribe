"""Git history as meeting context.

For the standup draft, what you actually did last week is already recorded in
your commit log. Reading it means the draft is grounded in real work rather
than in the model's idea of what a week of work sounds like.

Everything here degrades to an empty list: not being in a repo, not having git
installed, or pointing at a repo with no commits are all normal, not errors.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

TIMEOUT = 15


def _run(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def is_repo(path: Path) -> bool:
    return _run(["git", "rev-parse", "--is-inside-work-tree"], path) == "true"


def commits_since(
    repo: Path,
    since: str | None = None,
    author: str | None = None,
    limit: int = 60,
) -> list[str]:
    """Commit subjects since a date, optionally filtered to one author.

    `since` is an ISO date; git accepts relative forms too ("2 weeks ago").
    """
    repo = Path(repo)
    if not repo.is_dir() or not is_repo(repo):
        return []

    args = ["git", "log", f"--max-count={limit}", "--no-merges", "--pretty=format:%ad %s", "--date=short"]
    if since:
        args.append(f"--since={since}")
    if author:
        args.append(f"--author={author}")

    output = _run(args, repo)
    if not output:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def current_branch(repo: Path) -> str | None:
    return _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], Path(repo))


def repo_name(repo: Path) -> str:
    url = _run(["git", "remote", "get-url", "origin"], Path(repo))
    if url:
        return url.rstrip("/").split("/")[-1].removesuffix(".git")
    return Path(repo).name
