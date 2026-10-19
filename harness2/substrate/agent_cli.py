# SPDX-License-Identifier: Apache-2.0

"""Shared plumbing for substrates that run a coding-agent CLI inside a workspace directory.

Pi and DeepSeek Harness both read `AGENTS.md` and `.agents/skills/` from the directory they
run in and stream newline-delimited JSON events to stdout. What they share lives here: the
`agent_root` install mode (copy the editable tree to the workspace root), a portable
process-group runner with a wall-clock deadline (no GNU `timeout`), and tolerant JSONL reading.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from ..core import runmark as RM
from ..core import tree as HT
from ..core.tree import SurfaceSpec
from .base import Install

AGENT_ROOT = "agent_root"

TRUNC_FMT = "…(+{n} chars — render truncation)"


def clip(s, n: int) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\x00", "")
    if n <= 0 or len(s) <= n:
        return s
    return s[:n] + TRUNC_FMT.format(n=len(s) - n)


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Every JSON object line in `path`, in order; banners and torn lines are skipped."""
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    for ln in raw.replace("\x00", "").splitlines():
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            ev = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if isinstance(ev, dict):
            yield ev


def finish_render(lines: list[str], max_chars: int) -> str:
    rendered = "\n".join(lines).strip()
    if not rendered:
        return "(no parseable events in trajectory)"
    if len(rendered) > max_chars:
        rendered = rendered[:max_chars] + "\n" + TRUNC_FMT.format(n=len(rendered) - max_chars)
    return rendered


def stage_root(surface: SurfaceSpec, tree_dir: Path, entries: Sequence[str], spec) -> Install:
    """The `agent_root` install: stage exactly `entries` of the tree for a copy to the
    workspace root. There is no render step; the CLI reads these files itself."""
    if spec.install_mode != AGENT_ROOT:
        raise ValueError(
            f"install_mode {spec.install_mode!r} is not {AGENT_ROOT!r}, the only mode the "
            f"{surface.name} substrate supports"
        )
    tree_dir = Path(tree_dir)
    if not HT.is_tree(surface, tree_dir):
        raise ValueError(f"{tree_dir} is not a {surface.name} harness tree (no AGENTS.md)")
    staged = Path(tempfile.mkdtemp(prefix=f"harness2-{surface.name}-root-"))
    for entry in entries:
        src = tree_dir / entry
        if not src.exists():
            continue
        if src.is_dir():
            shutil.copytree(src, staged / entry, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(src, staged / entry)
    return Install(
        staging_dir=staged,
        fingerprint=RM.fingerprint(tree_dir),
        extras={"install_mode": AGENT_ROOT},
    )


def install_into(staging_dir: Path, workspace: Path) -> list[str]:
    """Copy a staged harness into a workspace root; return the installed relative paths."""
    installed: list[str] = []
    src_root = Path(staging_dir)
    for f in sorted(src_root.rglob("*")):
        if not f.is_file() or "__pycache__" in f.parts:
            continue
        rel = f.relative_to(src_root)
        dest = Path(workspace) / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
        installed.append(rel.as_posix())
    return installed


@dataclass(frozen=True)
class AgentExit:
    """How one CLI invocation ended."""

    returncode: int
    timed_out: bool
    wall_s: float


def run_agent(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout: int,
    grace: int = 30,
) -> AgentExit:
    """Run `argv` in its own process group with stdout/stderr captured to files.

    At the deadline the whole group gets SIGTERM, then SIGKILL after `grace` seconds, so a
    stuck tool subprocess cannot outlive the rollout.
    """
    t0 = time.time()
    timed_out = False
    with (
        open(stdout_path, "w", encoding="utf-8") as out,
        open(stderr_path, "w", encoding="utf-8") as err,
    ):
        proc = subprocess.Popen(
            list(argv),
            cwd=str(cwd),
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=max(1, int(timeout)))
        except subprocess.TimeoutExpired:
            timed_out = True
            _signal_group(proc, signal.SIGTERM)
            try:
                proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                _signal_group(proc, signal.SIGKILL)
                proc.wait()
    return AgentExit(proc.returncode, timed_out, round(time.time() - t0, 1))


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            pass


__all__ = [
    "AGENT_ROOT",
    "AgentExit",
    "clip",
    "finish_render",
    "install_into",
    "iter_jsonl",
    "run_agent",
    "stage_root",
]
