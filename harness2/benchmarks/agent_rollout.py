# SPDX-License-Identifier: Apache-2.0

"""One rollout of a coding-agent CLI substrate (`pi`, `dsh`) in a fresh scratch workspace.

A benchmark supplies two callbacks: `prepare(workspace)` lays out the task inputs and returns
the assignment message, and `collect(workspace)` copies the deliverables to where the
benchmark grades them. Everything between is shared: install the harness at the workspace
root, overlay runtime-owned files, run the CLI under a deadline, and record config.json,
the event stream and metrics.json. metrics.json is written last and only when the stream
holds events, so a dead model chain is never cached as a finished rollout.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .. import config
from ..core import tree as HT
from ..substrate.agent_cli import run_agent
from ..substrate.base import Install
from ..substrate.registry import SESSION_FILES


def scratch_dir(substrate_name: str, label: str) -> Path:
    """A fresh per-rollout scratch root, refused if it sits inside a git checkout."""
    root = config.SCRATCH
    d = root / substrate_name / f"{label}__{os.getpid()}"
    for parent in [root, *root.parents]:
        if (parent / ".git").exists():
            raise RuntimeError(
                f"HARNESS2_SCRATCH={root} is inside the git checkout {parent}; agents resolve "
                f"their project root through .git and would read that repository's "
                f"instructions. Point HARNESS2_SCRATCH outside any repository."
            )
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    return d


def install_harness(substrate, install: Install, workspace: Path) -> list[str]:
    """Copy the editable files of a staged harness into the workspace root.

    README stubs and anything off the editable surface stay behind, so the agent sees
    exactly the components the harness declares.
    """
    surface = substrate.surface()
    src = Path(install.staging_dir)
    installed: list[str] = []
    for f in sorted(src.rglob("*")):
        if not f.is_file() or "__pycache__" in f.parts:
            continue
        rel = f.relative_to(src).as_posix()
        if not HT.is_allowed(surface, rel):
            continue
        dest = workspace / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
        installed.append(rel)
    return installed


def run_rollout(
    substrate,
    *,
    install: Install,
    run_dir: Path,
    label: str,
    model: str,
    timeout: int,
    prepare: Callable[[Path], str],
    collect: Callable[[Path], list[str]],
    effort: str = "",
    runtime_files: Mapping[str, Path] | None = None,
    path_prepend: Sequence[Path] = (),
    extra_env: Mapping[str, str] | None = None,
    stream_name: str | None = None,
    keep_workspace: bool = False,
) -> dict:
    """Run one rollout and return its metrics; raise if the agent produced no events."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    scratch = scratch_dir(substrate.name, label)
    ws = scratch / "ws"
    ws.mkdir()
    effort = effort or config.SOLVER_EFFORT
    stream = run_dir / (stream_name or SESSION_FILES[substrate.name])
    stderr = run_dir / f"{substrate.name}_stderr.log"
    started = datetime.now(timezone.utc).isoformat()
    try:
        installed = install_harness(substrate, install, ws)
        for rel, src in (runtime_files or {}).items():
            dest = ws / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        prompt = prepare(ws)

        base_env = dict(os.environ)
        if path_prepend:
            base_env["PATH"] = os.pathsep.join(
                [*(str(p) for p in path_prepend), base_env.get("PATH", "")]
            )
        base_env.update(extra_env or {})
        argv, env = substrate.invocation(
            prompt, ws, scratch, model, effort=effort, base_env=base_env
        )
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "runner": substrate.name,
                    "model": model,
                    "reasoning_effort": effort,
                    "label": label,
                    "timeout": timeout,
                    "harness_fingerprint": install.fingerprint,
                    "harness_files": installed,
                    "runtime_files": sorted(runtime_files or {}),
                    "started_at": started,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        ended = run_agent(
            argv, cwd=ws, env=env, stdout_path=stream, stderr_path=stderr, timeout=timeout
        )
        delivered = collect(ws)
        summary = substrate.summarise(stream)
        if summary["n_events"] == 0:
            tail = stderr.read_text(encoding="utf-8", errors="replace").strip()[-800:]
            raise RuntimeError(
                f"{substrate.name} produced no events for {label} (exit {ended.returncode}"
                f"{', timed out' if ended.timed_out else ''}): {tail or 'no stderr'}"
            )
        rc = ended.returncode
        if rc == 0 and not delivered:
            rc = 1
        metrics = {
            "runner": substrate.name,
            "model": model,
            "label": label,
            "turn_count": summary["turns"],
            "tool_calls": summary["tool_calls"],
            "stream_errors": summary["errors"],
            "input_tokens": summary["input_tokens"],
            "cached_input_tokens": summary["cached_input_tokens"],
            "output_tokens": summary["output_tokens"],
            "reasoning_tokens": summary["reasoning_tokens"],
            "cost_usd": summary["cost_usd"],
            "wall_clock_seconds": ended.wall_s,
            "timed_out": ended.timed_out,
            "exit_code": rc,
            "finished_cleanly": rc == 0 and not ended.timed_out,
            "deliverables": delivered,
            "harness_files": installed,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics
    finally:
        if not keep_workspace:
            shutil.rmtree(scratch, ignore_errors=True)


def copy_tree_files(src: Path, dest: Path) -> list[str]:
    """Copy every file under `src` to `dest`; return the relative paths copied."""
    out: list[str] = []
    src, dest = Path(src), Path(dest)
    if not src.is_dir():
        return out
    for f in sorted(src.rglob("*")):
        if f.is_file():
            rel = f.relative_to(src)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            out.append(rel.as_posix())
    return out


__all__ = ["copy_tree_files", "install_harness", "run_rollout", "scratch_dir"]
