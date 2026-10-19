# SPDX-License-Identifier: Apache-2.0

"""PiSubstrate: the Pi coding agent (`pi`) as a solver substrate.

The editable tree is installed at the workspace root and loaded explicitly, never by
discovery, so a user's own Pi configuration cannot leak into a rollout:

    AGENTS.md                      -> --append-system-prompt (context-file discovery is off)
    .agents/skills/<name>/SKILL.md -> --skill <dir>          (skill discovery is off)
    .pi/extensions/<name>.ts       -> -e <file>              (extension discovery is off)
    scripts/<name>.py              -> run by the agent with python3

Each rollout gets an empty `PI_CODING_AGENT_DIR`, so credentials come from the provider's
environment variable (for example `OPENCODE_API_KEY` or `DEEPSEEK_API_KEY`). Point
`HARNESS2_PI_AGENT_DIR` at a dedicated agent directory to use saved logins instead.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping

from .. import config
from ..core.tree import SurfaceSpec
from .agent_cli import AGENT_ROOT, clip, finish_render, iter_jsonl, stage_root
from .base import Install, Launcher, RunRequest

PI_SURFACE = SurfaceSpec(
    name="pi",
    always_on=("AGENTS.md",),
    always_on_max=12_000,
    always_on_cap_reason=(
        "AGENTS.md is appended whole to the system prompt of every model request, so every "
        "token competes with the agent's own output budget"
    ),
    skill_dir=".agents/skills",
    skill_name_rule="must_match",
    trigger_hint=(
        'Fix it by adding a trigger line naming `{name}` to AGENTS.md, e.g. "When <the '
        'situation this skill handles>, load the `{name}` skill."'
    ),
    autotrigger_file="AGENTS.md",
    trigger_line="- When {desc}, load the `{name}` skill.\n",
    subagent_requires_mode=False,
    plugin_dir=".pi/extensions",
    hooks_file=None,
    compile_skill_scripts=True,
    reject_empty_always_on=True,
    stubs=(
        (
            "scripts/README.md",
            "<!-- c7: deterministic helpers, run as `python3 scripts/<name>.py`. "
            "Add one by emitting scripts/<name>.py. -->\n",
        ),
        (
            ".agents/skills/README.md",
            "<!-- c6: reusable workflows. Add one as .agents/skills/<name>/SKILL.md with YAML "
            "frontmatter `name: <name>` (lowercase letters, digits and hyphens, matching the "
            "directory) and `description:`. Only name and description are advertised; the body "
            "is read on demand. -->\n",
        ),
        (
            ".pi/extensions/README.md",
            "<!-- c8: Pi extensions, loaded into the agent process. The only lever that acts "
            "INSIDE the tool loop. Add one as .pi/extensions/<name>.ts exporting a default "
            "function (pi) that registers handlers with pi.on(...). -->\n",
        ),
    ),
)

PI_ROOT_ENTRIES = ("AGENTS.md", ".agents", ".pi", "scripts")

# Session markers a parent Pi process exports to its tools; a nested rollout must not inherit
# them.
_PARENT_SESSION_VARS = frozenset(
    {"PI_SESSION_ID", "PI_SESSION_FILE", "PI_PROVIDER", "PI_MODEL", "PI_REASONING_LEVEL"}
)

# Pi skill names: lowercase letters, digits and single hyphens (docs/skills.md).
_PI_SKILL_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_EXT_LOAD_FAIL = "Failed to load extension"


PI_PACKAGE = "@earendil-works/pi-coding-agent"
PI_VERSION = "1.3.0"


def pi_bin() -> str:
    """HARNESS2_PI_BIN, else `pi` on PATH, else the copy `harness2-setup` installs."""
    if os.environ.get("HARNESS2_PI_BIN"):
        return os.environ["HARNESS2_PI_BIN"]
    found = shutil.which("pi")
    if found:
        return found
    local = config.DEPENDENCIES / "pi" / "node_modules" / ".bin" / "pi"
    return str(local) if local.is_file() else "pi"


def split_effort(model: str, effort: str) -> str:
    """`provider/id` plus a thinking level, unless the model already carries one."""
    if not effort or re.search(r":(off|minimal|low|medium|high|xhigh|max)$", model):
        return model
    return f"{model}:{effort}"


class PiSubstrate:
    """Pi behind the one Substrate interface; the launcher body is benchmark-supplied."""

    name = "pi"

    def __init__(self, launcher: Launcher | None = None):
        self._launcher = launcher

    def surface(self) -> SurfaceSpec:
        return PI_SURFACE

    def materialise(self, tree_dir: Path, spec, ctx: Mapping[str, str]) -> Install:
        return stage_root(PI_SURFACE, tree_dir, PI_ROOT_ENTRIES, spec)

    def execute(
        self, task_ref: str, install: Install, run_ids: list[str], timeout: int
    ) -> list[Path]:
        if self._launcher is None:
            raise RuntimeError(
                "PiSubstrate was constructed without a launcher body; the benchmark adapter "
                "must supply one"
            )
        return self._launcher(
            RunRequest(task_ref=task_ref, run_ids=tuple(run_ids), timeout=int(timeout)), install
        )

    def argv(self, prompt: str, workspace: Path, model: str, *, effort: str = "high") -> list[str]:
        """The `pi` print-mode argv for one rollout in `workspace`, harness already installed."""
        ws = Path(workspace)
        cmd = [
            pi_bin(),
            "-p",
            "--mode",
            "json",
            "--no-session",
            "--no-approve",
            "--no-context-files",
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-themes",
        ]
        agents_md = ws / "AGENTS.md"
        if agents_md.is_file() and agents_md.read_text(errors="replace").strip():
            cmd += ["--append-system-prompt", str(agents_md)]
        skills = ws / PI_SURFACE.skill_dir
        for skill in sorted(skills.glob("*/SKILL.md")) if skills.is_dir() else ():
            cmd += ["--skill", str(skill.parent)]
        exts = ws / str(PI_SURFACE.plugin_dir)
        for ext in sorted(exts.glob("*.ts")) if exts.is_dir() else ():
            cmd += ["-e", str(ext)]
        cmd += ["--model", split_effort(model, effort), prompt]
        return cmd

    def env(self, base_env: Mapping[str, str], *, agent_dir: Path) -> dict[str, str]:
        """An isolated Pi process: its own agent directory and no background network work."""
        env = {k: v for k, v in base_env.items() if k not in _PARENT_SESSION_VARS}
        env["PI_CODING_AGENT_DIR"] = os.environ.get("HARNESS2_PI_AGENT_DIR") or str(agent_dir)
        env.setdefault("PI_SKIP_VERSION_CHECK", "1")
        env.setdefault("PI_OFFLINE", "1")
        env.setdefault("PI_TELEMETRY", "0")
        return env

    def invocation(
        self,
        prompt: str,
        workspace: Path,
        scratch: Path,
        model: str,
        *,
        effort: str,
        base_env: Mapping[str, str],
    ) -> tuple[list[str], dict[str, str]]:
        """(argv, env) for one rollout; `scratch` holds per-run state outside the workspace."""
        agent_dir = Path(scratch) / "pi-agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        return self.argv(prompt, workspace, model, effort=effort), self.env(
            base_env, agent_dir=agent_dir
        )

    def summarise(self, stream_path: Path) -> dict:
        return summarise(stream_path)

    def check_tree(self, tree: Mapping[str, str]) -> list[str]:
        """Pi-specific problems a generic validator cannot see: skill names Pi refuses, and
        extensions that fail to load (which stop Pi before the first model call)."""
        errs: list[str] = []
        skill_rx = re.compile(rf"{re.escape(PI_SURFACE.skill_dir)}/([^/]+)/SKILL\.md")
        for rel in tree:
            m = skill_rx.fullmatch(rel)
            if m and not _PI_SKILL_NAME.fullmatch(m.group(1)):
                errs.append(
                    f"{rel}: Pi skill names use lowercase letters, digits and single hyphens "
                    f"only; rename the directory (and its `name:`) to e.g. "
                    f"`{re.sub(r'[^a-z0-9]+', '-', m.group(1).lower()).strip('-') or 'skill'}`"
                )
        exts = {
            rel: src
            for rel, src in tree.items()
            if rel.startswith(f"{PI_SURFACE.plugin_dir}/") and rel.endswith(".ts")
        }
        if exts:
            errs.extend(check_extensions(exts))
        return errs

    def read_trajectory(
        self, stream_path: Path, *, obs_trunc: int, max_chars: int, args_trunc: int = 2000
    ) -> str:
        return render_events(
            Path(stream_path), obs_trunc=obs_trunc, max_chars=max_chars, args_trunc=args_trunc
        )

    def preflight(self) -> None:
        """The CLI must exist and answer `--version`; credentials are checked by the run."""
        binary = pi_bin()
        if not shutil.which(binary) and not Path(binary).is_file():
            raise RuntimeError(
                "pi is not installed; install it with `npm install -g "
                "@earendil-works/pi-coding-agent` or set HARNESS2_PI_BIN"
            )
        try:
            r = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"`{binary} --version` failed: {exc}") from exc
        if r.returncode != 0:
            raise RuntimeError(f"`{binary} --version` exited {r.returncode}: {r.stderr[-300:]}")


def check_extensions(exts: Mapping[str, str]) -> list[str]:
    """Load each extension in a throwaway Pi process that stops before any model call.

    Pi loads `-e` extensions before resolving `--model`, so pointing it at a model that
    cannot exist turns a startup into a free, offline load check: a parse error or a
    throwing factory prints "Failed to load extension"; a sound one reaches the model error.
    """
    errs: list[str] = []
    with tempfile.TemporaryDirectory(prefix="harness2-pi-ext-") as td:
        root = Path(td)
        (root / "agent").mkdir()
        for rel, src in sorted(exts.items()):
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(src, encoding="utf-8")
            env = {k: v for k, v in os.environ.items() if not k.endswith(("_API_KEY", "_TOKEN"))}
            env.update(
                PI_CODING_AGENT_DIR=str(root / "agent"),
                PI_OFFLINE="1",
                PI_SKIP_VERSION_CHECK="1",
                PI_TELEMETRY="0",
            )
            try:
                r = subprocess.run(
                    [
                        pi_bin(),
                        "-p",
                        "--no-session",
                        "--no-approve",
                        "--no-context-files",
                        "--no-extensions",
                        "--no-skills",
                        "-e",
                        str(path),
                        "--model",
                        "harness2-extension-check/none",
                        "load check",
                    ],
                    cwd=root,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                errs.append(f"{rel}: could not run the extension load check ({exc})")
                continue
            out = (r.stderr or "") + (r.stdout or "")
            if _EXT_LOAD_FAIL in out:
                line = next((ln for ln in out.splitlines() if _EXT_LOAD_FAIL in ln), out)
                detail = line.split(_EXT_LOAD_FAIL, 1)[-1].lstrip(": ").strip()
                detail = detail.replace(str(path), rel).replace(str(root), ".")
                errs.append(
                    f"{rel} fails to load in Pi ({detail[:300]}); Pi refuses to start with a "
                    f"broken extension, so this harness would produce no rollout at all"
                )
    return errs


def _text_parts(content, kind: str) -> str:
    if isinstance(content, str):
        return content if kind == "text" else ""
    if not isinstance(content, list):
        return ""
    out = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == kind:
            out.append(str(part.get(kind) or part.get("text") or ""))
    return "\n".join(t for t in out if t.strip())


def render_events(path: Path, *, obs_trunc: int, max_chars: int, args_trunc: int = 2000) -> str:
    """Render Pi's `--mode json` stream into the shared [ACTION]/[OBSERVATION] grammar.

    Only finalised `message_end` records are read, so streaming deltas never duplicate.
    """
    lines: list[str] = []
    for ev in iter_jsonl(path):
        et = ev.get("type", "")
        if et == "message_end":
            msg = ev.get("message")
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role == "assistant":
                think = _text_parts(content, "thinking")
                if think:
                    lines.append(f"[REASONING] {clip(think, obs_trunc)}")
                text = _text_parts(content, "text")
                if text:
                    lines.append(f"[ASSISTANT] {clip(text, obs_trunc)}")
                for part in content if isinstance(content, list) else ():
                    if isinstance(part, dict) and part.get("type") == "toolCall":
                        args = json.dumps(part.get("arguments"), ensure_ascii=False, default=str)
                        lines.append(f"[ACTION] {part.get('name')}: {clip(args, args_trunc)}")
                if msg.get("stopReason") == "error" or msg.get("errorMessage"):
                    lines.append(f"[ERROR] {clip(msg.get('errorMessage') or 'model error', 1000)}")
            elif role == "toolResult":
                tag = "OBSERVATION ERROR" if msg.get("isError") else "OBSERVATION"
                lines.append(f"[{tag}] {clip(_text_parts(content, 'text'), obs_trunc)}")
            elif role == "user":
                text = _text_parts(content, "text")
                if text:
                    lines.append(f"[USER] {clip(text, obs_trunc)}")
        elif et == "auto_retry_end" and not ev.get("success"):
            lines.append(f"[ERROR] {clip(ev.get('finalError') or 'retries exhausted', 1000)}")
        elif et == "compaction_end":
            lines.append("[HARNESS] context compacted")
    return finish_render(lines, max_chars)


def summarise(path: Path) -> dict:
    """Counts, token usage, cost and the final answer of one Pi stream, for metrics.json."""
    s = {
        "n_events": 0,
        "turns": 0,
        "tool_calls": 0,
        "errors": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
        "final_text": "",
    }
    for ev in iter_jsonl(path):
        s["n_events"] += 1
        et = ev.get("type")
        if et == "turn_end":
            s["turns"] += 1
        elif et == "tool_execution_start":
            s["tool_calls"] += 1
        elif et == "auto_retry_end" and not ev.get("success"):
            s["errors"] += 1
        elif et == "message_end":
            msg = ev.get("message") or {}
            if msg.get("role") != "assistant":
                continue
            if msg.get("stopReason") == "error":
                s["errors"] += 1
            u = msg.get("usage") or {}
            s["input_tokens"] += int(u.get("input") or 0)
            s["output_tokens"] += int(u.get("output") or 0)
            s["cached_input_tokens"] += int(u.get("cacheRead") or 0)
            s["reasoning_tokens"] += int(u.get("reasoning") or 0)
            cost = u.get("cost") or {}
            s["cost_usd"] += float(cost.get("total") or 0.0) if isinstance(cost, dict) else 0.0
            text = _text_parts(msg.get("content"), "text")
            if text:
                s["final_text"] = text
    s["cost_usd"] = round(s["cost_usd"], 6)
    return s


def final_message(path: Path) -> str:
    return summarise(path)["final_text"]


__all__ = [
    "PI_PACKAGE",
    "PI_VERSION",
    "PI_SURFACE",
    "PI_ROOT_ENTRIES",
    "PiSubstrate",
    "AGENT_ROOT",
    "check_extensions",
    "final_message",
    "pi_bin",
    "render_events",
    "summarise",
]
