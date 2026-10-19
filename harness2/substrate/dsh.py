# SPDX-License-Identifier: Apache-2.0

"""DshSubstrate: DeepSeek Harness (`dsh --profile headless`) as a solver substrate.

dsh reads `AGENTS.md` and `.agents/skills/<name>/SKILL.md` from the project root, which is
the workspace when no `.git` sits above it. Everything else comes from a per-rollout
`--patch` overlay that harness2 writes:

    model            -> the `agent-default-model` row (provider/model, reasoning effort)
    credentials      -> the provider's environment variable, read by dsh's pi-ai routes
    .dsh/plugins/*.ts -> one `insert` row per plugin, loaded by absolute path

Each rollout gets an empty `DSH_HOME` and `DSH_AGENTS_HOME`, so a user's own dsh profile,
skills and sessions never reach a run. `HARNESS2_DSH_PATCH` names an extra overlay applied
last, for custom gateways or other profile rows.
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
from .agent_cli import clip, finish_render, iter_jsonl, stage_root
from .base import Install, Launcher, RunRequest

DSH_VERSION = "0.2.1-rc.2"
DSH_PACKAGE = "@deepseek-ai/dsh"

DSH_SURFACE = SurfaceSpec(
    name="dsh",
    always_on=("AGENTS.md",),
    always_on_max=12_000,
    always_on_cap_reason=(
        "AGENTS.md is injected whole into the instructions of every turn, so every token "
        "competes with the agent's own output budget"
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
    plugin_dir=".dsh/plugins",
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
            "frontmatter `name: <name>` matching the directory and `description:`. Only name "
            "and description are advertised; the body is loaded with the `skill` tool. -->\n",
        ),
        (
            ".dsh/plugins/README.md",
            "<!-- c8: DeepSeek Harness plugins (Cordis), loaded into the agent process. The only "
            "lever that acts INSIDE the tool loop. Add one as .dsh/plugins/<name>.ts exporting "
            "`name` and `apply(ctx)`; register listeners with ctx.on('tools/pre-execute', ...). "
            "-->\n",
        ),
    ),
)

DSH_ROOT_ENTRIES = ("AGENTS.md", ".agents", ".dsh", "scripts")

_CHECK_PLUGIN_JS = r"""
const { pathToFileURL } = await import("node:url");
const m = await import(pathToFileURL(process.argv[1]).href);
const problems = [];
if (typeof m.apply !== "function") problems.push("it must export a function `apply(ctx)`");
if (typeof m.name !== "string" || !m.name) problems.push("it must export a string `name`");
if (problems.length) { console.error("PLUGIN_SHAPE: " + problems.join("; ")); process.exit(3); }
"""


def dsh_bin() -> str:
    """HARNESS2_DSH_BIN, else the copy `harness2-setup --substrate dsh` installs, else PATH."""
    if os.environ.get("HARNESS2_DSH_BIN"):
        return os.environ["HARNESS2_DSH_BIN"]
    local = config.DEPENDENCIES / "dsh" / "node_modules" / ".bin" / "dsh"
    if local.is_file():
        return str(local)
    return shutil.which("dsh") or "dsh"


def split_model(model: str) -> tuple[str, str]:
    """'provider/model' -> (provider, model); a bare id means DeepSeek's own API."""
    if "/" in model:
        provider, _, name = model.partition("/")
        return provider, name
    return "deepseek", model


def overlay_rows(model: str, *, effort: str, plugins: list[Path]) -> list[dict]:
    """The per-rollout `--patch` overlay: model, provider route and plugin inserts."""
    provider, name = split_model(model)
    row: dict = {"provider": provider, "model": name}
    if effort:
        row["reasoningEffort"] = effort
    rows: list[dict] = [{"id": "agent-default-model", "config": row}]
    if provider != "deepseek":
        rows.append({"id": "llm-pi-ai", "config": {"providers": {provider: {}}}})
    if plugins:
        rows.append(
            {
                "insert": [
                    {"id": f"harness2-plugin-{p.stem}", "name": str(p.resolve())} for p in plugins
                ]
            }
        )
    return rows


class DshSubstrate:
    """DeepSeek Harness behind the one Substrate interface; the launcher body is
    benchmark-supplied."""

    name = "dsh"

    def __init__(self, launcher: Launcher | None = None):
        self._launcher = launcher

    def surface(self) -> SurfaceSpec:
        return DSH_SURFACE

    def materialise(self, tree_dir: Path, spec, ctx: Mapping[str, str]) -> Install:
        return stage_root(DSH_SURFACE, tree_dir, DSH_ROOT_ENTRIES, spec)

    def execute(
        self, task_ref: str, install: Install, run_ids: list[str], timeout: int
    ) -> list[Path]:
        if self._launcher is None:
            raise RuntimeError(
                "DshSubstrate was constructed without a launcher body; the benchmark adapter "
                "must supply one"
            )
        return self._launcher(
            RunRequest(task_ref=task_ref, run_ids=tuple(run_ids), timeout=int(timeout)), install
        )

    def write_overlay(
        self, workspace: Path, dest: Path, model: str, *, effort: str = "high"
    ) -> Path:
        """Write the rollout's overlay. JSON is valid YAML, so no YAML library is needed."""
        plugins = sorted((Path(workspace) / str(DSH_SURFACE.plugin_dir)).glob("*.ts"))
        dest = Path(dest)
        dest.write_text(
            json.dumps(overlay_rows(model, effort=effort, plugins=plugins), indent=2) + "\n",
            encoding="utf-8",
        )
        return dest

    def argv(self, prompt: str, overlay: Path) -> list[str]:
        cmd = [dsh_bin(), "--profile", "headless", "--patch", str(overlay)]
        extra = os.environ.get("HARNESS2_DSH_PATCH")
        if extra:
            cmd += ["--patch", str(Path(extra).expanduser().resolve())]
        return cmd + ["--json", prompt]

    def env(self, base_env: Mapping[str, str], *, home: Path, agents_home: Path) -> dict[str, str]:
        env = dict(base_env)
        env["DSH_HOME"] = str(home)
        env["DSH_AGENTS_HOME"] = str(agents_home)
        env["DSH_PERMISSION_MODE"] = os.environ.get(
            "HARNESS2_DSH_PERMISSION_MODE", "danger-full-access"
        )
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
        scratch = Path(scratch)
        home, agents_home = scratch / "dsh-home", scratch / "dsh-agents"
        home.mkdir(parents=True, exist_ok=True)
        agents_home.mkdir(parents=True, exist_ok=True)
        overlay = self.write_overlay(workspace, scratch / "overlay.json", model, effort=effort)
        return self.argv(prompt, overlay), self.env(base_env, home=home, agents_home=agents_home)

    def summarise(self, stream_path: Path) -> dict:
        return summarise(stream_path)

    def check_tree(self, tree: Mapping[str, str]) -> list[str]:
        """Import each plugin in Node (types stripped) and check its `name`/`apply` exports.

        Importing evaluates the module top level but never calls `apply`, so nothing the
        plugin registers runs here.
        """
        plugins = {
            rel: src
            for rel, src in tree.items()
            if rel.startswith(f"{DSH_SURFACE.plugin_dir}/") and rel.endswith(".ts")
        }
        if not plugins:
            return []
        node = shutil.which("node")
        if node is None:
            return ["node is not on PATH, so dsh plugins cannot be checked or loaded"]
        errs: list[str] = []
        with tempfile.TemporaryDirectory(prefix="harness2-dsh-plugin-") as td:
            for rel, src in sorted(plugins.items()):
                path = Path(td) / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(src, encoding="utf-8")
                try:
                    r = subprocess.run(
                        [node, "--input-type=module", "-e", _CHECK_PLUGIN_JS, str(path)],
                        cwd=td,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                except (OSError, subprocess.TimeoutExpired) as exc:
                    errs.append(f"{rel}: could not run the plugin load check ({exc})")
                    continue
                if r.returncode != 0:
                    msg = (r.stderr or r.stdout).strip().replace(str(path), rel)
                    msg = msg.replace(td, ".")
                    shape = re.search(r"PLUGIN_SHAPE: (.+)", msg)
                    detail = shape.group(1) if shape else _first_error_line(msg)
                    errs.append(
                        f"{rel} does not load as a dsh plugin: {detail[:300]}. Plugins may import "
                        f"only Node built-ins (type-only imports are fine)"
                    )
        return errs

    def read_trajectory(
        self, stream_path: Path, *, obs_trunc: int, max_chars: int, args_trunc: int = 2000
    ) -> str:
        return render_events(
            Path(stream_path), obs_trunc=obs_trunc, max_chars=max_chars, args_trunc=args_trunc
        )

    def preflight(self) -> None:
        binary = dsh_bin()
        if not shutil.which(binary) and not Path(binary).is_file():
            raise RuntimeError(
                "dsh is not installed; run `harness2-setup <bench> --substrate dsh` or set "
                "HARNESS2_DSH_BIN"
            )
        try:
            r = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"`{binary} --version` failed: {exc}") from exc
        if r.returncode != 0:
            raise RuntimeError(f"`{binary} --version` exited {r.returncode}: {r.stderr[-300:]}")


def _first_error_line(msg: str) -> str:
    for ln in msg.splitlines():
        if re.search(r"(Error|error):", ln):
            return ln.strip()
    return msg.splitlines()[0] if msg else "unknown error"


def render_events(path: Path, *, obs_trunc: int, max_chars: int, args_trunc: int = 2000) -> str:
    """Render dsh's `--json` run events into the shared [ACTION]/[OBSERVATION] grammar."""
    lines: list[str] = []
    last_text = ""
    for ev in iter_jsonl(path):
        et = ev.get("type", "")
        if et == "thinking":
            if str(ev.get("text") or "").strip():
                lines.append(f"[REASONING] {clip(ev['text'], obs_trunc)}")
        elif et == "text":
            last_text = str(ev.get("text") or "")
            if last_text.strip():
                lines.append(f"[ASSISTANT] {clip(last_text, obs_trunc)}")
        elif et == "final":
            text = str(ev.get("text") or "")
            if text.strip() and text != last_text:
                lines.append(f"[ASSISTANT] {clip(text, obs_trunc)}")
        elif et == "tool_call":
            args = json.dumps(ev.get("input"), ensure_ascii=False, default=str)
            lines.append(f"[ACTION] {ev.get('tool')}: {clip(args, args_trunc)}")
        elif et == "tool_result":
            tag = "OBSERVATION ERROR" if ev.get("status") == "error" else "OBSERVATION"
            res = ev.get("result")
            if not isinstance(res, str):
                res = json.dumps(res, ensure_ascii=False, default=str)
            lines.append(f"[{tag}] {clip(res, obs_trunc)}")
        elif et == "error":
            lines.append(f"[ERROR] {clip(ev.get('message') or json.dumps(ev), 1000)}")
        elif et == "status" and ev.get("phase") == "turn_end":
            reason = (ev.get("reason") or {}).get("kind")
            if reason and reason != "completed":
                lines.append(f"[HARNESS] turn ended: {reason}")
    return finish_render(lines, max_chars)


def summarise(path: Path) -> dict:
    """Counts, token usage and the final answer of one dsh stream (dsh reports no cost)."""
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
        "outcome": "",
    }
    for ev in iter_jsonl(path):
        s["n_events"] += 1
        et = ev.get("type")
        if et == "tool_call":
            s["tool_calls"] += 1
        elif et == "error" or (et == "tool_result" and ev.get("status") == "error"):
            s["errors"] += 1
        elif et == "final":
            s["final_text"] = str(ev.get("text") or "")
        elif et == "status":
            if ev.get("phase") == "turn_end":
                s["turns"] += 1
                s["outcome"] = str((ev.get("reason") or {}).get("kind") or "")
            elif ev.get("phase") == "step_end":
                u = ev.get("usage") or {}
                s["input_tokens"] += int(u.get("inputTokens") or 0)
                s["output_tokens"] += int(u.get("outputTokens") or 0)
                s["cached_input_tokens"] += int(u.get("cacheReadTokens") or 0)
    return s


def final_message(path: Path) -> str:
    return summarise(path)["final_text"]


__all__ = [
    "DSH_PACKAGE",
    "DSH_ROOT_ENTRIES",
    "DSH_SURFACE",
    "DSH_VERSION",
    "DshSubstrate",
    "dsh_bin",
    "final_message",
    "overlay_rows",
    "render_events",
    "split_model",
    "summarise",
]
