# SPDX-License-Identifier: Apache-2.0

"""CustomAdapter: refine a harness on your own tasks, with no reference answers.

Every other adapter starts from somebody else's dataset and somebody else's verdict. This
one starts from a directory of your own assignments, so the only thing you have to supply is
work you actually care about:

    <tasks>/<area>/<task>/TASK.md     the assignment, verbatim, as the agent receives it
    <tasks>/<area>/<task>/files/**    optional reference material copied into the workspace
    <tasks>/<area>/<task>/check.py    optional check, run against the deliverables

Point `HARNESS2_CUSTOM_TASKS` at that directory. Nothing is required to be gradeable: with no
`check.py` and no judge configured a rollout is still executed, selected and reported, and
its measurement is simply absent (never a zero). With a judge configured, each deliverable is
rated against its own assignment, which needs no reference answer -- only the task text and
what the agent produced.

The seed is the harness the recursion starts from. It ships as a minimal AGENTS.md, and
`HARNESS2_CUSTOM_SEED` replaces it with a tree you exported from an earlier run, which is how
a harness refined on a public benchmark gets carried onto your own work.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .. import config
from ..core import tree as HT
from ..core.spec import BenchPrompts, BenchSpec, resolve_channel_model
from ..substrate import registry as SUB
from ..substrate.base import Install
from .agent_rollout import copy_tree_files, run_rollout

DEFAULT_TIMEOUT = 2400

# The assignment the agent is given. Kept deliberately plain: it names the folders and
# nothing about how the work will be looked at.
PROMPT = """=== TASK FOLDER ===
{task_folder}

=== INSTRUCTIONS ===
1. Read TASK.md in the task folder above.
2. Read whatever reference files in that folder the task refers to.
3. Complete the task as TASK.md specifies.
4. Save only the final deliverables to the output directory below. No intermediate or
   temporary files.

=== OUTPUT DIRECTORY ===
{output_dir}

IMPORTANT:
- Reference material is in {task_folder}; deliverables go in {output_dir}.
- Work only inside {workspace} (plus online sources the task asks you to consult).
- If the task is ambiguous, say which reading you chose and why before you proceed.
- If a file cannot be read directly (.xlsx, .docx, .db, .pptx), use code to extract it."""

SEED_AGENTS_MD = """\
You are a careful generalist. Read the assignment before you start, do the work it asks for,
and leave the deliverables in the output directory it names.
"""

_JUDGE_PROMPT = """\
You are rating one deliverable against the assignment that produced it. There is no reference
answer; judge only whether this deliverable does the assigned job well.

=== ASSIGNMENT ===
{task}

=== DELIVERABLE ===
{deliverable}

Reply with JSON only: {{"score": <integer 0-100>, "rationale": "<one or two sentences>"}}
100 means a practitioner would send it as-is. 0 means it does not attempt the assignment.
"""


def custom_prompts() -> BenchPrompts:
    """The work profile for arbitrary assignments: no profession, no domain overlay."""
    return BenchPrompts(
        work=(
            "complete the assignment in the task folder and leave the deliverables it asks "
            "for in the output directory, in the format it specifies"
        ),
        domain_word="area",
        deliverable_word="deliverable",
        inv_levers="",
        dom_levers="",
    )


def make_spec(substrate: str, *, runs_root: Path | None = None) -> BenchSpec:
    name = SUB.check(substrate)
    if not SUB.is_agent_cli(name):
        raise ValueError(
            f"the custom benchmark runs the agent-CLI substrates "
            f"{SUB.AGENT_CLI_SUBSTRATES}, got {name!r}: OpenCode and Codex need a benchmark-owned "
            f"checkout, render step and credential file, which `jb` and `lab` provide"
        )
    mode = "agent_root"
    return BenchSpec(
        name="custom",
        runs_root=runs_root or (config.RUNS / "custom"),
        install_mode=mode,
        prompts=custom_prompts(),
        channels=_channels(),
        domain_word="area",
        allow_subagents=True,
    )


def _channels() -> dict:
    ch = resolve_channel_model(None, None, effort="high")
    return {"proposer": ch, "selector": ch}


class CustomAdapter:
    """Your own tasks, executed by a Pi or DeepSeek Harness rollout and rated on demand."""

    def __init__(
        self,
        substrate: str,
        *,
        tasks_root: Path | str | None = None,
        seed_dir: Path | str | None = None,
        runs_root: Path | None = None,
        timeout: int | None = None,
    ):
        self.spec = make_spec(substrate, runs_root=runs_root)
        self.substrate = SUB.make(substrate, self._launch)
        self.tasks_root = Path(
            tasks_root or config.CUSTOM_TASKS or Path.cwd() / "custom_tasks"
        ).expanduser()
        self.seed_dir = Path(
            seed_dir or config.CUSTOM_SEED or config.SEEDS / "custom_default"
        ).expanduser()
        self.timeout = int(timeout or DEFAULT_TIMEOUT)
        self.store = self.spec.runs_root / "dataset"

    # -- tasks -------------------------------------------------------------------------

    def tasks(self, split: str = "all") -> list[str]:
        """Every `<area>/<task>` holding a TASK.md, sorted so a stream is reproducible."""
        if not self.tasks_root.is_dir():
            return []
        found = []
        for task_md in self.tasks_root.glob("*/*/TASK.md"):
            rel = task_md.parent.relative_to(self.tasks_root)
            if len(rel.parts) == 2:
                found.append(rel.as_posix())
        return sorted(found)

    def domain_of(self, task_id: str) -> str:
        return task_id.split("/", 1)[0]

    def task_prompt(self, task_id: str) -> str:
        return (self.tasks_root / task_id / "TASK.md").read_text(encoding="utf-8")

    def task_files(self, task_id: str) -> Path | None:
        d = self.tasks_root / task_id / "files"
        return d if d.is_dir() else None

    # -- execution ---------------------------------------------------------------------

    def base_harness(self) -> Path:
        """The seed this recursion starts from: a tree you exported, or the built-in one."""
        if not HT.is_tree(self.substrate.surface(), self.seed_dir):
            raise FileNotFoundError(
                f"seed harness {self.seed_dir} is not a {self.substrate.surface().name} tree "
                f"(it needs AGENTS.md); point HARNESS2_CUSTOM_SEED at one you exported with "
                f"harness2-export, or unset it to use the built-in seed"
            )
        return self.seed_dir

    def preflight(self) -> None:
        """The tasks, the seed and the CLI must all be usable before money is spent."""
        self.substrate.preflight()
        if not self.tasks_root.is_dir():
            raise RuntimeError(
                f"no custom tasks at {self.tasks_root}: create <area>/<task>/TASK.md there, or "
                f"point HARNESS2_CUSTOM_TASKS at a directory that has them"
            )
        if not self.tasks():
            raise RuntimeError(
                f"{self.tasks_root} holds no <area>/<task>/TASK.md, so there is nothing to run"
            )
        self.base_harness()

    def _out_dir(self, task_id: str, label: str) -> Path:
        return self.store / task_id / "model_output" / label

    def _traj_dir(self, task_id: str, label: str) -> Path:
        return self.store / task_id / "model_traj" / label

    def _launch(self, req, install: Install) -> list[Path]:
        if len(req.run_ids) != 1:
            raise ValueError(f"one rollout per invocation; got {req.run_ids}")
        task_id = req.task_ref
        label = req.run_ids[0].replace("/", "__")
        self.substrate.preflight()
        out, traj = self._out_dir(task_id, label), self._traj_dir(task_id, label)
        files = self.task_files(task_id)
        task_text = self.task_prompt(task_id)

        def prepare(ws: Path) -> str:
            task_folder = ws / "task_folder"
            task_folder.mkdir()
            (task_folder / "TASK.md").write_text(task_text, encoding="utf-8")
            if files is not None:
                shutil.copytree(files, task_folder / "files")
            (ws / "output").mkdir()
            return PROMPT.format(task_folder=task_folder, output_dir=ws / "output", workspace=ws)

        def collect(ws: Path) -> list[str]:
            return copy_tree_files(ws / "output", out)

        run_rollout(
            self.substrate,
            install=install,
            run_dir=traj,
            label=label,
            model=SUB.solver_model(self.substrate.name),
            timeout=int(req.timeout),
            prepare=prepare,
            collect=collect,
            stream_name=f"{task_id.replace('/', '_')}_{label}_{_stamp()}.jsonl",
        )
        if not self.delivered(out):
            raise RuntimeError(
                f"no deliverable for {task_id} @ {label} (see {traj}); left UNMARKED so the "
                f"next attempt re-rolls instead of being taken as a real zero"
            )
        (traj / ".adapt_complete").write_text("", encoding="utf-8")
        return [out]

    def execute(
        self, task_id: str, harness_dir: Path | None, run_id: str, *, timeout: int | None = None
    ) -> Path:
        label = run_id.replace("/", "__")
        out = self._out_dir(task_id, label)
        if (self._traj_dir(task_id, label) / ".adapt_complete").is_file() and self.delivered(out):
            return out
        tree_dir = Path(harness_dir) if harness_dir else self.base_harness()
        install = self.substrate.materialise(tree_dir, self.spec, {})
        self.substrate.execute(task_id, install, [run_id], int(timeout or self.timeout))
        return out

    # -- reading back ------------------------------------------------------------------

    def delivered(self, run_dir: Path) -> bool:
        d = Path(run_dir)
        return d.is_dir() and any(p.is_file() for p in d.rglob("*"))

    def trajectory(self, run_dir: Path, *, obs_trunc: int) -> str:
        task_id, label = self._task_and_label(run_dir)
        td = self._traj_dir(task_id, label)
        files = (
            sorted(td.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
            if td.is_dir()
            else []
        )
        if not files:
            return f"(no trajectory for {task_id} @ {label})"
        return self.substrate.read_trajectory(files[0], obs_trunc=obs_trunc, max_chars=8_000_000)

    def _task_and_label(self, run_dir: Path) -> tuple[str, str]:
        p = Path(run_dir)
        parts = p.parts
        if "model_output" not in parts:
            raise ValueError(f"{p} is not a custom run directory")
        i = parts.index("model_output")
        return "/".join(parts[i - 2 : i]), p.name

    def deliverable_files(self, run_dir: Path) -> dict[str, str]:
        d = Path(run_dir)
        if not d.is_dir():
            return {}
        out: dict[str, str] = {}
        for f in sorted(d.rglob("*")):
            if not f.is_file():
                continue
            try:
                out[f.relative_to(d).as_posix()] = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                out[f.relative_to(d).as_posix()] = f"(binary, {f.stat().st_size} bytes)"
        return out

    def deliverable_text(self, run_dir: Path) -> str:
        return "\n\n".join(f"=== {k} ===\n{v}" for k, v in self.deliverable_files(run_dir).items())

    def validate_tree(self, tree: dict[str, str]) -> None:
        """Nothing beyond the surface's own rules: these are your tasks, not a dataset's."""

    # -- measurement -------------------------------------------------------------------

    def score(self, run_dir: Path, task_id: str) -> dict | None:
        """A `check.py` verdict, else a judge's rating of the deliverable, else nothing.

        None means "not measured", which is not a zero and is never cached as one.
        """
        run_dir = Path(run_dir)
        if not self.delivered(run_dir):
            return {"n_passed": 0, "n_criteria": 1, "all_pass": False, "reason": "no deliverable"}
        check = self.tasks_root / task_id / "check.py"
        if check.is_file():
            return self._run_check(check, run_dir)
        got = self._judge(task_id, run_dir)
        return got

    @staticmethod
    def _run_check(check: Path, run_dir: Path) -> dict:
        """A check.py exits 0 for pass; its output is kept as the reason either way."""
        try:
            r = subprocess.run(
                [sys.executable, str(check), str(run_dir)],
                cwd=str(run_dir),
                capture_output=True,
                text=True,
                timeout=600,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return {"n_passed": 0, "n_criteria": 1, "all_pass": False, "reason": f"{exc}"}
        ok = r.returncode == 0
        return {
            "n_passed": 1 if ok else 0,
            "n_criteria": 1,
            "all_pass": ok,
            "reason": (r.stdout or r.stderr).strip()[-2000:],
        }

    def _judge(self, task_id: str, run_dir: Path) -> dict | None:
        """Rate a deliverable against its own assignment; no reference answer involved."""
        base, key, model = config.JB_JUDGE_API_BASE, config.JB_JUDGE_API_KEY, config.JUDGE_MODEL
        if not (base and key and model):
            return None
        deliverable = self.deliverable_text(run_dir)
        if not deliverable.strip():
            return None
        prompt = _JUDGE_PROMPT.format(
            task=self.task_prompt(task_id), deliverable=deliverable[:120_000]
        )
        try:
            got = _chat(base, key, model, prompt, config.judge_headers(base))
            message = got["choices"][0]["message"] or {}
            text = message.get("content") or message.get("reasoning_content") or ""
            rating = json.loads(_json_slice(text))
        except (OSError, KeyError, IndexError, TypeError, ValueError, urllib.error.URLError):
            return None
        score = int(rating.get("score", -1))
        if not 0 <= score <= 100:
            return None
        return {
            "n_passed": score,
            "n_criteria": 100,
            "all_pass": score == 100,
            "score": score,
            "reason": str(rating.get("rationale", ""))[:2000],
            "judge": model,
        }


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json_slice(text: str) -> str:
    """The first {...} block in `text`, since models decorate JSON with prose."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the judge reply")
    return text[start : end + 1]


def _chat(base_url: str, api_key: str, model: str, prompt: str, headers: dict[str, str]) -> dict:
    """One OpenAI-compatible chat completion, over the standard library only.

    No SDK: this adapter must work in the main environment, which has no judge dependency.
    """
    body = json.dumps(
        {
            "model": model,
            "temperature": 0,
            # A thinking judge spends part of this budget before it emits a verdict, and a
            # cut-off completion returns no content at all, which would read as a judge failure.
            "max_completion_tokens": 8192,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            # Some gateways front their API with a bot filter that rejects urllib's default
            # fingerprint outright, so the caller is named rather than left implicit.
            "User-Agent": "harness2-custom-judge/0.1",
            **headers,
        },
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def seed_harness(dest: Path) -> Path:
    """Write the built-in seed (one AGENTS.md) and return the directory."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "AGENTS.md").write_text(SEED_AGENTS_MD, encoding="utf-8")
    return dest


__all__ = [
    "CustomAdapter",
    "DEFAULT_TIMEOUT",
    "PROMPT",
    "SEED_AGENTS_MD",
    "custom_prompts",
    "make_spec",
    "seed_harness",
]
