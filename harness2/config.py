# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deployment settings and dependency versions. Shell values override .env."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent if (PKG.parent / "pyproject.toml").is_file() else Path.cwd()
THIRD_PARTY = ROOT / "third_party"
METHOD_VERSION = 2

DEPENDENCIES = ROOT / "dependencies"


class ConfigError(RuntimeError):
    """A required deployment value is missing; the message carries the remediation."""


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Load KEY=VALUE lines from `path` into os.environ with setdefault (the shell wins).

    Accepts blank lines, `#` comments, an optional `export ` prefix, and single- or
    double-quoted values. A missing or unreadable file is not an error.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or not all(c.isalnum() or c == "_" for c in key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        os.environ.setdefault(key, value)


load_dotenv()


def env(name: str, default=None):
    """Env override where a set-but-empty variable is the same as an unset one."""
    return os.environ.get(name) or default


def env_int(name: str, default: int) -> int:
    v = env(name)
    if v is None:
        return int(default)
    try:
        return int(v)
    except ValueError as e:
        raise ConfigError(f"{name} must be an integer, got {v!r}") from e


def require(name: str, *, needed_by: str) -> str:
    """The value of `name`, or ConfigError with the remediation line when it is unset/empty."""
    v = env(name)
    if not v:
        raise ConfigError(
            f"{name} is required for {needed_by}; set it in .env or the shell (see .env.example)"
        )
    return str(v)


RUNS = Path(env("HARNESS2_RUNS_DIR", ROOT / "runs")).expanduser().resolve()
# Per-rollout workspaces for CLI-agent substrates. The default lives in the system temp
# directory, outside any git checkout, so an agent that walks up to a project root finds none.
SCRATCH = (
    Path(env("HARNESS2_SCRATCH", Path(tempfile.gettempdir()) / "harness2-scratch"))
    .expanduser()
    .resolve()
)
# The custom benchmark: a directory of your own assignments, and an optional seed harness.
CUSTOM_TASKS = (
    Path(env("HARNESS2_CUSTOM_TASKS", "")).expanduser().resolve()
    if env("HARNESS2_CUSTOM_TASKS", "")
    else None
)
CUSTOM_SEED = (
    Path(env("HARNESS2_CUSTOM_SEED", "")).expanduser().resolve()
    if env("HARNESS2_CUSTOM_SEED", "")
    else None
)
SEEDS = PKG / "seeds"
VERTEX_PROJECT = env("GOOGLE_CLOUD_PROJECT", env("HARNESS2_VERTEX_PROJECT", ""))
VERTEX_REGION = env("GOOGLE_CLOUD_LOCATION", "global")
if VERTEX_PROJECT:
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", VERTEX_PROJECT)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", VERTEX_REGION)


SOLVER_MODEL_OC = env("HARNESS2_SOLVER_MODEL", "")
SOLVER_MODEL_CX = env("HARNESS2_CODEX_MODEL", SOLVER_MODEL_OC.split("/", 1)[-1])
SOLVER_MODEL_PI = env("HARNESS2_PI_MODEL", SOLVER_MODEL_OC)
SOLVER_MODEL_DSH = env("HARNESS2_DSH_MODEL", SOLVER_MODEL_OC)
SOLVER_EFFORT = env("HARNESS2_SOLVER_EFFORT", "high")
IMPROVER_MODEL = env("HARNESS2_IMPROVER_MODEL", SOLVER_MODEL_OC)
JUDGE_MODEL = env("HARNESS2_JUDGE_MODEL", "")

DEPENDENCY_PINS = {
    "job-bench-eval": "fa03a17f9ff662f136798a274e10011c3aa00fd0",
    "workbuddy-bench": "b516950be5b56eb3be406c2f76ee1c5111dcb57f",
    "harvey-labs": "b58a28ae9fde5ee4b810e53b4fd0a1915325a35d",
    "opencode": "23fb5e0516c99ac04a1aa46c193efda2e1b9bb24",
}
DEPENDENCY_ORIGINS = {
    "job-bench-eval": "https://github.com/Job-Bench/job-bench-eval.git",
    "workbuddy-bench": "https://github.com/Tencent/workbuddy-bench.git",
    "harvey-labs": "https://github.com/harveyai/harvey-labs.git",
    "opencode": "https://github.com/anomalyco/opencode.git",
}


def jb_repo() -> Path:
    return DEPENDENCIES / "job-bench-eval"


def wb_repo() -> Path:
    return DEPENDENCIES / "workbuddy-bench"


def lab_repo() -> Path:
    return DEPENDENCIES / "harvey-labs"


OPENCODE_DIR = DEPENDENCIES / "opencode"
BUN = DEPENDENCIES / "tools" / "node_modules" / ".bin" / "bun"
LAB_PYTHON = lab_repo() / ".venv" / "bin" / "python"
LAB_JUDGE_MODEL = (
    "vertex/" + JUDGE_MODEL if JUDGE_MODEL.startswith(("claude", "gemini")) else JUDGE_MODEL
)
LAB_DELIV_CAP = 60000
JB_JUDGE_PYTHON = jb_repo() / ".venv" / "bin" / "python"
JB_JUDGE_MODEL = (
    "google/" + JUDGE_MODEL
    if JUDGE_MODEL.startswith("gemini") and not env("HARNESS2_JUDGE_BASE_URL")
    else JUDGE_MODEL
)
_vertex_host = (
    "aiplatform.googleapis.com"
    if VERTEX_REGION == "global"
    else f"{VERTEX_REGION}-aiplatform.googleapis.com"
)
VERTEX_OPENAI_URL = (
    f"https://{_vertex_host}/v1/projects/{VERTEX_PROJECT}"
    f"/locations/{VERTEX_REGION}/endpoints/openapi"
)
JB_JUDGE_API_BASE = env("HARNESS2_JUDGE_BASE_URL", VERTEX_OPENAI_URL if VERTEX_PROJECT else "")
JB_JUDGE_API_KEY = env("HARNESS2_JUDGE_API_KEY", "")
JB_JUDGE_REASONING_EFFORT = "low"


# One judge session per process, so a gateway that routes on the header sees a stable shard.
_JUDGE_SESSION = f"harness2-judge-{uuid.uuid4()}"


def judge_headers(api_base: str) -> dict[str, str]:
    """Extra HTTP headers for an OpenAI-compatible judge endpoint.

    HARNESS2_JUDGE_HEADERS takes a JSON object. OpenCode's gateway routes on an
    `x-opencode-session` header and rejects requests without one, so a judge pointed at it
    gets one session id per process unless the variable already sets it.
    """
    raw = env("HARNESS2_JUDGE_HEADERS", "")
    try:
        headers = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise ConfigError(f"HARNESS2_JUDGE_HEADERS must be a JSON object: {exc}") from exc
    if not isinstance(headers, dict) or not all(isinstance(v, str) for v in headers.values()):
        raise ConfigError("HARNESS2_JUDGE_HEADERS must be a JSON object of string values")
    if "opencode.ai" in api_base and "x-opencode-session" not in headers:
        headers["x-opencode-session"] = _JUDGE_SESSION
    return headers


JB_REPORTING_JUDGE_DETAILS = ""
JB_OBS_TRUNC = 50000

CODEX_VERSION = "0.144.6"
CODEX_DIST = str(DEPENDENCIES / "codex" / "dist")
CODEX_BASE_URL = env("HARNESS2_CODEX_BASE_URL", "")
CODEX_CONTAINER_URL = env("HARNESS2_CODEX_CONTAINER_URL", CODEX_BASE_URL)
VERTEX_PROXY_KEY = env("HARNESS2_CODEX_API_KEY", "")
CODEX_HOME_DIR = DEPENDENCIES / "codex" / "home"

WB_SOLVER_MODEL = "harness2/solver"
WB_WEB_JUDGE = "harness2/judge"
WB_RENDER_PY = sys.executable
WB_OC_HARNESS_VER = "1.14.18"
WB_CX_HARNESS_VER = CODEX_VERSION
WB_MAX_JOBS = {"oc": 1, "cx": 1}
WB_MAX_TRIALS = {"oc": 2, "cx": 2}
WB_MODEL_CONNECTION = "local_proxy"

DEFAULT_PROPOSER_CHANNEL = env("HARNESS2_IMPROVER_CHANNEL", "opencode")
PROPOSER_MODEL_BY_CHANNEL = {
    "opencode": IMPROVER_MODEL,
    "claude_cli": IMPROVER_MODEL,
    "pi": IMPROVER_MODEL,
}
PHASE_WIDTH = 2
TRANSPORT_ATTEMPTS = 2
PROPOSER_CONCURRENCY = 2


def needs_opencode(substrate: str) -> bool:
    """OpenCode is needed when it executes rollouts or runs the improver sessions."""
    return substrate == "oc" or DEFAULT_PROPOSER_CHANNEL == "opencode"


def validate_models(*, substrate: str = "", judge: bool = False) -> None:
    """Fail fast on the model settings this run will use, before anything is spent.

    `substrate` narrows the executor check to the framework that will execute the rollouts,
    so a Pi-only setup does not need the OpenCode model set. Without it the OpenCode settings
    are checked, which is what every caller before the agent-CLI substrates needed.
    """
    from .substrate import registry as SUB  # local: registry builds on this module

    if DEFAULT_PROPOSER_CHANNEL not in PROPOSER_MODEL_BY_CHANNEL:
        raise ConfigError(
            f"HARNESS2_IMPROVER_CHANNEL={DEFAULT_PROPOSER_CHANNEL!r} is not one of "
            f"{tuple(PROPOSER_MODEL_BY_CHANNEL)}"
        )
    name = SUB.check(substrate or "oc")
    model = SUB.solver_model(name)
    if not model:
        raise ConfigError(
            f"Set {SUB.MODEL_VARS[name]} for {SUB.DESCRIPTIONS[name]} executor rollouts "
            f"(see .env.example)"
        )
    if name in SUB.MODEL_NEEDS_PROVIDER and "/" not in model:
        raise ConfigError(f"{SUB.MODEL_VARS[name]} must be provider/model, got {model!r}")
    if not IMPROVER_MODEL:
        raise ConfigError(
            "Set HARNESS2_IMPROVER_MODEL for the improver sessions (see .env.example)"
        )
    if "/" not in IMPROVER_MODEL:
        raise ConfigError(f"HARNESS2_IMPROVER_MODEL must be provider/model, got {IMPROVER_MODEL!r}")
    if any(m.startswith("google-vertex") for m in (model, IMPROVER_MODEL)):
        if not VERTEX_PROJECT:
            raise ConfigError("Set GOOGLE_CLOUD_PROJECT for Vertex AI")
    if judge:
        require("HARNESS2_JUDGE_MODEL", needed_by="benchmark judging")
