# SPDX-License-Identifier: Apache-2.0

"""The one list of solver substrates, so CLIs, adapters and docs cannot drift apart.

`oc` and `cx` are launched by benchmark-specific runners; `pi` and `dsh` are coding-agent
CLIs that share `benchmarks/agent_rollout.py`.
"""

from __future__ import annotations

from .. import config
from .base import Launcher

SUBSTRATES = ("oc", "cx", "pi", "dsh")
AGENT_CLI_SUBSTRATES = ("pi", "dsh")

DESCRIPTIONS = {
    "oc": "OpenCode",
    "cx": "Codex CLI",
    "pi": "Pi coding agent",
    "dsh": "DeepSeek Harness",
}

HELP = "task-executor substrate: " + " | ".join(f"{k} ({v})" for k, v in DESCRIPTIONS.items())

# The setting that names each substrate's executor model.
MODEL_VARS = {
    "oc": "HARNESS2_SOLVER_MODEL",
    "cx": "HARNESS2_CODEX_MODEL",
    "pi": "HARNESS2_PI_MODEL",
    "dsh": "HARNESS2_DSH_MODEL",
}

# Substrates whose CLI needs a provider-prefixed model id; Codex CLI and DeepSeek Harness
# resolve a bare id against their own configured provider.
MODEL_NEEDS_PROVIDER = frozenset({"oc", "pi"})

# The event-stream filename each substrate's rollout writes into its run directory.
SESSION_FILES = {
    "oc": "opencode_session.jsonl",
    "cx": "codex_session.jsonl",
    "pi": "pi_session.jsonl",
    "dsh": "dsh_session.jsonl",
}


def is_agent_cli(name: str) -> bool:
    return name in AGENT_CLI_SUBSTRATES


def check(name: str) -> str:
    if name not in SUBSTRATES:
        raise ValueError(f"unknown substrate {name!r}; expected one of {SUBSTRATES}")
    return name


def make(name: str, launcher: Launcher | None = None):
    """A substrate object by short name, with the benchmark's launcher body attached."""
    check(name)
    if name == "oc":
        from .opencode import OpencodeSubstrate

        return OpencodeSubstrate(launcher=launcher)
    if name == "cx":
        from .codex import CodexSubstrate

        return CodexSubstrate(launcher=launcher)
    if name == "pi":
        from .pi import PiSubstrate

        return PiSubstrate(launcher=launcher)
    from .dsh import DshSubstrate

    return DshSubstrate(launcher=launcher)


def solver_model(name: str) -> str:
    """The executor model id a substrate runs, in that substrate's own syntax."""
    return {
        "oc": config.SOLVER_MODEL_OC,
        "cx": config.SOLVER_MODEL_CX,
        "pi": config.SOLVER_MODEL_PI,
        "dsh": config.SOLVER_MODEL_DSH,
    }[check(name)]


__all__ = [
    "AGENT_CLI_SUBSTRATES",
    "DESCRIPTIONS",
    "HELP",
    "MODEL_NEEDS_PROVIDER",
    "MODEL_VARS",
    "SESSION_FILES",
    "SUBSTRATES",
    "check",
    "is_agent_cli",
    "make",
    "solver_model",
]
