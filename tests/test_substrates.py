# SPDX-License-Identifier: Apache-2.0

"""Regressions for the agent-CLI substrates, the custom benchmark and seed export.

These never start a model. They cover the parts that silently break a rollout before any
model is reached: argv/env construction, the overlay, the renderers, the load checks that
reject an unloadable harness, and the export/import round trip.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness2 import config
from harness2.benchmarks import agent_rollout as AR
from harness2.benchmarks import custom
from harness2.benchmarks.custom import CustomAdapter, make_spec, seed_harness
from harness2.cli import export_harness, setup
from harness2.cli._common import make_adapter
from harness2.core import runmark as RM
from harness2.core import tree as HT
from harness2.core.spec import BenchPrompts, BenchSpec, ChannelSpec, resolve_proposer
from harness2.preflight import check
from harness2.substrate import dsh, pi, registry
from harness2.substrate.agent_cli import clip, iter_jsonl, stage_root

# These checks run a real binary; on a host without it they are skipped rather than faked,
# so the suite says what it did not cover.
PI_INSTALLED = bool(shutil.which("pi"))
NODE_INSTALLED = bool(shutil.which("node"))


def _spec(substrate: str) -> BenchSpec:
    ch = ChannelSpec("opencode", "local/deterministic")
    return BenchSpec(
        name="t",
        runs_root=Path(tempfile.gettempdir()) / "harness2-test-runs",
        install_mode="agent_root",
        prompts=BenchPrompts(work="do the work"),
        channels={"proposer": ch, "selector": ch},
    )


class RegistryTests(unittest.TestCase):
    def test_every_substrate_is_described_and_resolvable(self):
        self.assertEqual(set(registry.SUBSTRATES), set(registry.DESCRIPTIONS))
        self.assertEqual(set(registry.SUBSTRATES), set(registry.SESSION_FILES))
        self.assertEqual(set(registry.SUBSTRATES), set(registry.MODEL_VARS))
        for name in registry.SUBSTRATES:
            self.assertEqual(registry.make(name).name, name)

    def test_unknown_substrate_is_rejected(self):
        with self.assertRaises(ValueError):
            registry.check("nope")

    def test_agent_cli_membership(self):
        self.assertTrue(registry.is_agent_cli("pi"))
        self.assertTrue(registry.is_agent_cli("dsh"))
        self.assertFalse(registry.is_agent_cli("oc"))


class AgentCliTests(unittest.TestCase):
    def test_iter_jsonl_skips_banners_and_torn_lines(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.jsonl"
            p.write_text(
                'banner text\n{"type":"turn_end"}\n{"type":"torn"\n[1,2]\n{"ok":true}\n',
                encoding="utf-8",
            )
            self.assertEqual([e.get("type") for e in iter_jsonl(p)], ["turn_end", None])

    def test_iter_jsonl_tolerates_a_missing_file(self):
        self.assertEqual(list(iter_jsonl(Path("/nonexistent/x.jsonl"))), [])

    def test_clip_is_bounded_and_labelled(self):
        out = clip("x" * 100, 10)
        self.assertLess(len(out), 60)
        self.assertIn("+90", out)
        self.assertEqual(clip("short", 10), "short")

    def test_stage_root_refuses_a_foreign_install_mode(self):
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            (t / "AGENTS.md").write_text("x", encoding="utf-8")
            spec = _spec("pi")
            object.__setattr__(spec, "install_mode", "codex_root")
            with self.assertRaises(ValueError):
                stage_root(pi.PI_SURFACE, t, pi.PI_ROOT_ENTRIES, spec)

    def test_stage_root_keeps_only_declared_entries(self):
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            (t / "AGENTS.md").write_text("role", encoding="utf-8")
            (t / "secrets.txt").write_text("no", encoding="utf-8")
            (t / "scripts").mkdir()
            (t / "scripts" / "a.py").write_text("print(1)", encoding="utf-8")
            install = stage_root(pi.PI_SURFACE, t, pi.PI_ROOT_ENTRIES, _spec("pi"))
            staged = sorted(
                p.relative_to(install.staging_dir).as_posix()
                for p in install.staging_dir.rglob("*")
                if p.is_file()
            )
            self.assertEqual(staged, ["AGENTS.md", "scripts/a.py"])
            self.assertIn("install_mode", install.extras)


class PiSubstrateTests(unittest.TestCase):
    def test_effort_suffix_is_added_once(self):
        self.assertEqual(pi.split_effort("p/m", "high"), "p/m:high")
        self.assertEqual(pi.split_effort("p/m:xhigh", "high"), "p/m:xhigh")
        self.assertEqual(pi.split_effort("p/m", ""), "p/m")

    def test_argv_is_fully_explicit_and_never_discovers(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            (ws / "AGENTS.md").write_text("role", encoding="utf-8")
            (ws / ".agents/skills/s1").mkdir(parents=True)
            (ws / ".agents/skills/s1/SKILL.md").write_text("---\nname: s1\n---\n", encoding="utf-8")
            (ws / ".pi/extensions").mkdir(parents=True)
            (ws / ".pi/extensions/e.ts").write_text("export default function(){}", encoding="utf-8")
            argv = pi.PiSubstrate().argv("do it", ws, "opencode-go/x", effort="high")
            for flag in (
                "--no-session",
                "--no-approve",
                "--no-context-files",
                "--no-extensions",
                "--no-skills",
                "--no-prompt-templates",
                "--no-themes",
            ):
                self.assertIn(flag, argv)
            self.assertEqual(argv[argv.index("--append-system-prompt") + 1], str(ws / "AGENTS.md"))
            self.assertIn(str(ws / ".agents/skills/s1"), argv)
            self.assertIn(str(ws / ".pi/extensions/e.ts"), argv)
            self.assertEqual(argv[argv.index("--model") + 1], "opencode-go/x:high")
            self.assertEqual(argv[-1], "do it")

    def test_empty_agents_md_is_not_appended(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            (ws / "AGENTS.md").write_text("   \n", encoding="utf-8")
            self.assertNotIn("--append-system-prompt", pi.PiSubstrate().argv("p", ws, "m"))

    def test_env_isolated_and_parent_session_markers_dropped(self):
        env = pi.PiSubstrate().env(
            {"PI_SESSION_ID": "leak", "PI_MODEL": "leak", "PATH": "/bin", "KEEP": "1"},
            agent_dir=Path("/tmp/agent"),
        )
        for gone in ("PI_SESSION_ID", "PI_MODEL"):
            self.assertNotIn(gone, env)
        self.assertEqual(env["PATH"], "/bin")
        self.assertEqual(env["KEEP"], "1")
        self.assertEqual(env["PI_CODING_AGENT_DIR"], "/tmp/agent")
        self.assertEqual(env["PI_OFFLINE"], "1")

    def test_skill_names_pi_refuses_are_reported_with_a_fix(self):
        bad = {"AGENTS.md": "x", ".agents/skills/Bad_Name/SKILL.md": "---\nname: Bad_Name\n---\n"}
        problems = pi.PiSubstrate().check_tree(bad)
        self.assertEqual(len(problems), 1)
        self.assertIn("bad-name", problems[0])

    @unittest.skipUnless(PI_INSTALLED, "the pi CLI is not installed")
    def test_a_sound_extension_passes_the_offline_load_check(self):
        ext = {
            ".pi/extensions/ok.ts": (
                'export default function (pi: any) { pi.on("tool_call", async () => undefined); }\n'
            )
        }
        self.assertEqual(pi.check_extensions(ext), [])

    @unittest.skipUnless(PI_INSTALLED, "the pi CLI is not installed")
    def test_a_broken_extension_is_rejected_before_any_model_call(self):
        bad = {".pi/extensions/bad.ts": "export default function ( {\n"}
        problems = pi.check_extensions(bad)
        self.assertEqual(len(problems), 1)
        self.assertIn("fails to load", problems[0])
        self.assertIn(".pi/extensions/bad.ts", problems[0])


class PiRendererTests(unittest.TestCase):
    def _stream(self, td: str) -> Path:
        p = Path(td) / "s.jsonl"
        events = [
            {"type": "turn_end"},
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "hmm"},
                        {"type": "text", "text": "working"},
                        {"type": "toolCall", "name": "bash", "arguments": {"command": "ls"}},
                    ],
                    "usage": {
                        "input": 10,
                        "output": 4,
                        "cacheRead": 7,
                        "reasoning": 2,
                        "cost": {"total": 0.5},
                    },
                },
            },
            {
                "type": "message_end",
                "message": {
                    "role": "toolResult",
                    "content": [{"type": "text", "text": "boom"}],
                    "isError": True,
                },
            },
            {"type": "tool_execution_start"},
        ]
        p.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
        return p

    def test_render_uses_the_shared_grammar(self):
        with tempfile.TemporaryDirectory() as td:
            out = pi.render_events(self._stream(td), obs_trunc=100, max_chars=10_000)
            for tag in ("[REASONING]", "[ASSISTANT]", "[ACTION] bash:", "[OBSERVATION ERROR]"):
                self.assertIn(tag, out)

    def test_summarise_counts_and_totals(self):
        with tempfile.TemporaryDirectory() as td:
            s = pi.summarise(self._stream(td))
            self.assertEqual(s["turns"], 1)
            self.assertEqual(s["tool_calls"], 1)
            self.assertEqual(s["input_tokens"], 10)
            self.assertEqual(s["cached_input_tokens"], 7)
            self.assertEqual(s["cost_usd"], 0.5)
            self.assertEqual(s["final_text"], "working")
            self.assertGreater(s["n_events"], 0)

    def test_no_events_is_not_rendered_as_success(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "empty.jsonl"
            p.write_text("", encoding="utf-8")
            self.assertIn("no parseable events", pi.render_events(p, obs_trunc=10, max_chars=100))


class DshSubstrateTests(unittest.TestCase):
    def test_model_split(self):
        self.assertEqual(dsh.split_model("opencode-go/x"), ("opencode-go", "x"))
        # dsh registers its API-key adapter as `deepseek-official`; a bare id that said
        # `deepseek` fails with NO_ADAPTER before the first token.
        self.assertEqual(dsh.split_model("x"), ("deepseek-official", "x"))
        self.assertEqual(dsh.DEEPSEEK_PROVIDER, "deepseek-official")

    def test_overlay_carries_model_route_and_plugins(self):
        with tempfile.TemporaryDirectory() as td:
            plugin = Path(td) / "p.ts"
            plugin.write_text("export const name='p'; export function apply(){}", encoding="utf-8")
            rows = dsh.overlay_rows("opencode-go/x", effort="high", plugins=[plugin])
            self.assertEqual(
                rows[0],
                {
                    "id": "agent-default-model",
                    "config": {"provider": "opencode-go", "model": "x", "reasoningEffort": "high"},
                },
            )
            self.assertEqual(
                rows[1], {"id": "llm-pi-ai", "config": {"providers": {"opencode-go": {}}}}
            )
            self.assertEqual(rows[2]["insert"][0]["name"], str(plugin.resolve()))

    def test_native_provider_needs_no_route_row(self):
        rows = dsh.overlay_rows("x", effort="", plugins=[])
        self.assertEqual([r.get("id") for r in rows], ["agent-default-model"])
        self.assertNotIn("reasoningEffort", rows[0]["config"])
        self.assertEqual(rows[0]["config"]["provider"], "deepseek-official")
        # a prefixed id other than DeepSeek's own needs the pi-ai route registered
        self.assertEqual(
            [r.get("id") for r in dsh.overlay_rows("opencode-go/x", effort="", plugins=[])],
            ["agent-default-model", "llm-pi-ai"],
        )

    def test_overlay_written_from_the_installed_tree(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            (ws / ".dsh/plugins").mkdir(parents=True)
            (ws / ".dsh/plugins/a.ts").write_text(
                "export const name='a';export const apply=()=>{}", encoding="utf-8"
            )
            (ws / ".dsh/plugins/z.txt").write_text("ignored", encoding="utf-8")
            dest = dsh.DshSubstrate().write_overlay(ws, ws / "o.json", "deepseek/x")
            rows = json.loads(dest.read_text())
            self.assertEqual(rows[-1]["insert"][0]["id"], "harness2-plugin-a")
            self.assertEqual(len(rows[-1]["insert"]), 1)

    def test_argv_applies_the_extra_overlay_last(self):
        with patch.dict("os.environ", {"HARNESS2_DSH_PATCH": "/tmp/extra.yml"}):
            argv = dsh.DshSubstrate().argv("p", Path("/tmp/o.json"))
        self.assertEqual(argv[-1], "p")
        self.assertEqual(argv.count("--patch"), 2)
        second = argv.index("--patch", argv.index("--patch") + 1)
        self.assertEqual(Path(argv[second + 1]), Path("/tmp/extra.yml").resolve())
        self.assertIn("--json", argv)

    def test_env_isolates_home_and_sets_permission_mode(self):
        env = dsh.DshSubstrate().env({}, home=Path("/tmp/h"), agents_home=Path("/tmp/a"))
        self.assertEqual(env["DSH_HOME"], "/tmp/h")
        self.assertEqual(env["DSH_AGENTS_HOME"], "/tmp/a")
        self.assertEqual(env["DSH_PERMISSION_MODE"], "danger-full-access")

    @unittest.skipUnless(NODE_INSTALLED, "node is not installed")
    def test_plugin_shape_is_checked_offline(self):
        problems = dsh.DshSubstrate().check_tree({".dsh/plugins/x.ts": "export const name = 1;\n"})
        self.assertEqual(len(problems), 1)
        self.assertIn("dsh plugin", problems[0])

    @unittest.skipUnless(NODE_INSTALLED, "node is not installed")
    def test_sound_plugin_passes(self):
        src = "export const name = 'x';\nexport function apply(ctx: any) { void ctx; }\n"
        self.assertEqual(dsh.DshSubstrate().check_tree({".dsh/plugins/x.ts": src}), [])

    @unittest.skipUnless(NODE_INSTALLED, "node is not installed")
    def test_syntax_error_is_reported_against_the_relative_path(self):
        problems = dsh.DshSubstrate().check_tree({".dsh/plugins/bad.ts": "export function ( {\n"})
        self.assertEqual(len(problems), 1)
        self.assertIn(".dsh/plugins/bad.ts", problems[0])
        self.assertNotIn(tempfile.gettempdir(), problems[0])

    def test_renderer_and_summary(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.jsonl"
            events = [
                {"type": "thinking", "text": "t"},
                {"type": "text", "text": "answer"},
                {"type": "tool_call", "tool": "bash", "input": {"command": "ls"}},
                {"type": "tool_result", "status": "error", "result": "boom"},
                {
                    "type": "status",
                    "phase": "step_end",
                    "usage": {"inputTokens": 5, "outputTokens": 2, "cacheReadTokens": 3},
                },
                {"type": "status", "phase": "turn_end", "reason": {"kind": "completed"}},
                {"type": "final", "text": "answer"},
            ]
            p.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
            out = dsh.render_events(p, obs_trunc=100, max_chars=10_000)
            self.assertIn("[ACTION] bash:", out)
            self.assertIn("[OBSERVATION ERROR]", out)
            s = dsh.summarise(p)
            self.assertEqual((s["turns"], s["tool_calls"], s["outcome"]), (1, 1, "completed"))
            self.assertEqual((s["input_tokens"], s["cached_input_tokens"]), (5, 3))
            self.assertEqual(s["final_text"], "answer")
            self.assertEqual(out.count("[ASSISTANT]"), 1)


class RolloutTests(unittest.TestCase):
    def test_scratch_refuses_to_sit_inside_a_checkout(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".git").mkdir()
            with patch.object(config, "SCRATCH", repo / "scratch"), self.assertRaises(RuntimeError):
                AR.scratch_dir("pi", "x")

    def test_install_harness_drops_offsurface_and_stubs(self):
        class FakeSub:
            name = "pi"

            def surface(self):
                return pi.PI_SURFACE

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "staged"
            (src / "scripts").mkdir(parents=True)
            (src / "AGENTS.md").write_text("role", encoding="utf-8")
            (src / "scripts" / "README.md").write_text("stub", encoding="utf-8")
            (src / "scripts" / "ok.py").write_text("print(1)", encoding="utf-8")
            (src / "notes.txt").write_text("off-surface", encoding="utf-8")
            ws = Path(td) / "ws"
            ws.mkdir()
            from harness2.substrate.base import Install

            installed = AR.install_harness(
                FakeSub(), Install(staging_dir=src, fingerprint="f", extras={}), ws
            )
            self.assertEqual(sorted(installed), ["AGENTS.md", "scripts/ok.py"])
            self.assertFalse((ws / "notes.txt").exists())

    def test_copy_tree_files_reports_relative_paths(self):
        with tempfile.TemporaryDirectory() as td:
            src, dest = Path(td) / "out", Path(td) / "got"
            (src / "sub").mkdir(parents=True)
            (src / "sub/a.md").write_text("a", encoding="utf-8")
            self.assertEqual(AR.copy_tree_files(src, dest), ["sub/a.md"])
            self.assertEqual((dest / "sub/a.md").read_text(), "a")
            self.assertEqual(AR.copy_tree_files(Path(td) / "missing", dest), [])


class CustomBenchTests(unittest.TestCase):
    def _tasks(self, root: Path) -> None:
        t = root / "area" / "task-a"
        (t / "files").mkdir(parents=True)
        (t / "TASK.md").write_text("Do the thing.", encoding="utf-8")
        (t / "files" / "in.md").write_text("source", encoding="utf-8")

    def test_tasks_are_areas_over_task_directories(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._tasks(root)
            (root / "area" / "no-task").mkdir()
            a = CustomAdapter("pi", tasks_root=root, runs_root=root / "runs")
            self.assertEqual(a.tasks(), ["area/task-a"])
            self.assertEqual(a.domain_of("area/task-a"), "area")

    def test_no_tasks_root_yields_no_tasks_rather_than_an_error(self):
        a = CustomAdapter("pi", tasks_root=Path("/nonexistent/tasks"))
        self.assertEqual(a.tasks(), [])

    def test_opencode_and_codex_are_refused_with_the_reason(self):
        for name in ("oc", "cx"):
            with self.assertRaises(ValueError) as ctx:
                make_spec(name)
            self.assertIn("agent-CLI", str(ctx.exception))

    def test_seed_defaults_to_the_builtin_tree(self):
        with tempfile.TemporaryDirectory() as td:
            seed = seed_harness(Path(td) / "seed")
            a = CustomAdapter("pi", tasks_root=Path(td), seed_dir=seed)
            self.assertTrue(HT.is_tree(pi.PI_SURFACE, a.base_harness()))
            self.assertEqual(a.base_harness(), seed)

    def test_missing_seed_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as td:
            a = CustomAdapter("pi", tasks_root=Path(td), seed_dir=Path(td) / "nope")
            with self.assertRaises(FileNotFoundError) as ctx:
                a.base_harness()
            self.assertIn("AGENTS.md", str(ctx.exception))

    def test_preflight_names_what_is_missing(self):
        with tempfile.TemporaryDirectory() as td:
            a = CustomAdapter(
                "pi", tasks_root=Path(td) / "none", seed_dir=seed_harness(Path(td) / "s")
            )
            with patch.object(a.substrate, "preflight", lambda: None):
                with self.assertRaises(RuntimeError) as ctx:
                    a.preflight()
            self.assertIn("HARNESS2_CUSTOM_TASKS", str(ctx.exception))

    def test_deliverables_are_read_back_as_text(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "model_output" / "lbl"
            d.mkdir(parents=True)
            (d / "out.md").write_text("hello", encoding="utf-8")
            a = CustomAdapter("pi", tasks_root=Path(td))
            self.assertTrue(a.delivered(d))
            self.assertEqual(a.deliverable_files(d), {"out.md": "hello"})
            self.assertIn("hello", a.deliverable_text(d))

    def test_check_script_decides_the_verdict(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._tasks(root)
            check = root / "area" / "task-a" / "check.py"
            check.write_text(
                "import sys, pathlib\n"
                "p = pathlib.Path(sys.argv[1]) / 'ok.md'\n"
                "good = p.is_file() and p.read_text().strip() == 'x'\n"
                "print('ok' if good else 'wrong content')\n"
                "raise SystemExit(0 if good else 1)\n",
                encoding="utf-8",
            )
            a = CustomAdapter("pi", tasks_root=root)
            out = root / "model_output" / "lbl"
            out.mkdir(parents=True)
            (out / "ok.md").write_text("x", encoding="utf-8")
            got = a.score(out, "area/task-a")
            self.assertEqual((got["n_passed"], got["n_criteria"], got["all_pass"]), (1, 1, True))
            (out / "ok.md").write_text("not x", encoding="utf-8")
            got = a.score(out, "area/task-a")
            self.assertEqual((got["n_passed"], got["all_pass"]), (0, False))
            self.assertIn("wrong content", got["reason"])

    def test_no_deliverable_is_measured_not_judged(self):
        with tempfile.TemporaryDirectory() as td:
            a = CustomAdapter("pi", tasks_root=Path(td))
            Empty = Path(td) / "model_output" / "lbl"
            Empty.mkdir(parents=True)
            self.assertEqual(a.score(Empty, "area/task-a")["n_passed"], 0)

    def test_no_judge_configured_means_unmeasured_not_zero(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._tasks(root)
            out = root / "model_output" / "lbl"
            out.mkdir(parents=True)
            (out / "x.md").write_text("y", encoding="utf-8")
            a = CustomAdapter("pi", tasks_root=root)
            with (
                patch.object(config, "JUDGE_MODEL", ""),
                patch.object(config, "JB_JUDGE_API_KEY", ""),
            ):
                self.assertIsNone(a.score(out, "area/task-a"))


class ExportTests(unittest.TestCase):
    def test_export_then_import_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "src"
            (src / ".agents/skills/s").mkdir(parents=True)
            (src / "AGENTS.md").write_text(
                "role\n\nWhen X, load the `s` skill.\n", encoding="utf-8"
            )
            (src / ".agents/skills/s/SKILL.md").write_text(
                "---\nname: s\n---\nbody\n", encoding="utf-8"
            )
            (src / "offsurface.txt").write_text("no", encoding="utf-8")
            dest = root / "out"
            with patch("sys.stdout"):
                code = export_harness.main(
                    ["--from", str(src), "--substrate", "pi", "--to", str(dest)]
                )
            self.assertEqual(code, 0)
            self.assertEqual(HT.read_tree(pi.PI_SURFACE, dest), HT.read_tree(pi.PI_SURFACE, src))
            self.assertFalse((dest / "offsurface.txt").exists())
            self.assertTrue((dest / "harness2.json").is_file())
            a = CustomAdapter("pi", tasks_root=root, seed_dir=dest)
            self.assertTrue(HT.is_tree(pi.PI_SURFACE, a.base_harness()))

    def test_export_refuses_a_non_empty_destination_without_force(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "src"
            src.mkdir()
            (src / "AGENTS.md").write_text("role", encoding="utf-8")
            dest = root / "out"
            dest.mkdir()
            (dest / "keep.txt").write_text("keep", encoding="utf-8")
            with patch("sys.stdout"):
                self.assertEqual(export_harness.main(["--from", str(src), "--to", str(dest)]), 1)
                self.assertEqual(
                    export_harness.main(["--from", str(src), "--to", str(dest), "--force"]),
                    0,
                )
            self.assertFalse((dest / "keep.txt").exists())
            self.assertTrue((dest / "AGENTS.md").is_file())

    def test_export_refuses_a_tree_without_agents_md(self):
        with tempfile.TemporaryDirectory() as td:
            with patch("sys.stdout"):
                self.assertEqual(
                    export_harness.main(["--from", td, "--to", str(Path(td) / "o")]), 1
                )


if __name__ == "__main__":
    unittest.main()


class FakeSubstrate:
    """A substrate whose rollout is whatever the test wants, for runner-level rules."""

    name = "pi"

    def __init__(self, events, *, model_actual="", sleep=0):
        self._events = events
        self._model_actual = model_actual
        self._sleep = sleep

    def surface(self):
        return pi.PI_SURFACE

    def invocation(self, prompt, workspace, scratch, model, *, effort, base_env):
        code = f"import time; time.sleep({self._sleep})" if self._sleep else "pass"
        return [sys.executable, "-c", code], {}

    def summarise(self, path):
        return {
            "n_events": self._events,
            "turns": 1,
            "tool_calls": 0,
            "errors": 0,
            "input_tokens": 1,
            "output_tokens": 1,
            "cached_input_tokens": 0,
            "reasoning_tokens": 0,
            "cost_usd": 0.0,
            "final_text": "",
            "model_actual": self._model_actual,
        }


def _install(td: Path):
    from harness2.substrate.base import Install

    staged = td / "staged"
    staged.mkdir(parents=True, exist_ok=True)
    (staged / "AGENTS.md").write_text("role", encoding="utf-8")
    return Install(staging_dir=staged, fingerprint="f", extras={})


class RunnerRuleTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        self._patches = [
            patch.object(config, "SCRATCH", self.root / "scratch"),
            patch.object(config, "SOLVER_EFFORT", "high"),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        self._td.cleanup()

    def _run(self, sub, *, timeout=60, model="p/m", out_name="out.md"):
        def prepare(ws):
            return "do it"

        def collect(ws):
            if out_name:
                (ws / "output").mkdir(exist_ok=True)
                (ws / "output" / out_name).write_text("x", encoding="utf-8")
            return [out_name] if out_name else []

        return AR.run_rollout(
            sub,
            install=_install(self.root),
            run_dir=self.root / "run",
            label="lbl",
            model=model,
            timeout=timeout,
            prepare=prepare,
            collect=collect,
        )

    def test_a_sound_rollout_is_measured(self):
        m = self._run(FakeSubstrate(5))
        self.assertEqual(m["exit_code"], 0)
        self.assertTrue(m["finished_cleanly"])
        self.assertEqual(m["deliverables"], ["out.md"])

    def test_no_events_is_refused(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._run(FakeSubstrate(0))
        self.assertIn("no events", str(ctx.exception))

    def test_a_substituted_model_is_refused(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._run(FakeSubstrate(5, model_actual="p/other"))
        self.assertIn("but p/m was requested", str(ctx.exception))

    def test_the_reported_model_is_recorded(self):
        m = self._run(FakeSubstrate(5, model_actual="p/m"))
        self.assertEqual(m["model_actual"], "p/m")

    def test_a_timed_out_rollout_is_never_measured_as_finished(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._run(FakeSubstrate(5, sleep=30), timeout=1)
        self.assertIn("deadline", str(ctx.exception))
        # evidence survives the refusal, so the attempt can still be inspected
        metrics = json.loads((self.root / "run" / "metrics.json").read_text())
        self.assertTrue(metrics["timed_out"])
        self.assertFalse(metrics["finished_cleanly"])

    def test_offsurface_files_never_reach_the_workspace(self):
        staged = self.root / "staged2"
        (staged / "scripts").mkdir(parents=True)
        (staged / "AGENTS.md").write_text("role", encoding="utf-8")
        (staged / "notes.txt").write_text("no", encoding="utf-8")
        from harness2.substrate.base import Install

        ws = self.root / "ws"
        ws.mkdir()
        installed = AR.install_harness(
            FakeSubstrate(1), Install(staging_dir=staged, fingerprint="f", extras={}), ws
        )
        self.assertEqual(installed, ["AGENTS.md"])
        self.assertFalse((ws / "notes.txt").exists())


class _FakeSubstrate:
    """A substrate with no CLI: materialise and preflight are stubs, and execute runs the
    adapter's own launcher, which is the code under test."""

    name = "pi"

    def __init__(self, adapter):
        self._adapter = adapter

    def surface(self):
        return pi.PI_SURFACE

    def preflight(self):
        return None

    def materialise(self, tree_dir, spec, ctx):
        from harness2.substrate.base import Install

        d = Path(tempfile.mkdtemp()) / "staged"
        d.mkdir(parents=True)
        (d / "AGENTS.md").write_text("role", encoding="utf-8")
        return Install(staging_dir=d, fingerprint="f", extras={})

    def execute(self, task_id, install, run_ids, timeout):
        from harness2.substrate.base import RunRequest

        return self._adapter._launch(
            RunRequest(task_ref=task_id, run_ids=tuple(run_ids), timeout=timeout), install
        )


class CustomMarkerTests(unittest.TestCase):
    """The real launcher path, with only the model call stubbed out."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        (self.root / "area" / "t").mkdir(parents=True)
        (self.root / "area" / "t" / "TASK.md").write_text("do it", encoding="utf-8")
        self.seed = seed_harness(self.root / "seed")

    def tearDown(self):
        self._td.cleanup()

    def _adapter(self):
        a = CustomAdapter(
            "pi", tasks_root=self.root, seed_dir=self.seed, runs_root=self.root / "runs"
        )
        a.substrate = _FakeSubstrate(a)
        return a

    def _rollout(self, a, *, produce):
        """A stand-in for run_rollout that exercises prepare() and collect() for real."""
        calls = []

        def fake(substrate, *, install, run_dir, label, model, timeout, prepare, collect, **_):
            calls.append(label)
            ws = Path(tempfile.mkdtemp())
            prepare(ws)
            if produce:
                (ws / "output").mkdir(exist_ok=True)
                (ws / "output" / "g.md").write_text("good", encoding="utf-8")
            collect(ws)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
            return {"deliverables": []}

        return fake, calls

    def test_a_finished_rollout_is_marked_and_reused(self):
        a = self._adapter()
        fake, calls = self._rollout(a, produce=True)
        with patch.object(custom, "run_rollout", fake):
            out = a.execute("area/t", None, "area/t/base1")
            self.assertTrue(a.delivered(out))
            self.assertEqual(len(calls), 1)
            a.execute("area/t", None, "area/t/base1")
        self.assertEqual(len(calls), 1, "a finished rollout must not run twice")

    def test_an_empty_rollout_is_refused_and_left_unmarked(self):
        a = self._adapter()
        fake, _ = self._rollout(a, produce=False)
        with patch.object(custom, "run_rollout", fake), self.assertRaises(RuntimeError):
            a.execute("area/t", None, "area/t/base1")
        mark = a._traj_dir("area/t", "area__t__base1")
        self.assertFalse((mark / RM.MARK).exists(), "a refused rollout must stay unmarked")

    def test_a_re_roll_cannot_inherit_the_previous_attempt(self):
        a = self._adapter()
        good, _ = self._rollout(a, produce=True)
        with patch.object(custom, "run_rollout", good):
            out = a.execute("area/t", None, "area/t/base1")
        self.assertTrue((out / "g.md").is_file())

        # Same task and run id, a DIFFERENT harness, and an attempt that writes nothing:
        # without clearing the output dir this would be marked done and graded on g.md.
        other = self.root / "seed2"
        other.mkdir()
        (other / "AGENTS.md").write_text("a different role", encoding="utf-8")
        a2 = self._adapter()
        a2.seed_dir = other
        empty, _ = self._rollout(a2, produce=False)
        with patch.object(custom, "run_rollout", empty), self.assertRaises(RuntimeError):
            a2.execute("area/t", None, "area/t/base1")
        self.assertFalse((out / "g.md").exists(), "the stale artifact would be graded again")

    def test_a_changed_harness_rerolls_the_same_run_id(self):
        a = self._adapter()
        fake, calls = self._rollout(a, produce=True)
        other = self.root / "seed2"
        other.mkdir()
        (other / "AGENTS.md").write_text("a different role", encoding="utf-8")
        with patch.object(custom, "run_rollout", fake):
            a.execute("area/t", None, "area/t/base1")
            a.seed_dir = other
            a.execute("area/t", None, "area/t/base1")
        self.assertEqual(len(calls), 2)


class ScrubTests(unittest.TestCase):
    def test_credentials_are_stripped_before_a_load_check(self):
        from harness2.substrate.agent_cli import scrubbed_env

        env = scrubbed_env(
            {
                "OPENCODE_API_KEY": "a",
                "AWS_SECRET_ACCESS_KEY": "b",
                "GOOGLE_APPLICATION_CREDENTIALS": "c",
                "DSH_HOME": "d",
                "GITHUB_TOKEN": "e",
                "PATH": "/bin",
                "LANG": "C",
            }
        )
        self.assertEqual(env, {"PATH": "/bin", "LANG": "C"})


class ProposerChannelTests(unittest.TestCase):
    """An executor that brings its own agent CLI must not demand an OpenCode checkout."""

    def _resolve(self, substrate, *, configured):
        with (
            patch.object(config, "_IMPROVER_CHANNEL_SET", bool(configured)),
            patch.object(config, "DEFAULT_PROPOSER_CHANNEL", configured or "opencode"),
            patch.object(
                config,
                "PROPOSER_MODEL_BY_CHANNEL",
                {
                    "opencode": "p/m",
                    "claude_cli": "p/m",
                    "pi": "p/m",
                },
            ),
        ):
            return resolve_proposer(substrate)

    def test_an_unset_channel_follows_the_executor(self):
        self.assertEqual(self._resolve("pi", configured="").kind, "pi")
        self.assertEqual(self._resolve("dsh", configured="").kind, "pi")
        self.assertEqual(self._resolve("oc", configured="").kind, "opencode")
        self.assertEqual(self._resolve("cx", configured="").kind, "opencode")

    def test_an_explicit_channel_always_wins(self):
        self.assertEqual(self._resolve("pi", configured="opencode").kind, "opencode")
        self.assertEqual(self._resolve("oc", configured="pi").kind, "pi")

    def test_opencode_is_not_required_for_a_pi_run(self):
        with (
            patch.object(config, "_IMPROVER_CHANNEL_SET", False),
            patch.object(config, "DEFAULT_PROPOSER_CHANNEL", "opencode"),
        ):
            self.assertFalse(config.needs_opencode("pi"))
            self.assertFalse(config.needs_opencode("dsh"))
            self.assertTrue(config.needs_opencode("oc"))
            self.assertTrue(config.needs_opencode("cx"))


class CustomSetupTests(unittest.TestCase):
    def test_setup_for_custom_does_not_crash(self):
        with patch("sys.stdout"):
            self.assertEqual(setup.main(["custom", "--substrate", "pi", "--dry-run"]), 0)

    def test_setup_for_a_dsh_run_also_installs_the_improver(self):
        """dsh improves with Pi by default, so installing only the executor strands the run."""
        with patch("sys.stdout") as out:
            self.assertEqual(setup.main(["custom", "--substrate", "dsh", "--dry-run"]), 0)
        printed = " ".join(str(c) for c in out.mock_calls)
        self.assertIn("@deepseek-ai/dsh", printed)
        self.assertTrue(
            "pi-coding-agent" in printed or "Using the installed Pi CLI" in printed,
            printed,
        )

    def test_setup_for_custom_refuses_a_substrate_it_cannot_run(self):
        # setup reports a refusal as an exit status with its reason, like every other failure
        with patch("sys.stdout"), patch("sys.stderr") as err:
            with self.assertRaises(SystemExit) as ctx:
                setup.main(["custom", "--substrate", "oc", "--dry-run"])
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("pi or dsh", " ".join(str(c) for c in err.mock_calls))

    def test_a_custom_run_needs_no_judge(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "area" / "t").mkdir(parents=True)
            (root / "area" / "t" / "TASK.md").write_text("do it", encoding="utf-8")
            a = CustomAdapter(
                "pi",
                tasks_root=root,
                seed_dir=seed_harness(root / "seed"),
                runs_root=root / "runs",
            )
            with (
                patch.object(a.substrate, "preflight", lambda: None),
                patch.object(config, "JUDGE_MODEL", ""),
                patch.object(config, "SOLVER_MODEL_PI", "p/m"),
                patch.object(config, "IMPROVER_MODEL", "p/m"),
            ):
                errors = check(a, judge=True)
            self.assertEqual([e for e in errors if "JUDGE" in e], [])

    def test_a_thin_area_is_flagged(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "only" / "t").mkdir(parents=True)
            (root / "only" / "t" / "TASK.md").write_text("do it", encoding="utf-8")
            a = CustomAdapter("pi", tasks_root=root)
            with patch("sys.stdout") as out:
                a._warn_thin_areas()
            self.assertIn("fewer than two tasks", " ".join(str(c) for c in out.mock_calls))


class UpstreamNoticeTests(unittest.TestCase):
    def test_every_changed_upstream_module_carries_a_modification_notice(self):
        root = Path(__file__).resolve().parents[1]
        google_headed = [
            p
            for p in (root / "harness2").rglob("*.py")
            if "Copyright 2026 Google LLC" in p.read_text(encoding="utf-8", errors="replace")
        ]
        self.assertTrue(google_headed, "expected upstream files to still carry their header")
        changed = [
            p
            for p in google_headed
            if "Modified in the noetion/harness2 fork" in p.read_text(encoding="utf-8")
        ]
        self.assertGreaterEqual(len(changed), 10)


class SubstratePairGuardTests(unittest.TestCase):
    """A benchmark that cannot run a substrate must say so before installing anything."""

    def test_lab_refuses_an_agent_cli_substrate_before_cloning_it(self):
        for name in ("pi", "dsh"):
            with patch("sys.stdout"), patch("sys.stderr") as err:
                with self.assertRaises(SystemExit) as ctx:
                    setup.main(["lab", "--substrate", name, "--dry-run"])
            self.assertEqual(ctx.exception.code, 1)
            self.assertIn("oc or cx", " ".join(str(c) for c in err.mock_calls))

    def test_the_adapter_refuses_it_too(self):
        with self.assertRaises(ValueError):
            make_adapter("lab", None, "pi")


class CustomJudgeOptionalTests(unittest.TestCase):
    """A custom run with check.py decides its own verdict; judging must not demand a model."""

    def test_judge_cli_does_not_require_a_judge_model_for_custom(self):
        from harness2.cli import judge

        with tempfile.TemporaryDirectory() as td:
            with (
                patch.object(config, "RUNS", Path(td)),
                patch.object(config, "JUDGE_MODEL", ""),
                patch.object(config, "SOLVER_MODEL_PI", "p/m"),
                patch.object(config, "IMPROVER_MODEL", "p/m"),
                patch("sys.stdout"),
            ):
                with self.assertRaises(SystemExit) as ctx:
                    judge.main(["--bench", "custom", "--substrate", "pi", "--k", "1"])
        # SystemExit carries the reason as its code; it got that far, so no judge was demanded
        self.assertIn("nothing to judge", str(ctx.exception.code))
        self.assertNotIn("HARNESS2_JUDGE_MODEL", str(ctx.exception.code))

    def test_other_benchmarks_still_require_one(self):
        from harness2.cli import judge

        with tempfile.TemporaryDirectory() as td:
            with (
                patch.object(config, "RUNS", Path(td)),
                patch.object(config, "JUDGE_MODEL", ""),
                patch.object(config, "SOLVER_MODEL_CX", "m"),
                patch.object(config, "IMPROVER_MODEL", "p/m"),
                patch("sys.stdout"),
            ):
                with self.assertRaises(config.ConfigError) as ctx:
                    judge.main(["--bench", "jb", "--substrate", "cx", "--k", "1"])
        self.assertIn("HARNESS2_JUDGE_MODEL", str(ctx.exception))
