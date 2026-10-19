# SPDX-License-Identifier: Apache-2.0

"""Regressions for the agent-CLI substrates, the custom benchmark and seed export.

These never start a model. They cover the parts that silently break a rollout before any
model is reached: argv/env construction, the overlay, the renderers, the load checks that
reject an unloadable harness, and the export/import round trip.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness2 import config
from harness2.benchmarks import agent_rollout as AR
from harness2.benchmarks.custom import CustomAdapter, make_spec, seed_harness
from harness2.cli import export_harness
from harness2.core import tree as HT
from harness2.core.spec import BenchPrompts, BenchSpec, ChannelSpec
from harness2.substrate import dsh, pi, registry
from harness2.substrate.agent_cli import clip, iter_jsonl, stage_root


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

    def test_a_sound_extension_passes_the_offline_load_check(self):
        ext = {
            ".pi/extensions/ok.ts": (
                'export default function (pi: any) { pi.on("tool_call", async () => undefined); }\n'
            )
        }
        self.assertEqual(pi.check_extensions(ext), [])

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
        self.assertEqual(dsh.split_model("x"), ("deepseek", "x"))

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
        rows = dsh.overlay_rows("deepseek/x", effort="", plugins=[])
        self.assertEqual([r.get("id") for r in rows], ["agent-default-model"])
        self.assertNotIn("reasoningEffort", rows[0]["config"])

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

    def test_plugin_shape_is_checked_offline(self):
        problems = dsh.DshSubstrate().check_tree({".dsh/plugins/x.ts": "export const name = 1;\n"})
        self.assertEqual(len(problems), 1)
        self.assertIn("dsh plugin", problems[0])

    def test_sound_plugin_passes(self):
        src = "export const name = 'x';\nexport function apply(ctx: any) { void ctx; }\n"
        self.assertEqual(dsh.DshSubstrate().check_tree({".dsh/plugins/x.ts": src}), [])

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
