<h3 align="center">
  <img src="assets/Harness2-wordmark.png" alt="Harness²" width="360"><br>
  Recursive Agent Harnessing for an Open World
</h3>

<p align="center">
  <a href="https://arxiv.org/pdf/2610.14224"><img src="https://img.shields.io/badge/Paper-arXiv-EA4335?logo=arxiv&logoColor=white" alt="Paper"></a>
  <a href="https://harness2.github.io"><img src="https://img.shields.io/badge/Website-harness2.github.io-4285F4?logo=googlechrome&logoColor=white" alt="Website"></a>
  <a href="docs/BENCHMARKS.md"><img src="https://img.shields.io/badge/Docs-Benchmark%20guides-FBBC04" alt="Benchmark guides"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-34A853" alt="License"></a>
</p>

> **Fork notice.** This is an unofficial fork of
> [google-research/harness2](https://github.com/google-research/harness2), not affiliated
> with Google or the paper's authors. It adds `pi` (Pi coding agent) and `dsh` (DeepSeek
> Harness) executor substrates for JobBench and the custom benchmark, a `custom` benchmark
> for your own tasks without reference answers, and `harness2-export` /
> `HARNESS2_CUSTOM_SEED` for carrying a harness between runs. The paper's results below were
> produced upstream with the OpenCode and Codex executors. The additions here have been run
> end to end on a handful of tasks with one model provider; they are not benchmarked, and
> LAB and WorkBuddy-Bench do not support the new substrates.

<p align="center">
  <a href="docs/BENCHMARKS.md#jobbench">
    <img src="assets/harness2-setup.gif" alt="Install, set up, and run JobBench: uv pip install -e '.[render]', harness2-setup jb, then harness2 --bench jb --k 1 --mode parallel. Click for prerequisites and commands." width="640">
  </a>
</p>

A **harness** is the system around a model that enables it to act as an agent:
prompts, tools, memory, skills, orchestration, and more. **Harnessing** is the
process of optimizing that system.

Harness² is a plug-and-play framework for **recursive agent harnessing** in an
open world, where agents encounter tasks unseen during harness design. It
connects two levels of recursion while keeping model weights fixed:

* **Task-level recursion** refines the execution harness for the current task
  through a contrastive **Propose–Probe–Compose** step: propose edits, probe
  their effects on the same task, and compose a refined harness from behavioral
  contrasts between executions, without ground-truth feedback.
* **Harnessing-level recursion** updates a persistent *improvement harness*
  with adopted harnesses and evidence-supported lessons from each task to guide
  future refinement.

<p align="center">
  <a href="assets/harness2_pipeline.png">
    <img src="assets/harness2_pipeline.png" alt="Harness² pipeline. Task-level recursion: an agent executor runs the initial harness, a harness editor proposes invariant and domain edits, candidate probes execute them, and a harness composer combines them into the final harness. Harnessing-level recursion: an experience reflector reflects on the step and updates the improvement harness used for the next task." width="100%">
  </a>
</p>

Across three professional benchmarks and four model–harness configurations, a
single Harness² step improves GLM-5 and Gemini 3.5 Flash over their base
harnesses, with gains of up to 11.5 points on WorkBuddy-Bench, 4.5 points on
JobBench, and 11.8 percentage points on LAB's partial pass rate. Increasing the
task-level recursion budget brings further gains.

The release supports **parallel and sequential recursion**, **OpenCode and
Codex executors**, and **JobBench, LAB, and WorkBuddy-Bench**.

## Video overview

https://github.com/user-attachments/assets/76b10548-b14b-4c8f-9aa5-a93662df4d56

## Contents

- [Video overview](#video-overview)
- [Harness components](#harness-components)
- [Install](#install)
- [Run a benchmark](#run-a-benchmark)
- [Executor substrates: Pi and DeepSeek Harness](#executor-substrates-pi-and-deepseek-harness)
- [Your own tasks](#your-own-tasks)
- [Experiments](#experiments)
- [Web access](#web-access)
- [Results and resume](#results-and-resume)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)

## Harness components

Harness² exposes eight editable components around the model: **system prompt,
guardrails, memory, tool guidance, subagents, skills, scripts, and plugins**.
They control context construction, the tool interface, delegation, and behavior
inside the tool loop.

<p align="center">
  <a href="assets/components.png">
    <img src="assets/components.png" alt="Eight harness components grouped by runtime role: system prompt, guardrails, memory, and skills construct context; tool guidance and scripts define the tool interface; subagents orchestrate delegated work; plugins control middleware." width="100%">
  </a>
</p>

## Install

Use **Linux, Python 3.12, and [uv](https://docs.astral.sh/uv/getting-started/installation/)**.
From the repository root:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e '.[render]'
harness2-demo --output runs/demo --k 2
```

The local demo runs both recursion modes with deterministic editors and real
Python executions. It requires no model credentials.

The custom benchmark with the `pi` or `dsh` executor needs no Google Cloud
project, no Docker and no OpenCode checkout: one CLI, one provider key, and a
directory of your own tasks. That path is exercised on macOS as well as Linux.
LAB and WorkBuddy-Bench keep the requirements below.

For model-backed benchmarks, install **Node.js 20+, the Google Cloud CLI**, and
the [benchmark dependencies](docs/SETUP.md#requirements). LAB needs Pandoc and
Poppler; WorkBuddy-Bench needs Docker. Then configure your models:

```bash
cp .env.example .env
# Edit .env with your Vertex AI project and model IDs.
gcloud auth application-default login
```

See the [setup guide](docs/SETUP.md) for model settings, authentication
alternatives, and the [Codex executor](docs/SETUP.md#codex-executor).

## Run a benchmark

The following example installs LAB and runs one task, then evaluates and reports
its result. Model-backed runs incur API costs; start with a small task selection.

```bash
harness2-setup lab
harness2-check --bench lab
harness2 --bench lab --k 1 --mode parallel \
  --only-domains antitrust-competition --max-tasks 1
harness2-judge --bench lab --k 1 --mode parallel
harness2-report --bench lab --k 1 --mode parallel
```

Sequential recursion runs the same commands with `--mode sequential`. `--k` is
the task-level recursion budget; see [Experiments](#experiments). Judging and
reporting must use the same benchmark, executor, subset, `--k`, `--mode`, and
seed as the refinement run. Add `--dry-run` to `harness2` to preview task
selection without model calls.

| Benchmark | Tasks | Install | Guide |
|---|---|---|---|
| JobBench | Professional work | `harness2-setup jb` | [JobBench](docs/BENCHMARKS.md#jobbench) |
| LAB | Legal work | `harness2-setup lab` | [LAB](docs/BENCHMARKS.md#lab) |
| WorkBuddy-Bench | Office, web, and code | `harness2-setup wb` | [WorkBuddy](docs/BENCHMARKS.md#workbuddy-bench) |

Choose one benchmark per command. WorkBuddy also takes `--subset office`,
`--subset web`, or `--subset code` on every command after setup. The
[benchmark guides](docs/BENCHMARKS.md) cover each one's data, domains, and
grading.

## Executor substrates: Pi and DeepSeek Harness

`--substrate` chooses what runs the task. Four are wired in; `pi` has been
exercised end to end on JobBench, and `dsh` on the custom benchmark:

| Substrate | CLI | Notes |
|---|---|---|
| `oc` | OpenCode | Upstream default; runs from a pinned checkout through bun |
| `cx` | Codex CLI | Pinned binary under `dependencies/codex` |
| `pi` | [Pi](https://github.com/earendil-works/pi) | `npm install -g @earendil-works/pi-coding-agent`, or let setup install 1.3.0 into `dependencies/pi` |
| `dsh` | [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) | Setup installs 0.2.1-rc.2 into `dependencies/dsh` |

```bash
harness2-setup jb --substrate pi
harness2-check --bench jb --substrate pi
harness2 --bench jb --substrate pi --k 1 --only-domains technical_writers --max-tasks 1
harness2-judge --bench jb --substrate pi --k 1
```

`pi` and `dsh` run as coding-agent CLIs in a scratch workspace, not from a
pinned checkout: each rollout copies the harness to the workspace root, gets an
empty agent home (`PI_CODING_AGENT_DIR` or `DSH_HOME`) and explicit skill and
extension flags, so your own configuration never leaks into a run. The
workspace lives under `HARNESS2_SCRATCH` (default: the system temp directory)
and is refused if it would sit inside a git checkout, because both agents find
their project root by walking up to `.git`.

Model access is configuration, not code. `HARNESS2_PI_MODEL` and
`HARNESS2_DSH_MODEL` default to `HARNESS2_SOLVER_MODEL`, and both CLIs read their
credentials from the provider's environment variable (`OPENCODE_API_KEY`,
`DEEPSEEK_API_KEY`, ...). Nothing here is tied to a particular vendor, but only an
OpenCode Go provider has been exercised end to end, so treat another gateway as
the first thing to smoke-test.

Both substrates also validate a proposed harness while it is still a proposal,
by loading it offline with credentials stripped: Pi exits before its first model
call if an extension will not load, and silently drops a skill whose name it
cannot parse; dsh refuses to import a plugin that is not a module exporting
`name` and `apply`. Either failure is reported back to the improver instead of
being paid for as a rollout.

## Your own tasks

`--bench custom` refines a harness on work you care about, with no reference
answers and no dataset to download. Point it at a directory:

```
my-tasks/
  writing/
    exec-summary/
      TASK.md          the assignment, verbatim
      files/           optional reference material
      check.py         optional check, run against the deliverables
```

```bash
export HARNESS2_CUSTOM_TASKS=~/my-tasks
harness2-check --bench custom --substrate pi
harness2 --bench custom --substrate pi --k 1 --only-domains writing --max-tasks 1
harness2-judge --bench custom --substrate pi --k 1
```

`check.py` is optional and, when present, decides the verdict by exiting 0. With
no `check.py` and a judge configured, each deliverable is rated against its own
assignment: the task text plus what the agent produced, never a golden answer.
With neither, rollouts still run and are reported, and the measurement is absent
rather than zero. OpenCode and Codex are not supported here, because they need a
benchmark-owned checkout, render step and credential file.

Use several tasks per area. A single task gives the improver one example to
reason from, and it will write the specifics of that example into the harness.

### Carrying a harness between runs

`harness2-export` writes a harness out as a portable seed, and
`HARNESS2_CUSTOM_SEED` imports it as the tree the recursion starts from:

```bash
# a harness adopted while refining on JobBench
harness2-export --bench jb --k 1 --task technical_writers/task1 \
  --substrate pi --to ~/my-harness

# or any tree you wrote by hand
harness2-export --from ./some/harness --substrate pi --to ~/my-harness

# start from it on your own tasks
HARNESS2_CUSTOM_TASKS=~/my-tasks HARNESS2_CUSTOM_SEED=~/my-harness \
  harness2 --bench custom --substrate pi --k 1
```

The export is the editable tree itself plus `harness2.json`, which records the
source, the fingerprint, and the files it contains. Unset `HARNESS2_CUSTOM_SEED`
to fall back to the built-in one-file seed.

## Experiments

Scale task-level recursion in two ways: **parallel recursion** explores multiple
candidate harnesses and composes refinements from their execution evidence;
**sequential recursion** carries the refined harness and accumulated evidence
through successive Propose–Probe–Compose steps.

<p align="center">
  <a href="assets/harness2_recursion_modes.png">
    <img src="assets/harness2_recursion_modes.png" alt="Two ways to scale task-level recursion. Parallel: probe k general and k domain-specific candidate harnesses, then produce k compositions from the shared evidence. Sequential: carry the refined harness and accumulated evidence through k Propose–Probe–Compose steps." width="100%">
  </a>
</p>
<p align="center"><em>Parallel recursion explores wider; sequential recursion refines deeper.</em></p>

| Option | Behavior |
|---|---|
| `--k 1` | One Propose–Probe–Compose step |
| `--k K --mode parallel` | Probe `K` general and `K` domain-specific candidates, then produce `K` compositions |
| `--k K --mode sequential` | Refine the harness through `K` successive steps |
| `--substrate cx` | Use Codex instead of the default OpenCode executor |

After each task, harnessing-level recursion retains the adopted harness and
evidence-supported lessons to guide future refinement. See the
[experiment guide](docs/EXPERIMENTS.md) for task selection, concurrency,
reproducibility, and baseline comparisons.

## Web access

Executor network access is unrestricted, and Codex runs with its sandbox
disabled. Public benchmark rubrics may be accessible online. Before evaluation,
configure [network restrictions](docs/EXPERIMENTS.md#web-access) to prevent
access to benchmark answers and grading criteria.

## Results and resume

Results are saved under `runs/`, organized by benchmark and subset where
applicable. They include task summaries, deliverables, adopted harnesses,
reflection notes, and evaluation results.

Repeat the same command to resume. Use a new `HARNESS2_RUNS_DIR` when changing
models or task selection. See [results and resume](docs/EXPERIMENTS.md#results-and-resume)
for details; each command also supports `--help`.

## Citation

```bibtex
@article{xu2026harness2,
  title   = {Harness$^2$: Recursive Agent Harnessing for an Open World},
  author  = {Xu, Ruiyao and Chen, Yanfei and CuiZhu, Zhongying and Dalvi Mishra, Bhavana and Ming, Yifei and Yu, Han and Han, Rujun and Lee, Chen-Yu and Pfister, Tomas},
  journal = {arXiv preprint arXiv:2610.14224},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.14224}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Harness² is licensed under [Apache 2.0](LICENSE). Third-party code under
[`third_party/`](third_party/README.md), benchmarks, and datasets retain their
upstream licenses; see
[third-party notices](dependencies/README.md).

## Disclaimer

This is not an officially supported Google product. This project is not eligible
for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).
