# SPDX-License-Identifier: Apache-2.0

"""Export an adopted harness as a portable seed, and import it back onto your own tasks.

A harness is only worth refining if you can carry it somewhere else. `--from` takes any tree
directory (an adopted harness, a library entry, or one you wrote by hand); `--bench`/`--k`
resolve the harness a run adopted for a task, which is the usual source. The output is the
editable tree itself, stripped to the surface's own files, plus a manifest recording where it
came from and what it contains.

    harness2-export --from runs/jb/adapt/.../harnesses/technical_writers__task1/harness \\
        --substrate pi --to ~/my-harness

    harness2-export --bench jb --k 2 --task technical_writers/task1 --to ~/my-harness

Import is the other direction and needs no command: point HARNESS2_CUSTOM_SEED at the
exported directory and the custom benchmark starts its recursion from that tree.

    HARNESS2_CUSTOM_TASKS=~/my-tasks HARNESS2_CUSTOM_SEED=~/my-harness \\
        harness2 --bench custom --substrate pi --k 2
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from harness2.cli._common import HelpFormatter, library_root_for, make_adapter
from harness2.core import runmark as RM
from harness2.core import tree as HT
from harness2.substrate import registry as SUB


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="harness2-export", description=__doc__, formatter_class=HelpFormatter
    )
    src = p.add_argument_group("source: a tree directory, or a run's adopted harness")
    src.add_argument("--from", dest="src", type=Path, help="tree directory to export")
    src.add_argument("--bench", choices=("jb", "lab", "custom"), help="resolve from this bench")
    src.add_argument("--k", type=int, default=1, help="with --bench: which k's harness")
    src.add_argument("--task", help="with --bench: the task whose adopted harness to export")
    src.add_argument("--seed", type=int, default=None, help="with --bench: the task stream seed")
    src.add_argument(
        "--mode", default="parallel", help="with --bench: which mode's library to read"
    )
    p.add_argument("--substrate", choices=SUB.SUBSTRATES, default="pi", help=SUB.HELP)
    p.add_argument("--to", type=Path, required=True, help="destination directory")
    p.add_argument("--force", action="store_true", help="overwrite a non-empty destination")
    return p


def _adopted(args) -> Path:
    """The harness `--bench` adopted for `--task`, read out of the experience library."""
    if not args.task:
        raise ValueError("--bench needs --task: a harness is adopted per task, not per run")
    adapter = make_adapter(args.bench, None, args.substrate)
    from harness2.core import library as LIB
    from harness2.core.schedule import mode_token

    root = library_root_for(adapter, args.k, seed=args.seed)
    mode_dir = root / mode_token(args.mode)
    lib = LIB.Library(mode_dir, adapter.domain_of(args.task), adapter.substrate.surface())
    entry = lib.entry_dir(args.task) / "harness"
    if not entry.is_dir():
        raise FileNotFoundError(
            f"no adopted harness for {args.task} under {root}; list the entries in that "
            f"library's INDEX.md, or pass --from a tree directory instead"
        )
    return entry


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        src = args.src or _adopted(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    surface = SUB.make(args.substrate).surface()
    if not HT.is_tree(surface, src):
        print(f"FAIL: {src} is not a {surface.name} harness tree (it needs AGENTS.md)")
        return 1

    dest = args.to.expanduser().resolve()
    if dest.exists() and any(dest.iterdir()) and not args.force:
        print(f"FAIL: {dest} is not empty; pass --force to overwrite it")
        return 1
    tree = HT.read_tree(surface, src)
    dest.mkdir(parents=True, exist_ok=True)
    for stale in dest.glob("**/*"):
        if stale.is_file():
            stale.unlink()
    HT.write_tree(surface, tree, dest, stubs=False)

    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source": str(src),
        "substrate": args.substrate,
        "fingerprint": RM.fingerprint(Path(src)),
        "files": sorted(tree),
        "always_on_chars": len(HT.always_on_text(surface, tree)),
        "import_with": "HARNESS2_CUSTOM_SEED=" + str(dest),
    }
    (dest / "harness2.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"exported {len(tree)} file(s) to {dest}")
    for rel in sorted(tree):
        print(f"  {rel}  ({len(tree[rel])} chars)")
    print(f"import with: HARNESS2_CUSTOM_SEED={dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
