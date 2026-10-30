"""Offline measured-joint segmentation commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "fit", "infer"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--run", type=Path, required=True, help="New output directory")
        if name == "prepare":
            command.add_argument("--manifest", type=Path, required=True)
        else:
            command.add_argument("--input-run", type=Path, required=True)
        if name == "fit":
            command.add_argument("--references", type=Path, required=True)
        if name == "infer":
            command.add_argument("--templates", type=Path)
            command.add_argument("--training-replay", action="store_true",
                                 help="Allow demonstration overlap; never treated as evaluation")
    for name in ("report", "export-video"):
        command = commands.add_parser(name)
        command.add_argument("--run", type=Path, required=True)
        command.add_argument("--output", type=Path, help="New export directory")
        if name == "export-video":
            command.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        if args.command in {"prepare", "fit", "infer"}:
            from .joint.config import load_config
            config = load_config(args.config)
            if args.command == "prepare":
                from .joint.data import prepare_run
                rows = prepare_run(config, args.manifest, args.run)
                result = {"episodes": len(rows), "manifest": str(args.run / "manifest.json")}
            elif args.command == "fit":
                from .joint.pipeline import fit_run
                result = fit_run(config, args.input_run, args.references, args.run)
            else:
                from .joint.pipeline import infer_run
                result = infer_run(config, args.input_run, args.run, args.templates,
                                   training_replay=args.training_replay)
        else:
            from .joint.render import render_run, report_run
            if args.command == "export-video":
                if args.workers < 1:
                    parser.error("--workers must be positive")
                result = render_run(args.run, args.workers, args.output)
            else:
                result = report_run(args.run, args.output)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
