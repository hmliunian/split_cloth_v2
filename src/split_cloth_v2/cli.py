"""Prepare offline inputs, annotate serially, report and export existing results."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

from .config import load_config
from .data import load_manifest, prepare_episode
from .io_utils import fingerprint, now, read_json, sha256_file, write_json
from .provider import DashScopeProvider, MockProvider
from .search import run_search
from .video_export import export_run

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def source_digest() -> str:
    package = Path(__file__).parent
    paths = sorted(path for path in package.rglob("*") if path.suffix in (".py", ".txt"))
    return fingerprint({"source_files": {str(path.relative_to(package)): sha256_file(path) for path in paths}, "environment": _environment()})


def _environment() -> dict:
    return {"python": sys.version.split()[0], "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pyarrow", "av", "Pillow", "httpx", "PyYAML")}}


def _run_path(path: str | Path) -> Path:
    run = Path(path).resolve()
    runs = (PROJECT_ROOT / "runs").resolve()
    if not run.is_relative_to(runs) or run == runs:
        raise ValueError("run_directory_must_be_inside_this_repository_runs")
    return run


def _code_version():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def prepare(args) -> dict:
    run = _run_path(args.run)
    entries = load_manifest(args.manifest)
    config = load_config(args.config)
    config["execution_mode"] = "mock" if args.mock else "dashscope"
    if args.mock_decisions:
        if not args.mock:
            raise ValueError("mock_decisions_require_mock_mode")
        config["mock_decisions"] = read_json(args.mock_decisions)
    identity = {"config": config, "manifest": entries, "source_digest": source_digest()}
    run_fingerprint = fingerprint(identity)
    run_record = run / "run.json"
    if run.exists() and any(run.iterdir()):
        if not run_record.is_file() or read_json(run_record).get("run_fingerprint") != run_fingerprint:
            raise ValueError("existing_run_differs_use_a_new_run_id")
    else:
        run.mkdir(parents=True)
        write_json(run_record, {**identity, "run_fingerprint": run_fingerprint, "created_at": now(), "git_commit": _code_version(), "environment": _environment()})
        (run / "manifest.jsonl").write_text("".join(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n" for entry in entries), encoding="utf-8")
        (run / "prompt.txt").write_text(config["prompt_text"] + "\n", encoding="utf-8")
    summary = {"run_fingerprint": run_fingerprint, "execution_mode": config["execution_mode"], "episodes": [], "errors": []}
    for entry in entries:
        try:
            info = prepare_episode(entry, run / "episodes" / entry["key"], config)
            summary["episodes"].append({"key": entry["key"], "frame_count": info["frame_count"], "candidate_count": info["candidate_count"], "max_logical_requests": info["max_logical_requests"]})
            print(f"prepared {entry['key']}: {info['candidate_count']} candidates", flush=True)
        except Exception as exc:
            summary["errors"].append({"key": entry["key"], "error": str(exc), "error_type": type(exc).__name__})
            write_json(run / "reports/prepare_summary.json", summary)
            raise
    summary["episode_count"] = len(entries)
    summary["total_frames"] = sum(item["frame_count"] for item in summary["episodes"])
    summary["max_logical_requests"] = sum(item["max_logical_requests"] for item in summary["episodes"])
    summary["max_images"] = summary["max_logical_requests"] * len(config["context_offsets_s"])
    summary["status"] = "prepared"
    write_json(run / "reports/prepare_summary.json", summary)
    return summary


def _load_run(path, *, check_source=False):
    run = _run_path(path)
    record = read_json(run / "run.json")
    identity = {name: record[name] for name in ("config", "manifest", "source_digest")}
    if fingerprint(identity) != record["run_fingerprint"]:
        raise ValueError("frozen_run_record_changed")
    if check_source and record["source_digest"] != source_digest():
        raise ValueError("source_code_changed_use_a_new_run_id")
    if load_manifest(run / "manifest.jsonl") != record["manifest"]:
        raise ValueError("frozen_manifest_changed")
    if (run / "prompt.txt").read_text(encoding="utf-8").strip() != record["config"]["prompt_text"]:
        raise ValueError("frozen_prompt_changed")
    return run, record


def annotate(args) -> dict:
    run, record = _load_run(args.run, check_source=True)
    config = record["config"]
    if config["execution_mode"] == "mock":
        provider = MockProvider(config.get("mock_decisions"))
    else:
        credential_file = getattr(args, "credential_file", None)
        if credential_file and Path(credential_file).resolve().is_relative_to(PROJECT_ROOT):
            raise ValueError("credential_file_must_be_outside_repository")
        provider = DashScopeProvider(credential_file=credential_file)
    errors = []
    for entry in record["manifest"]:
        # Revalidate frozen inputs and preparation hashes before any paid call.
        prepare_episode(entry, run / "episodes" / entry["key"], config)
        result = run_search(run / "episodes" / entry["key"], provider, config, record["run_fingerprint"])
        print(f"annotated {entry['key']}: {result['status']}", flush=True)
        if result["status"] == "execution_error":
            errors.append(entry["key"])
    summary = report(argparse.Namespace(run=str(run)))
    if errors:
        raise RuntimeError("annotation_execution_errors: " + ", ".join(errors))
    return summary


def report(args) -> dict:
    run, record = _load_run(args.run)
    output_root = _export_root(run, getattr(args, "export_name", None))
    exports_path = output_root / "videos/exports.json"
    exports = read_json(exports_path) if exports_path.exists() else {}
    video_records = {}
    # Export indexes are lists of per-episode records under 'exports'.
    if isinstance(exports, dict):
        items = exports.get("videos", [])
        if isinstance(items, list):
            video_records = {item["key"]: item for item in items if "key" in item}
    episodes = []
    for entry in record["manifest"]:
        result_path = run / "episodes" / entry["key"] / "result.json"
        result = read_json(result_path) if result_path.exists() else {}
        result_matches_run = bool(result) and result.get("run_fingerprint") == record["run_fingerprint"] and result.get("key") == entry["key"] and result.get("frame_count") == entry["frame_count"]
        suffix = record["config"].get("video_export", {}).get("filename_suffix", "candidates")
        video_path = output_root / "videos" / f"{entry['key']}__{suffix}.mp4"
        sidecar = video_path.with_suffix(".json")
        video = read_json(sidecar) if sidecar.exists() else video_records.get(entry["key"], {})
        verification = video.get("verification") or {}
        verified_video = (
            video_path.is_file() and video.get("export_status") == "success"
            and verification.get("passed") is True
            and verification.get("frame_count") == entry["frame_count"]
            and video.get("output_sha256") == sha256_file(video_path)
            and result_path.is_file() and video.get("result_sha256") == sha256_file(result_path)
            and video.get("source_video_sha256") == entry["source_video_sha256"]
            and result_matches_run
        )
        slots = result.get("slots", {})
        episodes.append({
            "key": entry["key"], "frame_count": entry["frame_count"],
            "boundaries": {stage: (slots.get(stage) or {}).get("end_frame") for stage in ("stage1", "stage2", "stage3")},
            "status": (result.get("status", "not_run") if not result or result_matches_run else "execution_error"),
            "review_status": result.get("review_status", "unreviewed"),
            "has_uncertainty": result.get("has_uncertainty", False),
            "reasons": result.get("reasons", []) + (["result_run_fingerprint_mismatch"] if result and not result_matches_run else []),
            "execution_error": result.get("reasons") if result.get("status") == "execution_error" else None,
            "logical_calls": result.get("logical_calls", 0), "http_attempts": result.get("http_attempts", 0),
            "input_tokens": result.get("input_tokens", 0), "output_tokens": result.get("output_tokens", 0),
            "elapsed_s": result.get("elapsed_s", 0), "provider_elapsed_s": result.get("provider_elapsed_s", 0),
            "returned_models": result.get("returned_models", []),
            "video_path": str(video_path) if video_path.is_file() else None,
            "video_verified": verified_video, "video_verification": verification,
            "visual_assessment": "pending_human_review",
        })
    expected = load_manifest(PROJECT_ROOT / "docs/phase1_manifest.jsonl")
    expected_keys = {item["key"] for item in expected}
    manifest_matches_p1 = {entry["key"]: entry for entry in record["manifest"]} == {entry["key"]: entry for entry in expected}
    real_mode = record["config"]["execution_mode"] == "dashscope"
    summary = {"run_fingerprint": record["run_fingerprint"], "execution_mode": record["config"]["execution_mode"],
               "episode_count": len(episodes), "total_frames": sum(item["frame_count"] for item in episodes),
               "episodes": episodes, "model_accuracy": None,
               "p1_delivery_complete": real_mode and len(episodes) == 10
               and {item["key"] for item in episodes} == expected_keys and manifest_matches_p1
               and all(item["video_verified"] and item["status"] in ("complete", "needs_review") for item in episodes)}
    for field in ("logical_calls", "http_attempts", "input_tokens", "output_tokens", "elapsed_s", "provider_elapsed_s"):
        summary[field] = sum(item[field] for item in episodes)
    write_json(run / "reports/summary.json", summary)
    lines = ["# P1 experiment report", "", f"Execution mode: {summary['execution_mode']}",
             f"Episodes: {len(episodes)}; frames: {summary['total_frames']}",
             f"Delivery complete: {summary['p1_delivery_complete']}",
             "", "Model results remain unreviewed unless a separate human review exists. No accuracy is inferred from program completion.",
             "", "| Episode | Boundaries S1/S2/S3 | Program status | Review | Calls / HTTP | Tokens in/out | Video |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for item in episodes:
        boundaries = "/".join(str(item["boundaries"][stage]) for stage in ("stage1", "stage2", "stage3"))
        lines.append(f"| {item['key']} | {boundaries} | {item['status']} | {item['review_status']} | {item['logical_calls']}/{item['http_attempts']} | {item['input_tokens']}/{item['output_tokens']} | {item['video_path'] or 'pending'} |")
    lines += ["", "## Per-episode review notes", ""]
    for item in episodes:
        lines.append(f"- {item['key']}: visual review pending; reasons={item['reasons']}; execution_error={item['execution_error']}")
    (run / "reports/phase1_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _export_root(run: Path, name: str | None) -> Path:
    if name is None:
        return run
    if not name or Path(name).name != name or name in (".", "..") or not all(char.isalnum() or char in "_-" for char in name):
        raise ValueError("invalid_export_name")
    destination = run / "exports" / name
    if not destination.resolve().is_relative_to(run.resolve()):
        raise ValueError("export_directory_leaves_run")
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="Offline validation, candidate detection and image extraction")
    prepare_parser.add_argument("--manifest", required=True)
    prepare_parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    prepare_parser.add_argument("--run", required=True)
    prepare_parser.add_argument("--mock", action="store_true", help="Freeze this run as an explicit offline simulation")
    prepare_parser.add_argument("--mock-decisions")
    for name in ("annotate", "report", "export-video"):
        command = commands.add_parser(name, help="Calls paid VLM service unless run was prepared with --mock" if name == "annotate" else None)
        command.add_argument("--run", required=True)
        if name in ("report", "export-video"):
            command.add_argument("--export-name", help="Write/read a separate named export version while preserving prior exports")
        if name == "annotate":
            command.add_argument("--credential-file", help="Explicit external credential file; never copied or logged")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args)
        elif args.command == "annotate":
            result = annotate(args)
        elif args.command == "report":
            result = report(args)
        else:
            run, record = _load_run(args.run)
            export_root = _export_root(run, args.export_name)
            result = export_run(run, record["config"], export_dir=export_root if args.export_name else None)
            report(args)
            if result["failed_count"]:
                raise RuntimeError(f"video_export_failures: {result['failed_count']}")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
