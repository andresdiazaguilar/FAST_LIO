#!/usr/bin/env python3
"""Evaluate FAST-LIO2 trajectories on the internal Tinamu degeneracy datasets.

Input estimates and ground-truth files must be in TUM format:

    timestamp x y z qx qy qz qw

Unlike the GEODE helper, no body-frame transform is applied: the Tinamu ground
truth comes from the same SLAM session as the estimate (loop-closure-optimized
poses), so estimate and reference share the body frame and only Umeyama
alignment is needed.

Run it with the project venv Python:
/home/andres/semester_project/data/venv/bin/python \
  /home/andres/semester_project/data/GEODE_helper/script/evaluate_tinamu_fastlio.py \
  --est-dir /path/to/your/fastlio_tum_outputs \
  --recursive

The script looks for files whose name contains the sequence key, e.g.:
    cameroon_fail_short.txt
    senegal.txt
    silo_fail.txt
    valis_fail_2.txt

If your filenames differ, pass them explicitly:
  --estimate cameroon_fail_short=/path/to/cameroon_fastlio.txt \
  --estimate senegal=/path/to/senegal_fastlio.txt \
  --estimate silo_fail=/path/to/silo_fastlio.txt \
  --estimate valis_fail_2=/path/to/valis_fastlio.txt

Results are written by default to:
/home/andres/semester_project/data/tinamu_eval/results/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


SEQUENCE_ORDER = (
    "cameroon_fail_short",
    "senegal",
    "silo_fail",
    "valis_fail_2",
)

# Ground-truth TUM file inside each dataset's andres_results directory.
SEQUENCE_GT_RELPATH: Dict[str, str] = {
    "cameroon_fail_short": "andres_results/chunk_full_reset_100_to_400_optimized_poses_tum.txt",
    "senegal": "andres_results/chunks_bag_optimized_poses_tum.txt",
    "silo_fail": "andres_results/chunks_bag_optimized_poses_tum.txt",
    "valis_fail_2": "andres_results/chunks_bag_optimized_poses_tum.txt",
}

STAT_NAMES = ("max", "mean", "median", "min", "rmse", "sse", "std")


def parse_estimate_overrides(values: Sequence[str]) -> Dict[str, List[Path]]:
    overrides: Dict[str, List[Path]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--estimate must use SEQUENCE=/path/to/traj.txt")
        sequence, path_text = value.split("=", 1)
        if sequence not in SEQUENCE_ORDER:
            raise ValueError(f"Unknown sequence in --estimate: {sequence}")
        overrides.setdefault(sequence, []).append(Path(path_text).expanduser())
    return overrides


def discover_estimates(
    est_dir: Optional[Path],
    pattern: str,
    recursive: bool,
    overrides: Dict[str, List[Path]],
) -> Dict[str, List[Path]]:
    estimates: Dict[str, List[Path]] = {sequence: [] for sequence in SEQUENCE_ORDER}
    for sequence, paths in overrides.items():
        estimates[sequence].extend(paths)

    if est_dir is None:
        return estimates

    globber = est_dir.rglob if recursive else est_dir.glob
    for sequence in SEQUENCE_ORDER:
        user_pattern = pattern.format(sequence=sequence)
        matches = sorted(path for path in globber(user_pattern) if path.is_file())
        if not matches and user_pattern != f"*{sequence}*.txt":
            matches = sorted(path for path in globber(f"*{sequence}*.txt") if path.is_file())
        known = {path.resolve() for path in estimates[sequence] if path.exists()}
        estimates[sequence].extend(path for path in matches if path.resolve() not in known)

    return estimates


def count_data_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip() and not line.lstrip().startswith("#"))


def run_metric(command: Sequence[str]) -> Tuple[Dict[str, float], str, str, int]:
    process = subprocess.run(command, capture_output=True, text=True)
    stats = parse_evo_stats(process.stdout)
    return stats, process.stdout, process.stderr, process.returncode


def parse_evo_stats(output: str) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    for name in STAT_NAMES:
        match = re.search(rf"^\s*{name}\s+([-+0-9.eE]+)\s*$", output, flags=re.MULTILINE)
        if match:
            stats[name] = float(match.group(1))
    return stats


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_metric_command(
    executable: Path,
    gt_path: Path,
    estimate_path: Path,
    pose_relation: str,
    align: bool,
    t_max_diff: float,
    t_offset: float,
    extra_args: Sequence[str],
) -> List[str]:
    command = [
        str(executable),
        "tum",
        str(gt_path),
        str(estimate_path),
        "--pose_relation",
        pose_relation,
        "--t_max_diff",
        str(t_max_diff),
        "--t_offset",
        str(t_offset),
    ]
    if align:
        command.append("--align")
    command.extend(extra_args)
    return command


def safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)


def existing_executable(path: Optional[Path], name: str) -> Path:
    if path is not None:
        executable = path / name
        if executable.exists():
            return executable
    found = shutil.which(name)
    if found:
        return Path(found)
    raise FileNotFoundError(f"Could not find {name}. Pass --evo-bin-dir or add evo to PATH.")


def make_parser() -> argparse.ArgumentParser:
    default_dataset = Path("/home/andres/semester_project/data/datasets/Tinamu")
    default_evo_bin = Path("/home/andres/semester_project/data/venv/bin")
    parser = argparse.ArgumentParser(
        description="Run evo APE/RPE on FAST-LIO2 estimates over the internal Tinamu degeneracy datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--est-dir", type=Path, help="Directory containing FAST-LIO2 TUM .txt estimates.")
    parser.add_argument(
        "--estimate",
        action="append",
        default=[],
        metavar="SEQUENCE=PATH",
        help="Explicit trajectory path. Can be used multiple times.",
    )
    parser.add_argument(
        "--pattern",
        default="*{sequence}*.txt",
        help="Glob pattern used inside --est-dir. Use {sequence} as the sequence placeholder.",
    )
    parser.add_argument("--recursive", action="store_true", help="Search --est-dir recursively.")
    parser.add_argument("--dataset-dir", type=Path, default=default_dataset, help="Tinamu datasets root.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/andres/semester_project/data/tinamu_eval/results"),
        help="Directory for evo logs and the summary CSV.",
    )
    parser.add_argument("--evo-bin-dir", type=Path, default=default_evo_bin, help="Directory containing evo_ape/evo_rpe.")
    parser.add_argument("--t-max-diff", type=float, default=0.1, help="Maximum timestamp association difference.")
    parser.add_argument("--t-offset", type=float, default=0.0, help="Constant timestamp offset passed to evo.")
    parser.add_argument("--ape-relation", default="trans_part", help="evo APE pose relation.")
    parser.add_argument("--rpe-relation", default="trans_part", help="evo RPE pose relation.")
    parser.add_argument("--rpe-delta", type=float, default=1.0, help="RPE delta.")
    parser.add_argument(
        "--rpe-delta-unit",
        choices=("f", "d", "r", "m"),
        default="m",
        help="RPE delta unit: frames, degrees, radians, or meters.",
    )
    parser.add_argument(
        "--rpe-pairs-from-estimate",
        action="store_true",
        help="Let evo choose RPE pose pairs from the estimate instead of the reference trajectory.",
    )
    parser.add_argument("--no-align", action="store_true", help="Disable Umeyama alignment in evo.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed evo command.")
    return parser


def sequence_gt_path(dataset_dir: Path, sequence: str) -> Path:
    return dataset_dir / sequence / SEQUENCE_GT_RELPATH[sequence]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        overrides = parse_estimate_overrides(args.estimate)
        evo_ape = existing_executable(args.evo_bin_dir.expanduser(), "evo_ape")
        evo_rpe = existing_executable(args.evo_bin_dir.expanduser(), "evo_rpe")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    est_dir = args.est_dir.expanduser() if args.est_dir else None
    dataset_dir = args.dataset_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    logs_dir = output_dir / "logs"

    estimates = discover_estimates(est_dir, args.pattern, args.recursive, overrides)
    rows: List[Dict[str, object]] = []
    align = not args.no_align

    for sequence in SEQUENCE_ORDER:
        gt_path = sequence_gt_path(dataset_dir, sequence)
        sequence_estimates = estimates[sequence]
        if not gt_path.exists():
            print(f"[skip] {sequence}: missing GT file {gt_path}")
            continue
        if not sequence_estimates:
            print(f"[skip] {sequence}: no estimate found")
            continue

        for estimate_path in sequence_estimates:
            estimate_path = estimate_path.expanduser()
            row: Dict[str, object] = {
                "sequence": sequence,
                "input": str(estimate_path),
                "gt": str(gt_path),
            }
            if not estimate_path.exists():
                row["status"] = "missing_estimate"
                rows.append(row)
                print(f"[fail] {sequence}: missing estimate {estimate_path}")
                if args.fail_fast:
                    break
                continue

            try:
                pose_count = count_data_lines(estimate_path)
            except Exception as exc:
                row["status"] = "read_failed"
                row["error"] = str(exc)
                rows.append(row)
                print(f"[fail] {sequence}: cannot read {estimate_path}: {exc}")
                if args.fail_fast:
                    write_summary(output_dir / "summary.csv", rows)
                    return 1
                continue

            row["poses"] = pose_count

            ape_command = build_metric_command(
                evo_ape,
                gt_path,
                estimate_path,
                args.ape_relation,
                align,
                args.t_max_diff,
                args.t_offset,
                [],
            )
            rpe_extra_args = ["--delta", str(args.rpe_delta), "--delta_unit", args.rpe_delta_unit]
            if not args.rpe_pairs_from_estimate:
                rpe_extra_args.append("--pairs_from_reference")

            rpe_command = build_metric_command(
                evo_rpe,
                gt_path,
                estimate_path,
                args.rpe_relation,
                align,
                args.t_max_diff,
                args.t_offset,
                rpe_extra_args,
            )

            label = f"{sequence}__{safe_stem(estimate_path)}"
            ape_stats, ape_stdout, ape_stderr, ape_code = run_metric(ape_command)
            rpe_stats, rpe_stdout, rpe_stderr, rpe_code = run_metric(rpe_command)

            write_text(logs_dir / f"{label}.ape.log", ape_stdout + ape_stderr)
            write_text(logs_dir / f"{label}.rpe.log", rpe_stdout + rpe_stderr)

            for key, value in ape_stats.items():
                row[f"ape_{key}"] = value
            for key, value in rpe_stats.items():
                row[f"rpe_{key}"] = value
            row["ape_returncode"] = ape_code
            row["rpe_returncode"] = rpe_code
            row["status"] = "ok" if ape_code == 0 and rpe_code == 0 else "evo_failed"
            rows.append(row)

            ape_rmse = row.get("ape_rmse", "nan")
            rpe_rmse = row.get("rpe_rmse", "nan")
            print(f"[{row['status']}] {sequence}: APE RMSE={ape_rmse} m, RPE RMSE={rpe_rmse} m")

            if args.fail_fast and row["status"] != "ok":
                write_summary(output_dir / "summary.csv", rows)
                return 1

    summary_path = output_dir / "summary.csv"
    write_summary(summary_path, rows)
    print(f"\nSummary written to: {summary_path}")
    return 0 if all(row.get("status") in (None, "ok") for row in rows) else 1


def write_summary(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sequence",
        "status",
        "poses",
        "ape_rmse",
        "ape_mean",
        "ape_median",
        "ape_std",
        "ape_min",
        "ape_max",
        "rpe_rmse",
        "rpe_mean",
        "rpe_median",
        "rpe_std",
        "rpe_min",
        "rpe_max",
        "input",
        "gt",
        "error",
        "ape_returncode",
        "rpe_returncode",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    raise SystemExit(main())
