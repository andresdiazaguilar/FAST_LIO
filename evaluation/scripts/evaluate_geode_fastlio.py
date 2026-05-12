#!/usr/bin/env python3
"""Evaluate FAST-LIO2 trajectories on the local GEODE gamma subset.

Input trajectories must be in TUM format:

    timestamp x y z qx qy qz qw

The script converts gamma-device estimates into the GEODE evaluation frame where
needed, then runs evo APE and RPE against the dataset ground truth.

Run it with the project venv Python:
/home/andres/semester_project/data/venv/bin/python \
  /home/andres/semester_project/data/GEODE_helper/script/evaluate_geode_fastlio.py \
  --est-dir /path/to/your/fastlio_tum_outputs \
  --recursive

This assumes your FAST-LIO2 output files are TUM .txt files and their filenames contain the sequence names, e.g.:
Offroad1.txt
Shield_tunnel6.txt
Tunneling_tunnel1.txt
flat_surfaces_aggressive.txt
flat_surfaces_smooth.txt

If your filenames are different, run it explicitly:
/home/andres/semester_project/data/venv/bin/python \
  /home/andres/semester_project/data/GEODE_helper/script/evaluate_geode_fastlio.py \
  --estimate Offroad1=/path/to/offroad_fastlio.txt \
  --estimate Shield_tunnel6=/path/to/shield_fastlio.txt \
  --estimate Tunneling_tunnel1=/path/to/tunneling_fastlio.txt \
  --estimate flat_surfaces_aggressive=/path/to/flat_aggressive_fastlio.txt \
  --estimate flat_surfaces_smooth=/path/to/flat_smooth_fastlio.txt

Results are written here by default:
/home/andres/semester_project/data/geode_eval/results/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


SEQUENCE_ORDER = (
    "Offroad1",
    "Shield_tunnel6",
    "Tunneling_tunnel1",
    "flat_surfaces_aggressive",
    "flat_surfaces_smooth",
)


@dataclass(frozen=True)
class TransformSpec:
    name: str
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float

    def matrix(self) -> np.ndarray:
        rotation = quat_wxyz_to_matrix(self.qw, self.qx, self.qy, self.qz)
        translation = np.array([self.tx, self.ty, self.tz], dtype=float)
        return make_transform(rotation, translation)


GAMMA_GNSS = TransformSpec(
    name="gamma_gnss",
    qw=0.9998828,
    qx=-0.0057758,
    qy=0.0022253,
    qz=0.0140019,
    tx=0.0305,
    ty=-0.5959,
    tz=0.0902,
)

GAMMA_LEICA = TransformSpec(
    name="gamma_leica",
    qw=0.999901,
    qx=-0.00492765,
    qy=0.00575961,
    qz=0.0117651,
    tx=0.00947221,
    ty=-0.308202,
    tz=-0.365733,
)

SEQUENCE_TRANSFORMS = {
    "Offroad1": GAMMA_GNSS,
    "Shield_tunnel6": GAMMA_LEICA,
    "Tunneling_tunnel1": GAMMA_LEICA,
    "flat_surfaces_aggressive": None,
    "flat_surfaces_smooth": None,
}

STAT_NAMES = ("max", "mean", "median", "min", "rmse", "sse", "std")


def quat_wxyz_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0.0:
        raise ValueError("Quaternion has zero norm")
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return np.array(
        [
            [
                1.0 - 2.0 * qy * qy - 2.0 * qz * qz,
                2.0 * qx * qy - 2.0 * qz * qw,
                2.0 * qx * qz + 2.0 * qy * qw,
            ],
            [
                2.0 * qx * qy + 2.0 * qz * qw,
                1.0 - 2.0 * qx * qx - 2.0 * qz * qz,
                2.0 * qy * qz - 2.0 * qx * qw,
            ],
            [
                2.0 * qx * qz - 2.0 * qy * qw,
                2.0 * qy * qz + 2.0 * qx * qw,
                1.0 - 2.0 * qx * qx - 2.0 * qy * qy,
            ],
        ],
        dtype=float,
    )


def matrix_to_quat_xyzw(rotation: np.ndarray) -> Tuple[float, float, float, float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        qw = (rotation[2, 1] - rotation[1, 2]) / s
        qx = 0.25 * s
        qy = (rotation[0, 1] + rotation[1, 0]) / s
        qz = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        qw = (rotation[0, 2] - rotation[2, 0]) / s
        qx = (rotation[0, 1] + rotation[1, 0]) / s
        qy = 0.25 * s
        qz = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        qw = (rotation[1, 0] - rotation[0, 1]) / s
        qx = (rotation[0, 2] + rotation[2, 0]) / s
        qy = (rotation[1, 2] + rotation[2, 1]) / s
        qz = 0.25 * s

    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    return qx / norm, qy / norm, qz / norm, qw / norm


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


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


def transform_trajectory(input_path: Path, output_path: Path, spec: Optional[TransformSpec]) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if spec is None:
        shutil.copyfile(input_path, output_path)
        return count_data_lines(output_path)

    device_to_eval = np.linalg.inv(spec.matrix())
    line_count = 0
    with input_path.open("r", encoding="utf-8") as f_in, output_path.open("w", encoding="utf-8") as f_out:
        for line_number, line in enumerate(f_in, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 8:
                raise ValueError(f"{input_path}:{line_number}: expected 8 TUM columns")
            timestamp, x, y, z, qx, qy, qz, qw = map(float, parts[:8])
            rotation = quat_wxyz_to_matrix(qw, qx, qy, qz)
            translation = np.array([x, y, z], dtype=float)
            transformed = make_transform(rotation, translation) @ device_to_eval
            qx_out, qy_out, qz_out, qw_out = matrix_to_quat_xyzw(transformed[:3, :3])
            t_out = transformed[:3, 3]
            f_out.write(
                f"{timestamp:.9f} {t_out[0]:.9f} {t_out[1]:.9f} {t_out[2]:.9f} "
                f"{qx_out:.9f} {qy_out:.9f} {qz_out:.9f} {qw_out:.9f}\n"
            )
            line_count += 1
    return line_count


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


def sequence_gt_path(gt_dir: Path, sequence: str) -> Path:
    return gt_dir / f"{sequence}.txt"


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
    default_dataset = Path("/home/andres/semester_project/data/datasets/GEODE")
    default_evo_bin = Path("/home/andres/semester_project/data/venv/bin")
    parser = argparse.ArgumentParser(
        description="Convert GEODE gamma FAST-LIO2 estimates and compute APE/RPE with evo.",
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
    parser.add_argument("--dataset-dir", type=Path, default=default_dataset, help="GEODE dataset root.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/andres/semester_project/data/geode_eval/results"),
        help="Directory for transformed trajectories, evo logs, and summary CSV.",
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
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed conversion or evo command.")
    return parser


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
    gt_dir = dataset_dir / "groundtruth" / "traj"
    output_dir = args.output_dir.expanduser()
    transformed_dir = output_dir / "transformed"
    logs_dir = output_dir / "logs"

    estimates = discover_estimates(est_dir, args.pattern, args.recursive, overrides)
    rows: List[Dict[str, object]] = []
    align = not args.no_align

    for sequence in SEQUENCE_ORDER:
        gt_path = sequence_gt_path(gt_dir, sequence)
        spec = SEQUENCE_TRANSFORMS[sequence]
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
                "transform": spec.name if spec else "none",
            }
            if not estimate_path.exists():
                row["status"] = "missing_estimate"
                rows.append(row)
                print(f"[fail] {sequence}: missing estimate {estimate_path}")
                if args.fail_fast:
                    break
                continue

            output_name = f"{sequence}__{safe_stem(estimate_path)}.txt"
            transformed_path = transformed_dir / sequence / output_name
            try:
                pose_count = transform_trajectory(estimate_path, transformed_path, spec)
            except Exception as exc:
                row["status"] = "conversion_failed"
                row["error"] = str(exc)
                rows.append(row)
                print(f"[fail] {sequence}: conversion failed for {estimate_path}: {exc}")
                if args.fail_fast:
                    write_summary(output_dir / "summary.csv", rows)
                    return 1
                continue

            row["transformed"] = str(transformed_path)
            row["poses"] = pose_count

            ape_command = build_metric_command(
                evo_ape,
                gt_path,
                transformed_path,
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
                transformed_path,
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
        "transform",
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
        "transformed",
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
