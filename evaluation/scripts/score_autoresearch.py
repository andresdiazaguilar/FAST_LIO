#!/usr/bin/env python3
"""Compute the scalar autoresearch score from evo APE/RPE summaries.

The score is intentionally split into:
  1. per-sequence normalized APE/RPE ratios against a fixed baseline,
  2. per-dataset mean sequence score,
  3. a final scalar that mostly tracks the weaker dataset.

Lower is better. A score below zero means better than the baseline on average.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


PROJECT_DIR = Path("/home/andres/semester_project")


@dataclass(frozen=True)
class CsvSpec:
    dataset: str
    candidate_path: Path
    baseline_path: Path


@dataclass(frozen=True)
class SequenceScore:
    dataset: str
    sequence: str
    candidate_ape_rmse: float
    baseline_ape_rmse: float
    candidate_rpe_rmse: float
    baseline_rpe_rmse: float
    ape_norm: float
    rpe_norm: float
    score: float
    raw_score: float
    clipped: bool


def positive_float(value: str, field: str, sequence: str, path: Path) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {sequence}: {field} is not a float: {value!r}") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{path}: {sequence}: {field} must be finite and > 0, got {value!r}")
    return parsed


def read_summary(path: Path) -> Dict[str, Mapping[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)

    rows: Dict[str, Mapping[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sequence", "status", "ape_rmse", "rpe_rmse"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing required columns: {sorted(missing)}")

        for row in reader:
            sequence = (row.get("sequence") or "").strip()
            if not sequence:
                continue
            if sequence in rows:
                raise ValueError(f"{path}: duplicate sequence row: {sequence}")
            rows[sequence] = row

    if not rows:
        raise ValueError(f"{path}: no sequence rows found")
    return rows


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot average an empty list")
    return sum(values) / len(values)


def score_dataset(
    spec: CsvSpec,
    ape_weight: float,
    rpe_weight: float,
    missing_penalty: float,
    failed_penalty: float,
    seq_score_clip: float,
) -> List[SequenceScore]:
    candidate_rows = read_summary(spec.candidate_path)
    baseline_rows = read_summary(spec.baseline_path)

    scores: List[SequenceScore] = []
    all_sequences = sorted(set(candidate_rows) | set(baseline_rows))
    for sequence in all_sequences:
        candidate = candidate_rows.get(sequence)
        baseline = baseline_rows.get(sequence)

        if candidate is None or baseline is None:
            scores.append(
                SequenceScore(
                    dataset=spec.dataset,
                    sequence=sequence,
                    candidate_ape_rmse=math.nan,
                    baseline_ape_rmse=math.nan,
                    candidate_rpe_rmse=math.nan,
                    baseline_rpe_rmse=math.nan,
                    ape_norm=math.nan,
                    rpe_norm=math.nan,
                    score=missing_penalty,
                    raw_score=missing_penalty,
                    clipped=False,
                )
            )
            continue

        if candidate.get("status") != "ok" or baseline.get("status") != "ok":
            scores.append(
                SequenceScore(
                    dataset=spec.dataset,
                    sequence=sequence,
                    candidate_ape_rmse=math.nan,
                    baseline_ape_rmse=math.nan,
                    candidate_rpe_rmse=math.nan,
                    baseline_rpe_rmse=math.nan,
                    ape_norm=math.nan,
                    rpe_norm=math.nan,
                    score=failed_penalty,
                    raw_score=failed_penalty,
                    clipped=False,
                )
            )
            continue

        candidate_ape = positive_float(candidate["ape_rmse"], "ape_rmse", sequence, spec.candidate_path)
        baseline_ape = positive_float(baseline["ape_rmse"], "ape_rmse", sequence, spec.baseline_path)
        candidate_rpe = positive_float(candidate["rpe_rmse"], "rpe_rmse", sequence, spec.candidate_path)
        baseline_rpe = positive_float(baseline["rpe_rmse"], "rpe_rmse", sequence, spec.baseline_path)

        ape_norm = candidate_ape / baseline_ape
        rpe_norm = candidate_rpe / baseline_rpe
        raw_score = ape_weight * math.log(ape_norm) + rpe_weight * math.log(rpe_norm)
        score = max(-seq_score_clip, min(seq_score_clip, raw_score))
        scores.append(
            SequenceScore(
                dataset=spec.dataset,
                sequence=sequence,
                candidate_ape_rmse=candidate_ape,
                baseline_ape_rmse=baseline_ape,
                candidate_rpe_rmse=candidate_rpe,
                baseline_rpe_rmse=baseline_rpe,
                ape_norm=ape_norm,
                rpe_norm=rpe_norm,
                score=score,
                raw_score=raw_score,
                clipped=(score != raw_score),
            )
        )

    return scores


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute normalized APE/RPE aggregate score for FAST-LIO autoresearch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--geode-candidate",
        type=Path,
        default=PROJECT_DIR / "data/geode_eval/results_autoresearch/summary.csv",
        help="Candidate GEODE evo summary CSV.",
    )
    parser.add_argument(
        "--tinamu-candidate",
        type=Path,
        default=PROJECT_DIR / "data/tinamu_eval/results_autoresearch/summary.csv",
        help="Candidate Tinamu evo summary CSV.",
    )
    parser.add_argument(
        "--geode-baseline",
        type=Path,
        default=PROJECT_DIR / "data/baseline_performance/summary_geode_base_fastlio2.csv",
        help="Fixed baseline GEODE evo summary CSV.",
    )
    parser.add_argument(
        "--tinamu-baseline",
        type=Path,
        default=PROJECT_DIR / "data/baseline_performance/summary_tinamu_base_fastlio2.csv",
        help="Fixed baseline Tinamu evo summary CSV.",
    )
    parser.add_argument("--ape-weight", type=float, default=0.6, help="Weight for log-normalized APE RMSE.")
    parser.add_argument("--rpe-weight", type=float, default=0.4, help="Weight for log-normalized RPE RMSE.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Deprecated; kept for CLI compatibility. Max aggregation does not use temperature.",
    )
    parser.add_argument("--max-weight", type=float, default=0.8, help="Final score weight for worst dataset.")
    parser.add_argument(
        "--soft-worst-weight",
        type=float,
        help="Deprecated alias for --max-weight.",
    )
    parser.add_argument("--mean-weight", type=float, default=0.2, help="Final score weight for mean dataset score.")
    parser.add_argument("--missing-penalty", type=float, default=5.0, help="Sequence score used when a sequence is missing.")
    parser.add_argument("--failed-penalty", type=float, default=5.0, help="Sequence score used when evo status is not ok.")
    parser.add_argument(
        "--seq-score-clip",
        type=float,
        default=3.0,
        help=(
            "Symmetric clip applied to each per-sequence log-derived score "
            "(in [-clip, +clip]) before aggregation. Prevents one wildly "
            "divergent bag from swamping the aggregate. Pass 'inf' to disable. "
            "Does NOT clip missing/failed penalties — those stay sentinels."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of text.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)

    total_metric_weight = args.ape_weight + args.rpe_weight
    if not math.isclose(total_metric_weight, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        print(f"error: --ape-weight + --rpe-weight must equal 1.0, got {total_metric_weight}", file=sys.stderr)
        return 2

    max_weight = args.max_weight if args.soft_worst_weight is None else args.soft_worst_weight

    total_final_weight = max_weight + args.mean_weight
    if not math.isclose(total_final_weight, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        print(
            f"error: max weight + --mean-weight must equal 1.0, got {total_final_weight}",
            file=sys.stderr,
        )
        return 2

    if args.seq_score_clip <= 0.0 or math.isnan(args.seq_score_clip):
        print(
            f"error: --seq-score-clip must be > 0 (or inf to disable), got {args.seq_score_clip}",
            file=sys.stderr,
        )
        return 2

    specs = [
        CsvSpec("geode", args.geode_candidate.expanduser(), args.geode_baseline.expanduser()),
        CsvSpec("tinamu", args.tinamu_candidate.expanduser(), args.tinamu_baseline.expanduser()),
    ]

    try:
        sequence_scores: List[SequenceScore] = []
        for spec in specs:
            sequence_scores.extend(
                score_dataset(
                    spec,
                    args.ape_weight,
                    args.rpe_weight,
                    args.missing_penalty,
                    args.failed_penalty,
                    args.seq_score_clip,
                )
            )

        dataset_scores = {
            spec.dataset: mean(score.score for score in sequence_scores if score.dataset == spec.dataset) for spec in specs
        }
        dataset_score_values = list(dataset_scores.values())
        max_score = max(dataset_score_values)
        mean_score = mean(dataset_score_values)
        final_score = max_weight * max_score + args.mean_weight * mean_score
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    output = {
        "final_score": final_score,
        "max_dataset_score": max_score,
        "mean_dataset_score": mean_score,
        "dataset_scores": dataset_scores,
        "parameters": {
            "ape_weight": args.ape_weight,
            "rpe_weight": args.rpe_weight,
            "max_weight": max_weight,
            "mean_weight": args.mean_weight,
            "missing_penalty": args.missing_penalty,
            "failed_penalty": args.failed_penalty,
            "seq_score_clip": args.seq_score_clip,
        },
        "per_sequence": [
            {
                "dataset": score.dataset,
                "sequence": score.sequence,
                "score": score.score,
                "raw_score": score.raw_score,
                "clipped": score.clipped,
                "ape_norm": score.ape_norm,
                "rpe_norm": score.rpe_norm,
                "candidate_ape_rmse": score.candidate_ape_rmse,
                "baseline_ape_rmse": score.baseline_ape_rmse,
                "candidate_rpe_rmse": score.candidate_rpe_rmse,
                "baseline_rpe_rmse": score.baseline_rpe_rmse,
            }
            for score in sequence_scores
        ],
    }

    if args.json:
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        print(f"final_score: {final_score:.9f}")
        print(f"max_dataset_score: {max_score:.9f}")
        print(f"mean_dataset_score: {mean_score:.9f}")
        print("")
        print("dataset_scores:")
        for dataset, score in sorted(dataset_scores.items()):
            print(f"  {dataset}: {score:.9f}")
        print("")
        print("per_sequence:")
        for score in sorted(sequence_scores, key=lambda item: (item.dataset, item.sequence)):
            clip_tag = " [clipped]" if score.clipped else ""
            print(
                "  "
                f"{score.dataset}/{score.sequence}: "
                f"score={score.score:.9f}{clip_tag}, "
                f"raw={score.raw_score:.9f}, "
                f"ape_norm={score.ape_norm:.6g}, "
                f"rpe_norm={score.rpe_norm:.6g}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
