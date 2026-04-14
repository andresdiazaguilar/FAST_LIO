import math
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import rosbag


# -------------------------
# Configure runs here
# -------------------------
RUNS = [
    {
        "label": "D2 on",
        "bag_path": "/home/andres/semester_project/data/tinamu_failure_data/cameroon_fail_short/D2-LIO_outlier_rejection/d2_on.bag",
        "color": "tab:blue",
        "linestyle": "-",
    },
    {
        "label": "D2 off",
        "bag_path": "/home/andres/semester_project/data/tinamu_failure_data/cameroon_fail_short/D2-LIO_outlier_rejection/d2_off.bag",
        "color": "tab:orange",
        "linestyle": "--",
    },
]


# -------------------------
# Topics from laserMapping.cpp / structural_diagnostics.cpp
# -------------------------
deg_topic = "/fastlio/degeneracy"
post_deg_topic = "/fastlio/degeneracy_post"
eigvals_pre_translation_topic = "/fastlio/eigvals_pre_translation"
eigvals_pre_rotation_topic = "/fastlio/eigvals_pre_rotation"
eigvals_post_translation_topic = "/fastlio/eigvals_post_translation"
eigvals_post_rotation_topic = "/fastlio/eigvals_post_rotation"
inlier_ratio_topic = "/fastlio/inlier_ratio"
d2_outlier_stats_topic = "/fastlio/d2_outlier_stats"
edge_count_topic = "/struct_diag/edge_count"
plane_count_topic = "/struct_diag/plane_count"


# -------------------------
# Plot selection
# Use "all" or a set of keys.
# Available keys:
# "feature_counts", "condition_numbers", "min_eigenvalues",
# "translation_eigenvalues", "rotation_eigenvalues",
# "d2_motion", "structure_counts"
# -------------------------
ENABLED_PLOTS = {
    "feature_counts",
    "condition_numbers",
    "min_eigenvalues",
    "translation_eigenvalues",
    "rotation_eigenvalues",
    "d2_motion",
}


# -------------------------
# Figure saving
# -------------------------
SAVE_PLOTS = False
SAVE_PLOTS_DIR = "plots"
SAVE_PLOTS_FORMAT = "png"
SAVE_PLOTS_DPI = 300
SAVE_PLOTS_TIMESTAMPED_SUBDIR = True


# -------------------------
# Axis options
# -------------------------
LIMIT_X_AXIS = False
xmin, xmax = 0.0, 60.0

# Failure marker
# Editable failure start time in seconds on the normalized plot axis.
SHOW_FAILURE_MARKER = True
FAILURE_TIME = 145.0


def should_plot(key):
    return ENABLED_PLOTS == "all" or key in ENABLED_PLOTS


def _sanitize_filename(name):
    clean = re.sub(r"[^\w\s-]", "", name)
    clean = re.sub(r"[\s-]+", "_", clean.strip())
    return clean.lower() if clean else "figure"


def _figure_name(fig, idx):
    if fig._suptitle is not None and fig._suptitle.get_text():
        return f"{idx:02d}_{_sanitize_filename(fig._suptitle.get_text())}"
    if fig.axes and fig.axes[0].get_title():
        return f"{idx:02d}_{_sanitize_filename(fig.axes[0].get_title())}"
    return f"{idx:02d}_figure"


def save_all_open_figures():
    if not SAVE_PLOTS:
        return

    out_dir = SAVE_PLOTS_DIR
    if SAVE_PLOTS_TIMESTAMPED_SUBDIR:
        out_dir = os.path.join(out_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    fig_numbers = plt.get_fignums()
    if not fig_numbers:
        print("SAVE_PLOTS enabled, but no figures were created.")
        return

    existing_names = set()
    for idx, fig_num in enumerate(fig_numbers, start=1):
        fig = plt.figure(fig_num)
        base_name = _figure_name(fig, idx)
        name = base_name
        suffix = 2
        while name in existing_names:
            name = f"{base_name}_{suffix}"
            suffix += 1
        existing_names.add(name)

        file_path = os.path.join(out_dir, f"{name}.{SAVE_PLOTS_FORMAT}")
        fig.savefig(file_path, dpi=SAVE_PLOTS_DPI, bbox_inches="tight")

    print(f"Saved {len(fig_numbers)} figure(s) to: {out_dir}")


def maybe_apply_xlim(ax):
    if LIMIT_X_AXIS:
        ax.set_xlim(xmin, xmax)


def add_failure_marker(ax=None, linewidth=2, include_label=True):
    if not SHOW_FAILURE_MARKER:
        return
    if ax is None:
        ax = plt.gca()
    label = "failure start" if include_label else None
    ax.axvline(
        x=FAILURE_TIME,
        color="r",
        linestyle="--",
        linewidth=linewidth,
        label=label,
    )


def safe_div(num, den):
    if den is None or not math.isfinite(den) or abs(den) < 1e-12:
        return float("nan")
    return num / den


def normalize_time_series(run):
    t0_candidates = []
    for key in run:
        if key.startswith("t_") and run[key]:
            t0_candidates.append(run[key][0])

    t0 = min(t0_candidates) if t0_candidates else 0.0
    for key in list(run.keys()):
        if key.startswith("t_"):
            run[key] = [ti - t0 for ti in run[key]]
    run["t0"] = t0


def print_run_summary(run):
    label = run["label"]
    print(f"\n=== {label} ===")
    print(f"bag: {run['bag_path']}")
    print(f"degeneracy samples: {len(run['t_deg'])}")
    print(f"inlier ratio samples: {len(run['t_inlier_ratio'])}")
    print(f"post degeneracy samples: {len(run['t_post'])}")
    print(f"d2 stats samples: {len(run['t_d2'])}")

    if run["effective_feature_number"]:
        eff = run["effective_feature_number"]
        print(
            "effective features: "
            f"min={min(eff):.1f}, max={max(eff):.1f}, mean={sum(eff) / len(eff):.1f}"
        )

    valid_total = [v for v in run["total_candidate_features"] if math.isfinite(v)]
    if valid_total:
        print(
            "estimated pre-rejection features: "
            f"min={min(valid_total):.1f}, max={max(valid_total):.1f}, "
            f"mean={sum(valid_total) / len(valid_total):.1f}"
        )

    valid_inlier = [v for v in run["inlier_ratio"] if math.isfinite(v)]
    if valid_inlier:
        print(
            "inlier ratio: "
            f"min={min(valid_inlier):.3f}, max={max(valid_inlier):.3f}, "
            f"mean={sum(valid_inlier) / len(valid_inlier):.3f}"
        )

    if run["d2_filter_enabled"]:
        flags = sorted(set(int(v) for v in run["d2_filter_enabled"]))
        print(f"d2 filter flags seen in bag: {flags}")


def load_run(run_cfg):
    run = {
        "label": run_cfg["label"],
        "bag_path": run_cfg["bag_path"],
        "color": run_cfg.get("color", None),
        "linestyle": run_cfg.get("linestyle", "-"),
        "t_deg": [],
        "effective_feature_number": [],
        "cond_pre": [],
        "t_post": [],
        "post_eff": [],
        "cond_post": [],
        "t_inlier_ratio": [],
        "inlier_ratio": [],
        "total_candidate_features": [],
        "t_eig_pre_trans": [],
        "eig_pre_trans_1": [],
        "eig_pre_trans_2": [],
        "eig_pre_trans_3": [],
        "min_eig_pre_trans": [],
        "cond_eig_pre_trans": [],
        "t_eig_pre_rot": [],
        "eig_pre_rot_1": [],
        "eig_pre_rot_2": [],
        "eig_pre_rot_3": [],
        "min_eig_pre_rot": [],
        "cond_eig_pre_rot": [],
        "t_eig_post_trans": [],
        "eig_post_trans_1": [],
        "eig_post_trans_2": [],
        "eig_post_trans_3": [],
        "min_eig_post_trans": [],
        "cond_eig_post_trans": [],
        "t_eig_post_rot": [],
        "eig_post_rot_1": [],
        "eig_post_rot_2": [],
        "eig_post_rot_3": [],
        "min_eig_post_rot": [],
        "cond_eig_post_rot": [],
        "t_d2": [],
        "d2_filter_enabled": [],
        "d2_motion_ready": [],
        "d2_delta_t_norm": [],
        "d2_sin_half_dtheta": [],
        "d2_eff_features": [],
        "t_edge_count": [],
        "edge_count": [],
        "t_plane_count": [],
        "plane_count": [],
    }

    with rosbag.Bag(run_cfg["bag_path"], "r") as bag:
        for _, msg, _ in bag.read_messages(topics=[deg_topic]):
            data = msg.data
            if len(data) < 5:
                continue
            run["t_deg"].append(float(data[0]))
            run["effective_feature_number"].append(float(data[1]))
            run["cond_pre"].append(float(data[4]))

        for _, msg, _ in bag.read_messages(topics=[post_deg_topic]):
            data = msg.data
            if len(data) < 5:
                continue
            run["t_post"].append(float(data[0]))
            run["post_eff"].append(float(data[1]))
            run["cond_post"].append(float(data[4]))

        for _, msg, t in bag.read_messages(topics=[inlier_ratio_topic]):
            data = msg.data
            if isinstance(data, (list, tuple)) and len(data) >= 2:
                run["t_inlier_ratio"].append(float(data[0]))
                run["inlier_ratio"].append(float(data[1]))
            else:
                run["t_inlier_ratio"].append(t.to_sec())
                run["inlier_ratio"].append(float(data))

        for _, msg, _ in bag.read_messages(topics=[eigvals_pre_translation_topic]):
            data = msg.data
            if len(data) < 8:
                continue
            run["t_eig_pre_trans"].append(float(data[0]))
            run["eig_pre_trans_1"].append(float(data[2]))
            run["eig_pre_trans_2"].append(float(data[3]))
            run["eig_pre_trans_3"].append(float(data[4]))
            run["min_eig_pre_trans"].append(float(data[5]))
            run["cond_eig_pre_trans"].append(float(data[7]))

        for _, msg, _ in bag.read_messages(topics=[eigvals_pre_rotation_topic]):
            data = msg.data
            if len(data) < 8:
                continue
            run["t_eig_pre_rot"].append(float(data[0]))
            run["eig_pre_rot_1"].append(float(data[2]))
            run["eig_pre_rot_2"].append(float(data[3]))
            run["eig_pre_rot_3"].append(float(data[4]))
            run["min_eig_pre_rot"].append(float(data[5]))
            run["cond_eig_pre_rot"].append(float(data[7]))

        for _, msg, _ in bag.read_messages(topics=[eigvals_post_translation_topic]):
            data = msg.data
            if len(data) < 8:
                continue
            run["t_eig_post_trans"].append(float(data[0]))
            run["eig_post_trans_1"].append(float(data[2]))
            run["eig_post_trans_2"].append(float(data[3]))
            run["eig_post_trans_3"].append(float(data[4]))
            run["min_eig_post_trans"].append(float(data[5]))
            run["cond_eig_post_trans"].append(float(data[7]))

        for _, msg, _ in bag.read_messages(topics=[eigvals_post_rotation_topic]):
            data = msg.data
            if len(data) < 8:
                continue
            run["t_eig_post_rot"].append(float(data[0]))
            run["eig_post_rot_1"].append(float(data[2]))
            run["eig_post_rot_2"].append(float(data[3]))
            run["eig_post_rot_3"].append(float(data[4]))
            run["min_eig_post_rot"].append(float(data[5]))
            run["cond_eig_post_rot"].append(float(data[7]))

        for _, msg, _ in bag.read_messages(topics=[d2_outlier_stats_topic]):
            data = msg.data
            if len(data) < 6:
                continue
            run["t_d2"].append(float(data[0]))
            run["d2_filter_enabled"].append(float(data[1]))
            run["d2_motion_ready"].append(float(data[2]))
            run["d2_delta_t_norm"].append(float(data[3]))
            run["d2_sin_half_dtheta"].append(float(data[4]))
            run["d2_eff_features"].append(float(data[5]))

        for _, msg, t in bag.read_messages(topics=[edge_count_topic]):
            data = msg.data
            if isinstance(data, (list, tuple)) and len(data) >= 2:
                run["t_edge_count"].append(float(data[0]))
                run["edge_count"].append(float(data[1]))
            else:
                run["t_edge_count"].append(t.to_sec())
                run["edge_count"].append(float(data))

        for _, msg, t in bag.read_messages(topics=[plane_count_topic]):
            data = msg.data
            if isinstance(data, (list, tuple)) and len(data) >= 2:
                run["t_plane_count"].append(float(data[0]))
                run["plane_count"].append(float(data[1]))
            else:
                run["t_plane_count"].append(t.to_sec())
                run["plane_count"].append(float(data))

    for eff, ratio in zip(run["effective_feature_number"], run["inlier_ratio"]):
        run["total_candidate_features"].append(safe_div(eff, ratio))

    normalize_time_series(run)
    print_run_summary(run)
    return run


def plot_series(ax, run, times_key, values_key, label_suffix=""):
    times = run[times_key]
    values = run[values_key]
    if not times or not values:
        return

    label = run["label"]
    if label_suffix:
        label = f"{label} {label_suffix}"

    ax.plot(
        times,
        values,
        label=label,
        color=run["color"],
        linestyle=run["linestyle"],
        linewidth=1.8,
    )


loaded_runs = [load_run(run_cfg) for run_cfg in RUNS if os.path.exists(run_cfg["bag_path"])]

missing_bags = [run_cfg["bag_path"] for run_cfg in RUNS if not os.path.exists(run_cfg["bag_path"])]
if missing_bags:
    print("\nMissing bag files:")
    for bag_path in missing_bags:
        print(f"  {bag_path}")

if not loaded_runs:
    raise RuntimeError("No bag files were loaded. Update RUNS with valid bag paths.")


# =========================
# Feature counts
# =========================
if should_plot("feature_counts"):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle("D2 Outlier Filter Feature Comparison")

    axp = axs[0, 0]
    for run in loaded_runs:
        plot_series(axp, run, "t_deg", "effective_feature_number")
    add_failure_marker(axp)
    axp.set_title("Effective Features After Outlier Rejection")
    axp.set_ylabel("count")
    axp.grid(True)
    axp.legend()
    maybe_apply_xlim(axp)

    axp = axs[0, 1]
    for run in loaded_runs:
        plot_series(axp, run, "t_inlier_ratio", "inlier_ratio")
    add_failure_marker(axp)
    axp.set_title("Inlier Ratio")
    axp.set_ylabel("ratio")
    axp.grid(True)
    axp.legend()
    maybe_apply_xlim(axp)

    axp = axs[1, 0]
    for run in loaded_runs:
        plot_series(axp, run, "t_deg", "total_candidate_features")
    add_failure_marker(axp)
    axp.set_title("Estimated Features Before Outlier Rejection")
    axp.set_xlabel("time [s]")
    axp.set_ylabel("count")
    axp.grid(True)
    axp.legend()
    maybe_apply_xlim(axp)

    axp = axs[1, 1]
    for run in loaded_runs:
        plot_series(axp, run, "t_post", "post_eff")
    add_failure_marker(axp)
    axp.set_title("Posterior Effective Feature Count")
    axp.set_xlabel("time [s]")
    axp.set_ylabel("count")
    axp.grid(True)
    axp.legend()
    maybe_apply_xlim(axp)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


# =========================
# Condition numbers
# =========================
if should_plot("condition_numbers"):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle("Condition Number Comparison")

    axp = axs[0, 0]
    for run in loaded_runs:
        plot_series(axp, run, "t_deg", "cond_pre")
    add_failure_marker(axp)
    axp.set_title("Pose Condition Number (Pre)")
    axp.set_ylabel("condition number")
    axp.grid(True)
    axp.legend()
    maybe_apply_xlim(axp)

    axp = axs[0, 1]
    for run in loaded_runs:
        plot_series(axp, run, "t_post", "cond_post")
    add_failure_marker(axp)
    axp.set_title("Pose Condition Number (Post)")
    axp.set_ylabel("condition number")
    axp.grid(True)
    axp.legend()
    maybe_apply_xlim(axp)

    axp = axs[1, 0]
    for run in loaded_runs:
        plot_series(axp, run, "t_eig_pre_trans", "cond_eig_pre_trans", "(trans pre)")
        plot_series(axp, run, "t_eig_pre_rot", "cond_eig_pre_rot", "(rot pre)")
    add_failure_marker(axp)
    axp.set_title("3x3 Block Condition Numbers (Pre)")
    axp.set_xlabel("time [s]")
    axp.set_ylabel("condition number")
    axp.grid(True)
    axp.legend()
    maybe_apply_xlim(axp)

    axp = axs[1, 1]
    for run in loaded_runs:
        plot_series(axp, run, "t_eig_post_trans", "cond_eig_post_trans", "(trans post)")
        plot_series(axp, run, "t_eig_post_rot", "cond_eig_post_rot", "(rot post)")
    add_failure_marker(axp)
    axp.set_title("3x3 Block Condition Numbers (Post)")
    axp.set_xlabel("time [s]")
    axp.set_ylabel("condition number")
    axp.grid(True)
    axp.legend()
    maybe_apply_xlim(axp)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


# =========================
# Minimum eigenvalues
# =========================
if should_plot("min_eigenvalues"):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle("Minimum Eigenvalue Comparison")

    axp = axs[0, 0]
    for run in loaded_runs:
        plot_series(axp, run, "t_eig_pre_trans", "min_eig_pre_trans")
    add_failure_marker(axp)
    axp.set_title("Translation Min Eigenvalue (Pre)")
    axp.set_ylabel("lambda_min")
    axp.grid(True)
    axp.legend()
    maybe_apply_xlim(axp)

    axp = axs[0, 1]
    for run in loaded_runs:
        plot_series(axp, run, "t_eig_pre_rot", "min_eig_pre_rot")
    add_failure_marker(axp)
    axp.set_title("Rotation Min Eigenvalue (Pre)")
    axp.set_ylabel("lambda_min")
    axp.grid(True)
    axp.legend()
    maybe_apply_xlim(axp)

    axp = axs[1, 0]
    for run in loaded_runs:
        plot_series(axp, run, "t_eig_post_trans", "min_eig_post_trans")
    add_failure_marker(axp)
    axp.set_title("Translation Min Eigenvalue (Post)")
    axp.set_xlabel("time [s]")
    axp.set_ylabel("lambda_min")
    axp.grid(True)
    axp.legend()
    maybe_apply_xlim(axp)

    axp = axs[1, 1]
    for run in loaded_runs:
        plot_series(axp, run, "t_eig_post_rot", "min_eig_post_rot")
    add_failure_marker(axp)
    axp.set_title("Rotation Min Eigenvalue (Post)")
    axp.set_xlabel("time [s]")
    axp.set_ylabel("lambda_min")
    axp.grid(True)
    axp.legend()
    maybe_apply_xlim(axp)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


# =========================
# Translation eigenvalues
# =========================
if should_plot("translation_eigenvalues"):
    fig, axs = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Translation Eigenvalue Spectra")

    axp = axs[0]
    for run in loaded_runs:
        plot_series(axp, run, "t_eig_pre_trans", "eig_pre_trans_1", "(lambda1 pre)")
        plot_series(axp, run, "t_eig_pre_trans", "eig_pre_trans_2", "(lambda2 pre)")
        plot_series(axp, run, "t_eig_pre_trans", "eig_pre_trans_3", "(lambda3 pre)")
    add_failure_marker(axp)
    axp.set_title("Pre-Update Translation Eigenvalues")
    axp.set_ylabel("eigenvalue")
    axp.grid(True)
    axp.legend(ncol=2)
    maybe_apply_xlim(axp)

    axp = axs[1]
    for run in loaded_runs:
        plot_series(axp, run, "t_eig_post_trans", "eig_post_trans_1", "(lambda1 post)")
        plot_series(axp, run, "t_eig_post_trans", "eig_post_trans_2", "(lambda2 post)")
        plot_series(axp, run, "t_eig_post_trans", "eig_post_trans_3", "(lambda3 post)")
    add_failure_marker(axp)
    axp.set_title("Post-Update Translation Eigenvalues")
    axp.set_xlabel("time [s]")
    axp.set_ylabel("eigenvalue")
    axp.grid(True)
    axp.legend(ncol=2)
    maybe_apply_xlim(axp)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


# =========================
# Rotation eigenvalues
# =========================
if should_plot("rotation_eigenvalues"):
    fig, axs = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Rotation Eigenvalue Spectra")

    axp = axs[0]
    for run in loaded_runs:
        plot_series(axp, run, "t_eig_pre_rot", "eig_pre_rot_1", "(lambda1 pre)")
        plot_series(axp, run, "t_eig_pre_rot", "eig_pre_rot_2", "(lambda2 pre)")
        plot_series(axp, run, "t_eig_pre_rot", "eig_pre_rot_3", "(lambda3 pre)")
    add_failure_marker(axp)
    axp.set_title("Pre-Update Rotation Eigenvalues")
    axp.set_ylabel("eigenvalue")
    axp.grid(True)
    axp.legend(ncol=2)
    maybe_apply_xlim(axp)

    axp = axs[1]
    for run in loaded_runs:
        plot_series(axp, run, "t_eig_post_rot", "eig_post_rot_1", "(lambda1 post)")
        plot_series(axp, run, "t_eig_post_rot", "eig_post_rot_2", "(lambda2 post)")
        plot_series(axp, run, "t_eig_post_rot", "eig_post_rot_3", "(lambda3 post)")
    add_failure_marker(axp)
    axp.set_title("Post-Update Rotation Eigenvalues")
    axp.set_xlabel("time [s]")
    axp.set_ylabel("eigenvalue")
    axp.grid(True)
    axp.legend(ncol=2)
    maybe_apply_xlim(axp)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


# =========================
# D2 motion diagnostics
# =========================
if should_plot("d2_motion"):
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("D2 Motion Diagnostics")

    axp = axs[0]
    for run in loaded_runs:
        plot_series(axp, run, "t_d2", "d2_delta_t_norm")
    add_failure_marker(axp)
    axp.set_title("Inter-Scan Translation Norm")
    axp.set_ylabel("||delta_t|| [m]")
    axp.grid(True)
    axp.legend()
    maybe_apply_xlim(axp)

    axp = axs[1]
    for run in loaded_runs:
        plot_series(axp, run, "t_d2", "d2_sin_half_dtheta")
    add_failure_marker(axp)
    axp.set_title("sin(dtheta / 2)")
    axp.set_ylabel("value")
    axp.grid(True)
    axp.legend()
    maybe_apply_xlim(axp)

    axp = axs[2]
    for run in loaded_runs:
        plot_series(axp, run, "t_d2", "d2_filter_enabled")
    add_failure_marker(axp)
    axp.set_title("D2 Filter Flag Published By Node")
    axp.set_xlabel("time [s]")
    axp.set_ylabel("0 = off, 1 = on")
    axp.set_yticks([0.0, 1.0])
    axp.grid(True)
    axp.legend()
    maybe_apply_xlim(axp)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


# =========================
# Structural counts
# =========================
if should_plot("structure_counts"):
    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Structural Diagnostics Counts")

    axp = axs[0]
    for run in loaded_runs:
        plot_series(axp, run, "t_plane_count", "plane_count")
    add_failure_marker(axp)
    axp.set_title("Plane Count")
    axp.set_ylabel("count")
    axp.grid(True)
    axp.legend()
    maybe_apply_xlim(axp)

    axp = axs[1]
    for run in loaded_runs:
        plot_series(axp, run, "t_edge_count", "edge_count")
    add_failure_marker(axp)
    axp.set_title("Edge Count")
    axp.set_xlabel("time [s]")
    axp.set_ylabel("count")
    axp.grid(True)
    axp.legend()
    maybe_apply_xlim(axp)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


save_all_open_figures()
plt.show()
