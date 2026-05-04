import open3d as o3d
from pypcd4 import PointCloud
import numpy as np
import matplotlib.pyplot as plt
import os


def pcd_to_npy(pcd_file, npy_file=""):
    """
    Convert a .pcd file to .npy with columns:
    [x, y, z, intensity]
    """
    pcd = PointCloud.from_path(pcd_file)
    pc_data = pcd.pc_data

    if "intensity" in pc_data.dtype.names:
        intensity = pc_data["intensity"]
    else:
        print("[WARNING] No intensity field found in PCD. Using zeros.")
        intensity = np.zeros_like(pc_data["x"], dtype=np.float32)

    points = np.column_stack(
        (pc_data["x"], pc_data["y"], pc_data["z"], intensity)
    ).astype(np.float32)

    if npy_file == "":
        npy_file = pcd_file.replace(".pcd", ".npy")

    np.save(npy_file, points)
    print(f"[INFO] Saved numpy point cloud to: {npy_file}")
    return npy_file


def intensity_to_colors(intensity, cmap="viridis", clip_percentiles=(2, 98)):
    """
    Map intensity values to RGB colors.

    Parameters
    ----------
    intensity : np.ndarray
        1D array of intensity values.
    cmap : str
        Matplotlib colormap name. Examples: 'gray', 'viridis', 'plasma', 'inferno', 'turbo'
    clip_percentiles : tuple
        Lower and upper percentiles used for robust clipping.
    """
    intensity = intensity.astype(np.float32)

    lo = np.percentile(intensity, clip_percentiles[0])
    hi = np.percentile(intensity, clip_percentiles[1])

    if np.isclose(hi, lo):
        print("[WARNING] Intensity range is nearly constant. Using zeros for normalization.")
        norm = np.zeros_like(intensity, dtype=np.float32)
    else:
        intensity_clipped = np.clip(intensity, lo, hi)
        norm = (intensity_clipped - lo) / (hi - lo + 1e-8)

    colors = plt.get_cmap(cmap)(norm)[:, :3]
    return colors, lo, hi


def plot_pcd(
    npy_file,
    cmap="viridis",
    clip_percentiles=(2, 98),
    point_size=2.0,
    show_frame=True,
    frame_size=0.6,
):
    """
    Visualize a numpy point cloud with intensity-based coloring.

    Expected npy format:
    [x, y, z, intensity]
    """
    np_pcd = np.load(npy_file)

    if np_pcd.ndim != 2 or np_pcd.shape[1] < 4:
        raise ValueError(
            f"Expected npy shape [N, 4+] with columns [x, y, z, intensity], got {np_pcd.shape}"
        )

    xyz = np_pcd[:, :3].astype(np.float64)
    intensity = np_pcd[:, 3].astype(np.float32)

    print(f"[INFO] Loaded point cloud: {npy_file}")
    print(f"[INFO] Shape: {np_pcd.shape}")
    print(f"[INFO] Number of points: {xyz.shape[0]}")
    print(
        f"[INFO] Intensity stats -> min: {intensity.min():.3f}, "
        f"max: {intensity.max():.3f}, mean: {intensity.mean():.3f}, "
        f"std: {intensity.std():.3f}"
    )

    colors, lo, hi = intensity_to_colors(
        intensity,
        cmap=cmap,
        clip_percentiles=clip_percentiles,
    )

    print(
        f"[INFO] Robust normalization range from percentiles "
        f"{clip_percentiles[0]}-{clip_percentiles[1]}: [{lo:.3f}, {hi:.3f}]"
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    geometries = [pcd]

    if show_frame:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=frame_size, origin=[0, 0, 0]
        )
        geometries.insert(0, mesh_frame)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Intensity Viewer", width=1400, height=900)
    for geom in geometries:
        vis.add_geometry(geom)

    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.background_color = np.array([0.0, 0.0, 0.0])

    vis.run()
    vis.destroy_window()


def plot_pcd_direct_from_pcd(
    pcd_file,
    cmap="viridis",
    clip_percentiles=(2, 98),
    point_size=2.0,
    show_frame=True,
    frame_size=0.6,
):
    """
    Load directly from .pcd and visualize without manually keeping the .npy.
    """
    tmp_npy = pcd_to_npy(pcd_file, "")
    plot_pcd(
        tmp_npy,
        cmap=cmap,
        clip_percentiles=clip_percentiles,
        point_size=point_size,
        show_frame=show_frame,
        frame_size=frame_size,
    )


if __name__ == "__main__":
    # =========================
    # Choose your input here
    # =========================
    pcd_path = "/home/andres/semester_project/data/tinamu_failure_data/cameroon_fail_short/scan_100_to_400.pcd"
    # pcd_path = "/home/andres/semester_project/data/tinamu_failure_data/drift_gtc/scans.pcd"
    # pcd_path = "/home/andres/semester_project/data/tinamu_failure_data/silo_fail/scans.pcd"

    # =========================
    # Viewer settings
    # =========================
    cmap = "plasma"          # good options: "gray", "viridis", "plasma", "inferno", "turbo"
    clip_percentiles = (2, 98)
    point_size = 2.0
    show_frame = True
    frame_size = 5

    if not os.path.exists(pcd_path):
        raise FileNotFoundError(f"PCD file not found: {pcd_path}")

    # Convert once and visualize
    npy_path = pcd_to_npy(pcd_path, "")
    plot_pcd(
        npy_path,
        cmap=cmap,
        clip_percentiles=clip_percentiles,
        point_size=point_size,
        show_frame=show_frame,
        frame_size=frame_size,
    )