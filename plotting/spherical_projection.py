import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

try:
    import open3d as o3d
except ImportError:
    o3d = None


def rotation_matrix_from_euler(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    """
    Build a 3D rotation matrix from yaw, pitch, and roll in degrees.

    Rotation order:
    1. yaw around Z
    2. pitch around Y
    3. roll around X

    This rotates the viewing direction before spherical projection.
    """
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)

    r_yaw = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    r_pitch = np.array(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch)],
        ],
        dtype=np.float32,
    )

    r_roll = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(roll), -np.sin(roll)],
            [0.0, np.sin(roll), np.cos(roll)],
        ],
        dtype=np.float32,
    )

    return r_roll @ r_pitch @ r_yaw


def intensity_to_colors(intensity, cmap="viridis", clip_percentiles=(2, 98)):
    """
    Map intensity values to RGB colors using robust percentile clipping.
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


def require_open3d():
    """
    Ensure Open3D is available before attempting 3D visualization.
    """
    if o3d is None:
        raise ImportError(
            "open3d is required for the point cloud viewer. Install it with: pip install open3d"
        )


def create_projection_frame(
    rotation_matrix,
    translation,
    frame_size=1.0,
    origin_radius=0.08,
):
    """
    Create a world-space frame and origin marker for the virtual projection pose.

    The projection uses:
        p_camera = (p_world - t) @ R.T

    Therefore the virtual camera origin in world coordinates is t and its axes in
    world coordinates are given by R.
    """
    require_open3d()

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=frame_size, origin=[0.0, 0.0, 0.0]
    )
    frame.rotate(rotation_matrix.astype(np.float64), center=(0.0, 0.0, 0.0))
    frame.translate(translation.astype(np.float64))

    origin = o3d.geometry.TriangleMesh.create_sphere(radius=origin_radius)
    origin.compute_vertex_normals()
    origin.paint_uniform_color([1.0, 0.2, 0.2])
    origin.translate(translation.astype(np.float64))

    return frame, origin


def show_point_cloud_intensity_viewer(
    points,
    intensities,
    view_yaw_deg=0.0,
    view_pitch_deg=0.0,
    view_roll_deg=0.0,
    view_translate_x=0.0,
    view_translate_y=0.0,
    view_translate_z=0.0,
    cmap="plasma",
    clip_percentiles=(2, 98),
    point_size=2.0,
    show_world_frame=True,
    world_frame_size=1.0,
    show_projection_frame=True,
    projection_frame_size=1.0,
    projection_origin_radius=0.08,
):
    """
    Visualize the point cloud with intensity coloring and the virtual projection pose.
    """
    require_open3d()

    xyz = points.astype(np.float64)
    colors, lo, hi = intensity_to_colors(
        intensities,
        cmap=cmap,
        clip_percentiles=clip_percentiles,
    )

    print(
        f"[INFO] Viewer normalization range from percentiles "
        f"{clip_percentiles[0]}-{clip_percentiles[1]}: [{lo:.3f}, {hi:.3f}]"
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    geometries = [pcd]

    if show_world_frame:
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=world_frame_size, origin=[0.0, 0.0, 0.0]
        )
        geometries.append(world_frame)

    if show_projection_frame:
        rotation = rotation_matrix_from_euler(
            yaw_deg=view_yaw_deg,
            pitch_deg=view_pitch_deg,
            roll_deg=view_roll_deg,
        )
        translation = np.array(
            [view_translate_x, view_translate_y, view_translate_z],
            dtype=np.float64,
        )
        projection_frame, projection_origin = create_projection_frame(
            rotation_matrix=rotation,
            translation=translation,
            frame_size=projection_frame_size,
            origin_radius=projection_origin_radius,
        )
        geometries.extend([projection_frame, projection_origin])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Spherical Projection Intensity Viewer", width=1400, height=900)

    for geom in geometries:
        vis.add_geometry(geom)

    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.background_color = np.array([0.0, 0.0, 0.0])

    vis.run()
    vis.destroy_window()


def spherical_projection(
    points,
    intensities,
    image_width=1024,
    image_height=128,
    yaw_range=(-np.pi, np.pi),
    pitch_range=None,
    sort_by_depth=True,
    view_yaw_deg=0.0,
    view_pitch_deg=0.0,
    view_roll_deg=0.0,
    view_translate_x=0.0,
    view_translate_y=0.0,
    view_translate_z=0.0,
):
    """
    Create spherical projection images from a point cloud.

    Parameters
    ----------
    points : np.ndarray of shape [N, 3]
        Point cloud XYZ coordinates.
    intensities : np.ndarray of shape [N]
        Intensity values.
    image_width : int
        Width of output spherical image.
    image_height : int
        Height of output spherical image.
    yaw_range : tuple
        (min_yaw, max_yaw) in radians.
    pitch_range : tuple or None
        (min_pitch, max_pitch) in radians.
        If None, uses the min/max pitch found in the cloud.
    sort_by_depth : bool
        If True, sorts far-to-near so closer points overwrite farther ones.
    view_yaw_deg : float
        Rotation of the view around Z in degrees.
    view_pitch_deg : float
        Rotation of the view around Y in degrees.
    view_roll_deg : float
        Rotation of the view around X in degrees.
    view_translate_x : float
        Virtual camera translation along X in the input point-cloud frame.
    view_translate_y : float
        Virtual camera translation along Y in the input point-cloud frame.
    view_translate_z : float
        Virtual camera translation along Z in the input point-cloud frame.

    Returns
    -------
    intensity_image : np.ndarray [H, W]
        Spherical intensity image.
    range_image : np.ndarray [H, W]
        Spherical range image.
    index_image : np.ndarray [H, W]
        Stores original point indices used at each pixel, or -1.
    """
    translated_points = points - np.array(
        [view_translate_x, view_translate_y, view_translate_z],
        dtype=np.float32,
    )

    rotation = rotation_matrix_from_euler(
        yaw_deg=view_yaw_deg,
        pitch_deg=view_pitch_deg,
        roll_deg=view_roll_deg,
    )
    rotated_points = translated_points @ rotation.T

    x = rotated_points[:, 0]
    y = rotated_points[:, 1]
    z = rotated_points[:, 2]

    ranges = np.sqrt(x**2 + y**2 + z**2)

    valid = ranges > 1e-6
    x = x[valid]
    y = y[valid]
    z = z[valid]
    ranges = ranges[valid]
    intensities = intensities[valid]
    original_indices = np.arange(points.shape[0])[valid]

    yaw = np.arctan2(y, x)
    pitch = np.arcsin(z / ranges)

    yaw_min, yaw_max = yaw_range

    if pitch_range is None:
        pitch_min = np.min(pitch)
        pitch_max = np.max(pitch)
    else:
        pitch_min, pitch_max = pitch_range

    # Keep only points inside the requested angular FOV
    mask = (
        (yaw >= yaw_min) & (yaw <= yaw_max) &
        (pitch >= pitch_min) & (pitch <= pitch_max)
    )

    yaw = yaw[mask]
    pitch = pitch[mask]
    ranges = ranges[mask]
    intensities = intensities[mask]
    original_indices = original_indices[mask]

    # Normalize angles to image coordinates
    u = (yaw - yaw_min) / (yaw_max - yaw_min)
    v = (pitch - pitch_min) / (pitch_max - pitch_min)

    u = np.floor(u * (image_width - 1)).astype(np.int32)
    v = np.floor((1.0 - v) * (image_height - 1)).astype(np.int32)
    # note: (1-v) flips image so higher pitch is near top

    # Initialize outputs
    intensity_image = np.zeros((image_height, image_width), dtype=np.float32)
    range_image = np.full((image_height, image_width), np.inf, dtype=np.float32)
    index_image = np.full((image_height, image_width), -1, dtype=np.int32)

    # Sort far-to-near so nearer points overwrite farther ones
    if sort_by_depth:
        order = np.argsort(ranges)[::-1]
    else:
        order = np.arange(len(ranges))

    for i in order:
        px = u[i]
        py = v[i]
        r = ranges[i]

        if r < range_image[py, px]:
            range_image[py, px] = r
            intensity_image[py, px] = intensities[i]
            index_image[py, px] = original_indices[i]

    range_image[range_image == np.inf] = 0.0

    rotation_info = {
        "view_yaw_deg": float(view_yaw_deg),
        "view_pitch_deg": float(view_pitch_deg),
        "view_roll_deg": float(view_roll_deg),
        "view_translate_x": float(view_translate_x),
        "view_translate_y": float(view_translate_y),
        "view_translate_z": float(view_translate_z),
    }

    return intensity_image, range_image, index_image, (pitch_min, pitch_max), rotation_info


def normalize_robust(image, lower_percentile=2, upper_percentile=98):
    """
    Robust normalization to [0, 1], ignoring zeros as invalid pixels.
    """
    out = np.zeros_like(image, dtype=np.float32)

    valid = image > 0
    if not np.any(valid):
        return out

    vals = image[valid]
    lo = np.percentile(vals, lower_percentile)
    hi = np.percentile(vals, upper_percentile)

    if np.isclose(lo, hi):
        out[valid] = 1.0
        return out

    clipped = np.clip(image, lo, hi)
    out = (clipped - lo) / (hi - lo + 1e-8)
    out[~valid] = 0.0

    return out


def save_projection_images(
    intensity_image,
    range_image,
    output_dir,
    intensity_cmap="turbo",
):
    """
    Save visualization images for intensity and range.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Robust normalization
    intensity_norm = normalize_robust(intensity_image, 2, 98)
    range_norm = normalize_robust(range_image, 2, 98)

    # Save raw arrays
    np.save(os.path.join(output_dir, "intensity_image.npy"), intensity_image)
    np.save(os.path.join(output_dir, "range_image.npy"), range_image)

    # Save grayscale PNGs
    intensity_gray = (intensity_norm * 255).astype(np.uint8)
    range_gray = (range_norm * 255).astype(np.uint8)

    cv2.imwrite(os.path.join(output_dir, "intensity_gray.png"), intensity_gray)
    cv2.imwrite(os.path.join(output_dir, "range_gray.png"), range_gray)

    # Save colormap PNG using matplotlib
    cmap = plt.get_cmap(intensity_cmap)
    intensity_color = (cmap(intensity_norm)[:, :, :3] * 255).astype(np.uint8)
    intensity_color_bgr = cv2.cvtColor(intensity_color, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, "intensity_colormap.png"), intensity_color_bgr)

    print(f"[INFO] Saved outputs to: {output_dir}")


def show_projection(intensity_image, range_image, intensity_cmap="turbo"):
    """
    Display spherical projection results.
    """
    intensity_norm = normalize_robust(intensity_image, 2, 98)
    range_norm = normalize_robust(range_image, 2, 98)

    plt.figure(figsize=(14, 5))
    plt.imshow(intensity_norm, cmap=intensity_cmap, aspect="auto")
    plt.title("Spherical Intensity Projection")
    plt.xlabel("Yaw")
    plt.ylabel("Pitch")
    plt.colorbar(label="Normalized Intensity")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 5))
    plt.imshow(range_norm, cmap="viridis", aspect="auto")
    plt.title("Spherical Range Projection")
    plt.xlabel("Yaw")
    plt.ylabel("Pitch")
    plt.colorbar(label="Normalized Range")
    plt.tight_layout()
    plt.show()


def main():
    npy_path = "/home/andres/semester_project/data/tinamu_failure_data/cameroon_fail_short/scan_100_to_400.npy"
    output_dir = "/home/andres/semester_project/data/tinamu_failure_data/cameroon_fail_short/spherical_projection"
    view_yaw_deg =-10.0#z, blue
    view_pitch_deg = 0.0 #x red
    view_roll_deg = 25.0 +180 #x??#y, green
    view_translate_x = -6.0
    view_translate_y = -3.0
    view_translate_z = -6.0
    show_point_cloud_viewer = True
    viewer_cmap = "plasma"
    viewer_clip_percentiles = (2, 98)
    viewer_point_size = 2.0
    world_frame_size = 5.0
    projection_frame_size = 5.0
    projection_origin_radius = 0.2

    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"File not found: {npy_path}")

    cloud = np.load(npy_path)

    if cloud.ndim != 2 or cloud.shape[1] < 4:
        raise ValueError(
            f"Expected .npy with shape [N, 4+] containing [x, y, z, intensity], got {cloud.shape}"
        )

    points = cloud[:, :3].astype(np.float32)
    intensities = cloud[:, 3].astype(np.float32)

    print(f"[INFO] Loaded cloud: {cloud.shape[0]} points")
    print(
        f"[INFO] Intensity stats -> min: {intensities.min():.3f}, "
        f"max: {intensities.max():.3f}, mean: {intensities.mean():.3f}, "
        f"std: {intensities.std():.3f}"
    )

    intensity_image, range_image, index_image, pitch_range, rotation_info = spherical_projection(
        points=points,
        intensities=intensities,
        image_width=1024,
        image_height=128,
        yaw_range=(-np.pi, np.pi),
        pitch_range=None,   # auto from data
        sort_by_depth=True,
        view_yaw_deg=view_yaw_deg,
        view_pitch_deg=view_pitch_deg,
        view_roll_deg=view_roll_deg,
        view_translate_x=view_translate_x,
        view_translate_y=view_translate_y,
        view_translate_z=view_translate_z,
    )

    print(f"[INFO] Auto pitch range used: {pitch_range}")
    print(
        f"[INFO] View rotation used -> "
        f"yaw: {rotation_info['view_yaw_deg']:.3f} deg, "
        f"pitch: {rotation_info['view_pitch_deg']:.3f} deg, "
        f"roll: {rotation_info['view_roll_deg']:.3f} deg"
    )
    print(
        f"[INFO] View translation used -> "
        f"x: {rotation_info['view_translate_x']:.3f}, "
        f"y: {rotation_info['view_translate_y']:.3f}, "
        f"z: {rotation_info['view_translate_z']:.3f}"
    )

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "index_image.npy"), index_image)

    save_projection_images(
        intensity_image=intensity_image,
        range_image=range_image,
        output_dir=output_dir,
        intensity_cmap="plasma",
    )

    show_projection(
        intensity_image=intensity_image,
        range_image=range_image,
        intensity_cmap="plasma",
    )

    if show_point_cloud_viewer:
        show_point_cloud_intensity_viewer(
            points=points,
            intensities=intensities,
            view_yaw_deg=view_yaw_deg,
            view_pitch_deg=view_pitch_deg,
            view_roll_deg=view_roll_deg,
            view_translate_x=view_translate_x,
            view_translate_y=view_translate_y,
            view_translate_z=view_translate_z,
            cmap=viewer_cmap,
            clip_percentiles=viewer_clip_percentiles,
            point_size=viewer_point_size,
            show_world_frame=True,
            world_frame_size=world_frame_size,
            show_projection_frame=True,
            projection_frame_size=projection_frame_size,
            projection_origin_radius=projection_origin_radius,
        )


if __name__ == "__main__":
    main()
