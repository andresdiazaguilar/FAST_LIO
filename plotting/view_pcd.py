import open3d as o3d

# sudo mount -t drvfs F: /mnt/f

# pcd = o3d.io.read_point_cloud("/mnt/f/andres_data/senegal/results/chunks_0_to_800_optimized_map_downsampled_0.05m.pcd")
# pcd = o3d.io.read_point_cloud("/mnt/f/andres_data/senegal/results/chunks_0_to_800_optimized_poses.pcd")

# pcd = o3d.io.read_point_cloud("/home/andres/semester_project/data/usbdata/valis_fail_2/chunks_bag_optimized_poses.pcd")
pcd = o3d.io.read_point_cloud("/home/andres/semester_project/data/usbdata/valis_fail_2/results/chunk_4s_optimized_poses.pcd")


o3d.visualization.draw_geometries([pcd])