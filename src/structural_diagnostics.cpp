// ============================================================================
// structural_diagnostics.cpp — PCA-based local geometry classification
//
// For each point, find k nearest neighbors in the same frame (KD-tree),
// compute the 3×3 covariance of those neighbors, decompose into eigenvalues
// λ1 ≤ λ2 ≤ λ3, and classify:
//
//   EDGE  : λ2 / (λ3 + ε) < edge_thresh            → one dominant direction
//   PLANE : λ1 / (λ2 + ε) < plane_thresh
//           AND λ2 / (λ3 + ε) > plane_upper_thresh → two dominant directions
//   UNSTRUCTURED : everything else
//
// This module is diagnostics-only and does not affect the estimator.
// ============================================================================

#include "structural_diagnostics.h"

#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Eigenvalues>
#include <cmath>

// ============================================================================
// init — set up ROS publishers and load parameters
// ============================================================================
void StructuralDiagnostics::init(ros::NodeHandle& nh)
{
    // Load parameters (all under the "struct_diag/" namespace)
    nh.param<bool>  ("struct_diag/enable",              cfg_.enable,              true);
    nh.param<int>   ("struct_diag/k_neighbors",         cfg_.k_neighbors,         10);
    nh.param<double>("struct_diag/min_range",            cfg_.min_range,            0.5);
    nh.param<double>("struct_diag/edge_thresh",          cfg_.edge_thresh,          0.1);
    nh.param<double>("struct_diag/plane_thresh",         cfg_.plane_thresh,         0.1);
    nh.param<double>("struct_diag/plane_upper_thresh",   cfg_.plane_upper_thresh,   0.333333);
    nh.param<double>("struct_diag/eps",                  cfg_.eps,                  1e-6);

    // Advertise diagnostic point-cloud topics
    pub_edge_  = nh.advertise<sensor_msgs::PointCloud2>("/struct_diag/edge_cloud",  10);
    pub_plane_ = nh.advertise<sensor_msgs::PointCloud2>("/struct_diag/plane_cloud", 10);
    // Scalar counts for external plotting: [timestamp, count]
    pub_edge_count_  = nh.advertise<std_msgs::Float64MultiArray>("/struct_diag/edge_count", 10);
    pub_plane_count_ = nh.advertise<std_msgs::Float64MultiArray>("/struct_diag/plane_count", 10);

    ROS_INFO("[StructDiag] Initialised — k=%d  min_range=%.2f  "
             "edge_thresh=%.3f  plane_thresh=%.3f  plane_upper=%.3f  enable=%d",
             cfg_.k_neighbors, cfg_.min_range,
             cfg_.edge_thresh, cfg_.plane_thresh, cfg_.plane_upper_thresh,
             cfg_.enable);
}

// ============================================================================
// computeNeighborhoodCovariance
//   Given a set of point indices into `cloud`, compute the 3×3 sample
//   covariance of their (x,y,z) coordinates.
// ============================================================================
Eigen::Matrix3d StructuralDiagnostics::computeNeighborhoodCovariance(
    const PointCloudXYZI& cloud,
    const std::vector<int>& indices) const
{
    const int n = static_cast<int>(indices.size());

    // Compute centroid
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (int idx : indices)
    {
        mean(0) += cloud.points[idx].x;
        mean(1) += cloud.points[idx].y;
        mean(2) += cloud.points[idx].z;
    }
    mean /= static_cast<double>(n);

    // Accumulate scatter matrix
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (int idx : indices)
    {
        Eigen::Vector3d d;
        d(0) = cloud.points[idx].x - mean(0);
        d(1) = cloud.points[idx].y - mean(1);
        d(2) = cloud.points[idx].z - mean(2);
        cov += d * d.transpose();
    }
    cov /= static_cast<double>(n);
    return cov;
}

// ============================================================================
// classifyLocalStructure
//   Eigenvalues must be passed in ASCENDING order: λ1 ≤ λ2 ≤ λ3.
//   Returns: 1 = edge, 2 = plane, 0 = unstructured.
// ============================================================================
int StructuralDiagnostics::classifyLocalStructure(
    double lambda1, double lambda2, double lambda3) const
{
    const double ratio23 = lambda2 / (lambda3 + cfg_.eps);
    const double ratio12 = lambda1 / (lambda2 + cfg_.eps);

    // Edge-like: one dominant eigenvalue → λ2 much smaller than λ3
    if (ratio23 < cfg_.edge_thresh)
        return 1;

    // Plane-like: two dominant eigenvalues → λ1 much smaller than λ2, and λ2 close to λ3
    if (ratio12 < cfg_.plane_thresh && ratio23 > cfg_.plane_upper_thresh)
        return 2;

    return 0;  // unstructured
}

// ============================================================================
// analyze — run the full diagnostics pipeline on one frame
// ============================================================================
StructuralDiagResult StructuralDiagnostics::analyze(
    const PointCloudXYZI::Ptr& cloud_in,
    double timestamp)
{
    StructuralDiagResult res;

    if (!cfg_.enable || !cloud_in || cloud_in->empty())
        return res;

    // ------------------------------------------------------------------
    // 1. Pre-filter: remove NaN / too-close points
    // ------------------------------------------------------------------
    PointCloudXYZI::Ptr cloud_filtered(new PointCloudXYZI);
    cloud_filtered->reserve(cloud_in->size());

    const double min_range_sq = cfg_.min_range * cfg_.min_range;
    for (const auto& pt : cloud_in->points)
    {
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
            continue;
        double r2 = static_cast<double>(pt.x) * pt.x
                   + static_cast<double>(pt.y) * pt.y
                   + static_cast<double>(pt.z) * pt.z;
        if (r2 < min_range_sq)
            continue;
        cloud_filtered->push_back(pt);
    }

    res.total_valid_points = static_cast<int>(cloud_filtered->size());
    if (res.total_valid_points < cfg_.k_neighbors + 1)
        return res;

    // ------------------------------------------------------------------
    // 2. Build a KD-tree for the filtered frame
    // ------------------------------------------------------------------
    pcl::KdTreeFLANN<PointType> kdtree;
    kdtree.setInputCloud(cloud_filtered);

    std::vector<int>   nn_indices(cfg_.k_neighbors + 1);
    std::vector<float> nn_dists_sq(cfg_.k_neighbors + 1);

    // ------------------------------------------------------------------
    // 3. For every point, classify via local PCA
    // ------------------------------------------------------------------
    for (int i = 0; i < res.total_valid_points; ++i)
    {
        const PointType& pt = cloud_filtered->points[i];

        // k+1 because the query point itself is included in the result
        int found = kdtree.nearestKSearch(pt, cfg_.k_neighbors + 1,
                                          nn_indices, nn_dists_sq);
        if (found < cfg_.k_neighbors + 1)
            continue;

        // Covariance of the neighbourhood (including query point)
        Eigen::Matrix3d cov = computeNeighborhoodCovariance(*cloud_filtered,
                                                            nn_indices);

        // Eigenvalue decomposition (self-adjoint → real eigenvalues)
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
        // Eigen returns eigenvalues in ASCENDING order.
        Eigen::Vector3d ev = solver.eigenvalues();
        double lambda1 = ev(0);   // smallest
        double lambda2 = ev(1);
        double lambda3 = ev(2);   // largest

        int label = classifyLocalStructure(lambda1, lambda2, lambda3);

        if (label == 1)
        {
            res.edge_count++;
            res.edge_cloud->push_back(pt);
        }
        else if (label == 2)
        {
            res.plane_count++;
            res.plane_cloud->push_back(pt);
        }
        else
        {
            res.unstructured_count++;
        }
    }

    // Compute ratios
    if (res.total_valid_points > 0)
    {
        res.edge_ratio  = static_cast<double>(res.edge_count)  / res.total_valid_points;
        res.plane_ratio = static_cast<double>(res.plane_count) / res.total_valid_points;
    }

    // ------------------------------------------------------------------
    // 4. Publish and log
    // ------------------------------------------------------------------
    std_msgs::Header header;
    header.stamp.fromSec(timestamp);
    header.frame_id = "body";   // body-frame cloud
    publishAndLog(res, header);

    return res;
}

// ============================================================================
// publishAndLog — ROS_INFO line + PointCloud2 topics
// ============================================================================
void StructuralDiagnostics::publishAndLog(
    const StructuralDiagResult& res,
    const std_msgs::Header& header)
{
    ROS_INFO("[StructDiag] t=%.3f  valid=%d  edge=%d(%.2f%%)  plane=%d(%.2f%%)  "
             "unstruct=%d",
             header.stamp.toSec(),
             res.total_valid_points,
             res.edge_count,  res.edge_ratio  * 100.0,
             res.plane_count, res.plane_ratio * 100.0,
             res.unstructured_count);

    std_msgs::Float64MultiArray edge_count_msg;
    edge_count_msg.data.resize(2);
    edge_count_msg.data[0] = header.stamp.toSec();
    edge_count_msg.data[1] = static_cast<double>(res.edge_count);
    pub_edge_count_.publish(edge_count_msg);

    std_msgs::Float64MultiArray plane_count_msg;
    plane_count_msg.data.resize(2);
    plane_count_msg.data[0] = header.stamp.toSec();
    plane_count_msg.data[1] = static_cast<double>(res.plane_count);
    pub_plane_count_.publish(plane_count_msg);

    // Publish edge cloud
    if (!res.edge_cloud->empty())
    {
        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(*res.edge_cloud, msg);
        msg.header = header;
        pub_edge_.publish(msg);
    }

    // Publish plane cloud
    if (!res.plane_cloud->empty())
    {
        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(*res.plane_cloud, msg);
        msg.header = header;
        pub_plane_.publish(msg);
    }
}
