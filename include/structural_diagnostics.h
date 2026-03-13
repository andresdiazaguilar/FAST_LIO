#pragma once
// ============================================================================
// structural_diagnostics.h — Lightweight PCA-based point structure classifier
//
// For each point in an unorganized cloud, computes local geometry via
// eigenvalues of the k-NN covariance matrix and classifies as:
//   - EDGE   (line-like): one dominant eigenvalue
//   - PLANE  (surface):   two dominant eigenvalues, one small
//   - UNSTRUCTURED:       everything else
//
// This is a diagnostics-only module.  It does NOT modify the estimator.
// ============================================================================

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float64MultiArray.h>
#include <ros/ros.h>
#include <Eigen/Dense>

// Replicate the PointType definitions from common_lib.h to avoid
// multiple-definition linker errors (common_lib.h defines non-inline globals).
typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

// ---- Result struct for a single frame ----
struct StructuralDiagResult
{
    int    total_valid_points = 0;
    int    edge_count         = 0;
    int    plane_count        = 0;
    int    unstructured_count = 0;
    double edge_ratio         = 0.0;
    double plane_ratio        = 0.0;

    // Separated point clouds for publishing / visualization
    PointCloudXYZI::Ptr edge_cloud;
    PointCloudXYZI::Ptr plane_cloud;

    StructuralDiagResult()
        : edge_cloud(new PointCloudXYZI),
          plane_cloud(new PointCloudXYZI)
    {}
};

// ---- Configurable parameters ----
struct StructuralDiagConfig
{
    // KNN neighborhood size
    int   k_neighbors              = 10;

    // Pre-filter: minimum range from sensor origin (meters)
    double min_range               = 0.5;

    // Classification thresholds (see classifyLocalStructure)
    // Eigenvalues are ordered ASCENDING: lambda1 <= lambda2 <= lambda3.
    //   edge  if   lambda2 / (lambda3 + eps)  <  edge_thresh
    //   plane if   lambda1 / (lambda2 + eps)  <  plane_thresh
    //          AND lambda2 / (lambda3 + eps)  >  plane_upper_thresh
    double edge_thresh             = 0.1;        // 1 / 10.0
    double plane_thresh            = 0.1;        // 1 / 10.0
    double plane_upper_thresh      = 0.333333;   // 1 / 3.0

    // Numerical epsilon to avoid division by zero
    double eps                     = 1e-6;

    // If true, publishes edge/plane cloud topics and logs per-frame stats
    bool   enable                  = true;
};

// ---- Main class ----
class StructuralDiagnostics
{
public:
    StructuralDiagnostics() = default;

    /// Initialise ROS publishers and load parameters from the parameter server.
    void init(ros::NodeHandle& nh);

    /// Run the diagnostics on a single frame cloud (body frame).
    /// The cloud is treated as unorganized.
    StructuralDiagResult analyze(const PointCloudXYZI::Ptr& cloud_in,
                                 double timestamp);

    /// Access current config (e.g. for logging)
    const StructuralDiagConfig& config() const { return cfg_; }

private:
    // ---- helpers ----

    /// Compute 3x3 covariance of a neighbourhood given the point indices.
    Eigen::Matrix3d computeNeighborhoodCovariance(
        const PointCloudXYZI& cloud,
        const std::vector<int>& indices) const;

    /// Classify a point given the sorted eigenvalues (ascending).
    /// Returns 0=unstructured, 1=edge, 2=plane.
    int classifyLocalStructure(double lambda1,
                               double lambda2,
                               double lambda3) const;

    /// Print one ROS_INFO line per frame, publish clouds.
    void publishAndLog(const StructuralDiagResult& res,
                       const std_msgs::Header& header);

    StructuralDiagConfig cfg_;

    ros::Publisher pub_edge_;
    ros::Publisher pub_plane_;
    ros::Publisher pub_edge_count_;
    ros::Publisher pub_plane_count_;
};
