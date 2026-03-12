#ifndef CORNER_FEATURE_EXTRACTOR_HPP
#define CORNER_FEATURE_EXTRACTOR_HPP

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

/**
 * LIO-SAM-style corner and surface feature extractor for FAST-LIO.
 *
 * Works on unorganized point clouds by:
 *   1. Assigning each point to a scan ring via its elevation angle
 *   2. Sorting points within each ring by azimuth angle
 *   3. Computing range-based Laplacian curvature
 *   4. Marking occluded / parallel-beam points
 *   5. Extracting high-curvature (corner) and low-curvature (surface) features
 *      per-ring per-sector with non-maximum suppression
 *
 * The extracted corner and surface point clouds are in the same frame as the
 * input cloud (typically body frame), ready for downstream use (visualization,
 * map matching, Kalman filter measurement model, etc.).
 */
class CornerFeatureExtractor
{
public:
    // ===================== Configuration =====================
    int   N_SCANS;              // Number of scan rings (sensor-dependent)
    float edgeThreshold;        // Curvature above this → corner candidate
    float surfThreshold;        // Curvature below this → surface candidate
    int   maxCornersPerSector;  // Max corners extracted per sector per ring
    int   numSectors;           // Sectors per ring (azimuthal divisions)
    float surfLeafSize;         // VoxelGrid leaf size for surface downsampling
    float sensorMinRange;       // Blind-zone minimum range (metres)
    float verticalAngleBottom;  // Lower bound of vertical FOV (degrees)
    float verticalAngleTop;     // Upper bound of vertical FOV (degrees)

    // ===================== Output ============================
    PointCloudXYZI::Ptr cornerCloud;   // Extracted corner features
    PointCloudXYZI::Ptr surfaceCloud;  // Extracted surface features (downsampled per ring)
    int cornerCount;
    int surfaceCount;

    // ===================== Constructor =======================
    CornerFeatureExtractor()
        : N_SCANS(16)
        , edgeThreshold(1.0f)
        , surfThreshold(0.1f)
        , maxCornersPerSector(20)
        , numSectors(6)
        , surfLeafSize(0.4f)
        , sensorMinRange(0.5f)
        , verticalAngleBottom(-15.0f)
        , verticalAngleTop(15.0f)
        , cornerCount(0)
        , surfaceCount(0)
    {
        cornerCloud.reset(new PointCloudXYZI());
        surfaceCloud.reset(new PointCloudXYZI());
    }

    /**
     * Run the full extraction pipeline on an unorganized point cloud.
     * After this call, cornerCloud, surfaceCloud, cornerCount, surfaceCount
     * are populated.
     *
     * @param cloud  Input point cloud (body-frame, undistorted).
     */
    void extract(const PointCloudXYZI::Ptr &cloud)
    {
        cornerCloud->clear();
        surfaceCloud->clear();
        cornerCount = 0;
        surfaceCount = 0;

        if (!cloud || cloud->points.empty())
            return;

        organizeCloud(cloud);

        if (orderedCloud_->points.empty())
            return;

        calculateSmoothness();
        markOccludedPoints();
        extractFeatures();
    }

private:
    // ===================== Internal state ====================
    PointCloudXYZI::Ptr           orderedCloud_;
    std::vector<float>            pointRange_;
    std::vector<int>              pointColInd_;
    std::vector<int>              startRingIndex_;
    std::vector<int>              endRingIndex_;

    std::vector<float>            cloudCurvature_;
    std::vector<int>              cloudNeighborPicked_;
    std::vector<int>              cloudLabel_;

    struct SmoothInfo {
        float  value;
        size_t ind;
    };
    std::vector<SmoothInfo>       cloudSmoothness_;

    // ---------------------------------------------------------
    // Step 1 – Organize unstructured cloud into per-ring order
    // ---------------------------------------------------------
    void organizeCloud(const PointCloudXYZI::Ptr &cloud)
    {
        const int inputSize = static_cast<int>(cloud->points.size());

        struct PointMeta {
            int   origIndex;
            int   ring;
            float azimuth;   // degrees [0, 360)
            float range;
        };

        std::vector<PointMeta> metas;
        metas.reserve(inputSize);

        const float vertRange = verticalAngleTop - verticalAngleBottom;
        if (vertRange <= 0.0f)
            return;

        for (int i = 0; i < inputSize; ++i)
        {
            const PointType &pt = cloud->points[i];
            const float range = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
            if (range < sensorMinRange)
                continue;

            const float vertAngle =
                std::atan2(pt.z, std::sqrt(pt.x * pt.x + pt.y * pt.y)) * 180.0f / static_cast<float>(M_PI);

            const int ring = static_cast<int>(
                (vertAngle - verticalAngleBottom) / vertRange * static_cast<float>(N_SCANS));
            if (ring < 0 || ring >= N_SCANS)
                continue;

            float azimuth = std::atan2(pt.y, pt.x) * 180.0f / static_cast<float>(M_PI);
            if (azimuth < 0.0f)
                azimuth += 360.0f;

            PointMeta m;
            m.origIndex = i;
            m.ring      = ring;
            m.azimuth   = azimuth;
            m.range     = range;
            metas.push_back(m);
        }

        // Sort by ring, then by azimuth within each ring
        std::sort(metas.begin(), metas.end(),
                  [](const PointMeta &a, const PointMeta &b)
                  {
                      if (a.ring != b.ring) return a.ring < b.ring;
                      return a.azimuth < b.azimuth;
                  });

        const int orderedSize = static_cast<int>(metas.size());
        orderedCloud_.reset(new PointCloudXYZI());
        orderedCloud_->points.resize(orderedSize);
        pointRange_.resize(orderedSize);
        pointColInd_.resize(orderedSize);

        startRingIndex_.assign(N_SCANS, -1);
        endRingIndex_.assign(N_SCANS, -1);

        // Pseudo horizontal resolution for occlusion column checks
        static const int HORIZ_RESOLUTION = 1800;

        for (int i = 0; i < orderedSize; ++i)
        {
            orderedCloud_->points[i] = cloud->points[metas[i].origIndex];
            pointRange_[i]  = metas[i].range;
            pointColInd_[i] = static_cast<int>(metas[i].azimuth / 360.0f * HORIZ_RESOLUTION);

            const int ring = metas[i].ring;
            if (startRingIndex_[ring] < 0)
                startRingIndex_[ring] = i;
            endRingIndex_[ring] = i;
        }
    }

    // ---------------------------------------------------------
    // Step 2 – Compute range-based Laplacian curvature
    // ---------------------------------------------------------
    void calculateSmoothness()
    {
        const int cloudSize = static_cast<int>(orderedCloud_->points.size());
        cloudCurvature_.assign(cloudSize, 0.0f);
        cloudNeighborPicked_.assign(cloudSize, 0);
        cloudLabel_.assign(cloudSize, 0);
        cloudSmoothness_.resize(cloudSize);

        for (int i = 5; i < cloudSize - 5; ++i)
        {
            const float diffRange =
                  pointRange_[i - 5] + pointRange_[i - 4]
                + pointRange_[i - 3] + pointRange_[i - 2]
                + pointRange_[i - 1] - pointRange_[i] * 10.0f
                + pointRange_[i + 1] + pointRange_[i + 2]
                + pointRange_[i + 3] + pointRange_[i + 4]
                + pointRange_[i + 5];

            cloudCurvature_[i] = diffRange * diffRange;
        }

        // Populate smoothness vector (including boundary points initialised to 0)
        for (int i = 0; i < cloudSize; ++i)
        {
            cloudSmoothness_[i].value = cloudCurvature_[i];
            cloudSmoothness_[i].ind   = static_cast<size_t>(i);
        }
    }

    // ---------------------------------------------------------
    // Step 3 – Mark occluded & parallel-beam points
    // ---------------------------------------------------------
    void markOccludedPoints()
    {
        const int cloudSize = static_cast<int>(orderedCloud_->points.size());

        for (int i = 5; i < cloudSize - 6; ++i)
        {
            const float depth1 = pointRange_[i];
            const float depth2 = pointRange_[i + 1];
            const int columnDiff = std::abs(pointColInd_[i + 1] - pointColInd_[i]);

            if (columnDiff < 10)
            {
                if (depth1 - depth2 > 0.3f)
                {
                    cloudNeighborPicked_[i - 5] = 1;
                    cloudNeighborPicked_[i - 4] = 1;
                    cloudNeighborPicked_[i - 3] = 1;
                    cloudNeighborPicked_[i - 2] = 1;
                    cloudNeighborPicked_[i - 1] = 1;
                    cloudNeighborPicked_[i]     = 1;
                }
                else if (depth2 - depth1 > 0.3f)
                {
                    cloudNeighborPicked_[i + 1] = 1;
                    cloudNeighborPicked_[i + 2] = 1;
                    cloudNeighborPicked_[i + 3] = 1;
                    cloudNeighborPicked_[i + 4] = 1;
                    cloudNeighborPicked_[i + 5] = 1;
                    cloudNeighborPicked_[i + 6] = 1;
                }
            }

            // Parallel beam
            const float diff1 = std::abs(pointRange_[i - 1] - pointRange_[i]);
            const float diff2 = std::abs(pointRange_[i + 1] - pointRange_[i]);
            if (diff1 > 0.02f * pointRange_[i] && diff2 > 0.02f * pointRange_[i])
                cloudNeighborPicked_[i] = 1;
        }
    }

    // ---------------------------------------------------------
    // Step 4 – Extract corner & surface features per-ring/sector
    // ---------------------------------------------------------
    void extractFeatures()
    {
        PointCloudXYZI::Ptr surfScan(new PointCloudXYZI());
        PointCloudXYZI::Ptr surfScanDS(new PointCloudXYZI());

        pcl::VoxelGrid<PointType> downFilter;
        downFilter.setLeafSize(surfLeafSize, surfLeafSize, surfLeafSize);

        const int orderedSize = static_cast<int>(orderedCloud_->points.size());

        for (int i = 0; i < N_SCANS; ++i)
        {
            if (startRingIndex_[i] < 0 || endRingIndex_[i] < 0)
                continue;
            if (endRingIndex_[i] - startRingIndex_[i] < 10)
                continue;   // need ≥11 points for curvature window

            surfScan->clear();

            for (int j = 0; j < numSectors; ++j)
            {
                const int sp = (startRingIndex_[i] * (numSectors - j)
                              + endRingIndex_[i] * j) / numSectors;
                const int ep = (startRingIndex_[i] * (numSectors - 1 - j)
                              + endRingIndex_[i] * (j + 1)) / numSectors - 1;

                if (sp >= ep)
                    continue;

                // Sort sector by curvature (ascending)
                std::sort(cloudSmoothness_.begin() + sp,
                          cloudSmoothness_.begin() + ep + 1,
                          [](const SmoothInfo &a, const SmoothInfo &b)
                          { return a.value < b.value; });

                // --- Corner extraction (highest curvature first) ---
                int largestPickedNum = 0;
                for (int k = ep; k >= sp; --k)
                {
                    const int ind = static_cast<int>(cloudSmoothness_[k].ind);
                    if (ind < 0 || ind >= orderedSize)
                        continue;
                    if (cloudNeighborPicked_[ind] != 0)
                        continue;
                    if (cloudCurvature_[ind] <= edgeThreshold)
                        break;

                    ++largestPickedNum;
                    if (largestPickedNum <= maxCornersPerSector)
                    {
                        cloudLabel_[ind] = 1;
                        cornerCloud->push_back(orderedCloud_->points[ind]);
                    }
                    else
                    {
                        break;
                    }

                    cloudNeighborPicked_[ind] = 1;
                    for (int l = 1; l <= 5; ++l)
                    {
                        const int nb = ind + l;
                        if (nb >= orderedSize) break;
                        if (std::abs(pointColInd_[nb] - pointColInd_[nb - 1]) > 10)
                            break;
                        cloudNeighborPicked_[nb] = 1;
                    }
                    for (int l = -1; l >= -5; --l)
                    {
                        const int nb = ind + l;
                        if (nb < 0) break;
                        if (std::abs(pointColInd_[nb] - pointColInd_[nb + 1]) > 10)
                            break;
                        cloudNeighborPicked_[nb] = 1;
                    }
                }

                // --- Surface extraction (lowest curvature first) ---
                for (int k = sp; k <= ep; ++k)
                {
                    const int ind = static_cast<int>(cloudSmoothness_[k].ind);
                    if (ind < 0 || ind >= orderedSize)
                        continue;
                    if (cloudNeighborPicked_[ind] != 0)
                        continue;
                    if (cloudCurvature_[ind] >= surfThreshold)
                        break;

                    cloudLabel_[ind] = -1;
                    cloudNeighborPicked_[ind] = 1;

                    for (int l = 1; l <= 5; ++l)
                    {
                        const int nb = ind + l;
                        if (nb >= orderedSize) break;
                        if (std::abs(pointColInd_[nb] - pointColInd_[nb - 1]) > 10)
                            break;
                        cloudNeighborPicked_[nb] = 1;
                    }
                    for (int l = -1; l >= -5; --l)
                    {
                        const int nb = ind + l;
                        if (nb < 0) break;
                        if (std::abs(pointColInd_[nb] - pointColInd_[nb + 1]) > 10)
                            break;
                        cloudNeighborPicked_[nb] = 1;
                    }
                }

                // All non-corner points → surface candidates
                for (int k = sp; k <= ep; ++k)
                {
                    const int ind = static_cast<int>(cloudSmoothness_[k].ind);
                    if (ind < 0 || ind >= orderedSize)
                        continue;
                    if (cloudLabel_[ind] <= 0)
                        surfScan->push_back(orderedCloud_->points[ind]);
                }
            }

            // Downsample surface features for this ring
            surfScanDS->clear();
            downFilter.setInputCloud(surfScan);
            downFilter.filter(*surfScanDS);

            *surfaceCloud += *surfScanDS;
        }

        cornerCount  = static_cast<int>(cornerCloud->points.size());
        surfaceCount = static_cast<int>(surfaceCloud->points.size());
    }
};

#endif // CORNER_FEATURE_EXTRACTOR_HPP
