#pragma once
// ============================================================================
// drpm_degeneracy.h
//
// Port of the reference implementation published by the DRPM authors
// (https://github.com/ntnu-arl/drpm, src/degeneracy.h) so that it can be used
// inside FAST-LIO2 without pulling in an extra dependency.  The math,
// function names, template signatures and internal variable names are kept
// identical to the reference implementation whenever possible, so that the
// code here can be cross-checked against the paper's open-source release line
// by line.
//
// Reference: J. Hatleskog, K. Alexis, "Probabilistic Degeneracy Detection for
// Point-to-Plane Error Minimization", arXiv:2410.10784.
//
// The only intentional changes relative to the upstream header are:
//   * wrapped in an include guard / #pragma once (original is a plain header),
//   * the NaN warning inside ComputeSignalToNoiseProbabilities uses ROS_WARN
//     instead of std::cout (so it is visible in the FAST-LIO2 console),
//   * a couple of `inline` qualifiers so the non-template helper links cleanly
//     when the header is included from multiple translation units.
// ============================================================================

#include <boost/math/distributions/normal.hpp>
#include <cmath>
#include <tuple>
#include <vector>

#include <Eigen/Eigenvalues>
#include <ros/ros.h>

namespace degeneracy {

template <typename T>
using VectorVector3 = std::vector<Eigen::Matrix<T, 3, 1>>;

template <typename T>
using VectorMatrix3 = std::vector<Eigen::Matrix<T, 3, 3>>;

template <typename T>
inline Eigen::Matrix<T, 3, 3> VectorToSkew(const Eigen::Matrix<T, 3, 1>& vector) {
  Eigen::Matrix<T, 3, 3> skew;
  skew << 0, -vector.z(), vector.y(), vector.z(), 0, -vector.x(), -vector.y(), vector.x(), 0;
  return skew;
}

template <typename T, typename Q>
auto ComputeNoiseEstimate(const VectorVector3<Q>& points, const VectorVector3<Q>& normals, const std::vector<Q>& weights, const VectorMatrix3<T>& normal_covariances, const Eigen::Matrix<T, 6, 6>& U, const T& stdevPoints) {
  using Vector3 = Eigen::Matrix<T, 3, 1>;
  using Matrix3 = Eigen::Matrix<T, 3, 3>;
  using Vector6 = Eigen::Matrix<T, 6, 1>;
  using Matrix6 = Eigen::Matrix<T, 6, 6>;

  Matrix6 mean = Matrix6::Zero();
  Vector6 variance = Vector6::Zero();

  const size_t nPoints = points.size();

  for (size_t i = 0; i < nPoints; i++) {
    const Vector3 point = points[i].template cast<T>();
    const Vector3 normal = normals[i].template cast<T>();
    const Matrix3 nx = VectorToSkew<T>(normal);
    const Matrix3 px = VectorToSkew<T>(point);
    const T w = weights[i];

    // Coefficient matrix for epsilon and eta
    Matrix6 B = Matrix6::Zero();
    B.block(0, 0, 3, 3) = -nx;
    B.block(0, 3, 3, 3) = px * nx;
    B.block(3, 3, 3, 3) = nx;

    // Covariance matrix for epsilon and eta
    Matrix6 N = Matrix6::Zero();
    N.block(0, 0, 3, 3) = Matrix3::Identity() * std::pow(stdevPoints, 2);
    N.block(3, 3, 3, 3) = normal_covariances[i];

    Matrix6 contribution_to_mean = (B * N * B.transpose()) * w;

    mean.noalias() += contribution_to_mean.eval();

    // v hat weighted by w
    Vector6 v = Vector6::Zero();
    v.head(3) = std::sqrt(w) * px * normal;
    v.tail(3) = std::sqrt(w) * normal;

    // Compute variance in the directions given by U
    for (size_t k = 0; k < 6; k++) {
      const Vector6 u = U.col(k);
      const T a = (u.transpose() * contribution_to_mean * u).value();
      const T b = (u.transpose() * v).value();
      const T contribution_to_variance = 2 * std::pow(a, 2) + 4 * a * std::pow(b, 2);
      variance[k] += contribution_to_variance;
    }
  }

  return std::make_tuple(mean, variance);
}

template <typename T>
Eigen::Matrix<T, 6, 1> ComputeSignalToNoiseProbabilities(const Eigen::Matrix<T, 6, 6>& measured_information_matrix,
                                                         const Eigen::Matrix<T, 6, 6>& estimated_noise_mean,
                                                         const Eigen::Matrix<T, 6, 1>& estimated_noise_variances,
                                                         const Eigen::Matrix<T, 6, 6>& U,
                                                         const T& snr_factor) {
  typedef Eigen::Matrix<T, 6, 1> Vector6;

  Vector6 probabilities = Vector6::Zero();

  for (size_t k = 0; k < 6; k++) {
    const Vector6 u = U.col(k);
    const T measurement = (u.transpose() * measured_information_matrix * u).value();
    const T expected_noise = (u.transpose() * estimated_noise_mean * u).value();
    const T variance = estimated_noise_variances[k];
    const T stdev = variance > T(0.0) ? std::sqrt(variance) : T(0.0);
    const T test_point = measurement / (T(1.0) + snr_factor);

    const bool any_nan = std::isnan(expected_noise) || std::isnan(stdev) || std::isnan(test_point);

    if (!any_nan) {
      T probability = T(0.0);
      if (stdev > T(1e-12)) {
        probability = boost::math::cdf(
            boost::math::normal_distribution<T>(expected_noise, stdev), test_point);
      } else {
        probability = test_point >= expected_noise ? T(1.0) : T(0.0);
      }

      probabilities[k] = probability;
    } else {
      ROS_WARN_THROTTLE(1.0,
          "[DRPM] NaN value in probability calculation - stDev: %g  test_point: %g  expected_noise: %g",
          static_cast<double>(stdev), static_cast<double>(test_point), static_cast<double>(expected_noise));
      probabilities[k] = 0.0;
    }
  }

  return probabilities;
}

template <typename T>
Eigen::Matrix<T, 6, 1> SolveWithSnrProbabilities(
    const Eigen::Matrix<T, 6, 6>& U,
    const Eigen::Matrix<T, 6, 1>& eigenvalues,
    const Eigen::Matrix<T, 6, 1>& rhs,
    const Eigen::Matrix<T, 6, 1>& snr_probabilities) {
  typedef typename Eigen::Matrix<T, 6, 1> Vector6;

  Vector6 d_psinv = Vector6::Zero();

  for (size_t i = 0; i < 6; i++) {
    const T eigenvalue = eigenvalues[i];
    const T p = snr_probabilities[i];
    d_psinv[i] = p / eigenvalue;
  }

  Vector6 perturbation = U * d_psinv.asDiagonal() * U.transpose() * rhs;

  return perturbation;
}

}  // namespace degeneracy
