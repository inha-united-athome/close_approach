#include "close_approach/target_selector.hpp"

#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>

namespace {

constexpr float kPi = 3.14159265358979323846F;

float clampPositive(float value, float fallback) {
  return std::isfinite(value) && value > 0.0F ? value : fallback;
}

Eigen::Vector2f normalizedOr(const Eigen::Vector2f &v,
                             const Eigen::Vector2f &fallback) {
  const float n = v.norm();
  if (!std::isfinite(n) || n < 1e-6F) {
    return fallback;
  }
  return v / n;
}

float angleDiff(float a, float b) {
  float d = a - b;
  while (d > kPi) {
    d -= 2.0F * kPi;
  }
  while (d < -kPi) {
    d += 2.0F * kPi;
  }
  return std::abs(d);
}

std::vector<Eigen::Vector2f>
toPoints2D(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  std::vector<Eigen::Vector2f> points;
  points.reserve(cloud->points.size());
  for (const auto &pt : cloud->points) {
    points.emplace_back(pt.x, pt.y);
  }
  return points;
}

std::vector<cv::Point2f> toCvPoints(const std::vector<Eigen::Vector2f> &points) {
  std::vector<cv::Point2f> cv_points;
  cv_points.reserve(points.size());
  for (const auto &p : points) {
    cv_points.emplace_back(p.x(), p.y());
  }
  return cv_points;
}

float convexHullArea(const std::vector<Eigen::Vector2f> &points) {
  const auto cv_points = toCvPoints(points);
  if (cv_points.size() < 3) {
    return 0.0F;
  }

  std::vector<cv::Point2f> hull;
  cv::convexHull(cv_points, hull);
  if (hull.size() < 3) {
    return 0.0F;
  }
  return static_cast<float>(std::abs(cv::contourArea(hull)));
}

OBB obbFromAxes(const std::vector<Eigen::Vector2f> &points,
                Eigen::Vector2f axis1) {
  OBB obb;
  axis1 = normalizedOr(axis1, Eigen::Vector2f(1.0F, 0.0F));
  Eigen::Vector2f axis2(-axis1.y(), axis1.x());

  float min1 = std::numeric_limits<float>::max();
  float max1 = -std::numeric_limits<float>::max();
  float min2 = std::numeric_limits<float>::max();
  float max2 = -std::numeric_limits<float>::max();
  for (const auto &p : points) {
    const float s1 = p.dot(axis1);
    const float s2 = p.dot(axis2);
    min1 = std::min(min1, s1);
    max1 = std::max(max1, s1);
    min2 = std::min(min2, s2);
    max2 = std::max(max2, s2);
  }

  obb.axis1 = axis1;
  obb.axis2 = axis2;
  obb.length1 = std::max(0.0F, max1 - min1);
  obb.length2 = std::max(0.0F, max2 - min2);
  obb.center = ((min1 + max1) * 0.5F) * axis1 +
               ((min2 + max2) * 0.5F) * axis2;
  return obb;
}

TargetFitMetrics computeMetrics(const std::vector<Eigen::Vector2f> &points,
                                const OBB &obb, float edge_threshold) {
  TargetFitMetrics metrics;
  metrics.rect_area = std::max(0.0F, obb.length1 * obb.length2);
  metrics.hull_area = convexHullArea(points);
  if (metrics.rect_area > 1e-6F) {
    metrics.fill_ratio = metrics.hull_area / metrics.rect_area;
  }

  if (points.empty() || obb.length1 < 1e-6F || obb.length2 < 1e-6F) {
    return metrics;
  }

  const float c1 = obb.center.dot(obb.axis1);
  const float c2 = obb.center.dot(obb.axis2);
  const float min1 = c1 - obb.length1 * 0.5F;
  const float max1 = c1 + obb.length1 * 0.5F;
  const float min2 = c2 - obb.length2 * 0.5F;
  const float max2 = c2 + obb.length2 * 0.5F;

  float sum_dist = 0.0F;
  int support_count = 0;
  for (const auto &p : points) {
    const float s1 = p.dot(obb.axis1);
    const float s2 = p.dot(obb.axis2);
    const float edge_dist =
        std::min({std::abs(s1 - min1), std::abs(s1 - max1),
                  std::abs(s2 - min2), std::abs(s2 - max2)});
    sum_dist += edge_dist;
    if (edge_dist <= edge_threshold) {
      ++support_count;
    }
  }

  metrics.mean_edge_dist = sum_dist / static_cast<float>(points.size());
  metrics.support_ratio =
      static_cast<float>(support_count) / static_cast<float>(points.size());
  return metrics;
}

void computeTargetEdgeSupport(const std::vector<Eigen::Vector2f> &points,
                              const TargetEdge &edge, float edge_threshold,
                              TargetFitMetrics &metrics) {
  if (points.empty() || edge.target_length < 1e-6F) {
    return;
  }

  const Eigen::Vector2f axis =
      normalizedOr(edge.target_axis, Eigen::Vector2f(0.0F, 1.0F));
  const float half_length = edge.target_length * 0.5F;
  float sum_dist = 0.0F;
  int support_count = 0;
  for (const auto &p : points) {
    const Eigen::Vector2f delta = p - edge.target_center;
    const float along = delta.dot(axis);
    const float along_clamped = std::clamp(along, -half_length, half_length);
    const Eigen::Vector2f closest = edge.target_center + along_clamped * axis;
    const float dist = (p - closest).norm();
    sum_dist += dist;
    if (dist <= edge_threshold) {
      ++support_count;
    }
  }

  metrics.target_edge_mean_dist =
      sum_dist / static_cast<float>(points.size());
  metrics.target_edge_support_ratio =
      static_cast<float>(support_count) / static_cast<float>(points.size());
}

TargetEdge makeEdgeCandidate(const Eigen::Vector2f &center,
                             const Eigen::Vector2f &axis,
                             const Eigen::Vector2f &normal, float length,
                             const Eigen::Vector2f &anchor) {
  TargetEdge edge;
  edge.target_center = center;
  edge.target_axis = normalizedOr(axis, Eigen::Vector2f(0.0F, 1.0F));
  edge.normal_axis = normalizedOr(normal, Eigen::Vector2f(1.0F, 0.0F));
  edge.target_length = std::max(0.0F, length);

  if (edge.normal_axis.dot(edge.target_center - anchor) < 0.0F) {
    edge.normal_axis = -edge.normal_axis;
  }
  return edge;
}

std::array<TargetEdge, 2> makeSupportedLShapeEdges(
    const OBB &obb, int side_u, int side_v, const Eigen::Vector2f &anchor) {
  const Eigen::Vector2f u_normal = side_u == 0 ? -obb.axis1 : obb.axis1;
  const Eigen::Vector2f v_normal = side_v == 0 ? -obb.axis2 : obb.axis2;

  return {
      makeEdgeCandidate(obb.center + u_normal * (obb.length1 * 0.5F),
                        obb.axis2, u_normal, obb.length2, anchor),
      makeEdgeCandidate(obb.center + v_normal * (obb.length2 * 0.5F),
                        obb.axis1, v_normal, obb.length1, anchor),
  };
}

TargetSelector::EdgeProjection intersectAnchorRayWithTargetEdge(
    const TargetEdge &edge, const Eigen::Vector2f &anchor,
    float segment_tolerance) {
  TargetSelector::EdgeProjection projection;
  const Eigen::Vector2f axis =
      normalizedOr(edge.target_axis, Eigen::Vector2f(0.0F, 1.0F));
  if (std::abs(axis.y()) < 1e-4F) {
    return projection;
  }

  const float half_length = edge.target_length * 0.5F;
  const float s = (anchor.y() - edge.target_center.y()) / axis.y();
  const float s_clamped = std::clamp(s, -half_length, half_length);
  projection.projected = edge.target_center + s * axis;
  projection.ray_x = projection.projected.x() - anchor.x();
  projection.segment_penalty = std::abs(s - s_clamped);
  projection.valid = std::isfinite(projection.ray_x) &&
                     std::isfinite(projection.segment_penalty) &&
                     projection.ray_x >= 0.0F &&
                     projection.segment_penalty <= segment_tolerance;
  return projection;
}

struct LShapeFit {
  bool valid = false;
  OBB obb;
  std::array<TargetEdge, 2> supported_edges;
  TargetFitMetrics metrics;
};

LShapeFit fitLShape(const std::vector<Eigen::Vector2f> &points,
                    const Eigen::Vector2f &anchor,
                    const TargetSelectorParams &params) {
  LShapeFit result;
  if (points.size() < 3) {
    return result;
  }

  const float step = params.l_shape_angle_step_deg * kPi / 180.0F;
  const float sigma = clampPositive(params.l_shape_sigma, 0.03F);
  const float sigma2 = sigma * sigma;
  float best_score = -std::numeric_limits<float>::max();
  float best_theta = 0.0F;
  int best_side_u = 0;
  int best_side_v = 0;

  for (float theta = 0.0F; theta < kPi; theta += step) {
    const Eigen::Vector2f u(std::cos(theta), std::sin(theta));
    const Eigen::Vector2f v(-u.y(), u.x());

    float min_u = std::numeric_limits<float>::max();
    float max_u = -std::numeric_limits<float>::max();
    float min_v = std::numeric_limits<float>::max();
    float max_v = -std::numeric_limits<float>::max();
    std::vector<std::array<float, 2>> projected;
    projected.reserve(points.size());
    for (const auto &p : points) {
      const float pu = p.dot(u);
      const float pv = p.dot(v);
      projected.push_back({pu, pv});
      min_u = std::min(min_u, pu);
      max_u = std::max(max_u, pu);
      min_v = std::min(min_v, pv);
      max_v = std::max(max_v, pv);
    }

    const std::array<float, 2> u_sides = {min_u, max_u};
    const std::array<float, 2> v_sides = {min_v, max_v};
    for (int side_u = 0; side_u < 2; ++side_u) {
      for (int side_v = 0; side_v < 2; ++side_v) {
        float support_score = 0.0F;
        float mean_dist = 0.0F;
        for (const auto &pv : projected) {
          const float d_u = std::abs(pv[0] - u_sides[side_u]);
          const float d_v = std::abs(pv[1] - v_sides[side_v]);
          const float d = std::min(d_u, d_v);
          support_score += std::exp(-(d * d) / (2.0F * sigma2));
          mean_dist += d;
        }
        mean_dist /= static_cast<float>(projected.size());
        const float score = support_score - 0.2F * mean_dist / sigma;
        if (score > best_score) {
          best_score = score;
          best_theta = theta;
          best_side_u = side_u;
          best_side_v = side_v;
        }
      }
    }
  }

  result.obb = obbFromAxes(points, Eigen::Vector2f(std::cos(best_theta),
                                                   std::sin(best_theta)));
  result.valid = result.obb.length1 > 1e-5F && result.obb.length2 > 1e-5F;
  if (!result.valid) {
    return result;
  }

  result.supported_edges =
      makeSupportedLShapeEdges(result.obb, best_side_u, best_side_v, anchor);
  result.metrics = computeMetrics(points, result.obb, params.edge_threshold);
  return result;
}

float ratioCost(float value, float reference) {
  if (value <= 1e-6F || reference <= 1e-6F) {
    return 1.0F;
  }
  return std::abs(std::log(value / reference));
}

std::string stateName(TargetSelector::TrackState state) {
  if (state == TargetSelector::TrackState::LOCKED) {
    return "LOCKED";
  }
  if (state == TargetSelector::TrackState::LOST) {
    return "LOST";
  }
  return "ACQUIRE";
}

} // namespace

void TargetSelector::setParameters(const TargetSelectorParams &params) {
  params_ = params;
  params_.cluster_tolerance = std::max(params_.cluster_tolerance, 0.001F);
  params_.min_cluster_size = std::max(params_.min_cluster_size, 1);
  params_.max_cluster_size =
      std::max(params_.max_cluster_size, params_.min_cluster_size);
  params_.min_cluster_area = std::max(params_.min_cluster_area, 0.0F);
  params_.edge_threshold = std::max(params_.edge_threshold, 0.001F);
  params_.l_shape_angle_step_deg =
      std::clamp(params_.l_shape_angle_step_deg, 0.1F, 10.0F);
  params_.l_shape_sigma = std::max(params_.l_shape_sigma, 0.001F);
  params_.ray_segment_tolerance =
      std::max(params_.ray_segment_tolerance, 0.0F);
  params_.min_target_edge_support_ratio =
      std::clamp(params_.min_target_edge_support_ratio, 0.0F, 1.0F);
  params_.min_fill_ratio = std::clamp(params_.min_fill_ratio, 0.0F, 1.0F);
  params_.acquire_confirm_frames =
      std::max(params_.acquire_confirm_frames, 1);
  params_.relock_confirm_frames = std::max(params_.relock_confirm_frames, 1);
  params_.max_lost_frames = std::max(params_.max_lost_frames, 1);
  params_.acquire_hit_gate = std::max(params_.acquire_hit_gate, 0.01F);
  params_.acquire_yaw_gate = std::max(params_.acquire_yaw_gate, 0.01F);
  params_.lock_hit_gate = std::max(params_.lock_hit_gate, 0.01F);
  params_.lock_yaw_gate = std::max(params_.lock_yaw_gate, 0.01F);
  params_.lock_min_length_ratio =
      std::max(params_.lock_min_length_ratio, 0.01F);
  params_.lock_max_length_ratio =
      std::max(params_.lock_max_length_ratio, params_.lock_min_length_ratio);
  params_.lock_min_area_ratio = std::max(params_.lock_min_area_ratio, 0.01F);
  params_.lock_max_area_ratio =
      std::max(params_.lock_max_area_ratio, params_.lock_min_area_ratio);
  params_.lock_min_normal_dot =
      std::clamp(params_.lock_min_normal_dot, -1.0F, 1.0F);
}

void TargetSelector::reset() {
  state_ = TrackState::ACQUIRE;
  acquire_count_ = 0;
  relock_count_ = 0;
  lost_count_ = 0;
  has_pending_acquire_ = false;
  has_pending_relock_ = false;
  pending_acquire_ = Candidate{};
  pending_relock_ = Candidate{};
  locked_hit_odom_ = Eigen::Vector2f(0.0F, 0.0F);
  locked_normal_odom_ = Eigen::Vector2f(1.0F, 0.0F);
  locked_yaw_ = 0.0F;
  locked_length_ = 0.0F;
  locked_area_ = 0.0F;
}

bool TargetSelector::select(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                            const Eigen::Affine2f &base_to_odom,
                            TargetSelectorResult &result) {
  result = TargetSelectorResult{};
  result.state = stateName(state_);
  result.locked = state_ == TrackState::LOCKED;
  result.lost_count = lost_count_;

  if (!cloud || cloud->empty()) {
    result.reason = "empty cloud";
    return false;
  }

  const Eigen::Vector2f anchor_base = currentAnchorBase(base_to_odom);
  const auto candidates = extractCandidates(cloud, anchor_base, base_to_odom);
  if (candidates.empty()) {
    ++lost_count_;
    result.lost_count = lost_count_;
    result.reason = "no valid candidates";
    if (state_ != TrackState::ACQUIRE) {
      state_ = TrackState::LOST;
    }
    if (lost_count_ >= params_.max_lost_frames) {
      state_ = TrackState::ACQUIRE;
      has_pending_relock_ = false;
      relock_count_ = 0;
    }
    result.state = stateName(state_);
    return false;
  }

  if (state_ == TrackState::ACQUIRE) {
    return updateAcquire(chooseAcquireCandidate(candidates), result);
  }

  Candidate locked_candidate = chooseLockedCandidate(candidates);
  if (!locked_candidate.valid) {
    ++lost_count_;
    result.lost_count = lost_count_;
    result.reason = "no candidate passed lock gate";
    state_ = TrackState::LOST;
    if (lost_count_ >= params_.max_lost_frames) {
      state_ = TrackState::ACQUIRE;
      has_pending_relock_ = false;
      relock_count_ = 0;
    }
    result.state = stateName(state_);
    return false;
  }

  if (state_ == TrackState::LOST) {
    return updateLost(locked_candidate, result);
  }

  lost_count_ = 0;
  relock_count_ = 0;
  has_pending_relock_ = false;
  fillResult(locked_candidate, result);
  result.valid = true;
  result.locked = true;
  result.state = "LOCKED";
  return true;
}

std::vector<TargetSelector::Candidate> TargetSelector::extractCandidates(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
    const Eigen::Vector2f &anchor_base,
    const Eigen::Affine2f &base_to_odom) const {
  auto tree = std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
  tree->setInputCloud(cloud);

  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(params_.cluster_tolerance);
  ec.setMinClusterSize(params_.min_cluster_size);
  ec.setMaxClusterSize(params_.max_cluster_size);
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud);

  std::vector<pcl::PointIndices> cluster_indices;
  ec.extract(cluster_indices);

  std::vector<Candidate> candidates;
  candidates.reserve(cluster_indices.size());
  for (std::size_t i = 0; i < cluster_indices.size(); ++i) {
    auto cluster_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    cluster_cloud->points.reserve(cluster_indices[i].indices.size());
    for (const int index : cluster_indices[i].indices) {
      cluster_cloud->points.push_back(cloud->points[index]);
    }
    cluster_cloud->width =
        static_cast<std::uint32_t>(cluster_cloud->points.size());
    cluster_cloud->height = 1;
    cluster_cloud->is_dense = false;

    Candidate candidate =
        makeCandidate(i, cluster_cloud, anchor_base, base_to_odom);
    if (candidate.valid) {
      candidates.push_back(candidate);
    }
  }
  return candidates;
}

TargetSelector::Candidate TargetSelector::makeCandidate(
    std::size_t cluster_id, const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
    const Eigen::Vector2f &anchor_base,
    const Eigen::Affine2f &base_to_odom) const {
  Candidate candidate;
  candidate.cluster_id = cluster_id;
  candidate.cloud = cloud;

  if (!cloud || cloud->points.size() < 3) {
    return candidate;
  }

  float xmin = std::numeric_limits<float>::max();
  float xmax = -std::numeric_limits<float>::max();
  float ymin = std::numeric_limits<float>::max();
  float ymax = -std::numeric_limits<float>::max();
  for (const auto &pt : cloud->points) {
    xmin = std::min(xmin, pt.x);
    xmax = std::max(xmax, pt.x);
    ymin = std::min(ymin, pt.y);
    ymax = std::max(ymax, pt.y);
  }
  candidate.area = std::max(0.0F, (xmax - xmin) * (ymax - ymin));
  if (candidate.area < params_.min_cluster_area) {
    return candidate;
  }

  const auto points = toPoints2D(cloud);
  const LShapeFit fit = fitLShape(points, anchor_base, params_);
  if (!fit.valid || fit.metrics.fill_ratio < params_.min_fill_ratio) {
    return candidate;
  }

  TargetEdge best_edge;
  EdgeProjection best_projection;
  bool has_edge = false;
  float best_cost = std::numeric_limits<float>::max();
  for (const auto &edge : fit.supported_edges) {
    const EdgeProjection projection =
        intersectAnchorRayWithTargetEdge(edge, anchor_base,
                                         params_.ray_segment_tolerance);
    if (!projection.valid) {
      continue;
    }

    TargetFitMetrics metrics = fit.metrics;
    computeTargetEdgeSupport(points, edge, params_.edge_threshold, metrics);
    if (metrics.target_edge_support_ratio <
        params_.min_target_edge_support_ratio) {
      continue;
    }

    const float cost = projection.ray_x;
    if (cost < best_cost) {
      best_cost = cost;
      best_edge = edge;
      best_projection = projection;
      candidate.metrics = metrics;
      has_edge = true;
    }
  }

  if (!has_edge) {
    return candidate;
  }

  candidate.valid = true;
  candidate.obb = fit.obb;
  candidate.edge = best_edge;
  candidate.projection = best_projection;
  candidate.hit_base = best_projection.projected;
  candidate.hit_odom = base_to_odom * candidate.hit_base;
  candidate.edge.target_center = candidate.hit_base;
  candidate.normal_odom =
      normalizedOr(base_to_odom.linear() * candidate.edge.normal_axis,
                   Eigen::Vector2f(1.0F, 0.0F));
  candidate.yaw =
      std::atan2(candidate.normal_odom.y(), candidate.normal_odom.x());
  return candidate;
}

TargetSelector::Candidate TargetSelector::chooseAcquireCandidate(
    const std::vector<Candidate> &candidates) const {
  Candidate best;
  float best_cost = std::numeric_limits<float>::max();
  for (const auto &candidate : candidates) {
    const float cost = candidate.projection.ray_x -
                       params_.score_support_weight *
                           candidate.metrics.target_edge_support_ratio;
    if (cost < best_cost) {
      best_cost = cost;
      best = candidate;
      best.score = cost;
    }
  }
  return best;
}

TargetSelector::Candidate TargetSelector::chooseLockedCandidate(
    const std::vector<Candidate> &candidates) const {
  Candidate best;
  float best_score = std::numeric_limits<float>::max();
  for (auto candidate : candidates) {
    const float hit_dist = (candidate.hit_odom - locked_hit_odom_).norm();
    if (hit_dist > params_.lock_hit_gate) {
      continue;
    }

    const float yaw_diff = angleDiff(candidate.yaw, locked_yaw_);
    if (yaw_diff > params_.lock_yaw_gate) {
      continue;
    }

    const float normal_dot = candidate.normal_odom.dot(locked_normal_odom_);
    if (normal_dot < params_.lock_min_normal_dot) {
      continue;
    }

    const float length_ref = std::max(locked_length_, 1e-3F);
    const float length_ratio = candidate.edge.target_length / length_ref;
    if (length_ratio < params_.lock_min_length_ratio ||
        length_ratio > params_.lock_max_length_ratio) {
      continue;
    }

    const float area_ref = std::max(locked_area_, 1e-3F);
    const float area_ratio = candidate.area / area_ref;
    if (area_ratio < params_.lock_min_area_ratio ||
        area_ratio > params_.lock_max_area_ratio) {
      continue;
    }

    candidate.score =
        params_.score_hit_weight * hit_dist +
        params_.score_yaw_weight * yaw_diff +
        params_.score_length_weight *
            ratioCost(candidate.edge.target_length, length_ref) +
        params_.score_area_weight * ratioCost(candidate.area, area_ref) -
        params_.score_support_weight *
            candidate.metrics.target_edge_support_ratio;

    if (candidate.score < best_score) {
      best_score = candidate.score;
      best = candidate;
    }
  }
  return best;
}

void TargetSelector::lockTo(const Candidate &candidate,
                            TargetSelectorResult &result) {
  locked_hit_odom_ = candidate.hit_odom;
  locked_normal_odom_ = candidate.normal_odom;
  locked_yaw_ = candidate.yaw;
  locked_length_ = candidate.edge.target_length;
  locked_area_ = candidate.area;
  lost_count_ = 0;
  relock_count_ = 0;
  acquire_count_ = 0;
  has_pending_acquire_ = false;
  has_pending_relock_ = false;
  state_ = TrackState::LOCKED;

  fillResult(candidate, result);
  result.valid = true;
  result.locked = true;
  result.newly_locked = true;
  result.state = "LOCKED";
}

void TargetSelector::fillResult(const Candidate &candidate,
                                TargetSelectorResult &result) const {
  result.valid = candidate.valid;
  result.cluster_id = candidate.cluster_id;
  result.selected_cloud = candidate.cloud;
  result.obb = candidate.obb;
  result.edge = candidate.edge;
  result.metrics = candidate.metrics;
  result.hit_base = candidate.hit_base;
  result.hit_odom = candidate.hit_odom;
  result.area = candidate.area;
  result.score = candidate.score;
  result.lost_count = lost_count_;
}

Eigen::Vector2f
TargetSelector::currentAnchorBase(const Eigen::Affine2f &base_to_odom) const {
  if (state_ == TrackState::ACQUIRE) {
    return Eigen::Vector2f(0.0F, 0.0F);
  }
  return base_to_odom.inverse() * locked_hit_odom_;
}

bool TargetSelector::updateAcquire(const Candidate &candidate,
                                   TargetSelectorResult &result) {
  if (!candidate.valid) {
    result.reason = "no acquire candidate";
    return false;
  }

  if (!has_pending_acquire_) {
    pending_acquire_ = candidate;
    acquire_count_ = 1;
    has_pending_acquire_ = true;
  } else {
    const float hit_dist =
        (candidate.hit_odom - pending_acquire_.hit_odom).norm();
    const float yaw_diff = angleDiff(candidate.yaw, pending_acquire_.yaw);
    if (hit_dist <= params_.acquire_hit_gate &&
        yaw_diff <= params_.acquire_yaw_gate) {
      pending_acquire_ = candidate;
      ++acquire_count_;
    } else {
      pending_acquire_ = candidate;
      acquire_count_ = 1;
    }
  }

  if (acquire_count_ >= params_.acquire_confirm_frames) {
    lockTo(candidate, result);
    return true;
  }

  std::ostringstream reason;
  reason << "acquiring " << acquire_count_ << "/"
         << params_.acquire_confirm_frames;
  result.reason = reason.str();
  return false;
}

bool TargetSelector::updateLost(const Candidate &candidate,
                                TargetSelectorResult &result) {
  if (!candidate.valid) {
    result.reason = "no relock candidate";
    return false;
  }

  if (!has_pending_relock_) {
    pending_relock_ = candidate;
    relock_count_ = 1;
    has_pending_relock_ = true;
  } else {
    const float hit_dist =
        (candidate.hit_odom - pending_relock_.hit_odom).norm();
    const float yaw_diff = angleDiff(candidate.yaw, pending_relock_.yaw);
    if (hit_dist <= params_.acquire_hit_gate &&
        yaw_diff <= params_.acquire_yaw_gate) {
      pending_relock_ = candidate;
      ++relock_count_;
    } else {
      pending_relock_ = candidate;
      relock_count_ = 1;
    }
  }

  if (relock_count_ >= params_.relock_confirm_frames) {
    lockTo(candidate, result);
    result.newly_locked = false;
    return true;
  }

  std::ostringstream reason;
  reason << "relocking " << relock_count_ << "/"
         << params_.relock_confirm_frames;
  result.reason = reason.str();
  return false;
}
