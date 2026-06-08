#pragma once

#include "close_approach/edge_extractor.hpp"
#include "close_approach/plane_filter.hpp"

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

struct TargetSelectorParams {
  float cluster_tolerance = 0.03F;
  int min_cluster_size = 100;
  int max_cluster_size = 10000;
  float min_cluster_area = 0.09F;

  float edge_threshold = 0.03F;
  float l_shape_angle_step_deg = 1.0F;
  float l_shape_sigma = 0.03F;
  float ray_segment_tolerance = 0.03F;
  float min_target_edge_support_ratio = 0.02F;
  float min_fill_ratio = 0.50F;

  float circle_rmse_threshold = 0.04F;
  float circle_min_radius = 0.15F;
  float circle_max_radius = 0.8F;

  int acquire_confirm_frames = 3;
  int relock_confirm_frames = 3;
  int max_lost_frames = 8;
  float acquire_hit_gate = 0.15F;
  float acquire_yaw_gate = 0.17F;

  float lock_hit_gate = 0.30F;
  float lock_yaw_gate = 0.35F;
  float lock_min_length_ratio = 0.50F;
  float lock_max_length_ratio = 2.00F;
  float lock_min_area_ratio = 0.40F;
  float lock_max_area_ratio = 2.50F;
  float lock_min_normal_dot = 0.70F;

  float score_hit_weight = 2.0F;
  float score_yaw_weight = 1.0F;
  float score_length_weight = 0.5F;
  float score_area_weight = 0.3F;
  float score_support_weight = 0.5F;
};

struct TargetFitMetrics {
  float rect_area = 0.0F;
  float hull_area = 0.0F;
  float fill_ratio = 0.0F;
  float mean_edge_dist = std::numeric_limits<float>::max();
  float support_ratio = 0.0F;
  float target_edge_mean_dist = std::numeric_limits<float>::max();
  float target_edge_support_ratio = 0.0F;
};

struct TargetSelectorResult {
  bool valid = false;
  bool newly_locked = false;
  bool locked = false;
  bool is_round = false;
  int lost_count = 0;
  std::string state;
  std::string reason;

  std::size_t cluster_id = 0;
  pcl::PointCloud<pcl::PointXYZ>::Ptr selected_cloud;
  OBB obb;
  TargetEdge edge;
  TargetFitMetrics metrics;
  Eigen::Vector2f anchor_base{0.0F, 0.0F};
  Eigen::Vector2f hit_base{0.0F, 0.0F};
  Eigen::Vector2f hit_odom{0.0F, 0.0F};
  float area = 0.0F;
  float score = 0.0F;
};

class TargetSelector {
public:
  enum class TrackState {
    ACQUIRE,
    LOCKED,
    LOST
  };

  TargetSelector() = default;

  void setParameters(const TargetSelectorParams &params);
  void reset();

  bool select(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
              const Eigen::Affine2f &base_to_odom,
              TargetSelectorResult &result);

  TrackState state() const { return state_; }
  const Eigen::Vector2f &lockedHitOdom() const { return locked_hit_odom_; }

  struct EdgeProjection {
    Eigen::Vector2f projected{0.0F, 0.0F};
    float ray_x = std::numeric_limits<float>::max();
    float segment_penalty = std::numeric_limits<float>::max();
    bool valid = false;
  };

private:
  struct Candidate {
    bool valid = false;
    bool is_round = false;
    std::size_t cluster_id = 0;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    OBB obb;
    TargetEdge edge;
    TargetFitMetrics metrics;
    EdgeProjection projection;
    Eigen::Vector2f hit_base{0.0F, 0.0F};
    Eigen::Vector2f hit_odom{0.0F, 0.0F};
    Eigen::Vector2f normal_odom{1.0F, 0.0F};
    float area = 0.0F;
    float yaw = 0.0F;
    float score = std::numeric_limits<float>::max();
  };

  std::vector<Candidate>
  extractCandidates(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                    const Eigen::Vector2f &anchor_base,
                    const Eigen::Affine2f &base_to_odom) const;
  Candidate makeCandidate(std::size_t cluster_id,
                          const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                          const Eigen::Vector2f &anchor_base,
                          const Eigen::Affine2f &base_to_odom) const;
  Candidate chooseAcquireCandidate(const std::vector<Candidate> &candidates) const;
  Candidate chooseLockedCandidate(const std::vector<Candidate> &candidates) const;

  void lockTo(const Candidate &candidate, const Eigen::Affine2f &base_to_odom, TargetSelectorResult &result);
  void fillResult(const Candidate &candidate, TargetSelectorResult &result) const;
  void fillResultFromLocked(const Eigen::Affine2f &base_to_odom, TargetSelectorResult &result) const;
  bool handleLostFrame(const Eigen::Affine2f &base_to_odom, TargetSelectorResult &result);
  void resetAcquire();
  Eigen::Vector2f currentAnchorBase(const Eigen::Affine2f &base_to_odom) const;
  bool updateAcquire(const Candidate &candidate, const Eigen::Affine2f &base_to_odom, TargetSelectorResult &result);
  bool updateLost(const Candidate &candidate, TargetSelectorResult &result);

  TargetSelectorParams params_;
  TrackState state_ = TrackState::ACQUIRE;
  int acquire_count_ = 0;
  int relock_count_ = 0;
  int lost_count_ = 0;
  bool has_pending_acquire_ = false;
  bool has_pending_relock_ = false;
  Candidate pending_acquire_;
  Candidate pending_relock_;

  Eigen::Vector2f locked_hit_odom_{0.0F, 0.0F};
  Eigen::Vector2f locked_normal_odom_{1.0F, 0.0F};
  float locked_yaw_ = 0.0F;
  float locked_length_ = 0.0F;
  float locked_area_ = 0.0F;

  Eigen::Vector2f locked_obb_center_odom_{0.0F, 0.0F};
  Eigen::Vector2f locked_obb_axis1_odom_{1.0F, 0.0F};
  Eigen::Vector2f locked_obb_axis2_odom_{0.0F, 1.0F};
  float locked_obb_length1_ = 0.0F;
  float locked_obb_length2_ = 0.0F;
};
