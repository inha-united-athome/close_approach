#include "close_approach/roi_filter.hpp"

#include <Eigen/Geometry>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include <algorithm>
#include <limits>
#include <vector>

Filter::Filter() = default;

void Filter::setParameters(float leaf_size, int mean_k, float stddev_mul_thresh,
                           float ground_height, float cluster_tolerance,
                           int min_cluster_size, int max_cluster_size) {
  leaf_size_ = leaf_size;
  mean_k_ = mean_k;
  stddev_mul_thresh_ = stddev_mul_thresh;
  ground_height_ = ground_height;
  cluster_tolerance_ = cluster_tolerance;
  min_cluster_size_ = min_cluster_size;
  max_cluster_size_ = max_cluster_size;

  RCLCPP_INFO(rclcpp::get_logger("Filter"),
              "Filter parameters set: leaf_size=%.2f, mean_k=%d, "
              "stddev_mul_thresh=%.2f, ground_height=%.2f, "
              "cluster_tolerance=%.2f, min_cluster_size=%d, "
              "max_cluster_size=%d",
              leaf_size_, mean_k_, stddev_mul_thresh_, ground_height_,
              cluster_tolerance_, min_cluster_size_, max_cluster_size_);
}

void Filter::setCameraInfo(float fx, float fy, float cx, float cy) {
  fx_ = fx;
  fy_ = fy;
  cx_ = cx;
  cy_ = cy;

  RCLCPP_INFO(
      rclcpp::get_logger("Filter"),
      "Camera intrinsic parameters set: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f",
      fx_, fy_, cx_, cy_);
}

void Filter::roi_filter(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr &pointcloud_msg,
    const vision_msgs::msg::Detection2DArray::ConstSharedPtr &detection_msg,
    pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  auto source_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  pcl::fromROSMsg(*pointcloud_msg, *source_cloud);

  auto roi_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

  RCLCPP_INFO(rclcpp::get_logger("Filter"),
              "Starting ROI filtering with %zu detections.",
              detection_msg->detections.size());

  for (const auto &detection : detection_msg->detections) {
    const auto &bbox = detection.bbox;
    const float x_min =
        static_cast<float>(bbox.center.position.x - bbox.size_x / 2.0);
    const float x_max =
        static_cast<float>(bbox.center.position.x + bbox.size_x / 2.0);
    const float y_min =
        static_cast<float>(bbox.center.position.y - bbox.size_y / 2.0);
    const float y_max =
        static_cast<float>(bbox.center.position.y + bbox.size_y / 2.0);

    for (const auto &point : source_cloud->points) {
      if (point.z <= 0.0F) {
        continue;
      }

      const float u = (point.x * fx_) / point.z + cx_;
      const float v = (point.y * fy_) / point.z + cy_;

      if (u >= x_min && u <= x_max && v >= y_min && v <= y_max) {
        roi_cloud->points.push_back(point);
      }
    }
  }

  roi_cloud->width = static_cast<std::uint32_t>(roi_cloud->points.size());
  roi_cloud->height = 1;
  roi_cloud->is_dense = false;
  cloud = roi_cloud;
}

void Filter::roi_filter(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr &pointcloud_msg,
    const vision_msgs::msg::Detection2DArray::ConstSharedPtr &detection_msg,
    pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, int target_idx) {
  
  if (target_idx < 0 || target_idx >= detection_msg->detections.size()) {
    RCLCPP_WARN(rclcpp::get_logger("Filter"), "Invalid target index: %d", target_idx);
    return;
  }

  auto source_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  pcl::fromROSMsg(*pointcloud_msg, *source_cloud);

  auto roi_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

  const auto &detection = detection_msg->detections[target_idx];
  const auto &bbox = detection.bbox;
  const float x_min = static_cast<float>(bbox.center.position.x - bbox.size_x / 2.0);
  const float x_max = static_cast<float>(bbox.center.position.x + bbox.size_x / 2.0);
  const float y_min = static_cast<float>(bbox.center.position.y - bbox.size_y / 2.0);
  const float y_max = static_cast<float>(bbox.center.position.y + bbox.size_y / 2.0);

  for (const auto &point : source_cloud->points) {
    if (point.z <= 0.0F) {
      continue;
    }

    const float u = (point.x * fx_) / point.z + cx_;
    const float v = (point.y * fy_) / point.z + cy_;

    if (u >= x_min && u <= x_max && v >= y_min && v <= y_max) {
      roi_cloud->points.push_back(point);
    }
  }

  roi_cloud->width = static_cast<std::uint32_t>(roi_cloud->points.size());
  roi_cloud->height = 1;
  roi_cloud->is_dense = false;
  cloud = roi_cloud;
}

void Filter::voxel_downsampling(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
  voxel_grid.setInputCloud(cloud);
  voxel_grid.setLeafSize(leaf_size_, leaf_size_, leaf_size_);

  auto downsampled_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  voxel_grid.filter(*downsampled_cloud);
  cloud = downsampled_cloud;
}

void Filter::remove_outliers(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
  sor.setInputCloud(cloud);
  sor.setMeanK(mean_k_);
  sor.setStddevMulThresh(stddev_mul_thresh_);

  auto filtered_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  sor.filter(*filtered_cloud);
  cloud = filtered_cloud;
}

void Filter::remove_ground(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                           const geometry_msgs::msg::Transform &tf) {
  Eigen::Affine3f transform = Eigen::Affine3f::Identity();
  transform.translation() << tf.translation.x, tf.translation.y,
      tf.translation.z;

  const Eigen::Quaternionf rotation(
      static_cast<float>(tf.rotation.w), static_cast<float>(tf.rotation.x),
      static_cast<float>(tf.rotation.y), static_cast<float>(tf.rotation.z));
  transform.linear() = rotation.toRotationMatrix();

  auto transformed_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  pcl::transformPointCloud(*cloud, *transformed_cloud, transform);

  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud(transformed_cloud);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(ground_height_, std::numeric_limits<float>::max());

  auto ground_removed_cloud =
      std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  pass.filter(*ground_removed_cloud);
  cloud = ground_removed_cloud;
}

void Filter::cluster_points(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                            pcl::search::KdTree<pcl::PointXYZ>::Ptr &kdtree) {
  if (cloud->empty()) {
    RCLCPP_WARN(rclcpp::get_logger("Filter"),
                "Input cloud for clustering is empty!");
    return;
  }

  RCLCPP_INFO(rclcpp::get_logger("Filter"), "Before clustering: %zu points",
              cloud->points.size());

  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(cluster_tolerance_);
  ec.setMinClusterSize(min_cluster_size_);
  ec.setMaxClusterSize(max_cluster_size_);
  ec.setSearchMethod(kdtree);
  ec.setInputCloud(cloud);

  std::vector<pcl::PointIndices> cluster_indices;
  ec.extract(cluster_indices);

  RCLCPP_INFO(rclcpp::get_logger("Filter"), "Found %zu clusters",
              cluster_indices.size());

  auto clustered_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

  if (!cluster_indices.empty()) {
    // 로봇 전방(x축)과 가장 정렬된 클러스터 선택: centroid |y|가 최소인 것
    auto best_it = cluster_indices.begin();
    float best_abs_y = std::numeric_limits<float>::max();
    for (auto it = cluster_indices.begin(); it != cluster_indices.end(); ++it) {
      float sum_y = 0.0F;
      for (const auto &idx : it->indices) {
        sum_y += cloud->points[idx].y;
      }
      const float abs_y = std::abs(sum_y / static_cast<float>(it->indices.size()));
      if (abs_y < best_abs_y) {
        best_abs_y = abs_y;
        best_it = it;
      }
    }
    for (const auto &index : best_it->indices) {
      clustered_cloud->points.push_back(cloud->points[index]);
    }
    RCLCPP_INFO(rclcpp::get_logger("Filter"),
                "Selected cluster: %zu points, centroid |y|=%.3f",
                best_it->indices.size(), best_abs_y);
  } else {
    RCLCPP_WARN(rclcpp::get_logger("Filter"),
                "No clusters found! Check min_cluster_size (%d) and "
                "cluster_tolerance (%.3f)",
                min_cluster_size_, cluster_tolerance_);
  }

  cloud = clustered_cloud;
}

void Filter::projection_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr projection_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);

  for (size_t i = 0; i < cloud->points.size(); i++) {
    pcl::PointXYZ point;
    point.x = cloud->points[i].x;
    point.y = cloud->points[i].y;
    point.z = 0;
    projection_cloud->points.push_back(point);
  }

  cloud = projection_cloud;
}

void Filter::front_slicing(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  if (cloud->points.empty())
    return;
  // 1. 기준 축(여기선 X축)의 모든 값 모으기
  std::vector<float> x_values;
  x_values.reserve(cloud->points.size());
  for (const auto &pt : cloud->points) {
    x_values.push_back(pt.x);
  }
  // 2. 중앙값(Median) 계산
  size_t n = x_values.size() / 2;
  std::nth_element(x_values.begin(), x_values.begin() + n, x_values.end());
  float median_x = x_values[n];

  // 3. 중앙값 기준으로 '로봇과 가까운 쪽 절반'만 남기기
  pcl::PointCloud<pcl::PointXYZ>::Ptr sliced_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  sliced_cloud->points.reserve(cloud->points.size() / 2);
  for (size_t i = 0; i < cloud->points.size(); i++) {
    // 일반적인 로봇 베이스 좌표계에서는 X축이 전방(깊이) 방향입니다.
    // 로봇에 가까운 쪽이 X가 작은 쪽이라고 가정 (x <= median_x)
    // (로봇 세팅에 따라 반대면 x >= median_x 로 부호 뒤집기)
    if (cloud->points[i].x <= median_x) {
      pcl::PointXYZ point;
      point.x = cloud->points[i].x;
      point.y = cloud->points[i].y;
      point.z = cloud->points[i].z;
      sliced_cloud->points.push_back(point);
    }
  }
  cloud = sliced_cloud;
}

void Filter::front_slicing_quantile(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, float quantile) {
  if (cloud->points.empty()) return;

  std::vector<float> x_values;
  x_values.reserve(cloud->points.size());
  for (const auto &pt : cloud->points) {
    x_values.push_back(pt.x);
  }

  size_t n = static_cast<size_t>(x_values.size() * quantile);
  if (n >= x_values.size()) n = x_values.size() - 1;

  std::nth_element(x_values.begin(), x_values.begin() + n, x_values.end());
  float cutoff_x = x_values[n];

  pcl::PointCloud<pcl::PointXYZ>::Ptr sliced_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  sliced_cloud->points.reserve(cloud->points.size());
  for (const auto &pt : cloud->points) {
    if (pt.x <= cutoff_x) {
      sliced_cloud->points.push_back(pt);
    }
  }
  cloud = sliced_cloud;
}

void Filter::front_slicing_density(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, int num_bins, float density_drop_threshold) {
  if (cloud->points.empty()) return;

  float min_x = cloud->points[0].x;
  float max_x = cloud->points[0].x;
  for (const auto &pt : cloud->points) {
    if (pt.x < min_x) min_x = pt.x;
    if (pt.x > max_x) max_x = pt.x;
  }

  if (max_x <= min_x) return;

  std::vector<int> bins(num_bins, 0);
  float bin_size = (max_x - min_x) / num_bins;
  for (const auto &pt : cloud->points) {
    int bin_idx = static_cast<int>((pt.x - min_x) / bin_size);
    if (bin_idx >= num_bins) bin_idx = num_bins - 1;
    bins[bin_idx]++;
  }

  float cutoff_x = max_x;
  for (int i = 1; i < num_bins; i++) {
    if (bins[i - 1] > 0) {
      float drop_ratio = static_cast<float>(bins[i]) / static_cast<float>(bins[i - 1]);
      if (drop_ratio < density_drop_threshold) {
        cutoff_x = min_x + i * bin_size;
        break;
      }
    }
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr sliced_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  sliced_cloud->points.reserve(cloud->points.size());
  for (const auto &pt : cloud->points) {
    if (pt.x <= cutoff_x) {
      sliced_cloud->points.push_back(pt);
    }
  }
  cloud = sliced_cloud;
}
