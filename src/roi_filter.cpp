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
                            pcl::search::KdTree<pcl::PointXYZ>::Ptr &kdtree,
                            float min_cluster_area,
                            const Eigen::Vector2f *anchor_xy) {
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

  if (cluster_indices.empty()) {
    RCLCPP_WARN(rclcpp::get_logger("Filter"),
                "No clusters found! Check min_cluster_size (%d) and "
                "cluster_tolerance (%.3f)",
                min_cluster_size_, cluster_tolerance_);
    cloud = clustered_cloud;
    return;
  }

  /*
  각 클러스터의 centroid 와 2D AABB footprint area 계산.
  footprint_area >= min_cluster_area 인 후보만 남기고, 그 중 reference 점
  (anchor_xy if available, 없으면 로봇 원점)에 centroid 가 가장 가까운 것 선택.
  → 가까이서 시작한 작은 박스/사람도 잡고, 뒷벽 swap 방지.
  */
  struct Cand {
    std::size_t idx;
    Eigen::Vector2f centroid;
    float area;
  };
  std::vector<Cand> cands;
  cands.reserve(cluster_indices.size());

  float max_seen_area = 0.0F;
  for (std::size_t i = 0; i < cluster_indices.size(); ++i) {
    const auto &ci = cluster_indices[i];
    float sum_x = 0.0F, sum_y = 0.0F;
    float xmin = std::numeric_limits<float>::max();
    float xmax = -std::numeric_limits<float>::max();
    float ymin = std::numeric_limits<float>::max();
    float ymax = -std::numeric_limits<float>::max();
    for (const auto &id : ci.indices) {
      const auto &pt = cloud->points[id];
      sum_x += pt.x;
      sum_y += pt.y;
      xmin = std::min(xmin, pt.x);
      xmax = std::max(xmax, pt.x);
      ymin = std::min(ymin, pt.y);
      ymax = std::max(ymax, pt.y);
    }
    const float n = static_cast<float>(ci.indices.size());
    Cand c;
    c.idx = i;
    c.centroid = Eigen::Vector2f(sum_x / n, sum_y / n);
    c.area = (xmax - xmin) * (ymax - ymin);
    max_seen_area = std::max(max_seen_area, c.area);
    cands.push_back(c);
  }

  const Eigen::Vector2f ref =
      anchor_xy ? *anchor_xy : Eigen::Vector2f(0.0F, 0.0F);

  const Cand *best = nullptr;
  float best_d2 = std::numeric_limits<float>::max();
  for (const auto &c : cands) {
    if (c.area < min_cluster_area) continue;
    const float d2 = (c.centroid - ref).squaredNorm();
    if (d2 < best_d2) {
      best_d2 = d2;
      best = &c;
    }
  }

  if (!best) {
    RCLCPP_WARN(rclcpp::get_logger("Filter"),
                "No cluster meets min_area=%.3f (max seen=%.3f, n=%zu)",
                min_cluster_area, max_seen_area, cands.size());
    cloud = clustered_cloud;
    return;
  }

  const auto &chosen = cluster_indices[best->idx];
  for (const auto &id : chosen.indices) {
    clustered_cloud->points.push_back(cloud->points[id]);
  }
  RCLCPP_INFO(rclcpp::get_logger("Filter"),
              "Selected cluster: pts=%zu area=%.3f dist=%.3f%s",
              chosen.indices.size(), best->area, std::sqrt(best_d2),
              anchor_xy ? " (vs anchor)" : " (vs origin)");

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
