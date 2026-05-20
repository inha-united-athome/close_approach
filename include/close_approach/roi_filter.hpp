#pragma once

#include <geometry_msgs/msg/transform.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Dense>

class Filter {
public:
  Filter();

  void setParameters(float leaf_size, int mean_k, float stddev_mul_thresh,
                     float ground_height, float cluster_tolerance,
                     int min_cluster_size, int max_cluster_size);
  void setCameraInfo(float fx, float fy, float cx, float cy);
  void roi_filter(
      const sensor_msgs::msg::PointCloud2::ConstSharedPtr &pointcloud_msg,
      const vision_msgs::msg::Detection2DArray::ConstSharedPtr &detection_msg,
      pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
  void roi_filter(
      const sensor_msgs::msg::PointCloud2::ConstSharedPtr &pointcloud_msg,
      const vision_msgs::msg::Detection2DArray::ConstSharedPtr &detection_msg,
      pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, int target_idx);
  void voxel_downsampling(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
  void remove_outliers(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
  void remove_ground(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                     const geometry_msgs::msg::Transform &tf);
  /*
  Footprint (2D AABB xy) 면적이 min_cluster_area 이상인 클러스터들 중,
  reference point(anchor_xy or 로봇 원점)에 centroid가 가장 가까운 것을 선택.
  anchor_xy == nullptr 이면 base 원점(0,0) 기준.
  */
  void cluster_points(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                      pcl::search::KdTree<pcl::PointXYZ>::Ptr &kdtree,
                      float min_cluster_area,
                      const Eigen::Vector2f *anchor_xy);

  void projection_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);

  void front_slicing(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
  void front_slicing_quantile(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, float quantile = 0.8f);
  void front_slicing_density(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, int num_bins = 20, float density_drop_threshold = 0.5f);
private:
  float leaf_size_ = 0.02F;
  int mean_k_ = 50;
  float stddev_mul_thresh_ = 1.0F;
  float ground_height_ = 0.0F;

  float cluster_tolerance_ = 0.02F; // 2cm distance
  int min_cluster_size_ = 20;
  int max_cluster_size_ = 4000;

  float fx_ = 0.0F;
  float fy_ = 0.0F;
  float cx_ = 0.0F;
  float cy_ = 0.0F;
};
