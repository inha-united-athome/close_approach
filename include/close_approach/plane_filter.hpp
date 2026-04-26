#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

struct OBB {
  Eigen::Vector2f center;
  Eigen::Vector2f axis1;
  Eigen::Vector2f axis2;
  float length1;
  float length2;
};

struct OBB3D {
  Eigen::Vector3f center;
  Eigen::Matrix3f rotation; // Columns are the axes
  float length_x;
  float length_y;
  float length_z;
  
  Eigen::Vector3f min_aabb; // For easy IoU calculation
  Eigen::Vector3f max_aabb; // For easy IoU calculation
};

class PlaneFilter {
public:
  PlaneFilter() = default;

  OBB compute_OBB(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
  OBB3D compute_3D_OBB(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);

private:
};
