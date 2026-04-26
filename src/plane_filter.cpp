#include "close_approach/plane_filter.hpp"

OBB PlaneFilter::compute_OBB(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  OBB obb;
  if (cloud->empty()) {
    obb.length1 = 0;
    obb.length2 = 0;
    return obb;
  }

  // 1. Convert PCL cloud to OpenCV 2D points (ignoring Z)
  std::vector<cv::Point2f> cv_points;
  cv_points.reserve(cloud->points.size());
  for (const auto &pt : cloud->points) {
    cv_points.emplace_back(pt.x, pt.y);
  }

  // 2. Compute the Minimum Area Bounding Rectangle
  cv::RotatedRect rect = cv::minAreaRect(cv_points);

  obb.center[0] = rect.center.x;
  obb.center[1] = rect.center.y;

  // 3. Extract 4 corners of the rotated rectangle
  cv::Point2f corners[4];
  rect.points(corners);

  // 4. Determine axes and lengths
  // corners are typically ordered clockwise or counter-clockwise
  cv::Point2f diff1 = corners[1] - corners[0];
  cv::Point2f diff2 = corners[2] - corners[1];

  float len1 = cv::norm(diff1);
  float len2 = cv::norm(diff2);

  obb.length1 = len1;
  obb.length2 = len2;

  if (len1 > 1e-5f) {
    obb.axis1[0] = diff1.x / len1;
    obb.axis1[1] = diff1.y / len1;
  } else {
    obb.axis1 = Eigen::Vector2f(1.0f, 0.0f);
  }

  if (len2 > 1e-5f) {
    obb.axis2[0] = diff2.x / len2;
    obb.axis2[1] = diff2.y / len2;
  } else {
    obb.axis2 = Eigen::Vector2f(0.0f, 1.0f);
  }

  return obb;
}

OBB3D PlaneFilter::compute_3D_OBB(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  OBB3D obb;
  if (cloud->empty()) {
    obb.length_x = obb.length_y = obb.length_z = 0;
    return obb;
  }

  pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
  feature_extractor.setInputCloud(cloud);
  feature_extractor.compute();

  pcl::PointXYZ min_point_OBB;
  pcl::PointXYZ max_point_OBB;
  pcl::PointXYZ position_OBB;
  Eigen::Matrix3f rotational_matrix_OBB;

  feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB,
                           rotational_matrix_OBB);

  obb.center = position_OBB.getVector3fMap();
  obb.rotation = rotational_matrix_OBB;

  // OBB 3D frame 상에서의 극단점 차이로 각 축방향 길이(Dimensions)를 정확히
  // 계산합니다
  obb.length_x = max_point_OBB.x - min_point_OBB.x;
  obb.length_y = max_point_OBB.y - min_point_OBB.y;
  obb.length_z = max_point_OBB.z - min_point_OBB.z;

  return obb;
}