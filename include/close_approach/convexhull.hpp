#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/surface/convex_hull.h>

class ConvexHull {
public:
  ConvexHull() = default;

  void compute(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
};