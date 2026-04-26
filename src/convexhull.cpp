#include "close_approach/convexhull.hpp"

void ConvexHull::compute(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr new_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::ConvexHull<pcl::PointXYZ> chull;
  chull.setInputCloud(cloud);
  chull.reconstruct(*new_cloud);
  cloud = new_cloud;
}