#pragma once
#include <Eigen/Dense>

struct TargetEdge
{
  Eigen::Vector2f target_axis;
  Eigen::Vector2f normal_axis;
  Eigen::Vector2f target_center;
  float target_length;
};

class EdgeExtractor
{
public:
  EdgeExtractor() = default;

  TargetEdge extract_edges(const Eigen::Vector2f &center,
    const Eigen::Vector2f &axis1,
    const Eigen::Vector2f &axis2,
    float length1,
    float length2);

};
