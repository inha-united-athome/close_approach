#include "close_approach/edge_extractor.hpp"

TargetEdge EdgeExtractor::extract_edges(const Eigen::Vector2f &center,
    const Eigen::Vector2f &axis1,
    const Eigen::Vector2f &axis2,
    float length1,
    float length2) 
{
  bool axis1_is_horizontal;
  
  Eigen::Vector2f p_A, p_B;
  Eigen::Vector2f target_axis;
  Eigen::Vector2f normal_axis;
  float target_length;
  float dist_A, dist_B;
  Eigen::Vector2f target_center;
  TargetEdge target_edge;
    
  axis1_is_horizontal = std::abs(axis1.y()) > std::abs(axis2.y());
  if (axis1_is_horizontal)
  {
    p_A = center + (length2/2.0f) * axis2;
    p_B = center - (length2/2.0f) * axis2;
    target_axis = axis1;
    normal_axis = axis2;
    target_length = length1;
  }
  else
  {
    p_A = center + (length1/2.0f) * axis1;
    p_B = center - (length1/2.0f) * axis1;
    target_axis = axis2;
    normal_axis = axis1;
    target_length = length2;
  }

  dist_A = p_A.norm();
  dist_B = p_B.norm();

  if (dist_A > dist_B)
  {
    target_center = p_B;
  }
  else
  {
    target_center = p_A;
  }

  return {target_axis, normal_axis, target_center, target_length};
}