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

  /*
  PCA 가 normal_axis 부호를 ±로 들쭉날쭉 뱉어서 e_theta 가 매 프레임 부호
  플리핑 → PID 회전 명령이 진동해서 yaw 수렴이 망가짐. 클러스터에서 로봇 쪽
  (원점 -> target_center 의 반대 방향) 으로 향하도록 부호 고정.
  target_center 는 항상 로봇 앞쪽 (+x 쪽)에 있으므로, normal_axis 가
  target_center 와 같은 방향이면 (dot > 0) "안쪽(테이블 너머)" 향함, 아니면
  로봇쪽. 클러스터를 마주보는 정렬을 위해 안쪽 방향을 일관 채택.
  */
  if (normal_axis.dot(target_center) < 0.0f)
  {
    normal_axis = -normal_axis;
  }

  return {target_axis, normal_axis, target_center, target_length};
}