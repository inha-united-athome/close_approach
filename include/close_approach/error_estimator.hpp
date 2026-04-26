#pragma once

#include "close_approach/edge_extractor.hpp"
#include <Eigen/Dense>

struct SE2Error {
  float x;
  float y;
  float degree_theta;
};

class ErrorEstimator {
public:
  ErrorEstimator() = default;

  SE2Error estimate_error(const TargetEdge &target_edge);
};
