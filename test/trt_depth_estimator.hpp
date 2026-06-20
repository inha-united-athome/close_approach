#pragma once

#include <memory>
#include <string>

#include <opencv2/core.hpp>

// TensorRT wrapper for Depth Anything V2 Small.
class TrtDepthEstimator {
public:
  TrtDepthEstimator(const std::string &engine_path,
                    int requested_input_width,
                    int requested_input_height);
  ~TrtDepthEstimator();

  TrtDepthEstimator(const TrtDepthEstimator &) = delete;
  TrtDepthEstimator &operator=(const TrtDepthEstimator &) = delete;

  bool ready() const;
  const std::string &status() const;
  int inputWidth() const;
  int inputHeight() const;

  // Returns an affine-invariant inverse-depth map (CV_32FC1) at ROI size.
  // Larger values generally indicate closer content; this is not metric depth.
  bool infer(const cv::Mat &bgr_roi, cv::Mat &relative_depth,
             double &inference_ms);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
