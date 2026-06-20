#pragma once

#include <memory>
#include <string>

#include <opencv2/core.hpp>

// Ultralytics YOLO26-sem TensorRT engine wrapper.
// TensorRT headers are intentionally hidden behind PImpl so the rest of the
// package can still build on machines where TensorRT is not installed.
class TrtSemanticSegmenter {
public:
  TrtSemanticSegmenter(const std::string &engine_path,
                       int requested_input_width,
                       int requested_input_height);
  ~TrtSemanticSegmenter();

  TrtSemanticSegmenter(const TrtSemanticSegmenter &) = delete;
  TrtSemanticSegmenter &operator=(const TrtSemanticSegmenter &) = delete;

  bool ready() const;
  const std::string &status() const;
  int inputWidth() const;
  int inputHeight() const;

  // bgr_roi: OpenCV BGR image. class_map: CV_8UC1 Cityscapes train IDs.
  bool infer(const cv::Mat &bgr_roi, cv::Mat &class_map, double &inference_ms);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
