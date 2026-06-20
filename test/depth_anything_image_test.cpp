#include "trt_depth_estimator.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace {

cv::Mat normalizeDepth(const cv::Mat &depth) {
  double min_value = 0.0, max_value = 0.0;
  cv::minMaxLoc(depth, &min_value, &max_value);
  cv::Mat normalized;
  if (max_value - min_value < 1e-6) {
    normalized = cv::Mat::zeros(depth.size(), CV_32F);
  } else {
    depth.convertTo(normalized, CV_32F, 1.0 / (max_value - min_value),
                    -min_value / (max_value - min_value));
  }
  return normalized;
}

cv::Mat makePanel(const cv::Mat &image, const std::string &label) {
  cv::Mat panel;
  if (image.channels() == 1) cv::cvtColor(image, panel, cv::COLOR_GRAY2BGR);
  else panel = image.clone();
  cv::rectangle(panel, {0, 0}, {panel.cols, 34}, {0, 0, 0}, -1);
  cv::putText(panel, label, {10, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.65,
              {255, 255, 255}, 2, cv::LINE_AA);
  return panel;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "usage: depth_anything_image_test ENGINE IMAGE [OUTPUT]\n";
    return 2;
  }
  const std::string output_path = argc == 4 ? argv[3] : "depth_anything_test.jpg";
  const cv::Mat image = cv::imread(argv[2], cv::IMREAD_COLOR);
  if (image.empty()) {
    std::cerr << "image read failed: " << argv[2] << '\n';
    return 2;
  }

  TrtDepthEstimator estimator(argv[1], 518, 518);
  if (!estimator.ready()) {
    std::cerr << estimator.status() << '\n';
    return 1;
  }

  cv::Mat depth;
  double inference_ms = 0.0;
  if (!estimator.infer(image, depth, inference_ms)) {
    std::cerr << estimator.status() << '\n';
    return 1;
  }

  const cv::Mat depth_norm = normalizeDepth(depth);
  cv::Mat depth_u8, depth_color;
  depth_norm.convertTo(depth_u8, CV_8U, 255.0);
  cv::applyColorMap(depth_u8, depth_color, cv::COLORMAP_INFERNO);

  cv::Mat smooth, grad_x, grad_y, magnitude, grad_u8;
  cv::GaussianBlur(depth_norm, smooth, {5, 5}, 0.0);
  cv::Sobel(smooth, grad_x, CV_32F, 1, 0, 3);
  cv::Sobel(smooth, grad_y, CV_32F, 0, 1, 3);
  cv::magnitude(grad_x, grad_y, magnitude);
  double grad_max = 0.0;
  cv::minMaxLoc(magnitude, nullptr, &grad_max);
  magnitude.convertTo(grad_u8, CV_8U, grad_max > 1e-6 ? 255.0 / grad_max : 0.0);
  cv::Mat depth_only_edge;
  cv::threshold(grad_u8, depth_only_edge, 0.15 * 255.0, 255,
                cv::THRESH_BINARY);

  cv::Mat gray, rgb_edges;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, gray, {5, 5}, 0.0);
  cv::Canny(gray, rgb_edges, 50, 150);

  cv::Mat weighted(rgb_edges.size(), CV_8U, cv::Scalar(0));
  constexpr float near_floor = 0.20f;
  constexpr float near_gamma = 2.0f;
  constexpr float near_cutoff = 0.35f;
  cv::Mat nonlinear_proximity;
  cv::pow(depth_norm, near_gamma, nonlinear_proximity);
  for (int y = 0; y < weighted.rows; ++y) {
    const std::uint8_t *edge = rgb_edges.ptr<std::uint8_t>(y);
    const float *near_depth = nonlinear_proximity.ptr<float>(y);
    const float *raw_proximity = depth_norm.ptr<float>(y);
    std::uint8_t *dst = weighted.ptr<std::uint8_t>(y);
    for (int x = 0; x < weighted.cols; ++x) {
      if (edge[x]) {
        if (raw_proximity[x] < near_cutoff) continue;
        const float weight = near_floor + (1.0f - near_floor) * near_depth[x];
        dst[x] = cv::saturate_cast<std::uint8_t>(255.0f * weight);
      }
    }
  }

  std::vector<cv::Mat> panels = {
    makePanel(image, "RGB"),
    makePanel(depth_color, "Relative inverse depth"),
    makePanel(rgb_edges, "RGB Canny"),
    makePanel(weighted, "Canny x near-depth"),
    makePanel(depth_only_edge, "Depth-only edge")
  };
  cv::Mat comparison;
  cv::hconcat(panels, comparison);
  cv::putText(comparison, cv::format("TensorRT %.1f ms", inference_ms),
              {10, comparison.rows - 12}, cv::FONT_HERSHEY_SIMPLEX, 0.65,
              {255, 255, 255}, 2, cv::LINE_AA);
  if (!cv::imwrite(output_path, comparison)) {
    std::cerr << "output write failed: " << output_path << '\n';
    return 1;
  }
  std::cout << estimator.status() << '\n'
            << "depth range: " << cv::format("%.4f .. %.4f", 
                *std::min_element(depth.begin<float>(), depth.end<float>()),
                *std::max_element(depth.begin<float>(), depth.end<float>())) << '\n'
            << "inference: " << inference_ms << " ms\n"
            << "saved: " << output_path << '\n';
  return 0;
}
