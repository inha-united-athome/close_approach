#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <vector>
#include <cmath>

// 수평 엣지 기반 Yaw 추정 테스트 노드
//
// 대상: 테이블·선반·식기세척기·냉장고 등
// 원리: 수평 에지 각도 → yaw error
//       수평 에지 평균 y픽셀 → stop threshold 기준 (가까울수록 아래로 내려옴)
//
// 트랙바: Canny / Hough / ROI (Top, Bot, Left, Right) 전부 실시간 조정
//
// 키: q / ESC → 종료

static const char *kWinLines = "Lines";
static const char *kWinEdges = "Edges";

class EdgeYawViewer : public rclcpp::Node {
public:
  EdgeYawViewer() : Node("edge_yaw_viewer") {
    this->declare_parameter<std::string>("topic",
        "/camera/camera_head/color/image_raw/compressed");
    topic_ = this->get_parameter("topic").as_string();

    auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
        topic_, qos,
        [this](const sensor_msgs::msg::CompressedImage::ConstSharedPtr &msg) {
          const std::vector<uint8_t> &buf = msg->data;
          cv::Mat decoded = cv::imdecode(buf, cv::IMREAD_COLOR);
          if (decoded.empty()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "imdecode 실패 (빈 프레임)");
            return;
          }
          std::lock_guard<std::mutex> lk(mtx_);
          frame_ = std::move(decoded);
          has_frame_ = true;
        });

    cv::namedWindow(kWinLines, cv::WINDOW_NORMAL);
    cv::namedWindow(kWinEdges, cv::WINDOW_NORMAL);
    initTrackbars();

    RCLCPP_INFO(this->get_logger(), "Edge Yaw Viewer 시작. topic=%s", topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "s: YAML 저장  q: 종료");
  }

  void saveYaml() {
    namespace fs = std::filesystem;
    const fs::path save_dir = fs::current_path() / "config";
    fs::create_directories(save_dir);
    const fs::path yaml_path = save_dir / "edge_detector.yaml";

    std::ofstream f(yaml_path);
    f << "edge_detector:\n";
    f << "  ros__parameters:\n";
    f << "    topic: \"" << topic_ << "\"\n";
    f << "    canny_low: "     << tb_.canny_low     << "\n";
    f << "    canny_high: "    << tb_.canny_high    << "\n";
    f << "    hough_thresh: "  << tb_.hough_thresh  << "\n";
    f << "    min_length: "    << tb_.min_length    << "\n";
    f << "    max_gap: "       << tb_.max_gap       << "\n";
    f << "    roi_top_pct: "   << tb_.roi_top_pct   << "\n";
    f << "    roi_bot_pct: "   << tb_.roi_bot_pct   << "\n";
    f << "    roi_left_pct: "  << tb_.roi_left_pct  << "\n";
    f << "    roi_right_pct: " << tb_.roi_right_pct << "\n";
    f << "    stop_y_pct: "    << tb_.stop_y_pct    << "\n";
    f.close();

    RCLCPP_INFO(this->get_logger(), "저장 완료: %s", yaml_path.c_str());
    RCLCPP_INFO(this->get_logger(),
        "저장 완료 → edge_detector_node 에서 바로 사용 가능:\n"
        "  ros2 launch close_approach approach.launch.py");
  }

  void update() {
    cv::Mat frame;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      if (!has_frame_) return;
      frame = frame_.clone();
    }

    readTrackbars(frame.cols, frame.rows);

    // ── 전처리 ──────────────────────────────────────────
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, {5, 5}, 0);

    // ROI 마스크 (사각형)
    cv::Mat roi_mask = cv::Mat::zeros(gray.size(), CV_8U);
    roi_mask(cv::Rect(rx1_, ry1_, rx2_ - rx1_, ry2_ - ry1_)) = 255;

    // Canny
    cv::Mat edges;
    cv::Canny(gray, edges, tb_.canny_low, tb_.canny_high);
    cv::bitwise_and(edges, roi_mask, edges);

    // HoughLinesP
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, CV_PI / 180,
                    tb_.hough_thresh, tb_.min_length, tb_.max_gap);

    // ── 수평 계열 라인 추출 ──────────────────────────────
    // |angle| < 45° → 수평 계열
    // 길이 가중 평균 각도 → yaw error
    // 길이 가중 평균 y중점 → stop 조건용
    float yaw_deg   = 0.0f;
    float mean_y    = 0.0f;
    bool  yaw_valid = false;
    std::vector<size_t> horiz_idx;
    float w_sum = 0.0f, wa_sum = 0.0f, wy_sum = 0.0f;

    for (size_t i = 0; i < lines.size(); i++) {
      const auto &l = lines[i];
      const float dx = static_cast<float>(l[2] - l[0]);
      const float dy = static_cast<float>(l[3] - l[1]);
      float angle = std::atan2(dy, dx) * 180.0f / CV_PI;

      // 0° = 수평, ±90° = 수직. -90~+90 으로 정규화
      if (angle >  90.0f) angle -= 180.0f;
      if (angle < -90.0f) angle += 180.0f;

      // |angle| >= 50° (수직에 가까운 선) 버림
      if (std::abs(angle) >= 50.0f) continue;

      {
        const float len  = std::hypot(dx, dy);
        const float my   = (l[1] + l[3]) * 0.5f;
        horiz_idx.push_back(i);
        wa_sum += angle * len;
        wy_sum += my    * len;
        w_sum  += len;
      }
    }

    if (!horiz_idx.empty()) {
      yaw_deg   = wa_sum / w_sum;
      mean_y    = wy_sum / w_sum;
      yaw_valid = true;
    }

    drawLines(frame, lines, horiz_idx, yaw_deg, mean_y, yaw_valid);
    drawEdges(edges);
  }

private:
  struct Trackbars {
    int canny_low     = 50;
    int canny_high    = 150;
    int hough_thresh  = 40;
    int min_length    = 60;
    int max_gap       = 15;
    int roi_top_pct   = 20;
    int roi_bot_pct   = 80;
    int roi_left_pct  = 20;
    int roi_right_pct = 80;
    int stop_y_pct    = 70;  // 제어 종료 기준선 (이미지 높이 %)
  } tb_;

  // 픽셀 ROI / stop line (readTrackbars 에서 갱신)
  int rx1_ = 0, rx2_ = 640;
  int ry1_ = 0, ry2_ = 480;
  int stop_y_ = 336;  // default 70%

  void initTrackbars() {
    cv::createTrackbar("Canny Low",    kWinLines, nullptr, 500, nullptr);
    cv::createTrackbar("Canny High",   kWinLines, nullptr, 500, nullptr);
    cv::createTrackbar("Hough Thresh", kWinLines, nullptr, 200, nullptr);
    cv::createTrackbar("Min Length",   kWinLines, nullptr, 400, nullptr);
    cv::createTrackbar("Max Gap",      kWinLines, nullptr, 100, nullptr);
    cv::createTrackbar("ROI Top %",    kWinLines, nullptr, 100, nullptr);
    cv::createTrackbar("ROI Bot %",    kWinLines, nullptr, 100, nullptr);
    cv::createTrackbar("ROI Left %",   kWinLines, nullptr, 100, nullptr);
    cv::createTrackbar("ROI Right %",  kWinLines, nullptr, 100, nullptr);

    cv::setTrackbarPos("Canny Low",    kWinLines, tb_.canny_low);
    cv::setTrackbarPos("Canny High",   kWinLines, tb_.canny_high);
    cv::setTrackbarPos("Hough Thresh", kWinLines, tb_.hough_thresh);
    cv::setTrackbarPos("Min Length",   kWinLines, tb_.min_length);
    cv::setTrackbarPos("Max Gap",      kWinLines, tb_.max_gap);
    cv::setTrackbarPos("ROI Top %",    kWinLines, tb_.roi_top_pct);
    cv::setTrackbarPos("ROI Bot %",    kWinLines, tb_.roi_bot_pct);
    cv::createTrackbar("Stop Y %",     kWinLines, nullptr, 100, nullptr);

    cv::setTrackbarPos("ROI Left %",   kWinLines, tb_.roi_left_pct);
    cv::setTrackbarPos("ROI Right %",  kWinLines, tb_.roi_right_pct);
    cv::setTrackbarPos("Stop Y %",     kWinLines, tb_.stop_y_pct);
  }

  void readTrackbars(int img_w, int img_h) {
    tb_.canny_low     = cv::getTrackbarPos("Canny Low",    kWinLines);
    tb_.canny_high    = cv::getTrackbarPos("Canny High",   kWinLines);
    tb_.hough_thresh  = cv::getTrackbarPos("Hough Thresh", kWinLines);
    tb_.min_length    = cv::getTrackbarPos("Min Length",   kWinLines);
    tb_.max_gap       = cv::getTrackbarPos("Max Gap",      kWinLines);
    tb_.roi_top_pct   = cv::getTrackbarPos("ROI Top %",    kWinLines);
    tb_.roi_bot_pct   = cv::getTrackbarPos("ROI Bot %",    kWinLines);
    tb_.roi_left_pct  = cv::getTrackbarPos("ROI Left %",   kWinLines);
    tb_.roi_right_pct = cv::getTrackbarPos("ROI Right %",  kWinLines);
    tb_.stop_y_pct    = cv::getTrackbarPos("Stop Y %",     kWinLines);

    const float top   = std::clamp(tb_.roi_top_pct  / 100.0f, 0.0f, 0.98f);
    const float bot   = std::clamp(tb_.roi_bot_pct  / 100.0f, top + 0.01f, 1.0f);
    const float left  = std::clamp(tb_.roi_left_pct / 100.0f, 0.0f, 0.98f);
    const float right = std::clamp(tb_.roi_right_pct/ 100.0f, left + 0.01f, 1.0f);

    ry1_    = static_cast<int>(top   * img_h);
    ry2_    = static_cast<int>(bot   * img_h);
    rx1_    = static_cast<int>(left  * img_w);
    rx2_    = static_cast<int>(right * img_w);
    stop_y_ = static_cast<int>(tb_.stop_y_pct / 100.0f * img_h);
  }

  void drawLines(const cv::Mat &frame,
                 const std::vector<cv::Vec4i> &lines,
                 const std::vector<size_t> &horiz_idx,
                 float yaw_deg, float mean_y, bool yaw_valid) {
    cv::Mat disp = frame.clone();

    // ROI 박스
    cv::rectangle(disp, {rx1_, ry1_}, {rx2_, ry2_}, {100, 100, 255}, 2);

    // 비수평 라인 (회색)
    std::vector<bool> is_horiz(lines.size(), false);
    for (size_t i : horiz_idx) is_horiz[i] = true;
    for (size_t i = 0; i < lines.size(); i++) {
      if (!is_horiz[i]) {
        const auto &l = lines[i];
        cv::line(disp, {l[0], l[1]}, {l[2], l[3]}, {70, 70, 70}, 1, cv::LINE_AA);
      }
    }

    // 수평 계열 라인 (노란색)
    for (size_t i : horiz_idx) {
      const auto &l = lines[i];
      cv::line(disp, {l[0], l[1]}, {l[2], l[3]}, {0, 220, 255}, 2, cv::LINE_AA);
    }

    // ── Stop 기준선 (빨간 점선) ──────────────────────────
    // 전체 이미지 너비에 걸쳐 표시 (ROI 밖에서도 보이게)
    for (int x = 0; x < disp.cols; x += 10)
      cv::line(disp, {x, stop_y_}, {std::min(x + 6, disp.cols - 1), stop_y_},
               {0, 0, 220}, 2);
    char stop_label[32];
    std::snprintf(stop_label, sizeof(stop_label), "STOP y=%d", stop_y_);
    cv::putText(disp, stop_label, {disp.cols - 110, stop_y_ - 6},
                cv::FONT_HERSHEY_SIMPLEX, 0.45, {0, 0, 220}, 1, cv::LINE_AA);

    if (yaw_valid) {
      // 감지된 수평선 평균 y (초록 실선)
      cv::line(disp, {rx1_, static_cast<int>(mean_y)},
                     {rx2_, static_cast<int>(mean_y)},
               {0, 200, 80}, 1, cv::LINE_AA);

      // yaw 기울기 시각화 (ROI 중앙에)
      const int cx = (rx1_ + rx2_) / 2;
      const int cy = static_cast<int>(mean_y);
      const float rad = yaw_deg * CV_PI / 180.0f;
      const int arm = std::min((rx2_ - rx1_) / 3, 80);
      const int ex  = static_cast<int>(std::cos(rad) * arm);
      const int ey  = static_cast<int>(std::sin(rad) * arm);
      // 기준 수평선 (흰색)
      cv::line(disp, {cx - arm, cy}, {cx + arm, cy}, {180, 180, 180}, 1, cv::LINE_AA);
      // 감지된 기울기 (빨강)
      cv::line(disp, {cx - ex, cy - ey}, {cx + ex, cy + ey},
               {0, 60, 255}, 3, cv::LINE_AA);

      // stop 판정 여부
      const bool reached = (mean_y >= stop_y_);

      // ── 텍스트 ───────────────────────────────────────
      const cv::Scalar yaw_col = std::abs(yaw_deg) < 3.0f
                                 ? cv::Scalar(0, 220, 0) : cv::Scalar(0, 60, 255);
      char buf[80];
      // yaw error: ±90° 기준 (0° = 완전 수평 정렬)
      std::snprintf(buf, sizeof(buf), "Yaw error : %+.1f deg  (+-90 range)", yaw_deg);
      cv::putText(disp, buf, {10, 38},
                  cv::FONT_HERSHEY_SIMPLEX, 0.75, yaw_col, 2, cv::LINE_AA);

      std::snprintf(buf, sizeof(buf), "Edge mean_y: %.0f px  stop_y: %d px",
                    mean_y, stop_y_);
      cv::putText(disp, buf, {10, 65},
                  cv::FONT_HERSHEY_SIMPLEX, 0.65, {0, 200, 80}, 1, cv::LINE_AA);

      if (reached) {
        cv::putText(disp, ">>> STOP <<<", {disp.cols / 2 - 80, disp.rows / 2},
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, {0, 0, 255}, 3, cv::LINE_AA);
      }

      drawYawBar(disp, yaw_deg);
    } else {
      cv::putText(disp, "Yaw: N/A", {10, 38},
                  cv::FONT_HERSHEY_SIMPLEX, 0.85, {100, 100, 100}, 1, cv::LINE_AA);
    }

    // ROI 좌표 + 라인 수
    char info[120];
    std::snprintf(info, sizeof(info),
                  "ROI x[%d,%d] y[%d,%d] | lines:%zu horiz:%zu | q:종료",
                  rx1_, rx2_, ry1_, ry2_,
                  lines.size(), horiz_idx.size());
    cv::putText(disp, info, {8, disp.rows - 8},
                cv::FONT_HERSHEY_SIMPLEX, 0.42, {160, 160, 160}, 1, cv::LINE_AA);

    cv::imshow(kWinLines, disp);
  }

  void drawYawBar(cv::Mat &img, float yaw_deg) const {
    const int bw = 240, bh = 14;
    const int bx = img.cols / 2 - bw / 2;
    const int by = img.rows - 28;

    cv::rectangle(img, {bx, by}, {bx + bw, by + bh}, {50, 50, 50}, -1);
    cv::rectangle(img, {bx, by}, {bx + bw, by + bh}, {130, 130, 130}, 1);

    const float clamped = std::clamp(yaw_deg, -45.0f, 45.0f);
    const int   mid     = bx + bw / 2;
    const int   fill_w  = static_cast<int>(clamped / 45.0f * (bw / 2));
    const int   fill_x  = std::min(mid, mid + fill_w);
    const int   fill_xe = std::max(mid, mid + fill_w);
    if (fill_x != fill_xe) {
      const cv::Scalar c = std::abs(yaw_deg) < 3.0f
                           ? cv::Scalar(0, 200, 0) : cv::Scalar(0, 60, 255);
      cv::rectangle(img, {fill_x, by + 1}, {fill_xe, by + bh - 1}, c, -1);
    }
    cv::line(img, {mid, by}, {mid, by + bh}, {255, 255, 255}, 1);
    cv::putText(img, "-45", {bx - 28, by + 11},
                cv::FONT_HERSHEY_SIMPLEX, 0.35, {150,150,150}, 1);
    cv::putText(img, "+45", {bx + bw + 3, by + 11},
                cv::FONT_HERSHEY_SIMPLEX, 0.35, {150,150,150}, 1);
  }

  void drawEdges(const cv::Mat &edges) {
    cv::Mat colored;
    cv::cvtColor(edges, colored, cv::COLOR_GRAY2BGR);
    cv::rectangle(colored, {rx1_, ry1_}, {rx2_, ry2_}, {100, 100, 255}, 2);
    // stop 기준선
    for (int x = 0; x < colored.cols; x += 10)
      cv::line(colored, {x, stop_y_}, {std::min(x + 6, colored.cols - 1), stop_y_},
               {0, 0, 220}, 2);
    cv::imshow(kWinEdges, colored);
  }

  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr sub_;
  std::string topic_;
  cv::Mat frame_;
  std::mutex mtx_;
  bool has_frame_ = false;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<EdgeYawViewer>();

  while (rclcpp::ok()) {
    rclcpp::spin_some(node);
    node->update();
    int key = cv::waitKey(10);
    if (key == 'q' || key == 27) break;
    if (key == 's') node->saveYaml();
  }

  cv::destroyAllWindows();
  rclcpp::shutdown();
  return 0;
}
