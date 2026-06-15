#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

// BEV Viewer — 트랙바로 4점 실시간 조정 + s키로 YAML 저장
//
// 사용법:
//   ros2 run close_approach bev_viewer_node
//   ros2 run close_approach bev_viewer_node --ros-args --params-file config/bev_params.yaml
//
// 키:
//   s   : config/bev_params.yaml 저장
//   r   : 트랙바 기본값으로 초기화
//   q / ESC : 종료
//
// 파라미터:
//   topic      : 이미지 토픽
//   bev_width  : BEV 출력 너비 (px)
//   bev_height : BEV 출력 높이 (px)
//   src_points : 4점 좌표 8개 (x0 y0 x1 y1 x2 y2 x3 y3), 순서: TL TR BR BL

static const char *kWinOrig = "Original (BEV 영역 조정)";
static const char *kWinBev  = "BEV";

// 트랙바 레이블 (OpenCV 트랙바는 같은 창에 바 이름으로 구분)
static const char *kBarNames[8] = {
    "TL_x", "TL_y",
    "TR_x", "TR_y",
    "BR_x", "BR_y",
    "BL_x", "BL_y"
};

class BevViewer : public rclcpp::Node {
public:
  BevViewer() : Node("bev_viewer") {
    this->declare_parameter<std::string>("topic", "/camera/camera_head/color/image_raw");
    this->declare_parameter<int>("bev_width", 500);
    this->declare_parameter<int>("bev_height", 500);
    this->declare_parameter<std::vector<double>>("src_points", std::vector<double>(8, -1.0));

    topic_ = this->get_parameter("topic").as_string();
    bev_w_ = this->get_parameter("bev_width").as_int();
    bev_h_ = this->get_parameter("bev_height").as_int();

    // src_points 파라미터가 있으면 로드 (sentinel -1 이 아닌 경우)
    auto pre = this->get_parameter("src_points").as_double_array();
    if (pre.size() == 8 && pre[0] >= 0.0) {
      for (int i = 0; i < 4; i++) {
        pts_[i][0] = static_cast<int>(pre[i * 2]);
        pts_[i][1] = static_cast<int>(pre[i * 2 + 1]);
      }
      has_preset_ = true;
      RCLCPP_INFO(this->get_logger(), "src_points 파라미터 로드 완료");
    }

    auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        topic_, qos,
        [this](const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
          cv_bridge::CvImagePtr cv_ptr;
          try {
            cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
          } catch (const cv_bridge::Exception &e) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "cv_bridge 변환 실패: %s", e.what());
            return;
          }
          std::lock_guard<std::mutex> lock(frame_mutex_);
          frame_ = cv_ptr->image.clone();
          has_frame_ = true;
        });

    cv::namedWindow(kWinOrig, cv::WINDOW_NORMAL);
    cv::namedWindow(kWinBev, cv::WINDOW_NORMAL);

    RCLCPP_INFO(this->get_logger(), "BEV Viewer 시작. topic=%s", topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "트랙바로 TL/TR/BR/BL 조정 | s:저장 r:초기화 q:종료");
  }

  // main 루프에서 매 프레임 호출
  void update() {
    cv::Mat frame;
    {
      std::lock_guard<std::mutex> lock(frame_mutex_);
      if (!has_frame_) return;
      frame = frame_.clone();
    }

    if (!trackbars_ready_) {
      initTrackbars(frame.cols, frame.rows);
      trackbars_ready_ = true;
    }

    // 트랙바 현재값 읽기
    for (int i = 0; i < 4; i++) {
      pts_[i][0] = cv::getTrackbarPos(kBarNames[i * 2],     kWinOrig);
      pts_[i][1] = cv::getTrackbarPos(kBarNames[i * 2 + 1], kWinOrig);
    }

    drawOriginal(frame);
    drawBev(frame);
  }

  void saveYaml() {
    if (!trackbars_ready_) {
      RCLCPP_WARN(this->get_logger(), "아직 이미지를 받지 못했습니다.");
      return;
    }
    namespace fs = std::filesystem;
    const fs::path save_dir = fs::current_path() / "config";
    fs::create_directories(save_dir);
    const fs::path yaml_path = save_dir / "bev_params.yaml";

    std::ofstream f(yaml_path);
    f << "bev_viewer:\n";
    f << "  ros__parameters:\n";
    f << "    topic: \"" << topic_ << "\"\n";
    f << "    bev_width: " << bev_w_ << "\n";
    f << "    bev_height: " << bev_h_ << "\n";
    f << "    src_points: [";
    for (int i = 0; i < 4; i++) {
      f << pts_[i][0] << ".0, " << pts_[i][1] << ".0";
      if (i < 3) f << ", ";
    }
    f << "]\n";
    f.close();

    RCLCPP_INFO(this->get_logger(), "저장 완료: %s", yaml_path.c_str());
    RCLCPP_INFO(this->get_logger(),
        "다음 실행:\n  ros2 run close_approach bev_viewer_node --ros-args --params-file %s",
        yaml_path.c_str());
  }

  void resetTrackbars() {
    if (!trackbars_ready_) return;
    cv::Mat frame;
    {
      std::lock_guard<std::mutex> lock(frame_mutex_);
      if (!has_frame_) return;
      frame = frame_.clone();
    }
    setDefaultPts(frame.cols, frame.rows);
    applyPtsToTrackbars();
    RCLCPP_INFO(this->get_logger(), "트랙바 기본값으로 초기화");
  }

private:
  void initTrackbars(int w, int h) {
    if (!has_preset_) setDefaultPts(w, h);

    for (int i = 0; i < 4; i++) {
      cv::createTrackbar(kBarNames[i * 2],     kWinOrig, nullptr, w, nullptr);
      cv::createTrackbar(kBarNames[i * 2 + 1], kWinOrig, nullptr, h, nullptr);
    }
    applyPtsToTrackbars();

    img_w_ = w;
    img_h_ = h;
  }

  void setDefaultPts(int w, int h) {
    // 이미지 하단 사다리꼴 (카메라 전방 지면 영역 기본값)
    pts_[0][0] = w / 4;     pts_[0][1] = h / 2;  // TL
    pts_[1][0] = 3 * w / 4; pts_[1][1] = h / 2;  // TR
    pts_[2][0] = w;          pts_[2][1] = h;       // BR
    pts_[3][0] = 0;          pts_[3][1] = h;       // BL
  }

  void applyPtsToTrackbars() {
    for (int i = 0; i < 4; i++) {
      cv::setTrackbarPos(kBarNames[i * 2],     kWinOrig, pts_[i][0]);
      cv::setTrackbarPos(kBarNames[i * 2 + 1], kWinOrig, pts_[i][1]);
    }
  }

  void drawOriginal(const cv::Mat &frame) {
    cv::Mat disp = frame.clone();

    std::vector<cv::Point> poly(4);
    for (int i = 0; i < 4; i++)
      poly[i] = {pts_[i][0], pts_[i][1]};

    // 반투명 채우기
    cv::Mat overlay = disp.clone();
    cv::fillConvexPoly(overlay, poly, {0, 255, 0});
    cv::addWeighted(overlay, 0.15, disp, 0.85, 0, disp);

    // 외곽선 + 꼭짓점
    static const char *labels[] = {"TL", "TR", "BR", "BL"};
    for (int i = 0; i < 4; i++) {
      cv::line(disp, poly[i], poly[(i + 1) % 4], {0, 220, 0}, 2, cv::LINE_AA);
      cv::circle(disp, poly[i], 7, {0, 255, 0}, -1);
      cv::putText(disp, labels[i],
                  {pts_[i][0] + 9, pts_[i][1] - 7},
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 255, 0}, 2, cv::LINE_AA);
    }

    cv::putText(disp, "s: YAML 저장  r: 초기화  q: 종료",
                {8, disp.rows - 8},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, {40, 220, 220}, 1, cv::LINE_AA);

    cv::imshow(kWinOrig, disp);
  }

  void drawBev(const cv::Mat &frame) {
    const std::vector<cv::Point2f> src = {
        {(float)pts_[0][0], (float)pts_[0][1]},
        {(float)pts_[1][0], (float)pts_[1][1]},
        {(float)pts_[2][0], (float)pts_[2][1]},
        {(float)pts_[3][0], (float)pts_[3][1]},
    };
    const std::vector<cv::Point2f> dst = {
        {0.0f,          0.0f},
        {(float)bev_w_, 0.0f},
        {(float)bev_w_, (float)bev_h_},
        {0.0f,          (float)bev_h_},
    };
    cv::Mat H = cv::getPerspectiveTransform(src, dst);
    cv::Mat bev;
    cv::warpPerspective(frame, bev, H, {bev_w_, bev_h_});
    cv::imshow(kWinBev, bev);
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  std::string topic_;
  int bev_w_, bev_h_;
  int img_w_ = 0, img_h_ = 0;

  cv::Mat frame_;
  std::mutex frame_mutex_;
  bool has_frame_      = false;
  bool has_preset_     = false;
  bool trackbars_ready_= false;

  int pts_[4][2]{};  // [TL, TR, BR, BL][x, y]
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<BevViewer>();

  while (rclcpp::ok()) {
    rclcpp::spin_some(node);
    node->update();
    int key = cv::waitKey(10);
    if (key == 'q' || key == 27) break;
    if (key == 's') node->saveYaml();
    if (key == 'r') node->resetTrackbars();
  }

  cv::destroyAllWindows();
  rclcpp::shutdown();
  return 0;
}
