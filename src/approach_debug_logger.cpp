#include "close_approach/msg/approach_error.hpp"

#include <geometry_msgs/msg/twist.hpp>
#include <pcl/PCLPointCloud2.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>

class ApproachDebugLogger : public rclcpp::Node {
public:
  using ApproachError = close_approach::msg::ApproachError;
  using CloudMsg = sensor_msgs::msg::PointCloud2;
  using ImageMsg = sensor_msgs::msg::CompressedImage;

  ApproachDebugLogger() : Node("approach_debug_logger") {
    this->declare_parameter<std::string>(
        "output_dir", "/home/thor/inha_log/module/close_approach");
    this->declare_parameter<double>("sample_period_sec", 0.5);
    this->declare_parameter<bool>("save_images", true);
    this->declare_parameter<bool>("save_pcd", true);
    this->declare_parameter<std::string>("image_topic",
                                         "/approach/edge_debug/compressed");
    this->declare_parameter<std::string>("cloud_topic", "/approach/debug_cloud");

    this->get_parameter("output_dir", output_dir_);
    this->get_parameter("sample_period_sec", sample_period_sec_);
    this->get_parameter("save_images", save_images_);
    this->get_parameter("save_pcd", save_pcd_);
    this->get_parameter("image_topic", image_topic_);
    this->get_parameter("cloud_topic", cloud_topic_);

    auto qos_rel = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
    auto qos_be = rclcpp::QoS(rclcpp::KeepLast(5)).best_effort();

    // Latched to match the manager: receive current active state even if this
    // subscription matches after the manager already published it.
    auto active_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
    active_sub_ = this->create_subscription<std_msgs::msg::Bool>(
        "/approach/active", active_qos,
        std::bind(&ApproachDebugLogger::activeCallback, this,
                  std::placeholders::_1));
    state_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/approach/state", qos_rel,
        [this](const std_msgs::msg::String::SharedPtr msg) {
          std::lock_guard<std::mutex> lock(mutex_);
          state_ = msg->data;
        });
    edge_sub_ = this->create_subscription<ApproachError>(
        "/approach/edge_error", qos_rel,
        [this](const ApproachError::SharedPtr msg) {
          std::lock_guard<std::mutex> lock(mutex_);
          edge_ = *msg;
          has_edge_ = true;
        });
    pc_sub_ = this->create_subscription<ApproachError>(
        "/approach/pc_error", qos_rel,
        [this](const ApproachError::SharedPtr msg) {
          std::lock_guard<std::mutex> lock(mutex_);
          pc_ = *msg;
          has_pc_ = true;
        });
    pc_debug_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/approach/pc_debug", qos_rel,
        [this](const std_msgs::msg::String::SharedPtr msg) {
          std::lock_guard<std::mutex> lock(mutex_);
          pc_status_ = msg->data;
        });
    lidar_sub_ = this->create_subscription<ApproachError>(
        "/approach/lidar_error", qos_rel,
        [this](const ApproachError::SharedPtr msg) {
          std::lock_guard<std::mutex> lock(mutex_);
          lidar_ = *msg;
          has_lidar_ = true;
        });
    lidar_debug_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/approach/lidar_debug", qos_rel,
        [this](const std_msgs::msg::String::SharedPtr msg) {
          std::lock_guard<std::mutex> lock(mutex_);
          lidar_status_ = msg->data;
        });
    control_sub_ = this->create_subscription<ApproachError>(
        "/approach/control_error", qos_rel,
        [this](const ApproachError::SharedPtr msg) {
          std::lock_guard<std::mutex> lock(mutex_);
          control_ = *msg;
          has_control_ = true;
        });
    cmd_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", qos_rel,
        [this](const geometry_msgs::msg::Twist::SharedPtr msg) {
          std::lock_guard<std::mutex> lock(mutex_);
          cmd_ = *msg;
          has_cmd_ = true;
        });
    image_sub_ = this->create_subscription<ImageMsg>(
        image_topic_, qos_be, [this](const ImageMsg::SharedPtr msg) {
          std::lock_guard<std::mutex> lock(mutex_);
          image_ = *msg;
          has_image_ = true;
        });
    cloud_sub_ = this->create_subscription<CloudMsg>(
        cloud_topic_, qos_be, [this](const CloudMsg::SharedPtr msg) {
          std::lock_guard<std::mutex> lock(mutex_);
          cloud_ = *msg;
          has_cloud_ = true;
        });

    const auto period = std::chrono::duration<double>(sample_period_sec_);
    timer_ = this->create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(period),
        std::bind(&ApproachDebugLogger::sample, this));

    RCLCPP_INFO(this->get_logger(),
                "ApproachDebugLogger ready. output_dir=%s image_topic=%s cloud_topic=%s",
                output_dir_.c_str(), image_topic_.c_str(), cloud_topic_.c_str());
  }

private:
  static std::string timeForFilename() {
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now.time_since_epoch()) %
                    1000;
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &time);
#else
    localtime_r(&time, &tm);
#endif
    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y%m%d_%H%M%S") << "_" << std::setw(3)
       << std::setfill('0') << ms.count();
    return ss.str();
  }

  static double stampSec(const std_msgs::msg::Header &header) {
    return static_cast<double>(header.stamp.sec) +
           static_cast<double>(header.stamp.nanosec) * 1e-9;
  }

  static float deg(float rad) {
    return rad * 180.0F / 3.14159265358979323846F;
  }

  static std::string imageExtension(const std::string &format) {
    if (format.find("png") != std::string::npos ||
        format.find("PNG") != std::string::npos) {
      return ".png";
    }
    return ".jpg";
  }

  static std::string csvToken(std::string value) {
    for (char &ch : value) {
      if (ch == ',' || ch == '\n' || ch == '\r') {
        ch = ' ';
      }
    }
    return value;
  }

  void activeCallback(const std_msgs::msg::Bool::SharedPtr msg) {
    if (msg->data) {
      startSession();
    } else {
      stopSession();
    }
  }

  void startSession() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_) {
      return;
    }

    active_ = true;
    sample_index_ = 0;
    has_edge_ = false;
    has_pc_ = false;
    has_lidar_ = false;
    has_control_ = false;
    has_cmd_ = false;
    has_image_ = false;
    has_cloud_ = false;
    pc_status_ = "none";
    lidar_status_ = "none";
    edge_ = ApproachError{};
    pc_ = ApproachError{};
    lidar_ = ApproachError{};
    control_ = ApproachError{};
    cmd_ = geometry_msgs::msg::Twist{};
    image_ = ImageMsg{};
    cloud_ = CloudMsg{};
    session_start_time_ = this->now();
    warned_no_image_ = false;
    warned_no_cloud_ = false;
    session_dir_ = std::filesystem::path(output_dir_) /
                   ("approach_" + timeForFilename());
    try {
      std::filesystem::create_directories(session_dir_);
    } catch (const std::filesystem::filesystem_error &ex) {
      RCLCPP_WARN(this->get_logger(), "Failed to create debug directory: %s",
                  ex.what());
      active_ = false;
      return;
    }

    csv_path_ = session_dir_ / "samples.csv";
    csv_.open(csv_path_, std::ios::out | std::ios::trunc);
    if (!csv_.is_open()) {
      RCLCPP_WARN(this->get_logger(), "Failed to open debug CSV: %s",
                  csv_path_.c_str());
      active_ = false;
      return;
    }

    csv_ << "sample,t_sec,state,"
            "edge_stamp,edge_valid,edge_theta_rad,edge_theta_deg,edge_mean_y_px,"
            "pc_stamp,pc_valid,pc_surface_x,pc_x_error,pc_y_error,pc_status,"
            "lidar_stamp,lidar_valid,lidar_surface_x,lidar_x_error,lidar_status,"
            "control_stamp,control_valid,control_x_error,control_theta_rad,control_theta_deg,"
            "cmd_vx,cmd_wz,image_file,pcd_file\n";
    csv_.flush();

    RCLCPP_INFO(this->get_logger(), "Debug session started: %s",
                session_dir_.c_str());
  }

  void stopSession() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_) {
      return;
    }
    active_ = false;
    if (csv_.is_open()) {
      csv_.flush();
      csv_.close();
    }
    RCLCPP_INFO(this->get_logger(), "Debug session stopped: %s",
                session_dir_.c_str());
  }

  void sample() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_ || !csv_.is_open()) {
      return;
    }

    const std::uint64_t idx = sample_index_++;
    const std::string stem = sampleStem(idx);
    const rclcpp::Time now = this->now();
    const double t_sec = now.seconds();

    std::string image_file;
    if (save_images_ && has_image_) {
      image_file = stem + imageExtension(image_.format);
      const auto path = session_dir_ / image_file;
      std::ofstream out(path, std::ios::out | std::ios::binary);
      if (out.is_open()) {
        out.write(reinterpret_cast<const char *>(image_.data.data()),
                  static_cast<std::streamsize>(image_.data.size()));
      } else {
        image_file.clear();
      }
    }

    std::string pcd_file;
    if (save_pcd_ && has_cloud_) {
      pcd_file = stem + ".pcd";
      const auto path = session_dir_ / pcd_file;
      pcl::PCLPointCloud2 pcl_cloud;
      pcl_conversions::toPCL(cloud_, pcl_cloud);
      pcl::PCDWriter writer;
      if (writer.writeBinary(path.string(), pcl_cloud) != 0) {
        pcd_file.clear();
      }
    }

    const double session_age = (now - session_start_time_).seconds();
    const double warn_after = std::max(1.0, sample_period_sec_ * 4.0);
    if (save_images_ && !has_image_ && !warned_no_image_ &&
        session_age >= warn_after) {
      warned_no_image_ = true;
      RCLCPP_WARN(this->get_logger(),
                  "No debug image received %.1fs after session start on %s",
                  session_age, image_topic_.c_str());
    }
    if (save_pcd_ && !has_cloud_ && !warned_no_cloud_ &&
        session_age >= warn_after) {
      warned_no_cloud_ = true;
      RCLCPP_WARN(this->get_logger(),
                  "No debug cloud received %.1fs after session start on %s",
                  session_age, cloud_topic_.c_str());
    }

    csv_ << idx << ',' << std::fixed << std::setprecision(6) << t_sec << ','
         << state_ << ',' << (has_edge_ ? stampSec(edge_.header) : 0.0) << ','
         << (has_edge_ && edge_.valid) << ',' << edge_.theta_error << ','
         << deg(edge_.theta_error) << ',' << edge_.mean_y_px << ','
         << (has_pc_ ? stampSec(pc_.header) : 0.0) << ','
         << (has_pc_ && pc_.valid) << ',' << pc_.surface_distance_m << ','
         << pc_.x_error << ',' << pc_.y_error << ',' << csvToken(pc_status_)
         << ',' << (has_lidar_ ? stampSec(lidar_.header) : 0.0) << ','
         << (has_lidar_ && lidar_.valid) << ','
         << lidar_.surface_distance_m << ',' << lidar_.x_error << ','
         << csvToken(lidar_status_) << ','
         << (has_control_ ? stampSec(control_.header) : 0.0) << ','
         << (has_control_ && control_.valid) << ',' << control_.x_error << ','
         << control_.theta_error << ',' << deg(control_.theta_error) << ','
         << (has_cmd_ ? cmd_.linear.x : 0.0) << ','
         << (has_cmd_ ? cmd_.angular.z : 0.0) << ',' << image_file << ','
         << pcd_file << '\n';
    csv_.flush();
  }

  std::string sampleStem(std::uint64_t idx) const {
    std::ostringstream ss;
    ss << "sample_" << std::setw(6) << std::setfill('0') << idx;
    return ss.str();
  }

  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr active_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr state_sub_;
  rclcpp::Subscription<ApproachError>::SharedPtr edge_sub_;
  rclcpp::Subscription<ApproachError>::SharedPtr pc_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr pc_debug_sub_;
  rclcpp::Subscription<ApproachError>::SharedPtr lidar_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr lidar_debug_sub_;
  rclcpp::Subscription<ApproachError>::SharedPtr control_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
  rclcpp::Subscription<ImageMsg>::SharedPtr image_sub_;
  rclcpp::Subscription<CloudMsg>::SharedPtr cloud_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::mutex mutex_;
  std::string output_dir_;
  std::string image_topic_;
  std::string cloud_topic_;
  double sample_period_sec_ = 0.5;
  bool save_images_ = true;
  bool save_pcd_ = true;

  bool active_ = false;
  std::filesystem::path session_dir_;
  std::filesystem::path csv_path_;
  std::ofstream csv_;
  std::uint64_t sample_index_ = 0;
  rclcpp::Time session_start_time_;

  std::string state_ = "IDLE";
  std::string pc_status_ = "none";
  std::string lidar_status_ = "none";
  ApproachError edge_;
  ApproachError pc_;
  ApproachError lidar_;
  ApproachError control_;
  geometry_msgs::msg::Twist cmd_;
  ImageMsg image_;
  CloudMsg cloud_;
  bool has_edge_ = false;
  bool has_pc_ = false;
  bool has_lidar_ = false;
  bool has_control_ = false;
  bool has_cmd_ = false;
  bool has_image_ = false;
  bool has_cloud_ = false;
  bool warned_no_image_ = false;
  bool warned_no_cloud_ = false;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ApproachDebugLogger>());
  rclcpp::shutdown();
  return 0;
}
