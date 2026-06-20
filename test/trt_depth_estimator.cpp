#include "trt_depth_estimator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <opencv2/imgproc.hpp>

#ifdef CAFM_HAS_TENSORRT
#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#endif

#ifdef CAFM_HAS_TENSORRT
namespace {

class TrtDepthLogger final : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *message) noexcept override {
    if (severity <= Severity::kWARNING) last_message = message ? message : "";
  }
  std::string last_message;
};

template <typename T>
struct TrtDestroy {
  void operator()(T *ptr) const noexcept { delete ptr; }
};

std::string dimsString(const nvinfer1::Dims &dims) {
  std::ostringstream out;
  out << '[';
  for (int i = 0; i < dims.nbDims; ++i) {
    if (i) out << ',';
    out << dims.d[i];
  }
  out << ']';
  return out.str();
}

std::size_t volume(const nvinfer1::Dims &dims) {
  std::size_t size = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] <= 0) throw std::runtime_error("unresolved dynamic tensor shape");
    size *= static_cast<std::size_t>(dims.d[i]);
  }
  return size;
}

std::size_t elementSize(nvinfer1::DataType type) {
  switch (type) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT8: return 1;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kBOOL: return 1;
    default: throw std::runtime_error("unsupported TensorRT tensor dtype");
  }
}

float valueAt(const std::vector<std::uint8_t> &data,
              nvinfer1::DataType type, std::size_t index) {
  switch (type) {
    case nvinfer1::DataType::kFLOAT:
      return reinterpret_cast<const float *>(data.data())[index];
    case nvinfer1::DataType::kHALF:
      return __half2float(reinterpret_cast<const __half *>(data.data())[index]);
    case nvinfer1::DataType::kINT8:
      return reinterpret_cast<const std::int8_t *>(data.data())[index];
    case nvinfer1::DataType::kINT32:
      return static_cast<float>(reinterpret_cast<const std::int32_t *>(data.data())[index]);
    case nvinfer1::DataType::kBOOL:
      return data[index] ? 1.0f : 0.0f;
    default:
      throw std::runtime_error("unsupported TensorRT output dtype");
  }
}

void checkCuda(cudaError_t error, const char *what) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
  }
}

std::vector<char> readEngineBlob(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("engine file open failed: " + path);
  const auto end = file.tellg();
  if (end <= 0) throw std::runtime_error("engine file is empty: " + path);
  std::vector<char> bytes(static_cast<std::size_t>(end));
  file.seekg(0);
  file.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!file) throw std::runtime_error("engine file read failed: " + path);

  // Also accept Ultralytics-style metadata-prefixed engines.
  if (bytes.size() > 5) {
    std::uint32_t metadata_len = 0;
    std::memcpy(&metadata_len, bytes.data(), sizeof(metadata_len));
    const std::size_t offset = sizeof(metadata_len) + metadata_len;
    if (metadata_len > 1 && offset < bytes.size() && bytes[4] == '{') {
      return {bytes.begin() + static_cast<std::ptrdiff_t>(offset), bytes.end()};
    }
  }
  return bytes;
}

}  // namespace
#endif

class TrtDepthEstimator::Impl {
public:
  Impl(const std::string &engine_path, int requested_w, int requested_h) {
#ifdef CAFM_HAS_TENSORRT
    try {
      load(engine_path, requested_w, requested_h);
    } catch (const std::exception &e) {
      status_ = e.what();
      releaseBuffers();
    }
#else
    (void)engine_path;
    (void)requested_w;
    (void)requested_h;
    status_ = "TensorRT support was not found when close_approach was built";
#endif
  }

  ~Impl() {
#ifdef CAFM_HAS_TENSORRT
    releaseBuffers();
#endif
  }

  bool ready() const { return ready_; }
  const std::string &status() const { return status_; }
  int inputWidth() const { return input_w_; }
  int inputHeight() const { return input_h_; }

  bool infer(const cv::Mat &bgr_roi, cv::Mat &relative_depth,
             double &inference_ms) {
#ifdef CAFM_HAS_TENSORRT
    if (!ready_ || bgr_roi.empty()) return false;
    try {
      cv::Mat resized;
      cv::resize(bgr_roi, resized, {input_w_, input_h_}, 0.0, 0.0,
                 cv::INTER_CUBIC);
      prepareInput(resized);

      const auto start = std::chrono::steady_clock::now();
      checkCuda(cudaMemcpyAsync(input_device_, input_host_.data(), input_bytes_,
                                cudaMemcpyHostToDevice, stream_), "copy depth input");
      if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("Depth TensorRT enqueueV3 failed");
      }
      checkCuda(cudaMemcpyAsync(output_host_.data(), output_device_, output_bytes_,
                                cudaMemcpyDeviceToHost, stream_), "copy depth output");
      checkCuda(cudaStreamSynchronize(stream_), "synchronize depth inference");
      inference_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start).count();

      decodeOutput(bgr_roi.size(), relative_depth);
      return true;
    } catch (const std::exception &e) {
      status_ = e.what();
      return false;
    }
#else
    (void)bgr_roi;
    (void)relative_depth;
    (void)inference_ms;
    return false;
#endif
  }

private:
#ifdef CAFM_HAS_TENSORRT
  void load(const std::string &engine_path, int requested_w, int requested_h) {
    const std::vector<char> blob = readEngineBlob(engine_path);
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) throw std::runtime_error("TensorRT runtime creation failed");
    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    if (!engine_) {
      throw std::runtime_error("depth engine deserialization failed: " +
                               logger_.last_message);
    }
    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("depth execution context creation failed");

    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
      const char *name = engine_->getIOTensorName(i);
      if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
        if (!input_name_.empty()) throw std::runtime_error("expected one depth input");
        input_name_ = name;
      } else if (output_name_.empty()) {
        output_name_ = name;
      }
    }
    if (input_name_.empty() || output_name_.empty()) {
      throw std::runtime_error("depth TensorRT input/output not found");
    }

    nvinfer1::Dims input_dims = engine_->getTensorShape(input_name_.c_str());
    if (input_dims.nbDims != 4) {
      throw std::runtime_error("expected BCHW depth input, got " + dimsString(input_dims));
    }
    input_w_ = input_dims.d[3] > 0 ? input_dims.d[3] : requested_w;
    input_h_ = input_dims.d[2] > 0 ? input_dims.d[2] : requested_h;
    if (input_w_ <= 0 || input_h_ <= 0) {
      throw std::runtime_error("dynamic depth engine requires a positive input size");
    }
    if (!context_->setInputShape(input_name_.c_str(),
                                 nvinfer1::Dims4{1, 3, input_h_, input_w_})) {
      throw std::runtime_error("TensorRT rejected depth input shape");
    }

    input_dims = context_->getTensorShape(input_name_.c_str());
    output_dims_ = context_->getTensorShape(output_name_.c_str());
    input_type_ = engine_->getTensorDataType(input_name_.c_str());
    output_type_ = engine_->getTensorDataType(output_name_.c_str());
    if (input_type_ != nvinfer1::DataType::kFLOAT &&
        input_type_ != nvinfer1::DataType::kHALF) {
      throw std::runtime_error("depth input must be FP32 or FP16");
    }
    parseOutputShape();

    input_bytes_ = volume(input_dims) * elementSize(input_type_);
    output_bytes_ = volume(output_dims_) * elementSize(output_type_);
    input_host_.resize(input_bytes_);
    output_host_.resize(output_bytes_);
    checkCuda(cudaStreamCreate(&stream_), "create depth CUDA stream");
    checkCuda(cudaMalloc(&input_device_, input_bytes_), "allocate depth input");
    checkCuda(cudaMalloc(&output_device_, output_bytes_), "allocate depth output");
    if (!context_->setTensorAddress(input_name_.c_str(), input_device_) ||
        !context_->setTensorAddress(output_name_.c_str(), output_device_)) {
      throw std::runtime_error("depth setTensorAddress failed");
    }

    // Pay TensorRT's lazy kernel/JIT cost during node startup, not on the
    // first camera frame used for edge comparison.
    checkCuda(cudaMemsetAsync(input_device_, 0, input_bytes_, stream_),
              "warm up depth input");
    if (!context_->enqueueV3(stream_)) {
      throw std::runtime_error("depth TensorRT warmup failed");
    }
    checkCuda(cudaStreamSynchronize(stream_), "synchronize depth warmup");

    ready_ = true;
    status_ = "ready: input=" + dimsString(input_dims) +
              " output=" + dimsString(output_dims_);
  }

  void parseOutputShape() {
    if (output_dims_.nbDims == 4 && output_dims_.d[0] == 1 &&
        output_dims_.d[1] == 1) {
      output_h_ = output_dims_.d[2];
      output_w_ = output_dims_.d[3];
    } else if (output_dims_.nbDims == 3 && output_dims_.d[0] == 1) {
      output_h_ = output_dims_.d[1];
      output_w_ = output_dims_.d[2];
    } else if (output_dims_.nbDims == 2) {
      output_h_ = output_dims_.d[0];
      output_w_ = output_dims_.d[1];
    } else {
      throw std::runtime_error("expected dense depth output, got " +
                               dimsString(output_dims_));
    }
    if (output_w_ <= 0 || output_h_ <= 0) {
      throw std::runtime_error("invalid depth output shape");
    }
  }

  void prepareInput(const cv::Mat &bgr) {
    constexpr float mean[3] = {0.485f, 0.456f, 0.406f};
    constexpr float stddev[3] = {0.229f, 0.224f, 0.225f};
    const std::size_t plane = static_cast<std::size_t>(input_w_) * input_h_;

    auto normalized = [&](const cv::Vec3b &pixel, int rgb_channel) {
      const int bgr_channel = 2 - rgb_channel;
      return (pixel[bgr_channel] / 255.0f - mean[rgb_channel]) /
             stddev[rgb_channel];
    };

    if (input_type_ == nvinfer1::DataType::kFLOAT) {
      float *dst = reinterpret_cast<float *>(input_host_.data());
      for (int y = 0; y < input_h_; ++y) {
        const cv::Vec3b *row = bgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < input_w_; ++x) {
          const std::size_t p = static_cast<std::size_t>(y) * input_w_ + x;
          for (int c = 0; c < 3; ++c) dst[c * plane + p] = normalized(row[x], c);
        }
      }
    } else {
      __half *dst = reinterpret_cast<__half *>(input_host_.data());
      for (int y = 0; y < input_h_; ++y) {
        const cv::Vec3b *row = bgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < input_w_; ++x) {
          const std::size_t p = static_cast<std::size_t>(y) * input_w_ + x;
          for (int c = 0; c < 3; ++c) {
            dst[c * plane + p] = __float2half(normalized(row[x], c));
          }
        }
      }
    }
  }

  void decodeOutput(cv::Size roi_size, cv::Mat &relative_depth) const {
    cv::Mat raw(output_h_, output_w_, CV_32F);
    float *dst = raw.ptr<float>();
    const std::size_t count = static_cast<std::size_t>(output_h_) * output_w_;
    for (std::size_t i = 0; i < count; ++i) {
      const float value = valueAt(output_host_, output_type_, i);
      dst[i] = std::isfinite(value) ? value : 0.0f;
    }
    cv::resize(raw, relative_depth, roi_size, 0.0, 0.0, cv::INTER_CUBIC);
  }

  void releaseBuffers() noexcept {
    if (input_device_) cudaFree(input_device_);
    if (output_device_) cudaFree(output_device_);
    if (stream_) cudaStreamDestroy(stream_);
    input_device_ = nullptr;
    output_device_ = nullptr;
    stream_ = nullptr;
    ready_ = false;
  }

  TrtDepthLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime, TrtDestroy<nvinfer1::IRuntime>> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, TrtDestroy<nvinfer1::ICudaEngine>> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, TrtDestroy<nvinfer1::IExecutionContext>> context_;
  std::string input_name_;
  std::string output_name_;
  nvinfer1::Dims output_dims_{};
  nvinfer1::DataType input_type_ = nvinfer1::DataType::kFLOAT;
  nvinfer1::DataType output_type_ = nvinfer1::DataType::kFLOAT;
  cudaStream_t stream_ = nullptr;
  void *input_device_ = nullptr;
  void *output_device_ = nullptr;
  std::size_t input_bytes_ = 0;
  std::size_t output_bytes_ = 0;
  std::vector<std::uint8_t> input_host_;
  std::vector<std::uint8_t> output_host_;
#endif

  bool ready_ = false;
  std::string status_;
  int input_w_ = 0;
  int input_h_ = 0;
  int output_w_ = 0;
  int output_h_ = 0;
};

TrtDepthEstimator::TrtDepthEstimator(const std::string &engine_path,
                                     int requested_input_width,
                                     int requested_input_height)
    : impl_(std::make_unique<Impl>(engine_path, requested_input_width,
                                  requested_input_height)) {}

TrtDepthEstimator::~TrtDepthEstimator() = default;
bool TrtDepthEstimator::ready() const { return impl_->ready(); }
const std::string &TrtDepthEstimator::status() const { return impl_->status(); }
int TrtDepthEstimator::inputWidth() const { return impl_->inputWidth(); }
int TrtDepthEstimator::inputHeight() const { return impl_->inputHeight(); }
bool TrtDepthEstimator::infer(const cv::Mat &bgr_roi, cv::Mat &relative_depth,
                              double &inference_ms) {
  return impl_->infer(bgr_roi, relative_depth, inference_ms);
}
