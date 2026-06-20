#include "trt_semantic_segmenter.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <opencv2/imgproc.hpp>

#ifdef CAFM_HAS_TENSORRT
#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#endif

namespace {

#ifdef CAFM_HAS_TENSORRT
struct LetterboxInfo {
  int resized_w = 0;
  int resized_h = 0;
  int pad_left = 0;
  int pad_top = 0;
};

cv::Mat letterbox(const cv::Mat &src, int dst_w, int dst_h, LetterboxInfo &info) {
  const double scale = std::min(static_cast<double>(dst_w) / src.cols,
                                static_cast<double>(dst_h) / src.rows);
  info.resized_w = std::max(1, static_cast<int>(std::round(src.cols * scale)));
  info.resized_h = std::max(1, static_cast<int>(std::round(src.rows * scale)));
  info.pad_left = (dst_w - info.resized_w) / 2;
  info.pad_top = (dst_h - info.resized_h) / 2;

  cv::Mat resized;
  cv::resize(src, resized, {info.resized_w, info.resized_h}, 0.0, 0.0,
             cv::INTER_LINEAR);
  cv::Mat output(dst_h, dst_w, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(output(cv::Rect(info.pad_left, info.pad_top,
                                 info.resized_w, info.resized_h)));
  return output;
}
#endif

}  // namespace

#ifdef CAFM_HAS_TENSORRT
namespace {

class TrtLogger final : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *message) noexcept override {
    if (severity <= Severity::kWARNING) {
      last_message = message ? message : "";
    }
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
  std::size_t n = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] <= 0) throw std::runtime_error("unresolved dynamic tensor shape");
    n *= static_cast<std::size_t>(dims.d[i]);
  }
  return n;
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

void checkCuda(cudaError_t error, const char *what) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
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
      return data[index] != 0 ? 1.0f : 0.0f;
    default:
      throw std::runtime_error("unsupported TensorRT output dtype");
  }
}

// Ultralytics prepends: uint32 metadata_length + UTF-8 JSON + TRT blob.
std::vector<char> readEngineBlob(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("engine file open failed: " + path);
  const auto end = file.tellg();
  if (end <= 0) throw std::runtime_error("engine file is empty: " + path);
  std::vector<char> bytes(static_cast<std::size_t>(end));
  file.seekg(0);
  file.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!file) throw std::runtime_error("engine file read failed: " + path);

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

class TrtSemanticSegmenter::Impl {
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

  bool infer(const cv::Mat &bgr_roi, cv::Mat &class_map, double &inference_ms) {
#ifdef CAFM_HAS_TENSORRT
    if (!ready_ || bgr_roi.empty()) return false;
    try {
      LetterboxInfo box;
      const cv::Mat padded = letterbox(bgr_roi, input_w_, input_h_, box);
      prepareInput(padded);

      const auto start = std::chrono::steady_clock::now();
      checkCuda(cudaMemcpyAsync(input_device_, input_host_.data(), input_bytes_,
                                cudaMemcpyHostToDevice, stream_), "copy input");
      if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("TensorRT enqueueV3 failed");
      }
      checkCuda(cudaMemcpyAsync(output_host_.data(), output_device_, output_bytes_,
                                cudaMemcpyDeviceToHost, stream_), "copy output");
      checkCuda(cudaStreamSynchronize(stream_), "synchronize inference");
      inference_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start).count();

      decodeOutput(box, bgr_roi.size(), class_map);
      return true;
    } catch (const std::exception &e) {
      status_ = e.what();
      return false;
    }
#else
    (void)bgr_roi;
    (void)class_map;
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
      throw std::runtime_error("TensorRT engine deserialization failed (version/GPU mismatch): " +
                               logger_.last_message);
    }
    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("TensorRT execution context creation failed");

    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
      const char *name = engine_->getIOTensorName(i);
      if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
        if (!input_name_.empty()) throw std::runtime_error("expected exactly one model input");
        input_name_ = name;
      } else if (output_name_.empty()) {
        output_name_ = name;
      }
    }
    if (input_name_.empty() || output_name_.empty()) {
      throw std::runtime_error("TensorRT input/output tensor not found");
    }

    nvinfer1::Dims input_dims = engine_->getTensorShape(input_name_.c_str());
    if (input_dims.nbDims != 4) {
      throw std::runtime_error("expected BCHW input, got " + dimsString(input_dims));
    }
    if (input_dims.d[1] > 0 && input_dims.d[1] != 3) {
      throw std::runtime_error("expected 3 input channels");
    }
    input_w_ = input_dims.d[3] > 0 ? input_dims.d[3] : requested_w;
    input_h_ = input_dims.d[2] > 0 ? input_dims.d[2] : requested_h;
    if (input_w_ <= 0 || input_h_ <= 0) {
      throw std::runtime_error("dynamic engine requires positive semantic_input_width/height");
    }
    if (!context_->setInputShape(input_name_.c_str(),
                                 nvinfer1::Dims4{1, 3, input_h_, input_w_})) {
      throw std::runtime_error("TensorRT rejected requested input shape");
    }

    input_dims = context_->getTensorShape(input_name_.c_str());
    output_dims_ = context_->getTensorShape(output_name_.c_str());
    input_type_ = engine_->getTensorDataType(input_name_.c_str());
    output_type_ = engine_->getTensorDataType(output_name_.c_str());
    if (input_type_ != nvinfer1::DataType::kFLOAT &&
        input_type_ != nvinfer1::DataType::kHALF) {
      throw std::runtime_error("semantic model input must be FP32 or FP16");
    }
    validateOutputShape();

    input_bytes_ = volume(input_dims) * elementSize(input_type_);
    output_bytes_ = volume(output_dims_) * elementSize(output_type_);
    input_host_.resize(input_bytes_);
    output_host_.resize(output_bytes_);
    checkCuda(cudaStreamCreate(&stream_), "create CUDA stream");
    checkCuda(cudaMalloc(&input_device_, input_bytes_), "allocate input buffer");
    checkCuda(cudaMalloc(&output_device_, output_bytes_), "allocate output buffer");
    if (!context_->setTensorAddress(input_name_.c_str(), input_device_) ||
        !context_->setTensorAddress(output_name_.c_str(), output_device_)) {
      throw std::runtime_error("TensorRT setTensorAddress failed");
    }

    ready_ = true;
    status_ = "ready: input=" + dimsString(input_dims) +
              " output=" + dimsString(output_dims_);
  }

  void validateOutputShape() {
    if (output_dims_.nbDims == 4) {
      if (output_dims_.d[0] != 1 || output_dims_.d[1] <= 0 ||
          output_dims_.d[2] <= 0 || output_dims_.d[3] <= 0) {
        throw std::runtime_error("invalid semantic logits shape " + dimsString(output_dims_));
      }
      return;
    }
    if (output_dims_.nbDims == 3 && output_dims_.d[0] == 1 &&
        output_dims_.d[1] > 0 && output_dims_.d[2] > 0) {
      return;
    }
    throw std::runtime_error("expected [1,C,H,W] logits or [1,H,W] class map, got " +
                             dimsString(output_dims_));
  }

  void prepareInput(const cv::Mat &bgr) {
    const std::size_t plane = static_cast<std::size_t>(input_w_) * input_h_;
    if (input_type_ == nvinfer1::DataType::kFLOAT) {
      auto *dst = reinterpret_cast<float *>(input_host_.data());
      for (int y = 0; y < input_h_; ++y) {
        const cv::Vec3b *row = bgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < input_w_; ++x) {
          const std::size_t p = static_cast<std::size_t>(y) * input_w_ + x;
          dst[p] = row[x][2] / 255.0f;
          dst[plane + p] = row[x][1] / 255.0f;
          dst[2 * plane + p] = row[x][0] / 255.0f;
        }
      }
    } else {
      auto *dst = reinterpret_cast<__half *>(input_host_.data());
      for (int y = 0; y < input_h_; ++y) {
        const cv::Vec3b *row = bgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < input_w_; ++x) {
          const std::size_t p = static_cast<std::size_t>(y) * input_w_ + x;
          dst[p] = __float2half(row[x][2] / 255.0f);
          dst[plane + p] = __float2half(row[x][1] / 255.0f);
          dst[2 * plane + p] = __float2half(row[x][0] / 255.0f);
        }
      }
    }
  }

  cv::Rect outputCrop(const LetterboxInfo &box, int out_w, int out_h) const {
    const double sx = static_cast<double>(out_w) / input_w_;
    const double sy = static_cast<double>(out_h) / input_h_;
    int x = std::clamp(static_cast<int>(std::round(box.pad_left * sx)), 0, out_w - 1);
    int y = std::clamp(static_cast<int>(std::round(box.pad_top * sy)), 0, out_h - 1);
    int w = std::clamp(static_cast<int>(std::round(box.resized_w * sx)), 1, out_w - x);
    int h = std::clamp(static_cast<int>(std::round(box.resized_h * sy)), 1, out_h - y);
    return {x, y, w, h};
  }

  void decodeOutput(const LetterboxInfo &box, cv::Size roi_size, cv::Mat &class_map) const {
    if (output_dims_.nbDims == 3) {
      const int out_h = output_dims_.d[1];
      const int out_w = output_dims_.d[2];
      cv::Mat raw(out_h, out_w, CV_8U);
      for (std::size_t i = 0; i < static_cast<std::size_t>(out_h) * out_w; ++i) {
        raw.data[i] = cv::saturate_cast<std::uint8_t>(
            std::round(valueAt(output_host_, output_type_, i)));
      }
      cv::resize(raw(outputCrop(box, out_w, out_h)), class_map, roi_size,
                 0.0, 0.0, cv::INTER_NEAREST);
      return;
    }

    const int classes = output_dims_.d[1];
    const int out_h = output_dims_.d[2];
    const int out_w = output_dims_.d[3];
    const std::size_t plane = static_cast<std::size_t>(out_h) * out_w;
    const cv::Rect crop = outputCrop(box, out_w, out_h);
    cv::Mat best(roi_size, CV_32F,
                 cv::Scalar(-std::numeric_limits<float>::infinity()));
    class_map = cv::Mat::zeros(roi_size, CV_8U);

    // Ultralytics upsamples logits before argmax. Resize each class plane to
    // preserve that behavior instead of resizing an already-quantized map.
    for (int c = 0; c < classes; ++c) {
      cv::Mat logits(out_h, out_w, CV_32F);
      float *dst = logits.ptr<float>();
      const std::size_t offset = static_cast<std::size_t>(c) * plane;
      for (std::size_t i = 0; i < plane; ++i) {
        dst[i] = valueAt(output_host_, output_type_, offset + i);
      }
      cv::Mat resized;
      cv::resize(logits(crop), resized, roi_size, 0.0, 0.0, cv::INTER_LINEAR);
      for (int y = 0; y < roi_size.height; ++y) {
        const float *score = resized.ptr<float>(y);
        float *winner = best.ptr<float>(y);
        std::uint8_t *label = class_map.ptr<std::uint8_t>(y);
        for (int x = 0; x < roi_size.width; ++x) {
          if (score[x] > winner[x]) {
            winner[x] = score[x];
            label[x] = cv::saturate_cast<std::uint8_t>(c);
          }
        }
      }
    }
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

  TrtLogger logger_;
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
};

TrtSemanticSegmenter::TrtSemanticSegmenter(const std::string &engine_path,
                                           int requested_input_width,
                                           int requested_input_height)
    : impl_(std::make_unique<Impl>(engine_path, requested_input_width,
                                  requested_input_height)) {}

TrtSemanticSegmenter::~TrtSemanticSegmenter() = default;
bool TrtSemanticSegmenter::ready() const { return impl_->ready(); }
const std::string &TrtSemanticSegmenter::status() const { return impl_->status(); }
int TrtSemanticSegmenter::inputWidth() const { return impl_->inputWidth(); }
int TrtSemanticSegmenter::inputHeight() const { return impl_->inputHeight(); }
bool TrtSemanticSegmenter::infer(const cv::Mat &bgr_roi, cv::Mat &class_map,
                                 double &inference_ms) {
  return impl_->infer(bgr_roi, class_map, inference_ms);
}
