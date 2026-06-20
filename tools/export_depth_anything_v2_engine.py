#!/usr/bin/env python3
"""Export Depth Anything V2 Small (HF official integration) to TensorRT."""

from __future__ import annotations

import argparse
from pathlib import Path

def export_onnx(model_id: str, output: Path, input_size: int) -> None:
    # Keep these dependencies out of the TensorRT-only deployment path. A robot
    # rebuilding an engine from an existing ONNX file does not need PyTorch.
    import onnx
    import torch
    from torch import nn
    from transformers import AutoModelForDepthEstimation

    class DepthOnly(nn.Module):
        """Keep only the dense relative-depth tensor from the HF model output."""

        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            return self.model(pixel_values=pixel_values).predicted_depth.unsqueeze(1)

    print(f"Loading {model_id} ...")
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    wrapper = DepthOnly(model.eval()).cpu()
    dummy = torch.zeros(1, 3, input_size, input_size, dtype=torch.float32)

    print(f"Exporting ONNX: {output}")
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            dummy,
            str(output),
            input_names=["pixel_values"],
            output_names=["relative_depth"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes=None,
        )
    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)
    print(f"ONNX OK: {output.stat().st_size / 1024**2:.1f} MiB")


def build_engine(onnx_path: Path, engine_path: Path,
                 workspace_gib: float, fp16: bool) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT ONNX parse failed:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gib * (1 << 30))
    )
    if fp16:
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("This GPU does not report fast FP16 support")
        config.set_flag(trt.BuilderFlag.FP16)

    print(f"Building {'FP16' if fp16 else 'FP32'} TensorRT engine: {engine_path}")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    engine_path.write_bytes(bytes(serialized))
    print(f"TensorRT OK: {engine_path.stat().st_size / 1024**2:.1f} MiB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id", default="depth-anything/Depth-Anything-V2-Small-hf"
    )
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--onnx", type=Path,
                        default=Path("depth_anything_v2_vits.onnx"))
    parser.add_argument("--engine", type=Path,
                        default=Path("depth_anything_v2_vits.engine"))
    parser.add_argument("--workspace", type=float, default=2.0)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--reuse-onnx", action="store_true")
    args = parser.parse_args()

    if not args.reuse_onnx or not args.onnx.exists():
        export_onnx(args.model_id, args.onnx, args.input_size)
    build_engine(args.onnx, args.engine, args.workspace, not args.fp32)
    print(args.engine.resolve())


if __name__ == "__main__":
    main()
