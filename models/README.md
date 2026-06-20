# Runtime models

Generate the TensorRT engine on the target robot and place it here:

```bash
python3 tools/export_depth_anything_v2_engine.py \
  --reuse-onnx \
  --onnx depth_anything_v2_vits.onnx \
  --engine models/depth_anything_v2_vits.engine
```

TensorRT engine files are intentionally excluded from Git.
