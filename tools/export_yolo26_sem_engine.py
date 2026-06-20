#!/usr/bin/env python3
"""Export an Ultralytics YOLO26 semantic model to a TensorRT engine."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo26n-sem.pt")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--workspace", type=float, default=4.0,
                        help="TensorRT workspace in GiB")
    parser.add_argument("--device", default="0")
    parser.add_argument("--fp32", action="store_true",
                        help="Disable the default FP16 optimization")
    parser.add_argument("--dynamic", action="store_true")
    args = parser.parse_args()

    exported = YOLO(args.model).export(
        format="engine",
        imgsz=args.imgsz,
        batch=1,
        half=not args.fp32,
        dynamic=args.dynamic,
        workspace=args.workspace,
        device=args.device,
    )
    print(Path(exported).resolve())


if __name__ == "__main__":
    main()
