from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a YOLO detector on the detector_ai dataset.")
    parser.add_argument(
        "--data",
        default=str(Path("data") / "detector_ai" / "detector_ai_yolo" / "dataset.yaml"),
        help="Path to the generated dataset.yaml file.",
    )
    parser.add_argument("--model", default="data/models/yolo11n.pt", help="Base model or checkpoint to fine-tune.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--device", default=None, help="Training device, for example cpu or 0.")
    parser.add_argument("--project", default=str(Path("data") / "detector_ai" / "runs"), help="Output project directory.")
    parser.add_argument("--name", default="grimforest_detector", help="Run name.")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as exc:
        print("Ultralytics is required for YOLO training.")
        print(f"Import error: {exc}")
        return 1

    model = YOLO(args.model)
    train_kwargs = {
        "data": args.data,
        "epochs": int(args.epochs),
        "imgsz": int(args.imgsz),
        "project": args.project,
        "name": args.name,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
