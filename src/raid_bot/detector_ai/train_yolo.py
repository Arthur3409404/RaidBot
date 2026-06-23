from __future__ import annotations

import argparse
from pathlib import Path


def resolve_training_device(requested_device: str | None) -> str | None:
    """Normalize the requested device and gracefully handle CPU-only installs."""
    if requested_device is None:
        return None

    normalized = str(requested_device).strip()
    if not normalized:
        return None
    if normalized.lower() == "cpu":
        return "cpu"

    try:
        import torch
    except Exception:
        return normalized

    if torch.cuda.is_available():
        return normalized

    print(
        f"CUDA device '{normalized}' was requested, but this Python environment has CPU-only PyTorch "
        f"({torch.__version__}). Falling back to 'cpu'."
    )
    return "cpu"


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
    resolved_device = resolve_training_device(args.device)
    if resolved_device is not None:
        train_kwargs["device"] = resolved_device

    model.train(**train_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
