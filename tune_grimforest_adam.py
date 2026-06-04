from __future__ import annotations

import argparse
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

GROUP_ORDER = ["T1", "T2", "T3", "T4", "T5", "T6"]
AVOID_GROUP = "avoid"


@dataclass(frozen=True)
class LabeledRun:
    image_path: Path
    true_count: int


@dataclass(frozen=True)
class TemplateImage:
    name: str
    image: np.ndarray


@dataclass(frozen=True)
class Detection:
    x: int
    y: int
    width: int
    height: int
    score: float
    group: str
    template_name: str


def _load_labeled_runs(debug_dir: Path) -> list[LabeledRun]:
    runs: list[LabeledRun] = []
    for run_dir in sorted(debug_dir.glob("run_*")):
        meta_path = run_dir / "run_meta.json"
        image_path = run_dir / "pov_cyan_brighter_binary.png"
        if not (meta_path.exists() and image_path.exists()):
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        label = ((meta.get("label") or {}).get("true_object_count"))
        if isinstance(label, int):
            runs.append(LabeledRun(image_path=image_path, true_count=int(label)))
    return runs


def _select_training_runs(runs: list[LabeledRun], benchmark: bool) -> list[LabeledRun]:
    if not benchmark:
        return runs
    eligible = [r for r in runs if int(r.true_count) >= 5]
    if len(eligible) < 2:
        raise RuntimeError(
            "Benchmark mode requires at least 2 labeled runs with true_object_count >= 5."
        )
    eligible.sort(key=lambda r: r.image_path.parent.name)
    return eligible[:2]


def _load_binary_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path.as_posix())
    _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return bw


def _classify_template_group(path: Path) -> str | None:
    stem = path.stem.strip()
    lower = stem.lower()
    if lower == AVOID_GROUP or lower.startswith(f"{AVOID_GROUP}_"):
        return AVOID_GROUP
    match = re.match(r"^(t[1-6])(?:_|$)", lower)
    if not match:
        return None
    return match.group(1).upper()


def _load_templates_by_group(template_dir: Path) -> dict[str, list[TemplateImage]]:
    grouped: dict[str, list[TemplateImage]] = {group: [] for group in GROUP_ORDER}
    grouped[AVOID_GROUP] = []

    for path in sorted(template_dir.glob("*.png")):
        group = _classify_template_group(path)
        if group is None:
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        grouped[group].append(TemplateImage(name=path.name, image=bw))

    missing = [group for group in GROUP_ORDER if len(grouped[group]) == 0]
    if missing:
        raise RuntimeError(f"Missing template group(s) in {template_dir.as_posix()}: {missing}")
    if len(grouped[AVOID_GROUP]) == 0:
        raise RuntimeError(
            f"Missing '{AVOID_GROUP}' template group in {template_dir.as_posix()} "
            "(expected files like avoid.png or avoid_*.png)."
        )
    return grouped


def _bbox_iou(a: Detection, b: Detection) -> float:
    ax1, ay1 = int(a.x), int(a.y)
    ax2, ay2 = int(a.x + a.width), int(a.y + a.height)
    bx1, by1 = int(b.x), int(b.y)
    bx2, by2 = int(b.x + b.width), int(b.y + b.height)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (a.width * a.height) + (b.width * b.height) - inter
    return float(inter / union) if union > 0 else 0.0


def _boxes_intersect(a: Detection, b: Detection) -> bool:
    ax1, ay1 = int(a.x), int(a.y)
    ax2, ay2 = int(a.x + a.width), int(a.y + a.height)
    bx1, by1 = int(b.x), int(b.y)
    bx2, by2 = int(b.x + b.width), int(b.y + b.height)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return (ix2 - ix1) > 0 and (iy2 - iy1) > 0


def _nms(detections: list[Detection], iou_thresh: float) -> list[Detection]:
    sorted_dets = sorted(
        detections,
        key=lambda d: (-float(d.score), int(d.y), int(d.x), d.template_name),
    )
    kept: list[Detection] = []
    for det in sorted_dets:
        if all(_bbox_iou(det, prev) <= float(iou_thresh) for prev in kept):
            kept.append(det)
    return kept


def _peak_points(result: np.ndarray, threshold: float, topk_per_template: int) -> tuple[np.ndarray, np.ndarray]:
    mask = result >= float(threshold)
    if not bool(np.any(mask)):
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    dilated = cv2.dilate(result, np.ones((3, 3), dtype=np.float32))
    peak_mask = mask & (result >= dilated)
    ys, xs = np.where(peak_mask)
    if len(ys) == 0:
        ys, xs = np.where(mask)
    topk = int(topk_per_template)
    if topk > 0 and len(ys) > topk:
        values = result[ys, xs]
        idx = np.argpartition(values, -topk)[-topk:]
        ys = ys[idx]
        xs = xs[idx]
    return ys.astype(np.int32), xs.astype(np.int32)


def _detect_group(
    binary_img: np.ndarray,
    templates: list[TemplateImage],
    threshold: float,
    topk_per_template: int,
    nms_iou_thresh: float,
    group_name: str,
) -> list[Detection]:
    detections: list[Detection] = []
    binary_float = binary_img.astype(np.float32)
    for tmpl in templates:
        th, tw = tmpl.image.shape[:2]
        if tw >= binary_img.shape[1] or th >= binary_img.shape[0]:
            continue
        tmpl_float = tmpl.image.astype(np.float32)
        if float(np.std(tmpl_float)) < 1e-6:
            continue
        result = cv2.matchTemplate(binary_float, tmpl_float, cv2.TM_CCOEFF_NORMED)
        ys, xs = _peak_points(result, threshold=threshold, topk_per_template=topk_per_template)
        for y, x in zip(ys, xs):
            score = float(result[int(y), int(x)])
            detections.append(
                Detection(
                    x=int(x),
                    y=int(y),
                    width=int(tw),
                    height=int(th),
                    score=score,
                    group=group_name,
                    template_name=tmpl.name,
                )
            )
    detections = _nms(detections, iou_thresh=nms_iou_thresh)
    detections.sort(key=lambda d: (-float(d.score), int(d.y), int(d.x), d.template_name))
    return detections


def detect_grimforest_grouped(
    binary_img: np.ndarray,
    templates_by_group: dict[str, list[TemplateImage]],
    thresholds: dict[str, float],
    topk_per_template: int = 24,
    nms_iou_thresh: float = 0.25,
) -> dict:
    accepted_by_group: dict[str, list[Detection]] = {}
    for group in GROUP_ORDER:
        accepted_by_group[group] = _detect_group(
            binary_img=binary_img,
            templates=templates_by_group[group],
            threshold=_clamp01(thresholds[f"threshold_{group}"]),
            topk_per_template=int(topk_per_template),
            nms_iou_thresh=float(nms_iou_thresh),
            group_name=group,
        )

    avoid_detections = _detect_group(
        binary_img=binary_img,
        templates=templates_by_group[AVOID_GROUP],
        threshold=_clamp01(thresholds["threshold_avoid"]),
        topk_per_template=int(topk_per_template),
        nms_iou_thresh=float(nms_iou_thresh),
        group_name=AVOID_GROUP,
    )

    filtered_by_group: dict[str, list[Detection]] = {}
    for group in GROUP_ORDER:
        group_dets = accepted_by_group[group]
        filtered_by_group[group] = [
            det for det in group_dets if all(not _boxes_intersect(det, bad) for bad in avoid_detections)
        ]

    final_detections: list[Detection] = []
    for group in GROUP_ORDER:
        final_detections.extend(filtered_by_group[group])

    return {
        "detections": final_detections,
        "group_detections": filtered_by_group,
        "group_detections_before_avoid": accepted_by_group,
        "avoid_detections": avoid_detections,
    }


def _count_precision(true_count: int, pred_count: int) -> float:
    if pred_count <= 0:
        return 1.0 if int(true_count) == 0 else 0.0
    tp = min(int(true_count), int(pred_count))
    return float(tp / max(1, int(pred_count)))


def _count_f1(true_count: int, pred_count: int) -> float:
    tp = min(int(true_count), int(pred_count))
    fp = max(0, int(pred_count) - int(true_count))
    fn = max(0, int(true_count) - int(pred_count))
    denom = float((2 * tp) + fp + fn)
    if denom <= 0.0:
        return 1.0
    return float((2.0 * float(tp)) / denom)


def _evaluate_runs(
    runs: list[LabeledRun],
    templates_by_group: dict[str, list[TemplateImage]],
    thresholds: dict[str, float],
    loss_mode: str,
    topk_per_template: int,
    nms_iou_thresh: float,
    on_image_evaluated: Callable[[int], None] | None = None,
    executor: ThreadPoolExecutor | None = None,
) -> dict:
    def evaluate_one(run: LabeledRun) -> dict:
        bw = _load_binary_image(run.image_path)
        det_out = detect_grimforest_grouped(
            binary_img=bw,
            templates_by_group=templates_by_group,
            thresholds=thresholds,
            topk_per_template=topk_per_template,
            nms_iou_thresh=nms_iou_thresh,
        )
        pred_count = len(det_out["detections"])
        true_count = int(run.true_count)
        precision = _count_precision(true_count=true_count, pred_count=pred_count)
        f1 = _count_f1(true_count=true_count, pred_count=pred_count)
        if not (0.0 <= float(f1) <= 1.0):
            raise RuntimeError(f"F1 must be between 0 and 1. Got: {f1}")
        mse = float((pred_count - true_count) ** 2)
        one_minus_precision = float(1.0 - precision)
        one_minus_f1 = float(1.0 - f1)
        if loss_mode == "mse_plus_one_minus_f1":
            loss = float(mse + one_minus_f1)
        elif loss_mode == "mse":
            loss = mse
        else:
            loss = one_minus_precision
        return {
            "run": run.image_path.parent.name,
            "true": true_count,
            "pred": pred_count,
            "precision": precision,
            "f1": f1,
            "loss": loss,
            "mse": mse,
            "one_minus_precision": one_minus_precision,
            "one_minus_f1": one_minus_f1,
        }

    rows: list[dict] = []
    if executor is None or len(runs) <= 1:
        for run in runs:
            rows.append(evaluate_one(run))
            if on_image_evaluated is not None:
                on_image_evaluated(1)
    else:
        futures = [executor.submit(evaluate_one, run) for run in runs]
        for future in as_completed(futures):
            rows.append(future.result())
            if on_image_evaluated is not None:
                on_image_evaluated(1)

    rows.sort(key=lambda row: str(row.get("run", "")))

    n = len(rows)
    mean_precision = float(sum(r["precision"] for r in rows) / n) if n else 0.0
    mean_f1 = float(sum(r["f1"] for r in rows) / n) if n else 0.0
    mean_mse = float(sum(r["mse"] for r in rows) / n) if n else 0.0
    mean_one_minus_precision = float(sum(r["one_minus_precision"] for r in rows) / n) if n else 1.0
    mean_one_minus_f1 = float(sum(r["one_minus_f1"] for r in rows) / n) if n else 1.0
    if not (0.0 <= float(mean_f1) <= 1.0):
        raise RuntimeError(f"Mean F1 must be between 0 and 1. Got: {mean_f1}")
    if loss_mode == "mse_plus_one_minus_f1":
        mean_loss = float(mean_mse + mean_one_minus_f1)
    elif loss_mode == "mse":
        mean_loss = mean_mse
    else:
        mean_loss = mean_one_minus_precision
    return {
        "rows": rows,
        "n": n,
        "precision": mean_precision,
        "f1": mean_f1,
        "mse": mean_mse,
        "one_minus_precision": mean_one_minus_precision,
        "one_minus_f1": mean_one_minus_f1,
        "loss": mean_loss,
    }


def _threshold_vector_to_dict(vec: np.ndarray) -> dict[str, float]:
    return {
        "threshold_T1": float(vec[0]),
        "threshold_T2": float(vec[1]),
        "threshold_T3": float(vec[2]),
        "threshold_T4": float(vec[3]),
        "threshold_T5": float(vec[4]),
        "threshold_T6": float(vec[5]),
        "threshold_avoid": float(vec[6]),
    }


def _threshold_dict_to_vector(thresholds: dict[str, float]) -> np.ndarray:
    return np.array(
        [
            float(thresholds["threshold_T1"]),
            float(thresholds["threshold_T2"]),
            float(thresholds["threshold_T3"]),
            float(thresholds["threshold_T4"]),
            float(thresholds["threshold_T5"]),
            float(thresholds["threshold_T6"]),
            float(thresholds["threshold_avoid"]),
        ],
        dtype=np.float64,
    )


def _batched(items: list[LabeledRun], batch_size: int) -> Iterable[list[LabeledRun]]:
    size = max(1, int(batch_size))
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def _clip_thresholds(vec: np.ndarray) -> np.ndarray:
    return np.clip(vec, 0.0, 1.0)


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, float(value))))


def _render_progress_bar(epoch: int, epochs: int, done: int, total: int, width: int = 28) -> None:
    total_safe = max(1, int(total))
    done_clamped = min(total_safe, max(0, int(done)))
    ratio = float(done_clamped) / float(total_safe)
    filled = int(round(ratio * int(width)))
    filled = min(int(width), max(0, filled))
    bar = ("#" * filled) + ("-" * (int(width) - filled))
    print(
        f"\rEpoch {epoch}/{epochs} image eval progress [{bar}] {done_clamped}/{total_safe}",
        end="",
        flush=True,
    )


def _validate_initial_thresholds(args: argparse.Namespace) -> None:
    keys = [
        "threshold_T1",
        "threshold_T2",
        "threshold_T3",
        "threshold_T4",
        "threshold_T5",
        "threshold_T6",
        "threshold_avoid",
    ]
    for key in keys:
        value = float(getattr(args, key))
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{key} must be between 0 and 1, got {value}.")


def tune(args: argparse.Namespace) -> int:
    debug_dir = Path(args.debug_dir)
    template_dir = Path(args.template_dir)
    report_path = Path(args.report_path)

    runs_all = _load_labeled_runs(debug_dir)
    if not runs_all:
        print(f"No labeled runs found in {debug_dir.as_posix()}")
        return 1
    runs_train = _select_training_runs(runs_all, benchmark=bool(args.benchmark))
    templates_by_group = _load_templates_by_group(template_dir)

    if bool(args.benchmark):
        print(
            f"[Benchmark] Using {len(runs_train)} runs with true_object_count >= 5 "
            f"(from {len(runs_all)} labeled runs)."
        )
    else:
        print(f"[Full] Using all labeled runs for training: {len(runs_train)}")

    theta = _threshold_dict_to_vector(
        {
            "threshold_T1": float(args.threshold_T1),
            "threshold_T2": float(args.threshold_T2),
            "threshold_T3": float(args.threshold_T3),
            "threshold_T4": float(args.threshold_T4),
            "threshold_T5": float(args.threshold_T5),
            "threshold_T6": float(args.threshold_T6),
            "threshold_avoid": float(args.threshold_avoid),
        }
    )
    theta = _clip_thresholds(theta)

    m = np.zeros_like(theta, dtype=np.float64)
    v = np.zeros_like(theta, dtype=np.float64)
    rng = np.random.default_rng(int(args.seed))
    worker_count = int(args.workers)
    if worker_count <= 0:
        worker_count = max(1, (os.cpu_count() or 2) - 1)
    worker_count = min(worker_count, max(1, len(runs_train)))
    if worker_count > 1:
        cv2.setNumThreads(1)
    print(f"Evaluation workers: {worker_count}")

    executor = ThreadPoolExecutor(max_workers=worker_count) if worker_count > 1 else None

    try:
        initial_eval = _evaluate_runs(
            runs=runs_train,
            templates_by_group=templates_by_group,
            thresholds=_threshold_vector_to_dict(theta),
            loss_mode=str(args.loss),
            topk_per_template=int(args.topk_per_template),
            nms_iou_thresh=float(args.nms_iou_thresh),
            executor=executor,
        )
        best_theta = theta.copy()
        best_eval = initial_eval
        global_step = 0

        print(
            f"Initial: loss={initial_eval['loss']:.6f}, precision={initial_eval['precision']:.6f}, "
            f"f1={initial_eval['f1']:.6f}, mse={initial_eval['mse']:.6f}, "
            f"one_minus_f1={initial_eval['one_minus_f1']:.6f}, "
            f"one_minus_precision={initial_eval['one_minus_precision']:.6f}"
        )
        print(json.dumps(_threshold_vector_to_dict(theta), indent=2))

        for epoch in range(1, int(args.epochs) + 1):
            order = rng.permutation(len(runs_train))
            shuffled = [runs_train[int(i)] for i in order]
            batches = list(_batched(shuffled, batch_size=int(args.batch_size)))
            epoch_total_image_evals = sum((2 * len(batch)) + len(runs_train) for batch in batches)
            epoch_done_image_evals = 0

            def on_epoch_image_eval(count: int) -> None:
                nonlocal epoch_done_image_evals
                epoch_done_image_evals += int(count)
                _render_progress_bar(
                    epoch=epoch,
                    epochs=int(args.epochs),
                    done=epoch_done_image_evals,
                    total=epoch_total_image_evals,
                )

            _render_progress_bar(
                epoch=epoch,
                epochs=int(args.epochs),
                done=epoch_done_image_evals,
                total=epoch_total_image_evals,
            )
            for batch_idx, batch in enumerate(batches, start=1):
                global_step += 1
                delta_sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=theta.shape)
                delta = float(args.spsa_delta)
                theta_plus = _clip_thresholds(theta + (delta * delta_sign))
                theta_minus = _clip_thresholds(theta - (delta * delta_sign))

                plus_eval = _evaluate_runs(
                    runs=batch,
                    templates_by_group=templates_by_group,
                    thresholds=_threshold_vector_to_dict(theta_plus),
                    loss_mode=str(args.loss),
                    topk_per_template=int(args.topk_per_template),
                    nms_iou_thresh=float(args.nms_iou_thresh),
                    on_image_evaluated=on_epoch_image_eval,
                    executor=executor,
                )
                minus_eval = _evaluate_runs(
                    runs=batch,
                    templates_by_group=templates_by_group,
                    thresholds=_threshold_vector_to_dict(theta_minus),
                    loss_mode=str(args.loss),
                    topk_per_template=int(args.topk_per_template),
                    nms_iou_thresh=float(args.nms_iou_thresh),
                    on_image_evaluated=on_epoch_image_eval,
                    executor=executor,
                )

                grad_scalar = float(plus_eval["loss"] - minus_eval["loss"]) / max(1e-9, 2.0 * delta)
                grad = grad_scalar * delta_sign

                beta1 = float(args.adam_beta1)
                beta2 = float(args.adam_beta2)
                eps = float(args.adam_epsilon)
                lr = float(args.adam_lr)

                m = beta1 * m + (1.0 - beta1) * grad
                v = beta2 * v + (1.0 - beta2) * (grad * grad)
                m_hat = m / (1.0 - math.pow(beta1, global_step))
                v_hat = v / (1.0 - math.pow(beta2, global_step))
                theta = _clip_thresholds(theta - (lr * m_hat / (np.sqrt(v_hat) + eps)))

                train_eval = _evaluate_runs(
                    runs=runs_train,
                    templates_by_group=templates_by_group,
                    thresholds=_threshold_vector_to_dict(theta),
                    loss_mode=str(args.loss),
                    topk_per_template=int(args.topk_per_template),
                    nms_iou_thresh=float(args.nms_iou_thresh),
                    on_image_evaluated=on_epoch_image_eval,
                    executor=executor,
                )
                if float(train_eval["loss"]) < float(best_eval["loss"]):
                    best_eval = train_eval
                    best_theta = theta.copy()

                print()
                threshold_dict = _threshold_vector_to_dict(theta)
                print(
                    f"[epoch {epoch}/{args.epochs}] [batch {batch_idx}/{len(batches)}] "
                    f"step={global_step} loss={train_eval['loss']:.6f} precision={train_eval['precision']:.6f} "
                    f"f1={train_eval['f1']:.6f} mse={train_eval['mse']:.6f} "
                    f"one_minus_f1={train_eval['one_minus_f1']:.6f} "
                    f"one_minus_precision={train_eval['one_minus_precision']:.6f} "
                    f"T1={threshold_dict['threshold_T1']:.4f} T2={threshold_dict['threshold_T2']:.4f} "
                    f"T3={threshold_dict['threshold_T3']:.4f} T4={threshold_dict['threshold_T4']:.4f} "
                    f"T5={threshold_dict['threshold_T5']:.4f} T6={threshold_dict['threshold_T6']:.4f} "
                    f"avoid={threshold_dict['threshold_avoid']:.4f}"
                )
                _render_progress_bar(
                    epoch=epoch,
                    epochs=int(args.epochs),
                    done=epoch_done_image_evals,
                    total=epoch_total_image_evals,
                )
            print()

        best_thresholds = _threshold_vector_to_dict(best_theta)
        print("Best thresholds:")
        print(json.dumps(best_thresholds, indent=2))
        print(
            f"Best metrics: loss={best_eval['loss']:.6f}, precision={best_eval['precision']:.6f}, "
            f"f1={best_eval['f1']:.6f}, mse={best_eval['mse']:.6f}, "
            f"one_minus_f1={best_eval['one_minus_f1']:.6f}, "
            f"one_minus_precision={best_eval['one_minus_precision']:.6f}"
        )

        report = {
            "mode": "grim_forest",
            "benchmark": bool(args.benchmark),
            "debug_dir": debug_dir.as_posix(),
            "template_dir": template_dir.as_posix(),
            "loss_mode": str(args.loss),
            "optimizer": {
                "name": "adam_spsa",
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "adam_lr": float(args.adam_lr),
                "adam_beta1": float(args.adam_beta1),
                "adam_beta2": float(args.adam_beta2),
                "adam_epsilon": float(args.adam_epsilon),
                "spsa_delta": float(args.spsa_delta),
                "seed": int(args.seed),
                "workers": int(worker_count),
            },
            "detector": {
                "topk_per_template": int(args.topk_per_template),
                "nms_iou_thresh": float(args.nms_iou_thresh),
                "groups": GROUP_ORDER,
                "avoid_group": AVOID_GROUP,
            },
            "train_run_count": len(runs_train),
            "best_thresholds": best_thresholds,
            "best_metrics": {
                "loss": float(best_eval["loss"]),
                "precision": float(best_eval["precision"]),
                "f1": float(best_eval["f1"]),
                "mse": float(best_eval["mse"]),
                "one_minus_f1": float(best_eval["one_minus_f1"]),
                "one_minus_precision": float(best_eval["one_minus_precision"]),
            },
            "per_run": best_eval["rows"],
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved report: {report_path.as_posix()}")
        return 0
    finally:
        if executor is not None:
            executor.shutdown(wait=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune Grim Forest grouped template thresholds (T1..T6 + avoid) with Adam."
    )
    parser.add_argument("--debug-dir", default="debug/grim_forest_standalone")
    parser.add_argument("--template-dir", default="pic/grimforest_test")
    parser.add_argument(
        "--report-path",
        default="debug/grim_forest_standalone/grimforest_adam_threshold_report.json",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Train on only 2 runs where true_object_count >= 5.",
    )
    parser.add_argument(
        "--loss",
        choices=["mse_plus_one_minus_f1", "mse", "one_minus_precision"],
        default="mse_plus_one_minus_f1",
        help="Training objective.",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--adam-lr", type=float, default=3e-3)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--spsa-delta", type=float, default=0.03)
    parser.add_argument("--topk-per-template", type=int, default=24)
    parser.add_argument("--nms-iou-thresh", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=0, help="Image-evaluation workers. 0 = auto.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--threshold-T1", type=float, default=0.6, dest="threshold_T1")
    parser.add_argument("--threshold-T2", type=float, default=0.6, dest="threshold_T2")
    parser.add_argument("--threshold-T3", type=float, default=0.6, dest="threshold_T3")
    parser.add_argument("--threshold-T4", type=float, default=0.6, dest="threshold_T4")
    parser.add_argument("--threshold-T5", type=float, default=0.6, dest="threshold_T5")
    parser.add_argument("--threshold-T6", type=float, default=0.6, dest="threshold_T6")
    parser.add_argument("--threshold-avoid", type=float, default=0.72, dest="threshold_avoid")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _validate_initial_thresholds(args)
    return tune(args)


if __name__ == "__main__":
    raise SystemExit(main())
