from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from run_grim_forest_standalone import detect_grimforest_like_structures

BENCHMARK_TRUE_COUNT = 6
ITERATIONS = 1000


@dataclass(frozen=True)
class LabeledRun:
    image_path: Path
    true_count: int


@dataclass(frozen=True)
class SoftEvalRun:
    run_name: str
    true_count: int
    binary_img: np.ndarray


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


def _select_benchmark_run(
    runs: list[LabeledRun],
    benchmark_run: str,
) -> LabeledRun:
    if not runs:
        raise RuntimeError("No labeled runs available for benchmark mode.")

    if benchmark_run:
        selected = [r for r in runs if r.image_path.parent.name == benchmark_run]
        if not selected:
            raise RuntimeError(f"Benchmark run '{benchmark_run}' not found among labeled runs.")
        chosen = selected[0]
        if int(chosen.true_count) != int(BENCHMARK_TRUE_COUNT):
            raise RuntimeError(
                f"Benchmark run '{benchmark_run}' has true_object_count={int(chosen.true_count)}, "
                f"but benchmark mode requires true_object_count={int(BENCHMARK_TRUE_COUNT)}."
            )
        return chosen

    selected_by_label = [r for r in runs if int(r.true_count) == int(BENCHMARK_TRUE_COUNT)]
    if not selected_by_label:
        raise RuntimeError(
            f"No labeled run found with true_object_count={int(BENCHMARK_TRUE_COUNT)} for benchmark mode."
        )
    selected_by_label.sort(key=lambda r: r.image_path.parent.name)
    return selected_by_label[0]


def _write_template_set(template_dir: Path, templates: list[np.ndarray]) -> None:
    template_dir.mkdir(parents=True, exist_ok=True)
    for old in template_dir.glob("*.png"):
        old.unlink()
    for idx, t in enumerate(templates, start=1):
        out_path = template_dir / f"optimized_{idx:02d}.png"
        cv2.imwrite(str(out_path), np.clip(t, 0, 255).astype(np.uint8))


def _pack_templates_to_strip(templates: list[np.ndarray], tile_size: int = 64) -> np.ndarray:
    prepared: list[np.ndarray] = []
    for t in templates:
        if t.shape != (tile_size, tile_size):
            resized = cv2.resize(t, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
            prepared.append(np.clip(resized, 0, 255).astype(np.uint8))
        else:
            prepared.append(np.clip(t, 0, 255).astype(np.uint8))
    if not prepared:
        raise RuntimeError("Cannot pack empty template list into strip.")
    return np.concatenate(prepared, axis=1)


def _initialize_random_strip_matrix(seed: int, width: int = 256, height: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((height, width), dtype=np.float32)


def _strip_matrix_to_templates(matrix_01: np.ndarray, template_count: int = 4, tile_size: int = 64) -> list[np.ndarray]:
    if matrix_01.ndim != 2:
        raise RuntimeError("Template matrix must be 2D.")
    expected_width = int(template_count) * int(tile_size)
    if matrix_01.shape[0] != tile_size or matrix_01.shape[1] != expected_width:
        raise RuntimeError(
            f"Template matrix must be {expected_width}x{tile_size} (WxH), got "
            f"{matrix_01.shape[1]}x{matrix_01.shape[0]}."
        )
    matrix_01 = np.clip(matrix_01, 0.0, 1.0).astype(np.float32)
    templates: list[np.ndarray] = []
    for idx in range(template_count):
        x0 = idx * tile_size
        x1 = x0 + tile_size
        tile_01 = matrix_01[:, x0:x1]
        tile = np.clip(np.round(tile_01 * 255.0), 0, 255).astype(np.uint8)
        templates.append(tile)
    return templates


def _write_template_strip(path: Path, templates: list[np.ndarray], tile_size: int = 64) -> Path:
    strip = _pack_templates_to_strip(templates, tile_size=tile_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), strip)
    return path


def _boxes_overlap(a, b) -> bool:
    ax1, ay1 = int(a.x), int(a.y)
    ax2, ay2 = int(a.x + a.width), int(a.y + a.height)
    bx1, by1 = int(b.x), int(b.y)
    bx2, by2 = int(b.x + b.width), int(b.y + b.height)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return (ix2 - ix1) > 0 and (iy2 - iy1) > 0


def _suppress_overlapping_boxes(boxes: list) -> list:
    sorted_boxes = sorted(boxes, key=lambda b: float(getattr(b, "score", 0.0)), reverse=True)
    kept = []
    for box in sorted_boxes:
        if all(not _boxes_overlap(box, prev) for prev in kept):
            kept.append(box)
    return kept


def _load_binary_for_loss(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path.as_posix())
    _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return bw


def _prepare_soft_eval_runs(runs: list[LabeledRun]) -> list[SoftEvalRun]:
    prepared: list[SoftEvalRun] = []
    for run in runs:
        bw = _load_binary_for_loss(run.image_path)
        prepared.append(
            SoftEvalRun(
                run_name=run.image_path.parent.name,
                true_count=int(run.true_count),
                binary_img=bw,
            )
        )
    return prepared


def _sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _soft_count_from_templates(
    binary_img: np.ndarray,
    templates: list[np.ndarray],
    max_objects: int,
    soft_temperature: float,
    soft_topk_per_map: int,
    soft_score_center: float,
) -> float:
    binary_float = binary_img.astype(np.float32)
    scales = np.linspace(1.0, 3.2, 12)
    score_parts: list[np.ndarray] = []
    per_map_topk = max(1, int(soft_topk_per_map))

    for template in templates:
        for scale in scales:
            tw = max(8, int(template.shape[1] * float(scale)))
            th = max(8, int(template.shape[0] * float(scale)))
            if tw >= binary_img.shape[1] or th >= binary_img.shape[0]:
                continue
            template_gray = cv2.resize(template, (tw, th), interpolation=cv2.INTER_AREA)
            template_float = template_gray.astype(np.float32)
            if float(np.std(template_float)) < 1e-6:
                continue
            response = cv2.matchTemplate(binary_float, template_float, cv2.TM_CCOEFF_NORMED)
            flat = response.reshape(-1)
            if flat.size <= 0:
                continue
            if flat.size <= per_map_topk:
                score_parts.append(flat.astype(np.float32))
            else:
                top_idx = np.argpartition(flat, -per_map_topk)[-per_map_topk:]
                score_parts.append(flat[top_idx].astype(np.float32))

    if not score_parts:
        return 0.0

    scores = np.concatenate(score_parts, axis=0)
    keep_k = min(max(1, int(max_objects)), int(scores.size))
    if scores.size > keep_k:
        top_idx = np.argpartition(scores, -keep_k)[-keep_k:]
        scores = scores[top_idx]
    activations = _sigmoid((scores - float(soft_score_center)) / max(float(soft_temperature), 1e-6))
    return float(np.sum(activations))


def _evaluate_template_set_soft(
    runs: list[SoftEvalRun],
    templates: list[np.ndarray],
    max_objects: int,
    soft_temperature: float,
    soft_topk_per_map: int,
    soft_score_center: float,
    status_every_runs: int = 0,
    executor: ThreadPoolExecutor | None = None,
) -> dict:
    total_runs = len(runs)

    def evaluate_one(run: SoftEvalRun) -> dict:
        soft_pred = _soft_count_from_templates(
            binary_img=run.binary_img,
            templates=templates,
            max_objects=max_objects,
            soft_temperature=soft_temperature,
            soft_topk_per_map=soft_topk_per_map,
            soft_score_center=soft_score_center,
        )
        abs_err = abs(float(run.true_count) - float(soft_pred))
        return {
            "run": run.run_name,
            "true": int(run.true_count),
            "pred_soft": float(soft_pred),
            "abs_err_soft": float(abs_err),
        }

    rows: list[dict] = []
    if executor is None or total_runs <= 1:
        for idx, run in enumerate(runs, start=1):
            rows.append(evaluate_one(run))
            if status_every_runs > 0 and (idx % status_every_runs == 0 or idx == total_runs):
                print(f"  [eval-soft] processed runs: {idx}/{total_runs}", flush=True)
    else:
        for idx, row in enumerate(executor.map(evaluate_one, runs), start=1):
            rows.append(row)
            if status_every_runs > 0 and (idx % status_every_runs == 0 or idx == total_runs):
                print(f"  [eval-soft] processed runs: {idx}/{total_runs}", flush=True)

    n = len(rows)
    mae = (sum(r["abs_err_soft"] for r in rows) / n) if n else 0.0
    true_total = float(sum(int(r["true"]) for r in rows))
    pred_total_soft = float(sum(float(r["pred_soft"]) for r in rows))
    pred_total_rounded = float(sum(int(round(float(r["pred_soft"]))) for r in rows))
    return {
        "rows": rows,
        "mae": float(mae),
        "quality": float(mae),
        "true_total": float(true_total),
        "pred_total_soft": float(pred_total_soft),
        "pred_total_rounded": float(pred_total_rounded),
    }


def _evaluate_template_set(
    runs: list[LabeledRun],
    template_dir: Path,
    min_score: float,
    min_template: float,
    max_objects: int,
    expected_count: int,
    status_every_runs: int = 0,
) -> dict:
    rows: list[dict] = []
    params = {
        "max_objects": int(max_objects),
        "min_score": float(min_score),
        "min_template_score_raw": float(min_template),
    }
    total_runs = len(runs)
    for idx, run in enumerate(runs, start=1):
        boxes = detect_grimforest_like_structures(
            binary_image_path=str(run.image_path),
            reference_dir=str(template_dir),
            expected_count=int(expected_count),
            debug=False,
            detector_params=params,
        )
        boxes = _suppress_overlapping_boxes(boxes)
        pred = len(boxes)
        rows.append({"run": run.image_path.parent.name, "true": run.true_count, "pred": pred, "abs_err": abs(run.true_count - pred)})
        if status_every_runs > 0 and (idx % status_every_runs == 0 or idx == total_runs):
            print(f"  [eval] processed runs: {idx}/{total_runs}", flush=True)

    n = len(rows)
    mae = (sum(r["abs_err"] for r in rows) / n) if n else 0.0
    exact = sum(1 for r in rows if r["true"] == r["pred"])
    exact_rate = (exact / n) if n else 0.0
    over = sum(1 for r in rows if r["pred"] > r["true"])
    under = sum(1 for r in rows if r["pred"] < r["true"])
    quality = float(mae) + (1.0 - float(exact_rate)) * 0.6 + (float(over) / max(1, n)) * 0.15
    return {
        "rows": rows,
        "mae": float(mae),
        "exact": int(exact),
        "exact_rate": float(exact_rate),
        "over": int(over),
        "under": int(under),
        "quality": float(quality),
    }


def _optimize_template_matrix_adam(
    runs: list[LabeledRun],
    template_count: int,
    max_objects: int,
    seed: int,
    trials: int,
    adam_lr: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_epsilon: float,
    grad_delta: float,
    grad_samples: int,
    soft_temperature: float,
    soft_topk_per_map: int,
    soft_score_center: float,
    workers: int,
    status_every_runs: int = 0,
) -> tuple[np.ndarray, dict]:
    tile_size = 64
    strip_width = int(template_count) * tile_size
    if strip_width != 256 or tile_size != 64:
        raise RuntimeError(
            f"Adam strip optimization expects 4x64x64 templates (256x64 strip). "
            f"Got template_count={template_count}."
        )

    rng = np.random.default_rng(seed)
    soft_runs = _prepare_soft_eval_runs(runs)
    worker_count = int(workers)
    if worker_count <= 0:
        worker_count = max(1, (os.cpu_count() or 2) - 1)
    if len(soft_runs) <= 1:
        worker_count = 1

    # Avoid oversubscription: OpenCV + Python-level parallelism.
    cv2.setNumThreads(1)
    executor = ThreadPoolExecutor(max_workers=worker_count) if worker_count > 1 else None

    def evaluate_matrix(matrix_01: np.ndarray) -> dict:
        templates = _strip_matrix_to_templates(matrix_01, template_count=template_count, tile_size=tile_size)
        return _evaluate_template_set_soft(
            runs=soft_runs,
            max_objects=max_objects,
            templates=templates,
            soft_temperature=soft_temperature,
            soft_topk_per_map=soft_topk_per_map,
            soft_score_center=soft_score_center,
            status_every_runs=status_every_runs,
            executor=executor,
        )

    try:
        x = _initialize_random_strip_matrix(seed=seed, width=strip_width, height=tile_size)
        m = np.zeros_like(x, dtype=np.float32)
        v = np.zeros_like(x, dtype=np.float32)

        current_metrics = evaluate_matrix(x)
        best_x = x.copy()
        best_metrics = current_metrics

        for trial_idx in range(1, trials + 1):
            grad = np.zeros_like(x, dtype=np.float32)
            for _ in range(max(1, int(grad_samples))):
                delta = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=x.shape)
                plus = np.clip(x + float(grad_delta) * delta, 0.0, 1.0)
                minus = np.clip(x - float(grad_delta) * delta, 0.0, 1.0)
                plus_metrics = evaluate_matrix(plus)
                minus_metrics = evaluate_matrix(minus)
                scalar = float(plus_metrics["mae"] - minus_metrics["mae"]) / (2.0 * float(grad_delta))
                grad += (scalar * delta).astype(np.float32)
            grad /= float(max(1, int(grad_samples)))

            m = float(adam_beta1) * m + (1.0 - float(adam_beta1)) * grad
            v = float(adam_beta2) * v + (1.0 - float(adam_beta2)) * (grad * grad)
            m_hat = m / (1.0 - float(adam_beta1) ** trial_idx)
            v_hat = v / (1.0 - float(adam_beta2) ** trial_idx)
            step = float(adam_lr) * m_hat / (np.sqrt(v_hat) + float(adam_epsilon))

            # Keep all optimized values in [0, 1] after every Adam step.
            x = np.clip(x - step, 0.0, 1.0).astype(np.float32)
            current_metrics = evaluate_matrix(x)

            if float(current_metrics["mae"]) < float(best_metrics["mae"]):
                best_x = x.copy()
                best_metrics = current_metrics

            print(
                f"[{trial_idx}/{trials}] true_total={current_metrics['true_total']:.3f} "
                f"pred_total_soft={current_metrics['pred_total_soft']:.3f} "
                f"pred_total_rounded={current_metrics['pred_total_rounded']:.3f} "
                f"continuous_loss_mae={current_metrics['mae']:.5f} best_continuous_loss_mae={best_metrics['mae']:.5f}",
                flush=True,
            )

        return best_x, best_metrics
    finally:
        if executor is not None:
            executor.shutdown(wait=True)


def run(args: argparse.Namespace) -> int:
    debug_dir = Path(args.debug_dir)
    output_dir = Path(args.output_dir)
    runs = _load_labeled_runs(debug_dir)
    if not runs:
        print(f"No labeled runs found in {debug_dir.as_posix()}")
        return 1
    all_run_count = len(runs)
    if bool(args.benchmark):
        benchmark_run = _select_benchmark_run(
            runs=runs,
            benchmark_run=str(args.benchmark_run or "").strip(),
        )
        runs = [benchmark_run]
        print(
            f"[Benchmark] Using only run: {runs[0].image_path.parent.name} "
            f"(label={runs[0].true_count}) out of {all_run_count} labeled runs."
        )
    else:
        print(f"[Full] Using all labeled runs for optimization: {all_run_count}")

    template_count = int(args.template_count)
    print(f"Labeled runs: {len(runs)}")
    if template_count != 4:
        raise RuntimeError("Grim Forest Adam optimizer requires --template-count 4 for a 256x64 strip.")

    print("Optimizing 256x64 grayscale template strip with Adam...")
    best_matrix, best_soft_metrics = _optimize_template_matrix_adam(
        runs=runs,
        template_count=template_count,
        max_objects=int(args.max_objects),
        seed=int(args.seed),
        trials=int(args.trials),
        adam_lr=float(args.adam_lr),
        adam_beta1=float(args.adam_beta1),
        adam_beta2=float(args.adam_beta2),
        adam_epsilon=float(args.adam_epsilon),
        grad_delta=float(args.grad_delta),
        grad_samples=int(args.grad_samples),
        soft_temperature=float(args.soft_temperature),
        soft_topk_per_map=int(args.soft_topk_per_map),
        soft_score_center=float(args.soft_score_center),
        workers=int(args.workers),
        status_every_runs=int(args.status_every_runs),
    )
    best_templates = _strip_matrix_to_templates(best_matrix, template_count=template_count, tile_size=64)
    _write_template_set(output_dir, best_templates)
    strip_path = output_dir / "optimized_strip_256x64.png"
    _write_template_strip(strip_path, best_templates, tile_size=64)
    best_hard_metrics = _evaluate_template_set(
        runs=runs,
        template_dir=output_dir,
        min_score=float(args.min_score),
        min_template=float(args.min_template),
        max_objects=int(args.max_objects),
        expected_count=int(args.expected_count),
        status_every_runs=int(args.status_every_runs),
    )

    report = {
        "mode": "grim_forest",
        "benchmark": bool(args.benchmark),
        "benchmark_label": int(BENCHMARK_TRUE_COUNT),
        "benchmark_run": str(args.benchmark_run or "").strip(),
        "template_mode": "grayscale",
        "optimized_strip_path": strip_path.as_posix(),
        "debug_dir": debug_dir.as_posix(),
        "output_template_dir": output_dir.as_posix(),
        "template_count": int(args.template_count),
        "selected_pool_indices": [],
        "matrix_shape_hw": [int(best_matrix.shape[0]), int(best_matrix.shape[1])],
        "optimizer": {
            "name": "adam",
            "objective": "soft_count_mae",
            "lr": float(args.adam_lr),
            "beta1": float(args.adam_beta1),
            "beta2": float(args.adam_beta2),
            "epsilon": float(args.adam_epsilon),
            "grad_delta": float(args.grad_delta),
            "grad_samples": int(args.grad_samples),
            "iterations": int(args.trials),
            "soft_temperature": float(args.soft_temperature),
            "soft_topk_per_map": int(args.soft_topk_per_map),
            "soft_score_center": float(args.soft_score_center),
            "workers": int(args.workers),
        },
        "fixed_detector_params": {
            "max_objects": int(args.max_objects),
            "min_score": float(args.min_score),
            "min_template_score_raw": float(args.min_template),
        },
        "metrics_soft": {
            "mae": round(float(best_soft_metrics["mae"]), 6),
            "quality": round(float(best_soft_metrics["quality"]), 6),
            "true_total": round(float(best_soft_metrics["true_total"]), 6),
            "pred_total_soft": round(float(best_soft_metrics["pred_total_soft"]), 6),
            "pred_total_rounded": round(float(best_soft_metrics["pred_total_rounded"]), 6),
        },
        "metrics_hard": {
            "mae": round(float(best_hard_metrics["mae"]), 6),
            "exact": int(best_hard_metrics["exact"]),
            "exact_rate": round(float(best_hard_metrics["exact_rate"]), 6),
            "over": int(best_hard_metrics["over"]),
            "under": int(best_hard_metrics["under"]),
            "quality": round(float(best_hard_metrics["quality"]), 6),
        },
        "per_run_soft": best_soft_metrics["rows"],
        "per_run_hard": best_hard_metrics["rows"],
    }
    report_path = output_dir / "optimization_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved optimized templates to: {output_dir.as_posix()}")
    print(f"Saved report: {report_path.as_posix()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inverse-optimize Grim Forest templates from labeled runs.")
    parser.add_argument("--debug-dir", default="debug/grim_forest_standalone", help="Directory containing run_*/ labels and binary images.")
    parser.add_argument("--output-dir", default="pic/grimforest", help="Template output directory (PNG files will be replaced).")
    parser.add_argument("--template-count", type=int, default=4, help="Number of templates to output (must be 4 for 256x64 strip optimization).")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help=f"Benchmark mode: optimize/evaluate using exactly one labeled run with true_object_count={int(BENCHMARK_TRUE_COUNT)}.",
    )
    parser.add_argument(
        "--benchmark-run",
        default="",
        help=f"Exact run folder name (e.g. run_20260517_123456) for benchmark mode; it must have true_object_count={int(BENCHMARK_TRUE_COUNT)}.",
    )
    parser.add_argument("--trials", type=int, default=int(ITERATIONS), help="Maximum Adam update iterations.")
    parser.add_argument("--adam-lr", type=float, default=0.05, help="Adam learning rate.")
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="Adam beta1.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="Adam beta2.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-8, help="Adam epsilon.")
    parser.add_argument("--grad-delta", type=float, default=0.02, help="SPSA perturbation scale for gradient estimate.")
    parser.add_argument("--grad-samples", type=int, default=1, help="Number of SPSA samples averaged per Adam step.")
    parser.add_argument("--soft-temperature", type=float, default=0.08, help="Temperature for sigmoid soft-count activation.")
    parser.add_argument("--soft-topk-per-map", type=int, default=16, help="Top-K response values kept per template/scale map for soft count.")
    parser.add_argument("--soft-score-center", type=float, default=0.38, help="Center score for soft-count sigmoid activation.")
    parser.add_argument("--workers", type=int, default=0, help="Soft-loss forward-pass workers. 0 = auto (cpu_count - 1).")
    parser.add_argument("--min-score", type=float, default=0.48, help="Fixed detector min_score.")
    parser.add_argument("--min-template", type=float, default=0.00, help="Fixed detector min_template_score_raw.")
    parser.add_argument("--max-objects", type=int, default=12, help="Detector max_objects.")
    parser.add_argument("--expected-count", type=int, default=5, help="Expected structure count for detector internals.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--status-every-runs", type=int, default=0, help="If >0, print evaluation progress every N runs.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
