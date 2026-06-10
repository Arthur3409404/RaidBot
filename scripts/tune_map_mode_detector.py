from __future__ import annotations

import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import _bootstrap  # noqa: F401


def _mode_spec(mode: str) -> dict:
    if mode == "cursed_city":
        from raid_bot.modes.cursedcity_tools import detect_cursedcity_like_structures

        return {
            "debug_dir": Path("data") / "output" / "debug" / "cursed_city_standalone",
            "reference_dir": Path("data") / "assets" / "images" / "cursedcity",
            "detector": detect_cursedcity_like_structures,
        }
    if mode == "grim_forest":
        from raid_bot.modes.grimforest_tools import detect_grimforest_like_structures

        return {
            "debug_dir": Path("data") / "output" / "debug" / "grim_forest_standalone",
            "reference_dir": Path("data") / "assets" / "images" / "grimforest",
            "detector": detect_grimforest_like_structures,
        }
    raise ValueError(f"Unsupported mode: {mode}")


def _load_labeled_runs(debug_dir: Path) -> list[tuple[Path, int]]:
    runs = []
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
            runs.append((image_path, int(label)))
    return runs


def _evaluate_runs(
    detector,
    reference_dir: Path,
    runs: list[tuple[Path, int]],
    params: dict,
    trial_index: int | None = None,
    trial_total: int | None = None,
) -> dict:
    rows = []
    run_total = len(runs)
    for run_index, (image_path, true_count) in enumerate(runs, start=1):
        if trial_index is not None and trial_total is not None:
            print(
                f"Trial {trial_index}/{trial_total} | Run {run_index}/{run_total} | {image_path.parent.name}",
                end="\r",
                flush=True,
            )
        boxes = detector(
            binary_image_path=str(image_path),
            reference_dir=str(reference_dir),
            expected_count=12,
            debug=False,
            detector_params=params,
        )
        pred = len(boxes)
        rows.append(
            {
                "run": image_path.parent.name,
                "true": true_count,
                "pred": pred,
                "abs_err": abs(true_count - pred),
            }
        )

    n = len(rows)
    true_total = sum(r["true"] for r in rows)
    pred_total = sum(r["pred"] for r in rows)
    mae = (sum(r["abs_err"] for r in rows) / n) if n else 0.0
    exact = sum(1 for r in rows if r["true"] == r["pred"])
    exact_rate = (exact / n) if n else 0.0
    over = sum(1 for r in rows if r["pred"] > r["true"])
    under = sum(1 for r in rows if r["pred"] < r["true"])
    return {
        "rows": rows,
        "n": n,
        "true_total": true_total,
        "pred_total": pred_total,
        "mae": mae,
        "exact": exact,
        "exact_rate": exact_rate,
        "over": over,
        "under": under,
    }


def _quality(metrics: dict) -> float:
    # Lower is better.
    return (
        float(metrics["mae"]) * 1.0
        + (1.0 - float(metrics["exact_rate"])) * 0.6
        + (float(metrics["over"]) / max(1, int(metrics["n"]))) * 0.15
    )


def _evaluate_trial_worker(payload: dict) -> dict:
    mode = payload["mode"]
    params = payload["params"]
    runs = [(Path(p), int(y)) for p, y in payload["runs"]]
    spec = _mode_spec(mode)
    detector = spec["detector"]
    reference_dir = spec["reference_dir"]
    metrics = _evaluate_runs(detector, reference_dir, runs, params)
    score = _quality(metrics)
    return {
        "params": params,
        "metrics": metrics,
        "quality": score,
    }


def tune(mode: str) -> int:
    spec = _mode_spec(mode)
    debug_dir = spec["debug_dir"]
    reference_dir = spec["reference_dir"]
    detector = spec["detector"]

    runs = _load_labeled_runs(debug_dir)
    if not runs:
        print(f"No labeled runs found in {debug_dir.as_posix()}")
        return 1

    print(f"Mode: {mode}")
    print(f"Labeled runs: {len(runs)}")

    max_objects_values = [12]
    # For Cursed City Detection Algorithm Refinement
    # min_score_values = [x / 100 for x in range(45, 55, 1)]
    # min_template_values = [x / 100 for x in range(0, 40, 5)]
    min_score_values = [x / 100 for x in range(36, 60, 1)]
    min_template_values = [x / 100 for x in range(0, 1)]
    total_trials = len(max_objects_values) * len(min_score_values) * len(min_template_values)

    trial_payloads = []
    for max_objects in max_objects_values:
        for min_score in min_score_values:
            for min_template in min_template_values:
                params = {
                    "max_objects": max_objects,
                    "min_score": float(min_score),
                    "min_template_score_raw": float(min_template),
                }
                trial_payloads.append(
                    {
                        "mode": mode,
                        "params": params,
                        "runs": [(str(p), int(y)) for p, y in runs],
                    }
                )

    best = None
    done = 0
    worker_count = min(8, max(1, len(trial_payloads)))
    print(f"Using multiprocessing with {worker_count} workers across {total_trials} trials.")
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = [pool.submit(_evaluate_trial_worker, payload) for payload in trial_payloads]
        for future in as_completed(futures):
            done += 1
            trial = future.result()
            params = trial["params"]
            print(
                f"[{done}/{total_trials}] min_score={params['min_score']:.2f}, "
                f"min_template={params['min_template_score_raw']:.2f}, "
                f"mae={trial['metrics']['mae']:.4f}, exact={trial['metrics']['exact_rate']:.4f}, "
                f"true_total={trial['metrics']['true_total']}, pred_total={trial['metrics']['pred_total']}"
            )
            if best is None or trial["quality"] < best["quality"]:
                best = trial

    assert best is not None
    out = {
        "mode": mode,
        "debug_dir": debug_dir.as_posix(),
        "reference_dir": reference_dir.as_posix(),
        "labeled_runs": len(runs),
        "best_params": best["params"],
        "best_metrics": {
            "mae": round(best["metrics"]["mae"], 6),
            "exact": int(best["metrics"]["exact"]),
            "exact_rate": round(best["metrics"]["exact_rate"], 6),
            "over": int(best["metrics"]["over"]),
            "under": int(best["metrics"]["under"]),
        },
        "quality": round(best["quality"], 6),
        "per_run": best["metrics"]["rows"],
    }
    out_path = debug_dir / "best_params.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Best params:")
    print(json.dumps(out["best_params"], indent=2))
    print("Best metrics:")
    print(json.dumps(out["best_metrics"], indent=2))
    print(f"Saved: {out_path.as_posix()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune map-mode detector parameters from labeled runs.")
    parser.add_argument("--mode", choices=["cursed_city", "grim_forest"], required=True)
    args = parser.parse_args()
    return tune(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
