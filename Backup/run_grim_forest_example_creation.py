from __future__ import annotations

import json
import logging
import sys
import time
import traceback

from run_grim_forest_standalone import GrimForestStandaloneRunner

MOVE_STRENGTH = 1.0
STEP_PAUSE_SECONDS = 0.8


def _build_move_plan() -> list[str]:
    single_pass = (["right"] * 10) + ["up"] + (["left"] * 10) + ["up"]
    return single_pass + single_pass + single_pass + single_pass + single_pass + single_pass + single_pass


def _perform_move(runner: GrimForestStandaloneRunner, move_name: str) -> None:
    if move_name == "right":
        runner.window_tools.move_right(runner.window, strength=MOVE_STRENGTH)
        return
    if move_name == "left":
        runner.window_tools.move_left(runner.window, strength=MOVE_STRENGTH)
        return
    if move_name == "up":
        runner.window_tools.move_up(runner.window, strength=MOVE_STRENGTH)
        return
    raise ValueError(f"Unsupported move: {move_name}")


def run_example_creation() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log = logging.getLogger("GrimForestExampleCreation")
    moves = _build_move_plan()

    for index, move_name in enumerate(moves, start=1):
        log.info("Step %d/%d: capture run before move '%s'.", index, len(moves), move_name)
        runner = GrimForestStandaloneRunner(run_detector=False)
        runner.run()

        run_meta_path = runner.debug_dir / "run_meta.json"
        if run_meta_path.exists():
            meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
            meta["sequence"] = {"step_index": int(index), "step_total": int(len(moves)), "next_move": move_name}
            run_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        log.info("Step %d/%d: executing move '%s'.", index, len(moves), move_name)
        _perform_move(runner, move_name)
        time.sleep(STEP_PAUSE_SECONDS)

    log.info("Example creation sequence finished. Total captures=%d", len(moves))
    return 0


def main() -> int:
    try:
        return run_example_creation()
    except Exception as exc:
        logging.error("Grim forest example creation failed: %s", exc)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
