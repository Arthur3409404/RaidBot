# -*- coding: utf-8 -*-
"""Contour-first cyan detector with template-driven class scoring."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ALLOWED_CLASSES = tuple(f"{idx}_allow" for idx in range(1, 5))
AVOID_CLASSES = tuple(f"{idx}_avoid" for idx in range(1, 5))
CLASS_LABELS = ALLOWED_CLASSES + AVOID_CLASSES
_LABEL_RE = re.compile(r"(?P<label>[1-4]_(?:allow|avoid))(?:_\d+)?$", re.IGNORECASE)


def _as_odd(value: int, minimum: int = 1) -> int:
    value = max(int(value), int(minimum))
    return value if value % 2 else value + 1


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_ratio(num: float, den: float) -> float:
    return 0.0 if den <= 0 else float(num) / float(den)


def _score_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return round(numeric, 6) if math.isfinite(numeric) else None


def _label_from_path(path: Path) -> str | None:
    match = _LABEL_RE.match(path.stem.strip().lower())
    if not match:
        return None
    label = str(match.group("label"))
    return label if label in CLASS_LABELS else None


class CyanTemplateDetector:
    def __init__(self, mode_slug: str, setup: dict[str, Any] | None = None):
        self.mode_slug = str(mode_slug or "").strip().lower().replace(" ", "_")
        self.setup = setup if isinstance(setup, dict) else {}
        self.templates: dict[str, list[dict[str, Any]]] = {label: [] for label in CLASS_LABELS}
        self.template_stats: dict[str, dict[str, float]] = {label: {} for label in CLASS_LABELS}
        self.template_counts: dict[str, int] = {label: 0 for label in CLASS_LABELS}
        self.template_dirs: list[str] = []
        self._template_signature: tuple[tuple[str, int], ...] = tuple()

    def refresh_profile(self, force: bool = False) -> None:
        self._ensure_templates(force=force)

    def detection_settings(self) -> dict[str, Any]:
        cfg = self._cfg()
        self._ensure_templates(force=False)
        return {
            "profile_name": cfg["profile_name"],
            "hsv_lower": cfg["hsv_lower"],
            "hsv_upper": cfg["hsv_upper"],
            "blur_kernel": cfg["blur_kernel"],
            "open_kernel": cfg["open_kernel"],
            "close_kernel": cfg["close_kernel"],
            "min_area": cfg["min_area"],
            "max_area_ratio": cfg["max_area_ratio"],
            "min_perimeter": cfg["min_perimeter"],
            "min_side_ratio": cfg["min_side_ratio"],
            "max_side_ratio": cfg["max_side_ratio"],
            "min_aspect_ratio": cfg["min_aspect_ratio"],
            "max_aspect_ratio": cfg["max_aspect_ratio"],
            "min_corners": cfg["min_corners"],
            "max_corners": cfg["max_corners"],
            "min_circularity": cfg["min_circularity"],
            "max_circularity": cfg["max_circularity"],
            "min_solidity": cfg["min_solidity"],
            "min_extent": cfg["min_extent"],
            "max_extent": cfg["max_extent"],
            "safe_bounds": list(cfg["safe_bounds"]),
            "ignore_regions": [list(region) for region in cfg["ignore_regions"]],
            "template_dirs": list(self.template_dirs),
            "template_counts": dict(self.template_counts),
            "template_count_total": int(sum(self.template_counts.values())),
            "size_tolerance": cfg["size_tolerance"],
            "max_class_score": cfg["max_class_score"],
            "min_class_margin": cfg["min_class_margin"],
            "min_allow_confidence": cfg["min_allow_confidence"],
        }

    def detect(self, image_np: np.ndarray) -> dict[str, Any]:
        if image_np is None or np.asarray(image_np).size == 0:
            return {"mask": np.zeros((1, 1), dtype=np.uint8), "candidates": [], "all_candidates": []}

        cfg = self._cfg()
        self._ensure_templates(force=False)

        rgb = np.asarray(image_np)
        if rgb.ndim == 2:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
        elif rgb.shape[2] == 4:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)

        if cfg["blur_kernel"] > 1:
            rgb = cv2.GaussianBlur(rgb, (cfg["blur_kernel"], cfg["blur_kernel"]), 0)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array(cfg["hsv_lower"], dtype=np.uint8), np.array(cfg["hsv_upper"], dtype=np.uint8))
        if cfg["open_kernel"] > 1:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((cfg["open_kernel"], cfg["open_kernel"]), dtype=np.uint8))
        if cfg["close_kernel"] > 1:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((cfg["close_kernel"], cfg["close_kernel"]), dtype=np.uint8))

        h, w = mask.shape[:2]
        min_side = max(8, int(round(min(h, w) * cfg["min_side_ratio"])))
        max_side = max(min_side + 1, int(round(min(h, w) * cfg["max_side_ratio"])))
        max_area = max(float(cfg["min_area"]), float(h * w) * cfg["max_area_ratio"])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        all_candidates: list[dict[str, Any]] = []
        for contour in contours:
            candidate = self._candidate_from_contour(contour, mask, w, h, min_side, max_side, max_area, cfg)
            if candidate is not None:
                all_candidates.append(candidate)

        all_candidates.sort(
            key=lambda item: (
                bool(item.get("accepted")),
                float(item.get("confidence", 0.0)),
                float(item.get("mask_area", 0.0)),
            ),
            reverse=True,
        )
        candidates = [
            item
            for item in all_candidates
            if bool(item.get("accepted")) and str(item.get("predicted_class", "")).endswith("_allow")
        ]
        return {"mask": mask, "candidates": candidates, "all_candidates": all_candidates}

    def _candidate_from_contour(
        self,
        contour: np.ndarray,
        mask: np.ndarray,
        width: int,
        height: int,
        min_side: int,
        max_side: int,
        max_area: float,
        cfg: dict[str, Any],
    ) -> dict[str, Any] | None:
        area = float(cv2.contourArea(contour))
        if area <= 0:
            return None
        perimeter = float(cv2.arcLength(contour, True))
        x, y, w_box, h_box = cv2.boundingRect(contour)
        aspect = _safe_ratio(float(w_box), float(h_box))
        cx = x + w_box / 2.0
        cy = y + h_box / 2.0
        cx_rel = _safe_ratio(cx, width)
        cy_rel = _safe_ratio(cy, height)
        roi = mask[y : y + h_box, x : x + w_box]
        fill = _safe_ratio(float(np.count_nonzero(roi)), float(w_box * h_box))
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = _safe_ratio(area, hull_area)
        extent = _safe_ratio(area, float(w_box * h_box))
        circularity = _safe_ratio(4.0 * math.pi * area, perimeter * perimeter)
        approx = cv2.approxPolyDP(contour, max(1.0, perimeter * cfg["approx_epsilon_ratio"]), True)
        corners = int(len(approx))

        candidate = {
            "bbox_local": {"x": int(x), "y": int(y), "width": int(w_box), "height": int(h_box)},
            "center_local": {"x": round(float(cx), 3), "y": round(float(cy), 3)},
            "center_rel": {"x": round(float(cx_rel), 6), "y": round(float(cy_rel), 6)},
            "mask_area": round(area, 3),
            "blue_fill": round(fill, 6),
            "cyan_fill": round(fill, 6),
            "perimeter": round(perimeter, 6),
            "corners": corners,
            "circularity": round(circularity, 6),
            "solidity": round(solidity, 6),
            "extent": round(extent, 6),
            "aspect_ratio": round(aspect, 6),
            "is_convex": bool(cv2.isContourConvex(approx)),
            "contour": [[int(p[0][0]), int(p[0][1])] for p in approx],
            "predicted_class": None,
            "per_class_scores": {label: None for label in CLASS_LABELS},
            "confidence": 0.0,
            "score": 0.0,
            "best_class_score": None,
            "class_margin": None,
            "accepted": False,
            "reject_reason": "unknown",
        }

        if area < cfg["min_area"]:
            candidate["reject_reason"] = "area_too_small"
            return candidate
        if area > max_area:
            candidate["reject_reason"] = "area_too_large"
            return candidate
        if perimeter < cfg["min_perimeter"]:
            candidate["reject_reason"] = "perimeter_too_small"
            return candidate
        if w_box < min_side or h_box < min_side:
            candidate["reject_reason"] = "side_too_small"
            return candidate
        if max(w_box, h_box) > max_side:
            candidate["reject_reason"] = "side_too_large"
            return candidate
        if aspect < cfg["min_aspect_ratio"] or aspect > cfg["max_aspect_ratio"]:
            candidate["reject_reason"] = "aspect_ratio_out_of_range"
            return candidate
        if corners < cfg["min_corners"] or corners > cfg["max_corners"]:
            candidate["reject_reason"] = "corner_count_out_of_range"
            return candidate
        if circularity < cfg["min_circularity"] or circularity > cfg["max_circularity"]:
            candidate["reject_reason"] = "circularity_out_of_range"
            return candidate
        if solidity < cfg["min_solidity"]:
            candidate["reject_reason"] = "solidity_too_low"
            return candidate
        if extent < cfg["min_extent"] or extent > cfg["max_extent"]:
            candidate["reject_reason"] = "extent_out_of_range"
            return candidate
        if cfg["require_convex"] and not bool(candidate["is_convex"]):
            candidate["reject_reason"] = "non_convex"
            return candidate
        if not self._inside_safe_bounds(cx_rel, cy_rel, cfg):
            candidate["reject_reason"] = "outside_safe_bounds"
            return candidate
        if self._inside_ignore_regions(cx_rel, cy_rel, cfg):
            candidate["reject_reason"] = "inside_ignore_region"
            return candidate

        c_mask = self._candidate_mask(mask, contour, cfg["roi_margin_ratio"])
        if c_mask is None or np.count_nonzero(c_mask) == 0:
            candidate["reject_reason"] = "empty_candidate_mask"
            return candidate
        c_desc = self._mask_descriptor(self._normalize_mask(c_mask, cfg["canvas_size"]))
        if c_desc is None:
            candidate["reject_reason"] = "descriptor_failed"
            return candidate

        class_scores = self._class_scores(c_desc, area, perimeter, cfg)
        candidate["per_class_scores"] = {label: _score_or_none(class_scores.get(label, float("inf"))) for label in CLASS_LABELS}
        decision = self._decide(class_scores, cfg)
        candidate["predicted_class"] = decision["predicted_class"]
        candidate["confidence"] = round(decision["confidence"], 6)
        candidate["score"] = round(decision["confidence"], 6)
        candidate["best_class_score"] = _score_or_none(decision["best_class_score"])
        candidate["class_margin"] = _score_or_none(decision["class_margin"])
        candidate["accepted"] = bool(decision["accepted"])
        candidate["reject_reason"] = decision["reject_reason"]
        return candidate

    def _class_scores(self, c_desc: dict[str, Any], area: float, perimeter: float, cfg: dict[str, Any]) -> dict[str, float]:
        weights = cfg["weights"]
        scores = {label: float("inf") for label in CLASS_LABELS}
        for label in CLASS_LABELS:
            entries = self.templates.get(label, [])
            if not entries:
                continue
            per_template = []
            stats = self.template_stats.get(label, {})
            for template in entries:
                inter = int(np.count_nonzero(cv2.bitwise_and(c_desc["mask"], template["mask"])))
                union = int(np.count_nonzero(cv2.bitwise_or(c_desc["mask"], template["mask"])))
                xor = int(np.count_nonzero(cv2.bitwise_xor(c_desc["mask"], template["mask"])))
                iou = _safe_ratio(inter, union)
                xor_ratio = _safe_ratio(xor, union)
                hu_component = min(2.5, float(np.mean(np.abs(c_desc["hu"] - template["hu"]))) / 5.5)
                shape_component = min(2.5, float(cv2.matchShapes(c_desc["contour"], template["contour"], cv2.CONTOURS_MATCH_I1, 0.0)) * 3.0)
                descriptor_component = min(
                    2.5,
                    (
                        abs(c_desc["circularity"] - template["circularity"])
                        + abs(c_desc["solidity"] - template["solidity"])
                        + abs(c_desc["extent"] - template["extent"])
                    )
                    / 1.2,
                )
                size_component = self._size_penalty(area, perimeter, stats, cfg["size_tolerance"])
                per_template.append(
                    weights["hu"] * hu_component
                    + weights["shape"] * shape_component
                    + weights["iou"] * (1.0 - iou)
                    + weights["xor"] * xor_ratio
                    + weights["descriptor"] * descriptor_component
                    + weights["size"] * size_component
                )
            if per_template:
                per_template.sort()
                scores[label] = float(np.median(per_template[: min(3, len(per_template))]))
        return scores

    def _decide(self, scores: dict[str, float], cfg: dict[str, Any]) -> dict[str, Any]:
        finite = [(label, value) for label, value in scores.items() if math.isfinite(value)]
        if not finite:
            return {"predicted_class": None, "accepted": False, "reject_reason": "no_templates", "confidence": 0.0, "best_class_score": None, "class_margin": None}
        finite.sort(key=lambda item: item[1])
        best_label, best_score = finite[0]
        second = finite[1][1] if len(finite) > 1 else best_score + 1.0
        margin = float(second) - float(best_score)
        score_quality = _clamp(1.0 - _safe_ratio(best_score, cfg["max_class_score"]), 0.0, 1.0)
        margin_quality = _clamp(_safe_ratio(margin, max(cfg["min_class_margin"] * 2.0, 1e-6)), 0.0, 1.0)
        confidence = _clamp(score_quality * 0.68 + margin_quality * 0.32, 0.0, 1.0)

        accepted = False
        reject_reason: str | None = "unknown"
        if best_label in AVOID_CLASSES:
            reject_reason = "avoid_class"
        elif best_score > cfg["max_class_score"]:
            reject_reason = "low_confidence"
        elif margin < cfg["min_class_margin"]:
            reject_reason = "ambiguous"
        elif confidence < cfg["min_allow_confidence"]:
            reject_reason = "low_confidence"
        else:
            accepted = True
            reject_reason = None
        return {
            "predicted_class": best_label,
            "accepted": accepted,
            "reject_reason": reject_reason,
            "confidence": float(confidence),
            "best_class_score": float(best_score),
            "class_margin": float(margin),
        }

    def _size_penalty(self, area: float, perimeter: float, stats: dict[str, float], tol: float) -> float:
        if not stats:
            return 0.0
        tol = max(0.05, float(tol))
        a_low, a_high = float(stats["area_low"]), float(stats["area_high"])
        p_low, p_high = float(stats["perimeter_low"]), float(stats["perimeter_high"])

        def _axis_penalty(value: float, low: float, high: float) -> float:
            if value < low:
                return _safe_ratio(low - value, max(low, 1.0))
            if value > high:
                return _safe_ratio(value - high, max(high, 1.0))
            return 0.0

        return float(min(2.5, ((_axis_penalty(area, a_low, a_high) + _axis_penalty(perimeter, p_low, p_high)) * 0.5) / tol))

    def _inside_safe_bounds(self, x_rel: float, y_rel: float, cfg: dict[str, Any]) -> bool:
        x1, y1, x2, y2 = cfg["safe_bounds"]
        return x1 <= x_rel <= x2 and y1 <= y_rel <= y2

    def _inside_ignore_regions(self, x_rel: float, y_rel: float, cfg: dict[str, Any]) -> bool:
        for x1, y1, x2, y2 in cfg["ignore_regions"]:
            if x1 <= x_rel <= x2 and y1 <= y_rel <= y2:
                return True
        return False

    def _candidate_mask(self, mask: np.ndarray, contour: np.ndarray, margin_ratio: float) -> np.ndarray | None:
        x, y, w_box, h_box = cv2.boundingRect(contour)
        margin = max(2, int(round(max(w_box, h_box) * float(margin_ratio))))
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(mask.shape[1], x + w_box + margin), min(mask.shape[0], y + h_box + margin)
        if x2 <= x1 or y2 <= y1:
            return None
        local = contour.copy()
        local[:, :, 0] -= int(x1)
        local[:, :, 1] -= int(y1)
        out = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cv2.drawContours(out, [local], -1, 255, -1)
        return cv2.bitwise_and(out, mask[y1:y2, x1:x2])

    def _normalize_mask(self, mask: np.ndarray, canvas_size: int) -> np.ndarray:
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        ys, xs = np.where(binary > 0)
        canvas_size = max(32, int(canvas_size))
        if len(xs) == 0 or len(ys) == 0:
            return np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        crop = binary[y1:y2, x1:x2]
        margin = max(4, int(round(canvas_size * 0.12)))
        target = max(4, canvas_size - 2 * margin)
        c_h, c_w = crop.shape[:2]
        scale = min(_safe_ratio(target, c_w), _safe_ratio(target, c_h))
        r_w, r_h = max(2, int(round(c_w * scale))), max(2, int(round(c_h * scale)))
        resized = cv2.resize(crop, (r_w, r_h), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        ox, oy = (canvas_size - r_w) // 2, (canvas_size - r_h) // 2
        canvas[oy : oy + r_h, ox : ox + r_w] = resized
        angle = self._orientation_angle(canvas)
        if angle is not None:
            matrix = cv2.getRotationMatrix2D((canvas_size / 2.0, canvas_size / 2.0), -angle, 1.0)
            rotated = cv2.warpAffine(canvas, matrix, (canvas_size, canvas_size), flags=cv2.INTER_NEAREST)
            canvas = np.where(rotated > 0, 255, 0).astype(np.uint8)
        return canvas

    def _orientation_angle(self, mask: np.ndarray) -> float | None:
        moments = cv2.moments(mask, binaryImage=True)
        if moments.get("m00", 0.0) <= 0.0:
            return None
        mu20, mu02, mu11 = float(moments.get("mu20", 0.0)), float(moments.get("mu02", 0.0)), float(moments.get("mu11", 0.0))
        common = math.sqrt(max(0.0, (mu20 - mu02) ** 2 + 4.0 * (mu11 ** 2)))
        lam1 = max(1e-6, (mu20 + mu02 + common) * 0.5)
        lam2 = max(1e-6, (mu20 + mu02 - common) * 0.5)
        if lam1 / lam2 < 1.12:
            return None
        return float(math.degrees(0.5 * math.atan2(2.0 * mu11, (mu20 - mu02))))

    def _mask_descriptor(self, mask: np.ndarray) -> dict[str, Any] | None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        if area <= 1.0 or perimeter <= 1.0:
            return None
        hu = cv2.HuMoments(cv2.moments(contour)).flatten()
        hu = np.sign(hu) * np.log10(np.clip(np.abs(hu), 1e-12, None))
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        x, y, w_box, h_box = cv2.boundingRect(contour)
        return {
            "mask": mask,
            "contour": contour,
            "hu": hu,
            "circularity": _safe_ratio(4.0 * math.pi * area, perimeter * perimeter),
            "solidity": _safe_ratio(area, hull_area),
            "extent": _safe_ratio(area, float(w_box * h_box)),
        }

    def _ensure_templates(self, force: bool) -> None:
        cfg = self._cfg()
        files = self._template_files(cfg["template_dirs"])
        signature = tuple((path.as_posix(), int(path.stat().st_mtime_ns)) for path in files)
        if not force and signature == self._template_signature:
            return
        self._template_signature = signature
        self.templates = {label: [] for label in CLASS_LABELS}
        self.template_stats = {label: {} for label in CLASS_LABELS}
        self.template_counts = {label: 0 for label in CLASS_LABELS}
        self.template_dirs = [path.as_posix() for path in cfg["template_dirs"]]

        for path in files:
            label = _label_from_path(path)
            if not label:
                continue
            image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image_bgr is None or image_bgr.size == 0:
                continue
            template_mask = self._template_mask(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), cfg)
            if template_mask is None:
                continue
            descriptor = self._mask_descriptor(self._normalize_mask(template_mask, cfg["canvas_size"]))
            if descriptor is None:
                continue
            contours, _ = cv2.findContours(template_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            t_contour = max(contours, key=cv2.contourArea)
            descriptor["raw_area"] = float(cv2.contourArea(t_contour))
            descriptor["raw_perimeter"] = float(cv2.arcLength(t_contour, True))
            self.templates[label].append(descriptor)

        for label in CLASS_LABELS:
            entries = self.templates[label]
            self.template_counts[label] = len(entries)
            self.template_stats[label] = self._stats_for(entries, cfg["size_tolerance"])

    def _template_mask(self, image_rgb: np.ndarray, cfg: dict[str, Any]) -> np.ndarray | None:
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array(cfg["hsv_lower"], dtype=np.uint8), np.array(cfg["hsv_upper"], dtype=np.uint8))
        if cfg["open_kernel"] > 1:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((cfg["open_kernel"], cfg["open_kernel"]), dtype=np.uint8))
        if cfg["close_kernel"] > 1:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((cfg["close_kernel"], cfg["close_kernel"]), dtype=np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            _, fallback = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(fallback, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            mask = fallback
        out = np.zeros_like(mask)
        cv2.drawContours(out, [max(contours, key=cv2.contourArea)], -1, 255, -1)
        return out

    def _stats_for(self, entries: list[dict[str, Any]], tolerance: float) -> dict[str, float]:
        if not entries:
            return {}
        tolerance = max(0.05, float(tolerance))
        areas = np.asarray([float(item["raw_area"]) for item in entries], dtype=np.float32)
        perimeters = np.asarray([float(item["raw_perimeter"]) for item in entries], dtype=np.float32)
        area_mid, perimeter_mid = float(np.median(areas)), float(np.median(perimeters))
        area_low = max(1.0, min(float(areas.min()) * (1.0 - tolerance), area_mid * (1.0 - tolerance)))
        area_high = max(area_low + 1.0, max(float(areas.max()) * (1.0 + tolerance), area_mid * (1.0 + tolerance)))
        per_low = max(1.0, min(float(perimeters.min()) * (1.0 - tolerance), perimeter_mid * (1.0 - tolerance)))
        per_high = max(per_low + 1.0, max(float(perimeters.max()) * (1.0 + tolerance), perimeter_mid * (1.0 + tolerance)))
        return {"area_low": area_low, "area_high": area_high, "perimeter_low": per_low, "perimeter_high": per_high}

    def _template_files(self, directories: list[Path]) -> list[Path]:
        files: list[Path] = []
        for directory in directories:
            if directory.exists() and directory.is_dir():
                files.extend([path for path in directory.glob("*.png") if path.is_file()])
        files.sort(key=lambda path: path.as_posix())
        return files

    def _cfg(self) -> dict[str, Any]:
        hsv_lower = self.setup.get("cyan_test_hsv_lower", self.setup.get("cyan_ring_hsv_lower", self.setup.get("blue_glow_hsv_lower", [82, 18, 120])))
        hsv_upper = self.setup.get("cyan_test_hsv_upper", self.setup.get("cyan_ring_hsv_upper", self.setup.get("blue_glow_hsv_upper", [108, 255, 255])))
        safe_bounds = self.setup.get("cyan_test_safe_bounds", self.setup.get("blue_glow_safe_bounds", [0.0, 0.0, 1.0, 1.0]))
        if not isinstance(safe_bounds, (list, tuple)) or len(safe_bounds) != 4:
            safe_bounds = [0.0, 0.0, 1.0, 1.0]

        ignore_raw = self.setup.get("cyan_test_ignore_regions", self.setup.get("blue_glow_ignore_regions", []))
        ignore_regions = [tuple(float(c) for c in region) for region in ignore_raw if isinstance(region, (list, tuple)) and len(region) == 4]
        dirs_raw = self.setup.get("cyan_test_template_dirs")
        if not isinstance(dirs_raw, (list, tuple)):
            single = self.setup.get("cyan_test_template_dir")
            dirs_raw = [single] if single else [
                "png",
                str(Path("data") / "map_data" / self.mode_slug / "png"),
                str(Path("data") / "map_data" / self.mode_slug / "templates"),
                str(Path("data") / "map_data" / self.mode_slug / "structures"),
            ]
        dirs = [Path(str(item)) for item in dirs_raw if item]

        return {
            "profile_name": str(self.setup.get("cyan_test_profile_name", self.mode_slug)),
            "hsv_lower": [int(hsv_lower[0]), int(hsv_lower[1]), int(hsv_lower[2])],
            "hsv_upper": [int(hsv_upper[0]), int(hsv_upper[1]), int(hsv_upper[2])],
            "blur_kernel": _as_odd(int(self.setup.get("cyan_test_blur_kernel", 3))),
            "open_kernel": _as_odd(int(self.setup.get("cyan_test_open_kernel", 3))),
            "close_kernel": _as_odd(int(self.setup.get("cyan_test_close_kernel", 5))),
            "min_area": int(self.setup.get("cyan_test_min_area", self.setup.get("blue_min_area", 120))),
            "max_area_ratio": float(self.setup.get("cyan_test_max_area_ratio", self.setup.get("blue_max_area_ratio", 0.08))),
            "min_perimeter": float(self.setup.get("cyan_test_min_perimeter", 35.0)),
            "min_side_ratio": float(self.setup.get("cyan_test_min_side_ratio", self.setup.get("cyan_ring_min_side_ratio", self.setup.get("blue_glow_min_side_ratio", 0.009)))),
            "max_side_ratio": float(self.setup.get("cyan_test_max_side_ratio", self.setup.get("cyan_ring_max_side_ratio", self.setup.get("blue_glow_max_side_ratio", 0.18)))),
            "min_aspect_ratio": float(self.setup.get("cyan_test_min_aspect_ratio", self.setup.get("cyan_ring_min_aspect_ratio", self.setup.get("blue_glow_min_aspect_ratio", 0.42)))),
            "max_aspect_ratio": float(self.setup.get("cyan_test_max_aspect_ratio", self.setup.get("cyan_ring_max_aspect_ratio", self.setup.get("blue_glow_max_aspect_ratio", 2.45)))),
            "min_corners": int(self.setup.get("cyan_test_min_corners", 4)),
            "max_corners": int(self.setup.get("cyan_test_max_corners", 18)),
            "approx_epsilon_ratio": float(self.setup.get("cyan_test_approx_epsilon_ratio", 0.035)),
            "min_circularity": float(self.setup.get("cyan_test_min_circularity", 0.04)),
            "max_circularity": float(self.setup.get("cyan_test_max_circularity", 1.35)),
            "min_solidity": float(self.setup.get("cyan_test_min_solidity", 0.30)),
            "min_extent": float(self.setup.get("cyan_test_min_extent", 0.08)),
            "max_extent": float(self.setup.get("cyan_test_max_extent", 0.95)),
            "require_convex": bool(self.setup.get("cyan_test_require_convex", False)),
            "safe_bounds": tuple(float(v) for v in safe_bounds),
            "ignore_regions": ignore_regions,
            "roi_margin_ratio": float(self.setup.get("cyan_test_roi_margin_ratio", 0.20)),
            "canvas_size": int(self.setup.get("cyan_test_canvas_size", 96)),
            "size_tolerance": float(self.setup.get("cyan_test_size_tolerance", 0.20)),
            "max_class_score": float(self.setup.get("cyan_test_max_class_score", 1.15)),
            "min_class_margin": float(self.setup.get("cyan_test_min_class_margin", 0.12)),
            "min_allow_confidence": float(self.setup.get("cyan_test_min_allow_confidence", 0.62)),
            "template_dirs": dirs,
            "weights": {
                "hu": float(self.setup.get("cyan_test_weight_hu", 0.27)),
                "shape": float(self.setup.get("cyan_test_weight_shape", 0.24)),
                "iou": float(self.setup.get("cyan_test_weight_iou", 0.20)),
                "xor": float(self.setup.get("cyan_test_weight_xor", 0.14)),
                "descriptor": float(self.setup.get("cyan_test_weight_descriptor", 0.10)),
                "size": float(self.setup.get("cyan_test_weight_size", 0.05)),
            },
        }
