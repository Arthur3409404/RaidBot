# -*- coding: utf-8 -*-
"""Blue-glow stage detection with mode-specific, map-data-driven profiles."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _as_odd(value: int, minimum: int = 1) -> int:
    normalized = max(int(value), int(minimum))
    return normalized if normalized % 2 == 1 else normalized + 1


def _quantile(values: list[float], q: float, default: float = 0.0) -> float:
    if not values:
        return float(default)
    return float(np.quantile(np.asarray(values, dtype=np.float32), float(q)))


def _to_rectangles(values: Any) -> list[tuple[float, float, float, float]]:
    rectangles: list[tuple[float, float, float, float]] = []
    if not isinstance(values, (list, tuple)):
        return rectangles
    for item in values:
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            continue
        x1, y1, x2, y2 = [float(component) for component in item]
        rectangles.append((x1, y1, x2, y2))
    return rectangles


@dataclass
class BlueStageProfile:
    profile_name: str
    mode_slug: str
    hue_min: int
    hue_max: int
    sat_min: int
    val_min: int
    pale_sat_max: int
    pale_val_min: int
    blue_bias_min: int
    pale_blue_bias_min: int
    blue_minus_green_min: int
    min_area: int
    max_area_ratio: float
    min_fill: float
    max_fill: float
    min_side: int
    min_side_ratio: float
    min_aspect_ratio: float
    max_aspect_ratio: float
    max_side_ratio: float
    ring_thickness_ratio: float
    min_ring_score: float
    min_border_fill: float
    min_center_fill: float
    max_core_fill: float
    min_circularity: float
    min_solidity: float
    open_kernel: int
    close_kernel: int
    safe_bounds: tuple[float, float, float, float]
    ignore_regions: list[tuple[float, float, float, float]] = field(default_factory=list)
    accent_enabled: bool = False
    accent_hue_min: int = 82
    accent_hue_max: int = 108
    accent_sat_min: int = 20
    accent_val_min: int = 120
    accent_min_fill: float = 0.0
    accent_weight: float = 0.0
    min_score: float = 0.0
    reference_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "profile_name": self.profile_name,
            "mode_slug": self.mode_slug,
            "hue_min": int(self.hue_min),
            "hue_max": int(self.hue_max),
            "sat_min": int(self.sat_min),
            "val_min": int(self.val_min),
            "pale_sat_max": int(self.pale_sat_max),
            "pale_val_min": int(self.pale_val_min),
            "blue_bias_min": int(self.blue_bias_min),
            "pale_blue_bias_min": int(self.pale_blue_bias_min),
            "blue_minus_green_min": int(self.blue_minus_green_min),
            "min_area": int(self.min_area),
            "max_area_ratio": float(self.max_area_ratio),
            "min_fill": float(self.min_fill),
            "max_fill": float(self.max_fill),
            "min_side": int(self.min_side),
            "min_side_ratio": float(self.min_side_ratio),
            "min_aspect_ratio": float(self.min_aspect_ratio),
            "max_aspect_ratio": float(self.max_aspect_ratio),
            "max_side_ratio": float(self.max_side_ratio),
            "ring_thickness_ratio": float(self.ring_thickness_ratio),
            "min_ring_score": float(self.min_ring_score),
            "min_border_fill": float(self.min_border_fill),
            "min_center_fill": float(self.min_center_fill),
            "max_core_fill": float(self.max_core_fill),
            "min_circularity": float(self.min_circularity),
            "min_solidity": float(self.min_solidity),
            "open_kernel": int(self.open_kernel),
            "close_kernel": int(self.close_kernel),
            "safe_bounds": [float(value) for value in self.safe_bounds],
            "ignore_regions": [[float(component) for component in region] for region in self.ignore_regions],
            "accent_enabled": bool(self.accent_enabled),
            "accent_hue_min": int(self.accent_hue_min),
            "accent_hue_max": int(self.accent_hue_max),
            "accent_sat_min": int(self.accent_sat_min),
            "accent_val_min": int(self.accent_val_min),
            "accent_min_fill": float(self.accent_min_fill),
            "accent_weight": float(self.accent_weight),
            "min_score": float(self.min_score),
            "reference_summary": dict(self.reference_summary or {}),
        }
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any], defaults: "BlueStageProfile") -> "BlueStageProfile":
        if not isinstance(payload, dict):
            return defaults

        profile = BlueStageProfile(
            profile_name=str(payload.get("profile_name", defaults.profile_name)),
            mode_slug=str(payload.get("mode_slug", defaults.mode_slug)),
            hue_min=int(payload.get("hue_min", defaults.hue_min)),
            hue_max=int(payload.get("hue_max", defaults.hue_max)),
            sat_min=int(payload.get("sat_min", defaults.sat_min)),
            val_min=int(payload.get("val_min", defaults.val_min)),
            pale_sat_max=int(payload.get("pale_sat_max", defaults.pale_sat_max)),
            pale_val_min=int(payload.get("pale_val_min", defaults.pale_val_min)),
            blue_bias_min=int(payload.get("blue_bias_min", defaults.blue_bias_min)),
            pale_blue_bias_min=int(payload.get("pale_blue_bias_min", defaults.pale_blue_bias_min)),
            blue_minus_green_min=int(payload.get("blue_minus_green_min", defaults.blue_minus_green_min)),
            min_area=int(payload.get("min_area", defaults.min_area)),
            max_area_ratio=float(payload.get("max_area_ratio", defaults.max_area_ratio)),
            min_fill=float(payload.get("min_fill", defaults.min_fill)),
            max_fill=float(payload.get("max_fill", defaults.max_fill)),
            min_side=int(payload.get("min_side", defaults.min_side)),
            min_side_ratio=float(payload.get("min_side_ratio", defaults.min_side_ratio)),
            min_aspect_ratio=float(payload.get("min_aspect_ratio", defaults.min_aspect_ratio)),
            max_aspect_ratio=float(payload.get("max_aspect_ratio", defaults.max_aspect_ratio)),
            max_side_ratio=float(payload.get("max_side_ratio", defaults.max_side_ratio)),
            ring_thickness_ratio=float(payload.get("ring_thickness_ratio", defaults.ring_thickness_ratio)),
            min_ring_score=float(payload.get("min_ring_score", defaults.min_ring_score)),
            min_border_fill=float(payload.get("min_border_fill", defaults.min_border_fill)),
            min_center_fill=float(payload.get("min_center_fill", defaults.min_center_fill)),
            max_core_fill=float(payload.get("max_core_fill", defaults.max_core_fill)),
            min_circularity=float(payload.get("min_circularity", defaults.min_circularity)),
            min_solidity=float(payload.get("min_solidity", defaults.min_solidity)),
            open_kernel=int(payload.get("open_kernel", defaults.open_kernel)),
            close_kernel=int(payload.get("close_kernel", defaults.close_kernel)),
            safe_bounds=tuple(payload.get("safe_bounds", defaults.safe_bounds)),
            ignore_regions=_to_rectangles(payload.get("ignore_regions", defaults.ignore_regions)),
            accent_enabled=bool(payload.get("accent_enabled", defaults.accent_enabled)),
            accent_hue_min=int(payload.get("accent_hue_min", defaults.accent_hue_min)),
            accent_hue_max=int(payload.get("accent_hue_max", defaults.accent_hue_max)),
            accent_sat_min=int(payload.get("accent_sat_min", defaults.accent_sat_min)),
            accent_val_min=int(payload.get("accent_val_min", defaults.accent_val_min)),
            accent_min_fill=float(payload.get("accent_min_fill", defaults.accent_min_fill)),
            accent_weight=float(payload.get("accent_weight", defaults.accent_weight)),
            min_score=float(payload.get("min_score", defaults.min_score)),
            reference_summary=dict(payload.get("reference_summary") or {}),
        )
        return profile


def _default_profile(mode_slug: str) -> BlueStageProfile:
    normalized = str(mode_slug or "").strip().lower().replace(" ", "_")
    if normalized == "grim_forest":
        return BlueStageProfile(
            profile_name="grim_forest",
            mode_slug="grim_forest",
            hue_min=92,
            hue_max=126,
            sat_min=58,
            val_min=68,
            pale_sat_max=95,
            pale_val_min=145,
            blue_bias_min=16,
            pale_blue_bias_min=5,
            blue_minus_green_min=6,
            min_area=100,
            max_area_ratio=0.035,
            min_fill=0.08,
            max_fill=0.94,
            min_side=12,
            min_side_ratio=0.010,
            min_aspect_ratio=0.42,
            max_aspect_ratio=2.45,
            max_side_ratio=0.18,
            ring_thickness_ratio=0.16,
            min_ring_score=0.07,
            min_border_fill=0.12,
            min_center_fill=0.02,
            max_core_fill=0.58,
            min_circularity=0.09,
            min_solidity=0.40,
            open_kernel=3,
            close_kernel=5,
            safe_bounds=(0.08, 0.08, 0.91, 0.94),
            ignore_regions=[
                (0.00, 0.00, 0.10, 0.55),
                (0.88, 0.58, 1.00, 1.00),
            ],
            accent_enabled=True,
            accent_hue_min=84,
            accent_hue_max=108,
            accent_sat_min=24,
            accent_val_min=124,
            accent_min_fill=0.012,
            accent_weight=0.16,
            min_score=0.0,
        )

    return BlueStageProfile(
        profile_name="cursed_city",
        mode_slug="cursed_city",
        hue_min=82,
        hue_max=146,
        sat_min=24,
        val_min=58,
        pale_sat_max=98,
        pale_val_min=132,
        blue_bias_min=1,
        pale_blue_bias_min=4,
        blue_minus_green_min=-10,
        min_area=120,
        max_area_ratio=0.06,
            min_fill=0.10,
            max_fill=0.92,
            min_side=12,
            min_side_ratio=0.010,
            min_aspect_ratio=0.42,
            max_aspect_ratio=2.55,
            max_side_ratio=0.21,
            ring_thickness_ratio=0.16,
            min_ring_score=0.08,
            min_border_fill=0.12,
            min_center_fill=0.02,
            max_core_fill=0.62,
        min_circularity=0.06,
        min_solidity=0.35,
        open_kernel=3,
        close_kernel=5,
        safe_bounds=(0.05, 0.04, 0.95, 0.93),
        ignore_regions=[
            (0.88, 0.60, 1.00, 1.00),
        ],
        accent_enabled=True,
        accent_hue_min=82,
        accent_hue_max=106,
        accent_sat_min=18,
        accent_val_min=120,
        accent_min_fill=0.010,
        accent_weight=0.14,
        min_score=0.0,
    )


class BlueStageDetector:
    """Detect blue-encircled stage nodes with mode-specific profiles."""

    def __init__(
        self,
        mode_slug: str,
        map_data_root: str | Path = Path("data") / "map_data",
        profile_overrides: dict[str, Any] | None = None,
        profile_cache_name: str = "blue_stage_profile.json",
    ) -> None:
        self.mode_slug = str(mode_slug or "").strip().lower().replace(" ", "_")
        self.map_data_root = Path(map_data_root)
        self.profile_overrides = dict(profile_overrides or {})
        self.profile_cache_name = str(profile_cache_name)
        self.base_profile = _default_profile(self.mode_slug)
        self.mode_dir = self.map_data_root / self.base_profile.mode_slug
        self.profile_cache_path = self.mode_dir / self.profile_cache_name
        self.reference_paths = self._collect_reference_paths()
        self.profile = self._load_or_build_profile()

    def _collect_reference_paths(self) -> list[Path]:
        references: list[Path] = []
        last_session_dir = self.mode_dir / "last_session"
        if last_session_dir.exists():
            references.extend(sorted(path for path in last_session_dir.glob("viewports/*.png") if path.exists()))

        latest_file = self.mode_dir / "latest.json"
        if latest_file.exists():
            try:
                latest_payload = json.loads(latest_file.read_text(encoding="utf-8"))
                for viewport in latest_payload.get("viewports", []):
                    image_path = viewport.get("image_path")
                    if not image_path:
                        continue
                    resolved = Path(image_path)
                    if not resolved.is_absolute():
                        resolved = Path.cwd() / resolved
                    if resolved.exists():
                        references.append(resolved)
            except (OSError, json.JSONDecodeError):
                pass

        unique = {path.resolve() if path.exists() else path for path in references}
        return sorted(unique)

    def _reference_fingerprint(self, paths: list[Path]) -> dict[str, Any]:
        if not paths:
            return {"count": 0, "max_mtime_ns": 0, "min_mtime_ns": 0}
        mtimes = []
        for path in paths:
            try:
                mtimes.append(path.stat().st_mtime_ns)
            except OSError:
                continue
        if not mtimes:
            return {"count": len(paths), "max_mtime_ns": 0, "min_mtime_ns": 0}
        return {
            "count": len(paths),
            "max_mtime_ns": int(max(mtimes)),
            "min_mtime_ns": int(min(mtimes)),
        }

    def _load_cached_profile(self, fingerprint: dict[str, Any]) -> BlueStageProfile | None:
        if not self.profile_cache_path.exists():
            return None
        try:
            payload = json.loads(self.profile_cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        if not isinstance(payload, dict):
            return None
        if payload.get("reference_fingerprint") != fingerprint:
            return None
        cached_profile = payload.get("profile")
        if not isinstance(cached_profile, dict):
            return None
        return BlueStageProfile.from_dict(cached_profile, defaults=self.base_profile)

    def _save_cached_profile(self, profile: BlueStageProfile, fingerprint: dict[str, Any]) -> None:
        if not self.mode_dir.exists():
            self.mode_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "mode_slug": self.mode_slug,
            "reference_fingerprint": fingerprint,
            "profile": profile.to_dict(),
        }
        try:
            self.profile_cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        except OSError:
            pass

    def _load_or_build_profile(self) -> BlueStageProfile:
        fingerprint = self._reference_fingerprint(self.reference_paths)
        cached = self._load_cached_profile(fingerprint)
        if cached is not None:
            profile = self._apply_overrides(cached, self.profile_overrides)
            profile.reference_summary.setdefault("source", "cache")
            profile.reference_summary.setdefault("reference_count", len(self.reference_paths))
            return profile

        calibrated = self._calibrate_from_references(self.base_profile, self.reference_paths)
        profile = self._apply_overrides(calibrated, self.profile_overrides)
        self._save_cached_profile(profile, fingerprint)
        return profile

    def _collect_seed_stats(self, base_profile: BlueStageProfile, reference_paths: list[Path]) -> dict[str, Any]:
        hue_values: list[float] = []
        sat_values: list[float] = []
        val_values: list[float] = []
        blue_bias_values: list[float] = []
        blue_minus_green_values: list[float] = []
        blue_minus_red_values: list[float] = []
        area_values: list[float] = []
        area_ratio_values: list[float] = []
        fill_values: list[float] = []
        side_values: list[float] = []
        side_ratio_values: list[float] = []
        ring_values: list[float] = []
        image_areas: list[float] = []

        seed_hue_min = int(_clamp(base_profile.hue_min - 10, 0, 179))
        seed_hue_max = int(_clamp(base_profile.hue_max + 10, 0, 179))
        seed_sat_min = int(_clamp(base_profile.sat_min - 22, 8, 255))
        seed_val_min = int(_clamp(base_profile.val_min - 18, 35, 255))
        seed_pale_sat_max = int(_clamp(base_profile.pale_sat_max + 35, 40, 160))
        seed_pale_val_min = int(_clamp(base_profile.pale_val_min - 26, 105, 255))
        seed_blue_bias_min = int(_clamp(base_profile.blue_bias_min - 8, -12, 80))
        seed_pale_blue_bias_min = int(_clamp(base_profile.pale_blue_bias_min - 2, 1, 30))
        seed_blue_minus_green_min = int(_clamp(base_profile.blue_minus_green_min - 12, -30, 80))

        for path in reference_paths:
            image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
            red = image_rgb[:, :, 0].astype(np.int16)
            green = image_rgb[:, :, 1].astype(np.int16)
            blue = image_rgb[:, :, 2].astype(np.int16)
            blue_bias = blue - np.maximum(red, green)
            blue_minus_green = blue - green
            blue_minus_red = blue - red

            vivid_seed = (
                (hsv[:, :, 0] >= seed_hue_min)
                & (hsv[:, :, 0] <= seed_hue_max)
                & (hsv[:, :, 1] >= seed_sat_min)
                & (hsv[:, :, 2] >= seed_val_min)
                & (blue_bias >= seed_blue_bias_min)
                & (blue_minus_green >= seed_blue_minus_green_min)
            )
            pale_seed = (
                (hsv[:, :, 2] >= seed_pale_val_min)
                & (hsv[:, :, 1] <= seed_pale_sat_max)
                & (blue_minus_red >= seed_pale_blue_bias_min)
                & (blue_minus_green >= seed_blue_minus_green_min - 8)
            )
            seed_mask = vivid_seed | pale_seed
            if not np.any(seed_mask):
                continue

            hue_values.extend(hsv[:, :, 0][seed_mask].astype(np.float32).tolist())
            sat_values.extend(hsv[:, :, 1][seed_mask].astype(np.float32).tolist())
            val_values.extend(hsv[:, :, 2][seed_mask].astype(np.float32).tolist())
            blue_bias_values.extend(blue_bias[seed_mask].astype(np.float32).tolist())
            blue_minus_green_values.extend(blue_minus_green[seed_mask].astype(np.float32).tolist())
            blue_minus_red_values.extend(blue_minus_red[seed_mask].astype(np.float32).tolist())

            image_height, image_width = seed_mask.shape[:2]
            image_area = float(image_height * image_width)
            image_areas.append(image_area)

            mask_uint8 = (seed_mask.astype(np.uint8) * 255)
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                contour_area = float(cv2.contourArea(contour))
                if contour_area < 50.0 or contour_area > image_area * 0.12:
                    continue
                x, y, width, height = cv2.boundingRect(contour)
                if width < 8 or height < 8:
                    continue
                aspect_ratio = width / float(height)
                if aspect_ratio < 0.20 or aspect_ratio > 4.0:
                    continue

                roi_mask = mask_uint8[y : y + height, x : x + width]
                blue_fill = float(np.count_nonzero(roi_mask)) / max(1.0, float(width * height))
                if blue_fill < 0.04:
                    continue

                thickness = max(2, int(round(min(width, height) * float(base_profile.ring_thickness_ratio))))
                ring_kernel = np.ones((_as_odd(2 * thickness + 1), _as_odd(2 * thickness + 1)), dtype=np.uint8)
                core_mask = cv2.erode(roi_mask, ring_kernel)
                border_mask = cv2.subtract(roi_mask, core_mask)
                border_fill = float(np.count_nonzero(border_mask)) / max(1.0, float(border_mask.size))
                core_fill = float(np.count_nonzero(core_mask)) / max(1.0, float(core_mask.size))
                ring_score = border_fill - core_fill

                area_values.append(contour_area)
                area_ratio_values.append(contour_area / max(1.0, image_area))
                fill_values.append(blue_fill)
                side_values.append(float(min(width, height)))
                side_ratio_values.append(float(max(width, height)) / max(1.0, float(min(image_height, image_width))))
                ring_values.append(ring_score)

        return {
            "reference_count": len(reference_paths),
            "hue_values": hue_values,
            "sat_values": sat_values,
            "val_values": val_values,
            "blue_bias_values": blue_bias_values,
            "blue_minus_green_values": blue_minus_green_values,
            "blue_minus_red_values": blue_minus_red_values,
            "area_values": area_values,
            "area_ratio_values": area_ratio_values,
            "fill_values": fill_values,
            "side_values": side_values,
            "side_ratio_values": side_ratio_values,
            "ring_values": ring_values,
            "image_areas": image_areas,
        }

    def _calibrate_from_references(self, base_profile: BlueStageProfile, reference_paths: list[Path]) -> BlueStageProfile:
        profile = BlueStageProfile.from_dict(base_profile.to_dict(), defaults=base_profile)
        if not reference_paths:
            profile.reference_summary = {
                "source": "defaults",
                "reference_count": 0,
                "reference_paths": [],
            }
            return profile

        stats = self._collect_seed_stats(base_profile, reference_paths)
        hue_values = stats.get("hue_values", [])
        sat_values = stats.get("sat_values", [])
        val_values = stats.get("val_values", [])
        blue_bias_values = stats.get("blue_bias_values", [])
        blue_minus_green_values = stats.get("blue_minus_green_values", [])
        blue_minus_red_values = stats.get("blue_minus_red_values", [])
        area_values = stats.get("area_values", [])
        area_ratio_values = stats.get("area_ratio_values", [])
        fill_values = stats.get("fill_values", [])
        side_values = stats.get("side_values", [])
        side_ratio_values = stats.get("side_ratio_values", [])
        ring_values = stats.get("ring_values", [])

        if hue_values:
            profile.hue_min = int(_clamp(round(_quantile(hue_values, 0.05, profile.hue_min) - 4), 60, 170))
            profile.hue_max = int(_clamp(round(_quantile(hue_values, 0.95, profile.hue_max) + 4), profile.hue_min + 6, 175))
        if sat_values:
            profile.sat_min = int(_clamp(round(_quantile(sat_values, 0.10, profile.sat_min) - 15), 10, 150))
            profile.pale_sat_max = int(_clamp(round(_quantile(sat_values, 0.38, profile.pale_sat_max) + 16), 40, 150))
        if val_values:
            profile.val_min = int(_clamp(round(_quantile(val_values, 0.10, profile.val_min) - 12), 35, 190))
            profile.pale_val_min = int(_clamp(round(_quantile(val_values, 0.75, profile.pale_val_min) - 8), 105, 230))
        if blue_bias_values:
            profile.blue_bias_min = int(_clamp(round(_quantile(blue_bias_values, 0.20, profile.blue_bias_min) - 3), -10, 70))
        if blue_minus_green_values:
            profile.blue_minus_green_min = int(
                _clamp(round(_quantile(blue_minus_green_values, 0.20, profile.blue_minus_green_min) - 3), -25, 80)
            )
        if blue_minus_red_values:
            profile.pale_blue_bias_min = int(_clamp(round(_quantile(blue_minus_red_values, 0.10, profile.pale_blue_bias_min)), 1, 32))

        if area_values:
            profile.min_area = int(_clamp(round(_quantile(area_values, 0.12, profile.min_area) * 0.80), 60, 520))
        if area_ratio_values:
            ratio_target = _quantile(area_ratio_values, 0.95, profile.max_area_ratio) * 1.45
            profile.max_area_ratio = float(_clamp(ratio_target, 0.015, 0.12))
        if fill_values:
            profile.min_fill = float(_clamp(_quantile(fill_values, 0.08, profile.min_fill) * 0.80, 0.04, 0.28))
            profile.max_fill = float(_clamp(_quantile(fill_values, 0.95, profile.max_fill) * 1.04, 0.70, 0.98))
        if side_values:
            profile.min_side = int(_clamp(round(_quantile(side_values, 0.10, profile.min_side) * 0.82), 10, 42))
        if side_ratio_values:
            profile.max_side_ratio = float(_clamp(_quantile(side_ratio_values, 0.95, profile.max_side_ratio) * 1.30, 0.10, 0.32))
        if ring_values:
            profile.min_ring_score = float(_clamp(_quantile(ring_values, 0.35, profile.min_ring_score) * 0.72, 0.03, 0.22))
            profile.max_core_fill = float(_clamp(0.78 - _quantile(ring_values, 0.80, 0.0) * 0.42, 0.38, 0.75))

        profile.reference_summary = {
            "source": "map_data_png_references",
            "reference_count": int(stats.get("reference_count", 0)),
            "reference_paths": [path.as_posix() for path in reference_paths[:25]],
            "seed_pixel_count": len(hue_values),
            "component_count": len(area_values),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        return profile

    def _apply_overrides(self, profile: BlueStageProfile, overrides: dict[str, Any] | None) -> BlueStageProfile:
        if not overrides:
            return profile

        payload = profile.to_dict()
        for key, value in overrides.items():
            if value is None:
                continue
            if key in {"safe_bounds"} and isinstance(value, (list, tuple)) and len(value) == 4:
                payload[key] = [float(v) for v in value]
                continue
            if key == "ignore_regions":
                payload[key] = [[float(component) for component in region] for region in _to_rectangles(value)]
                continue
            payload[key] = value
        return BlueStageProfile.from_dict(payload, defaults=profile)

    def refresh_profile(self, force: bool = False) -> BlueStageProfile:
        if force:
            self.reference_paths = self._collect_reference_paths()
            self.profile = self._load_or_build_profile()
        return self.profile

    def detection_settings(self) -> dict[str, Any]:
        settings = self.profile.to_dict()
        settings["lower_hsv"] = [settings["hue_min"], settings["sat_min"], settings["val_min"]]
        settings["upper_hsv"] = [settings["hue_max"], 255, 255]
        return settings

    def _inside_allowed_region(self, center_x_rel: float, center_y_rel: float, profile: BlueStageProfile) -> bool:
        safe_x1, safe_y1, safe_x2, safe_y2 = profile.safe_bounds
        if center_x_rel < safe_x1 or center_x_rel > safe_x2:
            return False
        if center_y_rel < safe_y1 or center_y_rel > safe_y2:
            return False
        for region in profile.ignore_regions:
            x1, y1, x2, y2 = region
            if x1 <= center_x_rel <= x2 and y1 <= center_y_rel <= y2:
                return False
        return True

    def detect(self, image_np: np.ndarray) -> dict[str, Any]:
        profile = self.profile
        if image_np.size == 0:
            return {"mask": np.zeros((1, 1), dtype=np.uint8), "candidates": [], "profile": profile.to_dict()}

        if image_np.ndim == 2:
            rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        else:
            rgb = np.asarray(image_np)
            if rgb.shape[2] == 4:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        red = rgb[:, :, 0].astype(np.int16)
        green = rgb[:, :, 1].astype(np.int16)
        blue = rgb[:, :, 2].astype(np.int16)
        blue_bias = blue - np.maximum(red, green)
        blue_minus_red = blue - red
        blue_minus_green = blue - green

        vivid_mask = (
            (hsv[:, :, 0] >= int(profile.hue_min))
            & (hsv[:, :, 0] <= int(profile.hue_max))
            & (hsv[:, :, 1] >= int(profile.sat_min))
            & (hsv[:, :, 2] >= int(profile.val_min))
            & (blue_bias >= int(profile.blue_bias_min))
            & (blue_minus_green >= int(profile.blue_minus_green_min))
        )
        pale_mask = (
            (hsv[:, :, 2] >= int(profile.pale_val_min))
            & (hsv[:, :, 1] <= int(profile.pale_sat_max))
            & (blue_minus_red >= int(profile.pale_blue_bias_min))
            & (blue_minus_green >= int(profile.blue_minus_green_min) - 8)
        )

        cool_mask = (vivid_mask | pale_mask)
        accent_mask = np.zeros_like(cool_mask, dtype=bool)
        if bool(profile.accent_enabled):
            accent_mask = (
                (hsv[:, :, 0] >= int(profile.accent_hue_min))
                & (hsv[:, :, 0] <= int(profile.accent_hue_max))
                & (hsv[:, :, 1] >= int(profile.accent_sat_min))
                & (hsv[:, :, 2] >= int(profile.accent_val_min))
            )

        combined_mask = (cool_mask | accent_mask).astype(np.uint8) * 255
        open_kernel_size = _as_odd(profile.open_kernel, minimum=1)
        close_kernel_size = _as_odd(profile.close_kernel, minimum=1)
        if open_kernel_size > 1:
            combined_mask = cv2.morphologyEx(
                combined_mask,
                cv2.MORPH_OPEN,
                np.ones((open_kernel_size, open_kernel_size), dtype=np.uint8),
            )
        if close_kernel_size > 1:
            combined_mask = cv2.morphologyEx(
                combined_mask,
                cv2.MORPH_CLOSE,
                np.ones((close_kernel_size, close_kernel_size), dtype=np.uint8),
            )

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_height, image_width = combined_mask.shape[:2]
        image_area = float(image_height * image_width)
        max_area = max(float(profile.min_area), image_area * float(profile.max_area_ratio))
        max_side_px = max(10, int(round(min(image_height, image_width) * float(profile.max_side_ratio))))
        min_side_px = max(6, int(round(min(image_height, image_width) * float(profile.min_side_ratio))))

        candidates: list[dict[str, Any]] = []
        for contour in contours:
            contour_area = float(cv2.contourArea(contour))
            if contour_area < float(profile.min_area) or contour_area > max_area:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            if width < int(profile.min_side) or height < int(profile.min_side):
                continue
            if width < min_side_px or height < min_side_px:
                continue
            if max(width, height) > max_side_px:
                continue

            aspect_ratio = width / float(height)
            if aspect_ratio < float(profile.min_aspect_ratio) or aspect_ratio > float(profile.max_aspect_ratio):
                continue

            center_x = x + width / 2.0
            center_y = y + height / 2.0
            center_x_rel = float(center_x) / max(1.0, float(image_width))
            center_y_rel = float(center_y) / max(1.0, float(image_height))
            if not self._inside_allowed_region(center_x_rel, center_y_rel, profile):
                continue

            roi_mask = combined_mask[y : y + height, x : x + width]
            roi_cool_mask = cool_mask[y : y + height, x : x + width]
            roi_accent_mask = accent_mask[y : y + height, x : x + width]
            roi_bias = blue_bias[y : y + height, x : x + width]
            roi_hsv = hsv[y : y + height, x : x + width]
            combined_fill = float(np.count_nonzero(roi_mask)) / max(1.0, float(width * height))
            cool_fill = float(np.count_nonzero(roi_cool_mask)) / max(1.0, float(width * height))
            accent_fill = float(np.count_nonzero(roi_accent_mask)) / max(1.0, float(width * height))
            if combined_fill > float(profile.max_fill):
                continue
            if cool_fill < float(profile.min_fill) and accent_fill < float(profile.accent_min_fill):
                continue

            thickness = max(2, int(round(min(width, height) * float(profile.ring_thickness_ratio))))
            kernel_size = _as_odd(2 * thickness + 1, minimum=3)
            ring_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            core_mask = cv2.erode(roi_mask, ring_kernel)
            border_mask = cv2.subtract(roi_mask, core_mask)
            border_fill = float(np.count_nonzero(border_mask)) / max(1.0, float(border_mask.size))
            core_fill = float(np.count_nonzero(core_mask)) / max(1.0, float(core_mask.size))
            ring_score = border_fill - core_fill
            if border_fill < float(profile.min_border_fill):
                continue
            required_ring_score = float(profile.min_ring_score)
            if accent_fill >= float(profile.accent_min_fill):
                required_ring_score = required_ring_score * 0.78
            if ring_score < required_ring_score and core_fill > float(profile.max_core_fill):
                continue

            center_box_x1 = int(width * 0.30)
            center_box_x2 = int(width * 0.70)
            center_box_y1 = int(height * 0.30)
            center_box_y2 = int(height * 0.70)
            center_box = roi_mask[center_box_y1:center_box_y2, center_box_x1:center_box_x2]
            center_box_fill = float(np.count_nonzero(center_box)) / max(1.0, float(center_box.size))
            if center_box_fill < float(profile.min_center_fill):
                continue
            if center_box_fill > float(profile.max_core_fill):
                continue

            perimeter = float(cv2.arcLength(contour, True))
            circularity = 0.0 if perimeter <= 1.0 else float((4.0 * np.pi * contour_area) / (perimeter * perimeter))
            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            solidity = 0.0 if hull_area <= 1.0 else float(contour_area / hull_area)
            if circularity < float(profile.min_circularity) and solidity < float(profile.min_solidity):
                continue

            active_pixels = roi_mask > 0
            blue_strength = float(np.mean(roi_bias[active_pixels])) if np.any(active_pixels) else 0.0
            brightness = float(np.mean(roi_hsv[:, :, 2][active_pixels])) if np.any(active_pixels) else 0.0

            ring_component = float(_clamp((ring_score + 0.10) / 0.55, 0.0, 1.0))
            hollow_component = float(_clamp(1.0 - center_box_fill, 0.0, 1.0))
            fill_component = float(_clamp(1.0 - abs(combined_fill - 0.30) / 0.45, 0.0, 1.0))
            blue_component = float(_clamp((blue_strength + 8.0) / 48.0, 0.0, 1.0))
            bright_component = float(_clamp((brightness - 55.0) / 180.0, 0.0, 1.0))
            shape_component = float(_clamp(circularity, 0.0, 1.0))
            accent_component = float(_clamp(accent_fill / 0.22, 0.0, 1.0))
            accent_weight = float(_clamp(profile.accent_weight, 0.0, 0.35))
            score = (
                ring_component * (0.30 - accent_weight * 0.15)
                + hollow_component * (0.21 - accent_weight * 0.05)
                + fill_component * 0.14
                + blue_component * (0.13 - accent_weight * 0.10)
                + bright_component * 0.10
                + shape_component * 0.06
                + accent_component * accent_weight
            )
            if score < float(profile.min_score):
                continue

            candidates.append(
                {
                    "bbox_local": {"x": int(x), "y": int(y), "width": int(width), "height": int(height)},
                    "center_local": {"x": round(float(center_x), 3), "y": round(float(center_y), 3)},
                    "center_rel": {"x": round(center_x_rel, 6), "y": round(center_y_rel, 6)},
                    "mask_area": round(contour_area, 3),
                    "blue_fill": round(cool_fill, 6),
                    "combined_fill": round(combined_fill, 6),
                    "highlight_fill": round(accent_fill, 6),
                    "ring_score": round(float(ring_score), 6),
                    "core_fill": round(float(center_box_fill), 6),
                    "border_fill": round(float(border_fill), 6),
                    "circularity": round(float(circularity), 6),
                    "solidity": round(float(solidity), 6),
                    "blue_strength": round(float(blue_strength), 6),
                    "brightness": round(float(brightness), 6),
                    "score": round(float(score), 6),
                }
            )

        candidates.sort(
            key=lambda candidate: (
                -float(candidate.get("score", 0.0)),
                -float(candidate.get("ring_score", 0.0)),
                float(candidate.get("center_local", {}).get("y", 0.0)),
                float(candidate.get("center_local", {}).get("x", 0.0)),
            )
        )
        return {
            "mask": combined_mask,
            "candidates": candidates,
            "profile": profile.to_dict(),
        }


def detect_blue_stage_candidates(
    image_np: np.ndarray,
    mode_slug: str,
    map_data_root: str | Path = Path("data") / "map_data",
    profile_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    detector = BlueStageDetector(
        mode_slug=mode_slug,
        map_data_root=map_data_root,
        profile_overrides=profile_overrides,
    )
    return detector.detect(image_np)
