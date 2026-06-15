"""Collect champion icon training images from AyumiLove champion pages.

The collector downloads and caches a champion page, finds the main champion
image, crops the lower-left portrait area, resizes it to the model input size,
and records the result in a labels CSV.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, UnidentifiedImageError


IMAGE_WIDTH = 48
IMAGE_HEIGHT = 64
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
DEFAULT_RANKING_INDEX_URL = "https://ayumilove.net/raid-shadow-legends-list-of-champions-by-ranking/"

DEFAULT_USER_AGENT = (
    "cmp-identification-raid-icon-collector/0.1 "
    "(personal local ML experiment; respectful caching)"
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CropConfig:
    """Relative crop box for extracting a portrait from a full champion image.

    Values are fractions of the full image size in Pillow's crop-box order:
    left, top, right, bottom. The default targets the lower-left portrait/icon
    area used on the Relickeeper page.
    """

    left: float = 0.015
    top: float = 0.700
    right: float = 0.205
    bottom: float = 0.985

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "CropConfig":
        """Create a crop config from JSON-compatible relative coordinates."""
        return cls(
            left=float(data["left"]),
            top=float(data["top"]),
            right=float(data["right"]),
            bottom=float(data["bottom"]),
        )

    def to_dict(self) -> dict[str, float]:
        """Return JSON-compatible relative crop coordinates."""
        return {
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
        }

    def validate(self) -> None:
        """Raise ValueError if the relative crop box is malformed."""
        values = (self.left, self.top, self.right, self.bottom)
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"Crop values must be within [0, 1], got {values}")
        if self.left >= self.right or self.top >= self.bottom:
            raise ValueError(f"Invalid crop box order: {values}")

    def to_absolute(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Convert this relative crop box into pixel coordinates."""
        self.validate()
        box = (
            round(self.left * width),
            round(self.top * height),
            round(self.right * width),
            round(self.bottom * height),
        )
        left, top, right, bottom = box
        if right <= left or bottom <= top:
            raise ValueError(f"Crop box is empty for image size {width}x{height}: {box}")
        return box


@dataclass(frozen=True)
class CollectorConfig:
    """Runtime configuration for the data collector."""

    data_dir: Path = Path("data")
    user_agent: str = DEFAULT_USER_AGENT
    request_delay_seconds: float = 1.0
    timeout_seconds: float = 30.0
    crop: CropConfig | None = None

    @property
    def raw_pages_dir(self) -> Path:
        return self.data_dir / "raw" / "pages"

    @property
    def raw_images_dir(self) -> Path:
        return self.data_dir / "raw" / "images"

    @property
    def processed_icons_dir(self) -> Path:
        return self.data_dir / "processed" / "icons"

    @property
    def debug_dir(self) -> Path:
        return self.data_dir / "processed" / "debug"

    @property
    def labels_path(self) -> Path:
        return self.data_dir / "processed" / "labels.csv"

    @property
    def crop_config_path(self) -> Path:
        return self.data_dir / "processed" / "crop_config.json"

    @property
    def dataset_manifest_path(self) -> Path:
        return self.data_dir / "processed" / "dataset_manifest.csv"


@dataclass(frozen=True)
class CollectionResult:
    """Paths and metadata produced for one collected champion page."""

    label: str
    champion_name: str
    ranking_tier: str | None
    ranking_code: str | None
    ranking_faction_code: str | None
    ranking_faction_name: str | None
    ranking_rarity_code: str | None
    ranking_rarity_name: str | None
    ranking_role_code: str | None
    ranking_role_name: str | None
    ranking_affinity_code: str | None
    ranking_affinity_name: str | None
    ranking_order: int | None
    ranking_index_url: str | None
    source_url: str
    source_image_url: str | None
    raw_page_path: Path
    raw_image_path: Path
    processed_image_path: Path
    debug_image_path: Path
    crop_box: tuple[int, int, int, int]
    image_width: int
    image_height: int


@dataclass(frozen=True)
class ChampionSource:
    """Downloaded source assets and metadata for a champion page."""

    label: str
    champion_name: str
    source_url: str
    source_image_url: str
    raw_page_path: Path
    raw_image_path: Path


@dataclass(frozen=True)
class BatchCollectionResult:
    """Summary of a multi-page collection run."""

    index_url: str
    discovered_links: int
    attempted_links: int
    successful: int
    failed: int
    results: list[CollectionResult]
    errors: list[dict[str, str]]


@dataclass(frozen=True)
class UpdateCheckResult:
    """Summary of whether AyumiLove contains new champion pages."""

    update_available: bool
    missing_urls: list[str]
    collected_count: int
    available_count: int


class DataCollectionError(RuntimeError):
    """Raised when a champion page cannot be collected reliably."""


@dataclass(frozen=True)
class ChampionIndexEntry:
    """A champion guide URL paired with its ranking tier from the index page."""

    url: str
    ranking_tier: str | None
    ranking_code: str | None
    ranking_order: int | None


class AyumiLoveCollector:
    """Collect champion icons from AyumiLove pages with caching and delays."""

    def __init__(self, config: CollectorConfig | None = None) -> None:
        self.config = config or CollectorConfig()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.config.user_agent})

    def collect(
        self,
        url: str,
        *,
        force: bool = False,
        ranking_tier: str | None = None,
        ranking_code: str | None = None,
        ranking_order: int | None = None,
        ranking_index_url: str | None = None,
    ) -> CollectionResult:
        """Collect and process one champion page.

        Args:
            url: AyumiLove champion page URL.
            force: Redownload and regenerate outputs even if cached files exist.

        Returns:
            Metadata describing the collected sample.

        Raises:
            DataCollectionError: If required page content or image data is absent.
        """
        source = self.prepare_source(url, force=force)
        parsed_ranking_code = parse_ranking_code(ranking_code)
        processed_path = self.config.processed_icons_dir / f"{source.label}.png"
        debug_path = self.config.debug_dir / f"{source.label}_crop_debug.jpg"
        processed_existed = processed_path.exists()
        crop_config = self._active_crop_config()
        crop_box, image_width, image_height = self._process_image(
            source.raw_image_path,
            processed_path,
            debug_path,
            crop_config=crop_config,
            force=force,
        )

        result = CollectionResult(
            label=source.label,
            champion_name=source.champion_name,
            ranking_tier=ranking_tier,
            ranking_code=ranking_code,
            ranking_faction_code=parsed_ranking_code.faction_code,
            ranking_faction_name=parsed_ranking_code.faction_name,
            ranking_rarity_code=parsed_ranking_code.rarity_code,
            ranking_rarity_name=parsed_ranking_code.rarity_name,
            ranking_role_code=parsed_ranking_code.role_code,
            ranking_role_name=parsed_ranking_code.role_name,
            ranking_affinity_code=parsed_ranking_code.affinity_code,
            ranking_affinity_name=parsed_ranking_code.affinity_name,
            ranking_order=ranking_order,
            ranking_index_url=ranking_index_url,
            source_url=source.source_url,
            source_image_url=source.source_image_url,
            raw_page_path=source.raw_page_path,
            raw_image_path=source.raw_image_path,
            processed_image_path=processed_path,
            debug_image_path=debug_path,
            crop_box=crop_box,
            image_width=image_width,
            image_height=image_height,
        )
        if force or not processed_existed or not self._label_exists(source.label):
            self._upsert_label(result)
        else:
            LOGGER.info("Leaving existing label metadata unchanged for %s", source.label)
        self._sync_dataset_manifest()
        LOGGER.info("Collection complete: %s", processed_path)
        return result

    def prepare_source(self, url: str, *, force: bool = False) -> ChampionSource:
        """Download or reuse the page and main champion source image."""
        self._ensure_directories()
        LOGGER.info("Preparing champion source from %s", url)

        page_path = self._page_cache_path(url)
        html = self._fetch_text(url, page_path, force=force)
        soup = BeautifulSoup(html, "html.parser")

        champion_name = extract_champion_name(soup)
        label = normalize_label(champion_name)
        if not label:
            raise DataCollectionError(f"Could not normalize champion name: {champion_name!r}")
        LOGGER.info("Champion detected: %s (label: %s)", champion_name, label)

        image_url = find_main_champion_image_url(soup, url, champion_name)
        LOGGER.info("Main champion image candidate: %s", image_url)
        image_path = self._image_cache_path(image_url, label)
        self._fetch_binary(image_url, image_path, force=force)

        return ChampionSource(
            label=label,
            champion_name=champion_name,
            source_url=url,
            source_image_url=image_url,
            raw_page_path=page_path,
            raw_image_path=image_path,
        )

    def configure_crop_interactively(self, url: str, *, force: bool = False) -> CollectionResult:
        """Open the downloaded source image, let the user draw a crop, and save it."""
        source = self.prepare_source(url, force=force)
        crop_config = select_crop_interactively(source.raw_image_path)
        save_crop_config(crop_config, self.config.crop_config_path, force=force)

        processed_path = self.config.processed_icons_dir / f"{source.label}.png"
        debug_path = self.config.debug_dir / f"{source.label}_selected_crop_debug.jpg"
        processed_existed = processed_path.exists()
        crop_box, image_width, image_height = self._process_image(
            source.raw_image_path,
            processed_path,
            debug_path,
            crop_config=crop_config,
            force=force,
        )

        result = CollectionResult(
            label=source.label,
            champion_name=source.champion_name,
            ranking_tier=None,
            ranking_code=None,
            ranking_faction_code=None,
            ranking_faction_name=None,
            ranking_rarity_code=None,
            ranking_rarity_name=None,
            ranking_role_code=None,
            ranking_role_name=None,
            ranking_affinity_code=None,
            ranking_affinity_name=None,
            ranking_order=None,
            ranking_index_url=None,
            source_url=source.source_url,
            source_image_url=source.source_image_url,
            raw_page_path=source.raw_page_path,
            raw_image_path=source.raw_image_path,
            processed_image_path=processed_path,
            debug_image_path=debug_path,
            crop_box=crop_box,
            image_width=image_width,
            image_height=image_height,
        )
        if force or not processed_existed or not self._label_exists(source.label):
            self._upsert_label(result)
        else:
            LOGGER.info("Leaving existing label metadata unchanged for %s", source.label)
        self._sync_dataset_manifest()
        LOGGER.info("Saved selected crop debug image: %s", debug_path)
        return result

    def collect_from_index(
        self,
        index_url: str,
        *,
        force: bool = False,
        limit: int | None = None,
        stop_on_error: bool = False,
    ) -> BatchCollectionResult:
        """Collect champion icons from every champion guide link on an index page."""
        self._ensure_directories()
        index_path = self._page_cache_path(index_url)
        html = self._fetch_text(index_url, index_path, force=force)
        entries = extract_champion_index_entries(html, index_url)
        if limit is not None:
            if limit < 1:
                raise DataCollectionError("--limit must be at least 1")
            entries_to_collect = entries[:limit]
        else:
            entries_to_collect = entries

        LOGGER.info(
            "Discovered %s champion links on %s; collecting %s",
            len(entries),
            index_url,
            len(entries_to_collect),
        )
        if not self.config.crop_config_path.exists() and self.config.crop is None:
            LOGGER.warning(
                "No saved crop config found at %s; using built-in default crop. "
                "Run configure-crop first if you want a manually calibrated crop.",
                self.config.crop_config_path,
            )

        results: list[CollectionResult] = []
        errors: list[dict[str, str]] = []
        for index, entry in enumerate(entries_to_collect, start=1):
            LOGGER.info("Collecting champion %s/%s: %s", index, len(entries_to_collect), entry.url)
            try:
                results.append(
                    self.collect(
                        entry.url,
                        force=force,
                        ranking_tier=entry.ranking_tier,
                        ranking_code=entry.ranking_code,
                        ranking_order=entry.ranking_order,
                        ranking_index_url=index_url,
                    )
                )
            except (DataCollectionError, ValueError, OSError) as exc:
                message = str(exc)
                LOGGER.error("Failed to collect %s: %s", entry.url, message)
                errors.append({"url": entry.url, "error": message})
                if stop_on_error:
                    raise

        return BatchCollectionResult(
            index_url=index_url,
            discovered_links=len(entries),
            attempted_links=len(entries_to_collect),
            successful=len(results),
            failed=len(errors),
            results=results,
            errors=errors,
        )

    def reprocess_existing_icons(self, *, force: bool = False) -> BatchCollectionResult:
        """Re-crop existing label rows from cached raw images without network requests."""
        self._ensure_directories()
        if not self.config.labels_path.exists():
            raise DataCollectionError(f"Labels CSV not found: {self.config.labels_path}")
        if not force:
            raise DataCollectionError("Reprocessing overwrites processed icons/debug images; pass --force")

        labels = pd.read_csv(self.config.labels_path)
        required = {"label", "champion_name", "source_url", "raw_image_path", "processed_image_path"}
        missing = required.difference(labels.columns)
        if missing:
            raise DataCollectionError(f"Labels CSV is missing required columns: {sorted(missing)}")

        crop_config = self._active_crop_config()
        results: list[CollectionResult] = []
        errors: list[dict[str, str]] = []
        rows = labels.to_dict(orient="records")
        for index, row in enumerate(rows, start=1):
            label = str(row["label"])
            LOGGER.info("Reprocessing cached icon %s/%s: %s", index, len(rows), label)
            try:
                raw_image_path = Path(str(row["raw_image_path"]))
                processed_path = Path(str(row["processed_image_path"]))
                debug_path = self.config.debug_dir / f"{label}_crop_debug.jpg"
                crop_box, image_width, image_height = self._process_image(
                    raw_image_path,
                    processed_path,
                    debug_path,
                    crop_config=crop_config,
                    force=force,
                )
                result = CollectionResult(
                    label=label,
                    champion_name=str(row["champion_name"]),
                    ranking_tier=str(row["ranking_tier"]) if "ranking_tier" in row and pd.notna(row["ranking_tier"]) else None,
                    ranking_code=str(row["ranking_code"]) if "ranking_code" in row and pd.notna(row["ranking_code"]) else None,
                    ranking_faction_code=str(row["ranking_faction_code"]) if "ranking_faction_code" in row and pd.notna(row["ranking_faction_code"]) else None,
                    ranking_faction_name=str(row["ranking_faction_name"]) if "ranking_faction_name" in row and pd.notna(row["ranking_faction_name"]) else None,
                    ranking_rarity_code=str(row["ranking_rarity_code"]) if "ranking_rarity_code" in row and pd.notna(row["ranking_rarity_code"]) else None,
                    ranking_rarity_name=str(row["ranking_rarity_name"]) if "ranking_rarity_name" in row and pd.notna(row["ranking_rarity_name"]) else None,
                    ranking_role_code=str(row["ranking_role_code"]) if "ranking_role_code" in row and pd.notna(row["ranking_role_code"]) else None,
                    ranking_role_name=str(row["ranking_role_name"]) if "ranking_role_name" in row and pd.notna(row["ranking_role_name"]) else None,
                    ranking_affinity_code=str(row["ranking_affinity_code"]) if "ranking_affinity_code" in row and pd.notna(row["ranking_affinity_code"]) else None,
                    ranking_affinity_name=str(row["ranking_affinity_name"]) if "ranking_affinity_name" in row and pd.notna(row["ranking_affinity_name"]) else None,
                    ranking_order=int(row["ranking_order"]) if "ranking_order" in row and pd.notna(row["ranking_order"]) else None,
                    ranking_index_url=str(row["ranking_index_url"]) if "ranking_index_url" in row and pd.notna(row["ranking_index_url"]) else None,
                    source_url=str(row["source_url"]),
                    source_image_url=str(row["source_image_url"]) if "source_image_url" in row and pd.notna(row["source_image_url"]) else None,
                    raw_page_path=Path(""),
                    raw_image_path=raw_image_path,
                    processed_image_path=processed_path,
                    debug_image_path=debug_path,
                    crop_box=crop_box,
                    image_width=image_width,
                    image_height=image_height,
                )
                self._upsert_label(result)
                results.append(result)
            except (DataCollectionError, ValueError, OSError) as exc:
                message = str(exc)
                LOGGER.error("Failed to reprocess %s: %s", label, message)
                errors.append({"url": str(row.get("source_url", label)), "error": message})

        return BatchCollectionResult(
            index_url="labels.csv",
            discovered_links=len(rows),
            attempted_links=len(rows),
            successful=len(results),
            failed=len(errors),
            results=results,
            errors=errors,
        )

    def _ensure_directories(self) -> None:
        for directory in (
            self.config.raw_pages_dir,
            self.config.raw_images_dir,
            self.config.processed_icons_dir,
            self.config.debug_dir,
            self.config.labels_path.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def _fetch_text(self, url: str, path: Path, *, force: bool) -> str:
        if path.exists() and not force:
            LOGGER.info("Using cached page: %s", path)
            return path.read_text(encoding="utf-8")

        LOGGER.info("Downloading page: %s", url)
        self._respectful_delay()
        try:
            response = self.session.get(url, timeout=self.config.timeout_seconds)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise DataCollectionError(f"Failed to download page {url}: {exc}") from exc

        path.write_text(response.text, encoding="utf-8")
        return response.text

    def _fetch_binary(self, url: str, path: Path, *, force: bool) -> None:
        if path.exists() and not force:
            LOGGER.info("Using cached image: %s", path)
            return

        LOGGER.info("Downloading image: %s", url)
        self._respectful_delay()
        try:
            response = self.session.get(url, timeout=self.config.timeout_seconds)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise DataCollectionError(f"Failed to download image {url}: {exc}") from exc

        content_type = response.headers.get("Content-Type", "")
        if "image" not in content_type.lower():
            LOGGER.warning("Image response has unexpected content type: %s", content_type)
        path.write_bytes(response.content)

    def _process_image(
        self,
        raw_image_path: Path,
        processed_path: Path,
        debug_path: Path,
        *,
        crop_config: CropConfig,
        force: bool,
    ) -> tuple[tuple[int, int, int, int], int, int]:
        try:
            with Image.open(raw_image_path) as source:
                image = source.convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise DataCollectionError(f"Could not read image {raw_image_path}: {exc}") from exc

        width, height = image.size
        crop_box = crop_config.to_absolute(width, height)
        LOGGER.info("Cropping %s with box %s from %sx%s image", raw_image_path, crop_box, width, height)

        if processed_path.exists() and not force:
            LOGGER.info("Using existing processed image: %s", processed_path)
        else:
            icon = image.crop(crop_box).resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
            icon.save(processed_path)

        if debug_path.exists() and not force:
            LOGGER.info("Using existing crop debug image: %s", debug_path)
        else:
            save_crop_debug_image(image, crop_box, debug_path)
        return crop_box, IMAGE_WIDTH, IMAGE_HEIGHT

    def _active_crop_config(self) -> CropConfig:
        if self.config.crop is not None:
            LOGGER.info("Using crop configuration from CLI arguments")
            return self.config.crop
        if self.config.crop_config_path.exists():
            LOGGER.info("Using saved crop configuration: %s", self.config.crop_config_path)
            return load_crop_config(self.config.crop_config_path)
        LOGGER.info("Using built-in default crop configuration")
        return CropConfig()

    def _upsert_label(self, result: CollectionResult) -> None:
        row = {
            "label": result.label,
            "champion_name": result.champion_name,
            "ranking_tier": result.ranking_tier,
            "ranking_code": result.ranking_code,
            "ranking_faction_code": result.ranking_faction_code,
            "ranking_faction_name": result.ranking_faction_name,
            "ranking_rarity_code": result.ranking_rarity_code,
            "ranking_rarity_name": result.ranking_rarity_name,
            "ranking_role_code": result.ranking_role_code,
            "ranking_role_name": result.ranking_role_name,
            "ranking_affinity_code": result.ranking_affinity_code,
            "ranking_affinity_name": result.ranking_affinity_name,
            "ranking_order": result.ranking_order,
            "ranking_index_url": result.ranking_index_url,
            "source_url": result.source_url,
            "source_image_url": result.source_image_url,
            "raw_page_path": result.raw_page_path.as_posix(),
            "raw_image_path": result.raw_image_path.as_posix(),
            "processed_image_path": result.processed_image_path.as_posix(),
            "debug_image_path": result.debug_image_path.as_posix(),
            "crop_box": ",".join(str(value) for value in result.crop_box),
            "image_width": result.image_width,
            "image_height": result.image_height,
        }

        if self.config.labels_path.exists():
            labels = pd.read_csv(self.config.labels_path)
            labels = labels[labels["label"] != result.label]
            labels = pd.concat([labels, pd.DataFrame([row])], ignore_index=True)
        else:
            labels = pd.DataFrame([row])

        labels = labels.sort_values("label").reset_index(drop=True)
        labels.to_csv(self.config.labels_path, index=False)
        self._write_dataset_manifest(labels)
        LOGGER.info("Updated labels CSV: %s", self.config.labels_path)

    def _label_exists(self, label: str) -> bool:
        if not self.config.labels_path.exists():
            return False
        try:
            labels = pd.read_csv(self.config.labels_path)
        except (OSError, pd.errors.ParserError):
            return False
        return "label" in labels.columns and label in set(labels["label"].astype(str))

    def _page_cache_path(self, url: str) -> Path:
        parsed = urlparse(url)
        slug = Path(parsed.path.rstrip("/")).name or "page"
        return self.config.raw_pages_dir / f"{safe_stem(slug)}.html"

    def _image_cache_path(self, url: str, label: str) -> Path:
        parsed = urlparse(url)
        suffix = Path(parsed.path).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".img"
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
        return self.config.raw_images_dir / f"{label}_{digest}{suffix}"

    def _respectful_delay(self) -> None:
        if self.config.request_delay_seconds > 0:
            time.sleep(self.config.request_delay_seconds)

    def _write_dataset_manifest(self, labels: pd.DataFrame) -> None:
        """Write a convenience dataset manifest alongside the labels CSV."""
        manifest_columns = [
            column
            for column in (
                "label",
                "champion_name",
                "ranking_tier",
                "ranking_code",
                "ranking_faction_code",
                "ranking_faction_name",
                "ranking_rarity_code",
                "ranking_rarity_name",
                "ranking_role_code",
                "ranking_role_name",
                "ranking_affinity_code",
                "ranking_affinity_name",
                "ranking_order",
                "ranking_index_url",
                "source_url",
                "source_image_url",
                "raw_page_path",
                "raw_image_path",
                "processed_image_path",
                "debug_image_path",
                "crop_box",
                "image_width",
                "image_height",
            )
            if column in labels.columns
        ]
        manifest = labels.loc[:, manifest_columns].copy() if manifest_columns else labels.copy()
        manifest.to_csv(self.config.dataset_manifest_path, index=False)
        LOGGER.info("Updated dataset manifest: %s", self.config.dataset_manifest_path)

    def _sync_dataset_manifest(self) -> None:
        """Refresh the dataset manifest from the current labels CSV, if present."""
        if not self.config.labels_path.exists():
            return
        labels = pd.read_csv(self.config.labels_path)
        self._write_dataset_manifest(labels)


def check4updates(
    *,
    data_dir: Path = Path("data"),
    index_url: str = DEFAULT_RANKING_INDEX_URL,
) -> UpdateCheckResult:
    """Compare AyumiLove champion links against collected dataset URLs."""
    labels_path = Path(data_dir) / "processed" / "labels.csv"
    if not labels_path.exists():
        raise DataCollectionError(f"Labels CSV not found: {labels_path}")

    labels = pd.read_csv(labels_path)
    if "source_url" not in labels.columns:
        raise DataCollectionError(f"Labels CSV is missing required column: source_url")

    collected_urls = {
        normalize_url(str(url))
        for url in labels["source_url"].dropna().astype(str)
        if str(url).strip()
    }

    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT})
    try:
        response = session.get(index_url, timeout=30.0)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise DataCollectionError(f"Failed to fetch ranking page {index_url}: {exc}") from exc

    entries = extract_champion_index_entries(response.text, index_url)
    available_urls = [entry.url for entry in entries]
    missing_urls = [url for url in available_urls if normalize_url(url) not in collected_urls]

    if missing_urls:
        print("Update available")
        for url in missing_urls:
            print(url)
    else:
        print("No update available")

    return UpdateCheckResult(
        update_available=bool(missing_urls),
        missing_urls=missing_urls,
        collected_count=len(collected_urls),
        available_count=len(available_urls),
    )


def extract_champion_name(soup: BeautifulSoup) -> str:
    """Extract champion name, preferring the overview `NAME:` field."""
    text = soup.get_text("\n", strip=True)
    name_match = re.search(r"(?im)^NAME:\s*(.+?)\s*$", text)
    if name_match:
        return clean_champion_name(name_match.group(1))

    heading = soup.find(["h1", "h2"])
    if heading:
        heading_text = heading.get_text(" ", strip=True)
        heading_text = re.split(r"\s+-\s+|\s+\|\s+", heading_text, maxsplit=1)[0]
        heading_text = re.sub(r"\bRaid Shadow Legends\b", "", heading_text, flags=re.IGNORECASE)
        return clean_champion_name(heading_text)

    title = soup.find("title")
    if title:
        title_text = title.get_text(" ", strip=True)
        title_text = re.split(r"\s+-\s+|\s+\|\s+", title_text, maxsplit=1)[0]
        return clean_champion_name(title_text)

    raise DataCollectionError("Could not extract champion name from page")


def clean_champion_name(value: str) -> str:
    """Remove extra descriptors from a champion-name field."""
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"\s*\([^)]*\)\s*$", "", value).strip()
    return value


def normalize_label(value: str) -> str:
    """Convert a champion name into a lowercase ASCII-safe label."""
    ascii_text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_text.lower()).strip("_")
    return normalized


def safe_stem(value: str) -> str:
    """Create a filesystem-safe lowercase stem."""
    stem = normalize_label(Path(value).stem or value)
    return stem or hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def normalize_url(value: str) -> str:
    """Normalize a URL for stable comparisons."""
    parsed = urlparse(value)
    normalized = parsed._replace(query="", fragment="", params="").geturl().rstrip("/")
    return normalized + "/"


def extract_champion_links(html: str, index_url: str) -> list[str]:
    """Extract unique champion guide links from an AyumiLove ranking page."""
    return [entry.url for entry in extract_champion_index_entries(html, index_url)]


def extract_champion_index_entries(html: str, index_url: str) -> list[ChampionIndexEntry]:
    """Extract unique champion guide links and their ranking tiers from an index page."""
    soup = BeautifulSoup(html, "html.parser")
    entries: list[ChampionIndexEntry] = []
    seen: set[str] = set()
    guide_pattern = re.compile(
        r"/raid-shadow-legends-[a-z0-9-]+-skill-mastery-equip-guide/?$",
        re.IGNORECASE,
    )
    ranking_order = 0
    for heading in soup.find_all("h4"):
        tier = parse_ranking_tier(heading.get_text(" ", strip=True))
        if tier is None:
            continue
        next_ul = heading.find_next("ul")
        if next_ul is None:
            continue
        for anchor in next_ul.find_all("a", href=True):
            href = str(anchor["href"]).strip()
            ranking_code = extract_ranking_code(anchor.get_text(" ", strip=True))
            if ranking_code is None:
                continue
            absolute_url = urljoin(index_url, href)
            parsed = urlparse(absolute_url)
            normalized = parsed._replace(query="", fragment="", params="").geturl().rstrip("/") + "/"
            if parsed.netloc and "ayumilove.net" not in parsed.netloc.lower():
                continue
            if not guide_pattern.search(urlparse(normalized).path):
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            ranking_order += 1
            entries.append(
                ChampionIndexEntry(
                    url=normalized,
                    ranking_tier=tier,
                    ranking_code=ranking_code,
                    ranking_order=ranking_order,
                )
            )
    if not entries:
        raise DataCollectionError(f"No champion guide links found on index page: {index_url}")
    return entries


def extract_ranking_code(value: str) -> str | None:
    """Extract a ranking code such as `KR-MSF` from a tier-list link label."""
    match = re.search(r"\(([A-Z]{2,3}-[A-Z]{3})\)\s*$", value)
    if not match:
        return None
    return match.group(1).upper()


@dataclass(frozen=True)
class ParsedRankingCode:
    """Decoded ranking code details from a champion tier-list link."""

    faction_code: str | None
    faction_name: str | None
    rarity_code: str | None
    rarity_name: str | None
    role_code: str | None
    role_name: str | None
    affinity_code: str | None
    affinity_name: str | None


FACTION_NAMES = {
    "AR": "Argonites",
    "BA": "Barbarians",
    "BL": "Banner Lords",
    "DE": "Dark Elves",
    "DS": "Demonspawn",
    "DW": "Dwarves",
    "HE": "High Elves",
    "KR": "Knights Revenant",
    "LZ": "Lizardmen",
    "OR": "Orcs",
    "OT": "Ogryn Tribes",
    "SK": "Shadowkin",
    "SO": "The Sacred Order",
    "SY": "Sylvan Watchers",
    "SW": "Skinwalkers",
    "UH": "Undead Hordes",
}

RARITY_NAMES = {
    "M": "Mythical",
    "L": "Legendary",
    "E": "Epic",
    "R": "Rare",
    "U": "Uncommon",
    "C": "Common",
}

ROLE_NAMES = {
    "A": "Attack",
    "D": "Defense",
    "H": "HP",
    "S": "Support",
}

AFFINITY_NAMES = {
    "F": "Force",
    "M": "Magic",
    "S": "Spirit",
    "V": "Void",
}


def parse_ranking_code(value: str | None) -> ParsedRankingCode:
    """Decode a ranking code into faction, rarity, role, and affinity pieces."""
    if not value:
        return ParsedRankingCode(None, None, None, None, None, None, None, None)
    match = re.fullmatch(r"([A-Z]{2,3})-([A-Z])([A-Z])([A-Z])", value.strip().upper())
    if not match:
        return ParsedRankingCode(None, None, None, None, None, None, None, None)

    faction_code, rarity_code, role_code, affinity_code = match.groups()
    return ParsedRankingCode(
        faction_code=faction_code,
        faction_name=FACTION_NAMES.get(faction_code),
        rarity_code=rarity_code,
        rarity_name=RARITY_NAMES.get(rarity_code),
        role_code=role_code,
        role_name=ROLE_NAMES.get(role_code),
        affinity_code=affinity_code,
        affinity_name=AFFINITY_NAMES.get(affinity_code),
    )


def parse_ranking_tier(value: str) -> str | None:
    """Extract a tier label such as SS, S, A, B, C, or F from a heading."""
    match = re.search(r"\b(SS|S|A|B|C|F)\s+Rank\b", value, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).upper()


def load_crop_config(path: Path) -> CropConfig:
    """Load relative crop coordinates from JSON."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        crop_data = data.get("relative_crop_box", data)
        crop_config = CropConfig.from_dict(crop_data)
        crop_config.validate()
        return crop_config
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise DataCollectionError(f"Could not load crop config {path}: {exc}") from exc


def save_crop_config(crop_config: CropConfig, path: Path, *, force: bool) -> None:
    """Save relative crop coordinates without overwriting unless forced."""
    crop_config.validate()
    if path.exists() and not force:
        raise DataCollectionError(f"Refusing to overwrite existing crop config without --force: {path}")
    payload = {
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "relative_crop_box": crop_config.to_dict(),
        "format": "left, top, right, bottom as fractions of source image size",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("Saved crop configuration: %s", path)


def save_crop_debug_image(
    image: Image.Image,
    crop_box: tuple[int, int, int, int],
    debug_path: Path,
) -> None:
    """Save a copy of the source image with the crop rectangle drawn on top."""
    debug_image = image.copy()
    draw = ImageDraw.Draw(debug_image)
    line_width = max(3, round(min(image.size) * 0.006))
    draw.rectangle(crop_box, outline=(255, 0, 0), width=line_width)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug_image.save(debug_path, quality=92)


def select_crop_interactively(image_path: Path) -> CropConfig:
    """Display an image and return a user-drawn crop box as relative coordinates."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RectangleSelector
    except ModuleNotFoundError as exc:
        raise DataCollectionError(
            "Interactive crop selection requires matplotlib. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise DataCollectionError(f"Could not read image {image_path}: {exc}") from exc

    width, height = image.size
    selected_box: dict[str, tuple[float, float, float, float]] = {}

    def on_select(click_event: object, release_event: object) -> None:
        x1 = getattr(click_event, "xdata", None)
        y1 = getattr(click_event, "ydata", None)
        x2 = getattr(release_event, "xdata", None)
        y2 = getattr(release_event, "ydata", None)
        if None in (x1, y1, x2, y2):
            return
        left = min(float(x1), float(x2))
        right = max(float(x1), float(x2))
        top = min(float(y1), float(y2))
        bottom = max(float(y1), float(y2))
        selected_box["box"] = (
            max(0.0, min(left, width)),
            max(0.0, min(top, height)),
            max(0.0, min(right, width)),
            max(0.0, min(bottom, height)),
        )

    def on_key(event: object) -> None:
        if getattr(event, "key", None) in {"enter", "return"} and "box" in selected_box:
            plt.close(fig)

    fig, axis = plt.subplots()
    axis.imshow(image)
    axis.set_title("Draw champion icon crop, then press Enter")
    axis.set_axis_off()
    selector = RectangleSelector(
        axis,
        on_select,
        useblit=True,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=True,
        props={"facecolor": "none", "edgecolor": "red", "linewidth": 2},
    )
    selector.set_active(True)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    if "box" not in selected_box:
        raise DataCollectionError("No crop rectangle selected")

    left, top, right, bottom = selected_box["box"]
    crop_config = CropConfig(
        left=left / width,
        top=top / height,
        right=right / width,
        bottom=bottom / height,
    )
    crop_config.validate()
    LOGGER.info("Selected relative crop box: %s", crop_config.to_dict())
    return crop_config


def find_main_champion_image_url(
    soup: BeautifulSoup,
    page_url: str,
    champion_name: str,
) -> str:
    """Find the most likely large champion screenshot on an AyumiLove page."""
    candidates = list(_iter_image_candidates(soup, page_url))
    if not candidates:
        raise DataCollectionError("No image candidates found on page")

    label_tokens = set(normalize_label(champion_name).split("_"))
    scored = sorted(
        candidates,
        key=lambda candidate: _score_image_candidate(candidate, label_tokens),
        reverse=True,
    )
    best = scored[0]
    if _score_image_candidate(best, label_tokens) <= 0:
        raise DataCollectionError("Could not identify a plausible main champion image")
    return best["url"]


def _iter_image_candidates(soup: BeautifulSoup, page_url: str) -> Iterable[dict[str, str]]:
    for image in soup.find_all("img"):
        source = (
            image.get("data-src")
            or image.get("data-lazy-src")
            or image.get("data-original")
            or image.get("src")
        )
        if not source:
            srcset = image.get("srcset") or image.get("data-srcset")
            source = _largest_srcset_url(srcset) if srcset else None
        if not source:
            continue
        url = urljoin(page_url, source)
        text_bits = [
            url,
            image.get("alt", ""),
            image.get("title", ""),
            " ".join(image.get("class", [])),
            image.get("id", ""),
        ]
        yield {"url": url, "text": " ".join(text_bits)}


def _largest_srcset_url(srcset: str | None) -> str | None:
    if not srcset:
        return None
    parts = [part.strip().split(" ") for part in srcset.split(",") if part.strip()]
    if not parts:
        return None
    return parts[-1][0]


def _score_image_candidate(candidate: dict[str, str], label_tokens: set[str]) -> int:
    text = normalize_label(candidate["text"])
    url = candidate["url"].lower()
    score = 0
    for token in label_tokens:
        if token and token in text:
            score += 20
    if "champion" in text:
        score += 8
    if "guide" in text:
        score += 4
    if any(part in url for part in (".jpg", ".jpeg", ".png", ".webp")):
        score += 3
    if any(bad in text for bad in ("avatar", "logo", "banner", "youtube", "facebook", "rating")):
        score -= 15
    return score
