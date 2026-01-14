# -*- coding: utf-8 -*-
"""
Utility library for UI automation, OCR, and image processing.

Created on Sat Oct 25 13:46:00 2025
@author: Arthur
"""

import re
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pyautogui
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


class TextObject:
    """Represents a detected text with its approximate screen position."""

    def __init__(self, text: Optional[str], mean_pos_x: float, mean_pos_y: float):
        self.text = text
        self.mean_pos_x = mean_pos_x
        self.mean_pos_y = mean_pos_y


# -------------------- Color Detection -------------------- #

def hex_to_bgr(hex_color: str) -> np.ndarray:
    """Convert HEX color string to BGR NumPy array."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return np.array([b, g, r], dtype=np.int16)


def detect_red_or_green_circle_stable(
    region_coords: Tuple[int, int, int, int],
    samples: int = 40,
    required_ratio: float = 0.75,
    min_pixels: int = 8,
    green_hex: str = "3CB043",
    red_hex: str = "C0392B",
    tolerance: int = 45
) -> Optional[str]:
    """
    Detect small red or green circles in a region with stable color sampling.

    Args:
        region_coords: (x, y, width, height) of the screen region.
        samples: Number of screenshots to sample.
        required_ratio: Fraction of samples that must detect the color.
        min_pixels: Minimum pixels to consider detection in one sample.
        green_hex: HEX code of green target color.
        red_hex: HEX code of red target color.
        tolerance: Max color distance to match.

    Returns:
        "red", "green", or None if no reliable detection.
    """
    green_bgr = hex_to_bgr(green_hex)
    red_bgr = hex_to_bgr(red_hex)

    green_hits = red_hits = 0
    threshold = int(samples * required_ratio)

    for _ in range(samples):
        screenshot = pyautogui.screenshot(region=region_coords)
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR).astype(np.int16)

        green_diff = np.linalg.norm(frame - green_bgr, axis=2)
        red_diff = np.linalg.norm(frame - red_bgr, axis=2)

        if np.count_nonzero(green_diff <= tolerance) >= min_pixels:
            green_hits += 1
        if np.count_nonzero(red_diff <= tolerance) >= min_pixels:
            red_hits += 1

        # Early exit if threshold met
        if green_hits >= threshold:
            return "green"
        if red_hits >= threshold:
            return "red"

    return None


def check_area_color(window, search_area: Tuple[float, float, float, float]) -> str:
    """
    Detects the dominant color in a relative screen area.

    Returns one of: "grey", "yellow", "green", "red", "black"
    """
    if not window:
        return "grey"

    rel_left, rel_top, rel_width, rel_height = search_area
    abs_left = window.left + int(rel_left * window.width)
    abs_top = window.top + int(rel_top * window.height)
    abs_width = int(rel_width * window.width)
    abs_height = int(rel_height * window.height)

    screenshot = pyautogui.screenshot(region=(abs_left, abs_top, abs_width, abs_height))
    img_np = np.array(screenshot)
    r, g, b = img_np.mean(axis=(0, 1))

    # Threshold rules
    if 40 < r < 60 and 70 < g < 100 and 90 < b < 110:
        return "grey"
    if 150 < r < 170 and 110 < g < 140 and 0 < b < 30:
        return "yellow"

    return "grey"  # fallback


# -------------------- Image Comparison -------------------- #

def compare_pngs(path1: str, path2: str) -> Optional[float]:
    """
    Compare two PNG images and return SSIM similarity score.

    Returns:
        float between 0-1 or None if images cannot be compared.
    """
    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        print(f"Error: Invalid image path(s): {path1}, {path2}")
        return None
    if img1.shape != img2.shape:
        print("Images have different sizes — cannot compare.")
        return None
    score, _ = ssim(img1, img2, full=True)
    return score


# -------------------- Template Matching -------------------- #

def get_similarities_in_relative_area(
    window,
    search_area: Tuple[float, float, float, float],
    path_to_template: str,
    threshold: float = 0.8,
    scales: Optional[List[float]] = None
) -> List[TextObject]:
    """
    Detect occurrences of a template image within a relative window area.

    Returns a list of TextObjects with positions.
    """
    if not window:
        return []

    scales = scales or [0.25, 0.5, 0.75, 1.0]
    rel_left, rel_top, rel_width, rel_height = search_area

    abs_left = window.left + int(rel_left * window.width)
    abs_top = window.top + int(rel_top * window.height)
    abs_width = int(rel_width * window.width)
    abs_height = int(rel_height * window.height)

    screenshot = pyautogui.screenshot(region=(abs_left, abs_top, abs_width, abs_height))
    search_img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)

    template_orig = cv2.imread(path_to_template, cv2.IMREAD_GRAYSCALE)
    if template_orig is None:
        raise FileNotFoundError(path_to_template)

    text_objects = []
    used_points = []

    for scale in scales:
        t_w = int(template_orig.shape[1] * scale)
        t_h = int(template_orig.shape[0] * scale)
        if t_w > search_img.shape[1] or t_h > search_img.shape[0]:
            continue
        template = cv2.resize(template_orig, (t_w, t_h), interpolation=cv2.INTER_AREA)

        result = cv2.matchTemplate(search_img, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshold)

        for pt in zip(*loc[::-1]):
            center_x, center_y = pt[0] + t_w / 2, pt[1] + t_h / 2
            abs_x, abs_y = abs_left + center_x, abs_top + center_y

            # Suppress near-duplicates
            if any(abs(abs_x - ux) < t_w * 0.5 and abs(abs_y - uy) < t_h * 0.5 for ux, uy in used_points):
                continue

            used_points.append((abs_x, abs_y))
            text_objects.append(TextObject(text=None, mean_pos_x=abs_x, mean_pos_y=abs_y))

    return text_objects


# -------------------- OCR Text Extraction -------------------- #

def get_text_from_image(reader, image) -> str:
    """
    Extracts text from an image using EasyOCR.

    Returns concatenated strings with confidence > 0.6.
    """
    result = reader.readtext(np.array(image))
    return "\n".join([text for (_, text, conf) in result if conf > 0.6])


def get_text_in_full_area(reader, coords: Optional[Tuple[int, int, int, int]]) -> Optional[str]:
    """Capture text from the entire window area."""
    if not coords:
        return None
    screenshot = pyautogui.screenshot(region=coords)
    return get_text_from_image(reader, screenshot)


def get_text_in_relative_area(
    reader,
    window,
    search_area: Tuple[float, float, float, float],
    power_detection: bool = False,
    faction_detection: bool = False
) -> List[TextObject]:
    """
    Capture text from a relative area and return as TextObjects.
    """
    if not window:
        return []

    rel_left, rel_top, rel_width, rel_height = search_area
    abs_left = window.left + int(rel_left * window.width)
    abs_top = window.top + int(rel_top * window.height)
    abs_width = int(rel_width * window.width)
    abs_height = int(rel_height * window.height)

    screenshot = pyautogui.screenshot(region=(abs_left, abs_top, abs_width, abs_height))
    image_np = np.array(screenshot)

    if power_detection:
        results = reader.readtext(image_np, allowlist='0123456789.,KkMmLUCHARluchar ')
    elif faction_detection:
        results = reader.readtext(image_np, allowlist='36')
    else:
        results = reader.readtext(image_np)

    text_objects = []
    for bbox, text, confidence in results:
        if confidence < 0.5:
            continue
        xs, ys = zip(*bbox)
        mean_x = sum(xs) / 4
        mean_y = sum(ys) / 4
        text_objects.append(TextObject(text=text, mean_pos_x=abs_left + mean_x, mean_pos_y=abs_top + mean_y))

    return text_objects

def get_text_from_cluster_area(reader, window, search_areas, power_detection=False):
    """
    Extract text from multiple cluster areas in a window.
    Preserves original behavior from old library.
    
    Args:
        reader: EasyOCR Reader
        window: WindowObject
        search_areas: dict of {pos: list of relative areas}
        power_detection: limit OCR to digits/letters for power levels
    Returns:
        List[TextObject]
    """
    if not window:
        return []

    all_text_objects = []

    for pos in search_areas:
        for area in search_areas[pos]:
            text_objs = get_text_in_relative_area(
                reader,
                window,
                area,
                power_detection=power_detection
            )

            # Optional: filter objects as in old implementation
            for obj in text_objs:
                text = obj.text or ""
                has_number = bool(re.search(r"\d", text))
                is_luchar = text.strip().lower() == "luchar"
                if not has_number and not is_luchar:
                    continue
                all_text_objects.append(obj)

    return all_text_objects

# -------------------- TextObject Utilities -------------------- #

def filter_text_objects(text_objects: List[TextObject]) -> List[TextObject]:
    """Clean and structure OCR text objects into expected patterns."""
    def has_number(text: str) -> bool:
        return bool(re.search(r"\d", text))

    def is_luchar(text: str) -> bool:
        return text.strip().lower() == "luchar"

    # Remove short entries
    text_objects = [obj for obj in text_objects if len(obj.text.strip()) >= 2]

    cleaned = []
    i = 0
    while i < len(text_objects):
        current = text_objects[i]
        next_obj = text_objects[i + 1] if i + 1 < len(text_objects) else None
        curr_text = current.text.strip()

        if next_obj and is_luchar(curr_text) and is_luchar(next_obj.text):
            cleaned.append(current)
            i += 2
            continue

        if has_number(curr_text):
            if next_obj and is_luchar(next_obj.text):
                cleaned.append(current)
            i += 1
            continue

        cleaned.append(current)
        i += 1

    # Enforce alternating number → luchar
    structured = []
    expect_number = True
    for obj in cleaned:
        text = obj.text.strip()
        if expect_number and has_number(text):
            structured.append(obj)
            expect_number = False
        elif not expect_number and is_luchar(text):
            structured.append(obj)
            expect_number = True

    return structured


def find_clusters(text_objects: List[TextObject], num_majorities: int = 2, tolerance: int = 8) -> List[TextObject]:
    """Cluster TextObjects by horizontal position and return top clusters."""
    text_objects = [obj for obj in text_objects if len(obj.text) >= 3]
    if not text_objects:
        return []

    sorted_objs = sorted(text_objects, key=lambda o: o.mean_pos_x)
    clusters = []
    current_cluster = []
    current_x = None

    for obj in sorted_objs:
        if current_x is None:
            current_x = obj.mean_pos_x
            current_cluster = [obj]
        elif abs(obj.mean_pos_x - current_x) <= tolerance:
            current_cluster.append(obj)
            current_x = sum(o.mean_pos_x for o in current_cluster) / len(current_cluster)
        else:
            clusters.append(current_cluster)
            current_cluster = [obj]
            current_x = obj.mean_pos_x
    if current_cluster:
        clusters.append(current_cluster)

    clusters.sort(key=len, reverse=True)
    top_clusters = clusters[:num_majorities]

    top_objs_set = {obj for cluster in top_clusters for obj in cluster}
    return [obj for obj in text_objects if obj in top_objs_set]


# -------------------- Visualization -------------------- #

def visualize_text_detection(reader, coords: Tuple[int, int, int, int]):
    """Draw detected OCR text over window region."""
    if not coords:
        return

    screenshot = pyautogui.screenshot(region=coords)
    image_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    result = reader.readtext(np.array(screenshot))

    for bbox, text, conf in result:
        if conf > 0.6:
            pts = np.array(bbox, dtype=np.int32)
            cv2.polylines(image_cv, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            x, y = pts[0]
            cv2.putText(image_cv, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    plt.title("Detected Text")
    plt.axis("off")
    plt.show()


def visualize_search_area(coords, search_area):
    """
    Draws a rectangle over a search area in the window for debugging.

    Args:
        coords (tuple): (left, top, width, height)
        search_area (list[float]): [rel_left, rel_top, rel_width, rel_height]
    """
    if not coords:
        print("No window to visualize.")
        return

    left, top, width, height = coords
    rel_left, rel_top, rel_width, rel_height = search_area

    screenshot = pyautogui.screenshot(region=coords)
    image_cv = np.array(screenshot)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    x = int(rel_left * width)
    y = int(rel_top * height)
    w = int(rel_width * width)
    h = int(rel_height * height)

    cv2.rectangle(image_cv, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(image_cv, "Search Area", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    plt.title("Relative Search Area")
    plt.axis("off")
    plt.show()