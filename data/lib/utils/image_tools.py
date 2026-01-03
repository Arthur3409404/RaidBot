# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 13:46:00 2025

@author: Arthur
"""

import cv2
import numpy as np
import pyautogui
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import re


class TextObject:
    def __init__(self, text, mean_pos_x, mean_pos_y):
        self.text = text
        self.mean_pos_x = mean_pos_x
        self.mean_pos_y = mean_pos_y
        

def detect_red_or_green_circle_stable(
    region_coords,
    samples=40,
    required_ratio=0.75,
    min_pixels=8,
    green_hex="3CB043",   # typical UI green
    red_hex="C0392B",     # typical UI red
    tolerance=45
):
    """
    Uses rapid sampling + color distance consistency to detect
    small red/green circles in noisy UI regions.

    region_coords: (x, y, w, h)
    Returns: "red", "green", or None
    """

    def hex_to_bgr(hex_color):
        hex_color = hex_color.lstrip("#")
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return np.array([b, g, r], dtype=np.int16)

    green_bgr = hex_to_bgr(green_hex)
    red_bgr = hex_to_bgr(red_hex)

    green_hits = 0
    red_hits = 0

    for _ in range(samples):
        screenshot = pyautogui.screenshot(region=region_coords)
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR).astype(np.int16)

        # Color distance masks
        green_diff = np.linalg.norm(frame - green_bgr, axis=2)
        red_diff = np.linalg.norm(frame - red_bgr, axis=2)

        green_pixels = np.count_nonzero(green_diff <= tolerance)
        red_pixels = np.count_nonzero(red_diff <= tolerance)

        if green_pixels >= min_pixels:
            green_hits += 1
        if red_pixels >= min_pixels:
            red_hits += 1

    threshold = int(samples * required_ratio)

    if red_hits >= threshold:
        return "red"
    if green_hits >= threshold:
        return "green"

    return None


def check_area_color(window, search_area):
    """
    Recognize the general color of a relative screen area inside a window.
    
    Returns one of: "grey", "yellow", "green", "red", "black"
    
    search_area: (rel_left, rel_top, rel_width, rel_height), values between 0 and 1
    window: object with .left, .top, .width, .height
    """
    # Capture the area as numpy array
    rel_left, rel_top, rel_width, rel_height = search_area
    abs_left = window.left + int(rel_left * window.width)
    abs_top = window.top + int(rel_top * window.height)
    abs_width = int(rel_width * window.width)
    abs_height = int(rel_height * window.height)

    screenshot = pyautogui.screenshot(region=(abs_left, abs_top, abs_width, abs_height))
    img_np = np.array(screenshot)

    # Compute average color
    avg_color = img_np.mean(axis=(0, 1))  # RGB
    r, g, b = avg_color
    
    # Simple thresholds for main colors
    # Thresholds for main colors
    if 40 < r < 60 and 70< g < 100 and 90 < b < 110:
        return "grey"
    if 150 < r < 170 and 110< g < 140 and 0 < b < 30:
        return "yellow"

    return "grey"  # fallback



def compare_pngs(path1, path2):
    """
    Compares two PNG images and returns their SSIM (similarity) score.

    Args:
        path1 (str): Path to the first image.
        path2 (str): Path to the second image.

    Returns:
        float | None: SSIM score between 0 and 1, or None if images differ in size.
    """
    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        print("Error: One or both image paths are invalid.")
        return None

    if img1.shape != img2.shape:
        print("Images have different sizes — cannot compare.")
        return None

    score, _ = ssim(img1, img2, full=True)
    return score


def get_text_from_image(reader, image):
    """
    Uses EasyOCR to extract text from an image.

    Args:
        reader: easyocr.Reader instance.
        image: PIL image or NumPy array.

    Returns:
        str: Concatenated detected text with confidence > 0.6.
    """
    result = reader.readtext(np.array(image))
    return "\n".join([text for (_, text, conf) in result if conf > 0.6])


def get_text_in_full_area(reader, coords):
    """
    Captures text from the entire window area.

    Args:
        reader: easyocr.Reader instance.
        coords (tuple): (left, top, width, height) of the window.

    Returns:
        str | None: Extracted text or None if coords missing.
    """
    if not coords:
        print("No window coordinates found.")
        return None

    screenshot = pyautogui.screenshot(region=coords)
    return get_text_from_image(reader, screenshot)


def get_text_in_relative_area(reader, window, search_area, powerdetection=False, factiondetection = False):
    """
    Takes a screenshot of a relative area within the game window and extracts text using OCR.

    Args:
        reader: easyocr.Reader instance.
        coords (tuple): (left, top, width, height) of the window.
        search_area (list[float]): [rel_left, rel_top, rel_width, rel_height].
        powerdetection (bool): If True, limit OCR allowlist to digits/letters for power levels.

    Returns:
        list[dict]: Each entry is {"text": str, "mean_pos_x": float, "mean_pos_y": float}.
    """
    if not window:
        #print("No window coordinates found.")
        return []

    rel_left, rel_top, rel_width, rel_height = search_area

    # Convert relative to absolute window coordinates
    abs_left = window.left + int(rel_left * window.width)
    abs_top = window.top + int(rel_top * window.height)
    abs_width = int(rel_width * window.width)
    abs_height = int(rel_height * window.height)

    screenshot = pyautogui.screenshot(region=(abs_left, abs_top, abs_width, abs_height))
    image_np = np.array(screenshot)

    if powerdetection:
        results = reader.readtext(image_np, allowlist='0123456789.,KkMmLUCHARluchar ')

    if factiondetection:
        results = reader.readtext(image_np, allowlist='36')
    else:
        results = reader.readtext(image_np)

    text_objects = []
    for bbox, text, confidence in results:
        if confidence < 0.5:
            continue
        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]

        mean_x = sum(xs) / 4
        mean_y = sum(ys) / 4

        abs_x = abs_left + mean_x
        abs_y = abs_top + mean_y

        text_obj =TextObject(text=text, mean_pos_x=abs_x, mean_pos_y=abs_y)
        text_objects.append(text_obj)

    return text_objects

def get_text_from_cluster_area(reader, window, search_areas, powerdetection = False):
    if not window:
        return []

    all_text_objects = []

    for pos in search_areas:
        for search_area in search_areas[pos]:

            rel_left, rel_top, rel_width, rel_height = search_area

            # Convert relative to absolute window coordinates
            abs_left = window.left + int(rel_left * window.width)
            abs_top = window.top + int(rel_top * window.height)
            abs_width = int(rel_width * window.width)
            abs_height = int(rel_height * window.height)

            screenshot = pyautogui.screenshot(
                region=(abs_left, abs_top, abs_width, abs_height)
            )
            image_np = np.array(screenshot)

            if powerdetection:
                results = reader.readtext(
                    image_np,
                    allowlist="0123456789.,KkMmLUCHARluchar "
                )
            else:
                results = reader.readtext(image_np)

            for bbox, text, confidence in results:
                if confidence < 0.5:
                    continue

                # Filter: must contain a number OR be 'luchar' (any case)
                has_number = bool(re.search(r"\d", text))
                is_luchar = text.strip().lower() == "luchar"

                if not has_number and not is_luchar:
                    continue

                xs = [point[0] for point in bbox]
                ys = [point[1] for point in bbox]

                mean_x = sum(xs) / 4
                mean_y = sum(ys) / 4

                abs_x = abs_left + mean_x
                abs_y = abs_top + mean_y

                text_obj = TextObject(
                    text=text,
                    mean_pos_x=abs_x,
                    mean_pos_y=abs_y
                )

                all_text_objects.append(text_obj)
                
    return all_text_objects


def filter_text_objects(text_objects):
    def has_number(text):
        return bool(re.search(r"\d", text))

    def is_luchar(text):
        return text.strip().lower() == "luchar"

    # 1. Remove entries with < 2 characters
    text_objects = [
        obj for obj in text_objects if len(obj.text.strip()) >= 2
    ]

    cleaned = []
    i = 0

    while i < len(text_objects):
        current = text_objects[i]
        curr_text = current.text.strip()
        next_obj = text_objects[i + 1] if i + 1 < len(text_objects) else None

        # Case 1: luchar followed by luchar → drop second
        if next_obj and is_luchar(curr_text) and is_luchar(next_obj.text):
            cleaned.append(current)
            i += 2
            continue

        # Case 2 (NEW):
        # number → keep ONLY if next is luchar
        if has_number(curr_text):
            if next_obj and is_luchar(next_obj.text):
                cleaned.append(current)
            # skip number regardless
            i += 1
            continue

        # Keep luchar (handled later by alternation enforcement)
        cleaned.append(current)
        i += 1

    # 2. Enforce alternating pattern: number → luchar → number → luchar
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

def visualize_text_detection(reader, coords):
    """
    Visualizes OCR text detection on the full window region.

    Args:
        reader: easyocr.Reader instance.
        coords (tuple): (left, top, width, height) of the window.
    """
    if not coords:
        print("No window to capture.")
        return

    screenshot = pyautogui.screenshot(region=coords)
    image_cv = np.array(screenshot)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    result = reader.readtext(image_cv)

    for (bbox, text, conf) in result:
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
    
    
def find_clusters( text_objects, num_majorities=2, tolerance=8):
    """
    Groups text_objects by proximity of mean_pos_x within tolerance,
    finds top `num_majorities` clusters with most objects,
    returns a cleaned list of text_objects in original order
    including only those belonging to the top clusters.
    """

    # Step 0: Filter out objects with fewer than 3 digits
    text_objects = [obj for obj in text_objects if len(obj.text) >= 3]
    
    if not text_objects:
        return []  # Early exit if no objects remain

    # Step 1: Sort objects by mean_pos_x to form clusters
    sorted_objs = sorted(text_objects, key=lambda o: o.mean_pos_x)

    clusters = []
    current_cluster = []
    current_x = None

    for obj in sorted_objs:
        if current_x is None:
            current_x = obj.mean_pos_x
            current_cluster = [obj]
        else:
            if abs(obj.mean_pos_x - current_x) <= tolerance:
                current_cluster.append(obj)
                # Update cluster center to mean of members' mean_pos_x
                current_x = sum(o.mean_pos_x for o in current_cluster) / len(current_cluster)
            else:
                clusters.append(current_cluster)
                current_cluster = [obj]
                current_x = obj.mean_pos_x
    if current_cluster:
        clusters.append(current_cluster)

    # Step 2: Sort clusters by size descending, take top N
    clusters.sort(key=len, reverse=True)
    top_clusters = clusters[:num_majorities]

    # Flatten the top clusters into a set for quick membership checking
    top_objs_set = set()
    for cluster in top_clusters:
        for obj in cluster:
            top_objs_set.add(obj)

    # Step 3: Filter original list preserving order
    cleaned_list = [obj for obj in text_objects if obj in top_objs_set]

    # Optional: print cluster info
    #print()
    #print(cleaned_list)
    
    return cleaned_list    
    