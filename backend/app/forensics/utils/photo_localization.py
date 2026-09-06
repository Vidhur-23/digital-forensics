"""
Photo region localization utility for passport and ID documents.
"""

import cv2
import numpy as np


def locate_photo_region(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Locates the portrait photo bounding box in a passport image.
    Uses Canny edge detection, dilation, and external contour filtering by area & aspect ratio.

    Parameters:
        image (np.ndarray): Input BGR or grayscale document image.

    Returns:
        tuple (x, y, w, h) or None if no candidate matches.
    """
    if image is None or image.size == 0:
        return None

    img_h, img_w = image.shape[:2]
    img_area = img_h * img_w
    if img_area == 0:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_bbox = None
    max_area_ratio = 0.0

    for cnt in contours:
        contour_area = cv2.contourArea(cnt)
        area_ratio = contour_area / img_area
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h) if h > 0 else 0.0

        if 0.03 <= area_ratio <= 0.15 and 0.5 <= aspect_ratio <= 1.0:
            if area_ratio > max_area_ratio:
                max_area_ratio = area_ratio
                best_bbox = (x, y, w, h)

    return best_bbox
