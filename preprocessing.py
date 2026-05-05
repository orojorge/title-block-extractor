"""
Pure functions for PDF-to-image conversion, resizing, and cropping.
"""

import logging
from pathlib import Path
import pymupdf
from PIL import Image

logger = logging.getLogger("preprocessing")

DPI = 150
CROP_SIZE = 560
THUMB_SIZE = 912


def pdf_to_image(pdf_path: str, dpi: int = DPI) -> Image.Image:
    """Render the first page of a PDF to a PIL image at the requested DPI."""
    logger.debug(f"pdf_to_image() called — path: {pdf_path}")
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    doc = pymupdf.open(pdf_path)
    if doc.page_count == 0:
        raise ValueError(f"PDF has no pages: {pdf_path}")
    scale = dpi / 72
    pix = doc[0].get_pixmap(matrix=pymupdf.Matrix(scale, scale))
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    logger.debug(f"pdf_to_image() done — output size: {img.size}")
    return img


def resize_shortest_side(image: Image.Image, size: int) -> Image.Image:
    """Resize an image so its shortest side matches `size`, preserving aspect ratio."""
    logger.debug(f"resize_shortest_side() called — input size: {image.size}, target: {size}")
    w, h = image.size
    if w <= h:
        new_w, new_h = size, round(h * size / w)
    else:
        new_w, new_h = round(w * size / h), size
    result = image.resize((new_w, new_h), Image.LANCZOS)
    logger.debug(f"resize_shortest_side() done — output size: {result.size}")
    return result


def crop_bottom_right(image: Image.Image) -> Image.Image:
    """Return the bottom-right quadrant as a fast title-block heuristic crop."""
    logger.debug(f"crop_bottom_right() called — input size: {image.size}")
    w, h = image.size
    result = image.crop((w // 2, h // 2, w, h))
    logger.debug(f"crop_bottom_right() done — output size: {result.size}")
    return result


def prepare_inputs(pdf_path: str, crop_size: int = CROP_SIZE, thumb_size: int = THUMB_SIZE) -> tuple[Image.Image, Image.Image]:
    """Prepare both model inputs: a focused crop and a resized full-page thumbnail."""
    logger.debug(f"prepare_inputs() called — path: {pdf_path}, crop_size: {crop_size}, thumb_size: {thumb_size}")
    full = pdf_to_image(pdf_path)
    thumbnail = resize_shortest_side(full, thumb_size)
    crop = resize_shortest_side(crop_bottom_right(full), crop_size)
    logger.debug(f"prepare_inputs() done — crop: {crop.size}, thumbnail: {thumbnail.size}")
    return crop, thumbnail
