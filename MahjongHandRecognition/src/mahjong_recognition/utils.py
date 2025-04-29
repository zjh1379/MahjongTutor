"""
Utility functions for mahjong hand recognition
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Union


def preprocess_image(image_path: str, target_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    """
    Preprocess image for model input
    
    Args:
        image_path: Path to the image
        target_size: Target size for the model input
        
    Returns:
        Preprocessed image as numpy array
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image from {image_path}")
    
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to target size
    img = cv2.resize(img, target_size)
    
    # Normalize pixel values to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    return img


def segment_tiles(image: np.ndarray) -> List[np.ndarray]:
    """
    Segment individual tiles from an image containing multiple tiles
    
    Args:
        image: Input image containing multiple tiles
        
    Returns:
        List of segmented tile images
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply binary thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    min_area = 500
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # Sort contours from left to right
    valid_contours = sorted(valid_contours, key=lambda cnt: cv2.boundingRect(cnt)[0])
    
    # Extract each tile
    tiles = []
    for cnt in valid_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        tile = image[y:y+h, x:x+w]
        # Resize tile to target size
        tile = cv2.resize(tile, (64, 64))
        tiles.append(tile)
    
    return tiles


def is_mahjong_image(image: np.ndarray) -> bool:
    """
    Check if an image contains mahjong tiles
    
    Args:
        image: Input image
        
    Returns:
        True if the image likely contains mahjong tiles, False otherwise
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply binary thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    min_area = 500
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # Check if there are enough rectangle-like contours
    rectangle_contours = 0
    for cnt in valid_contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
        if len(approx) == 4:  # Rectangle has 4 vertices
            rectangle_contours += 1
    
    # If there are at least 3 rectangle-like contours, it's likely a mahjong image
    return rectangle_contours >= 3


def format_result(tile_labels: List[str]) -> str:
    """
    Format recognition results in standard notation
    
    Args:
        tile_labels: List of recognized tile labels
        
    Returns:
        Formatted string in standard notation
    """
    # Group tiles by suit
    suits = {'m': [], 'p': [], 's': [], 'z': []}
    
    for label in tile_labels:
        if '万' in label:
            number = label.replace('万', '')
            suits['m'].append(number)
        elif '筒' in label:
            number = label.replace('筒', '')
            suits['p'].append(number)
        elif '索' in label:
            number = label.replace('索', '')
            suits['s'].append(number)
        else:
            # Handle honor tiles
            suits['z'].append(label)
    
    # Sort each suit
    for suit in ['m', 'p', 's']:
        # Convert to integers for sorting
        suits[suit] = sorted([int(n) for n in suits[suit]])
        # Convert back to strings
        suits[suit] = [str(n) for n in suits[suit]]
    
    # Honor tiles have special ordering: 东南西北白发中
    honor_order = {'东': 1, '南': 2, '西': 3, '北': 4, '白': 5, '发': 6, '中': 7}
    suits['z'] = sorted(suits['z'], key=lambda x: honor_order.get(x, 8))
    
    # Build the output string
    result = []
    
    for suit, label in [('m', 'm'), ('p', 'p'), ('s', 's')]:
        if suits[suit]:
            result.append(''.join(suits[suit]) + label)
    
    if suits['z']:
        result.append(''.join(suits['z']))
    
    return ' '.join(result) 