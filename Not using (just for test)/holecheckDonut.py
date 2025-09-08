import cv2
import numpy as np
import os
import sys
import math
from dataclasses import dataclass
from typing import Tuple, Dict

def _preprocess_roi(roi_image, size=(100, 100)):
    """
    Resizes the ROI image to a consistent size.
    
    Args:
        roi_image (np.ndarray): The region of interest (ROI) image.
        size (tuple): The target size for the image (width, height).
        
    Returns:
        np.ndarray: The resized ROI image.
    """
    if roi_image is None or roi_image.size == 0:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    
    return cv2.resize(roi_image, size, interpolation=cv2.INTER_AREA)

def _refine_roi_center(roi_bgr: np.ndarray, max_shift: int = 8):
    """
    Refine the center inside a ROI by searching for a point that is 'dark + has sharp edges'.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return (0, 0)

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cx, cy = w // 2, h // 2 

    best_score = -1e9
    best_xy = (cx, cy)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)

    patch_r = 15
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            x = np.clip(cx + dx, 0, w - 1)
            y = np.clip(cy + dy, 0, h - 1)
            y1, y2 = max(0, y - patch_r), min(h, y + patch_r)
            x1, x2 = max(0, x - patch_r), min(w, x + patch_r)

            patch = gray[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            dark = 255.0 - float(np.mean(patch))
            edge = float(np.mean(mag[y1:y2, x1:x2]))
            score = dark + 0.5 * edge

            if score > best_score:
                best_score = score
                best_xy = (int(x), int(y))
    return best_xy

def _make_masks(H: int, W: int, cx: int, cy: int, r_in: int):
    """
    Creates a circular mask for a given center and radius.
    
    Args:
        H (int): Height of the image.
        W (int): Width of the image.
        cx (int): Center x-coordinate.
        cy (int): Center y-coordinate.
        r_in (int): Inner radius of the circle.
        
    Returns:
        np.ndarray: The circular mask.
    """
    mask_in = np.zeros((H, W), np.uint8)
    cv2.circle(mask_in, (cx, cy), r_in, 255, -1)
    return mask_in

def isBlockedHole(roi_image):
    """
    Checks if a hole is blocked using a combination of darkness and texture analysis.
    
    Args:
        roi_image (np.ndarray): The region of interest (ROI) image of a single hole.
        
    Returns:
        tuple: A tuple containing:
               - bool: True if the hole is blocked, False otherwise.
               - output_image (np.ndarray): The image with status drawn.
    """
    preprocessed_image = _preprocess_roi(roi_image)
    
    # Use the new function to find the refined center
    cx, cy = _refine_roi_center(preprocessed_image)

    # Use a hardcoded radius for the analysis mask for now
    r_in = 10
    
    # Create a circular mask at the refined center
    mask = _make_masks(preprocessed_image.shape[0], preprocessed_image.shape[1], cx, cy, r_in)
    
    gray_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2GRAY)
    
    # Analyze Darkness
    mean_intensity = cv2.mean(gray_image, mask=mask)[0]
    
    # Analyze Texture (flatness) using variance of Laplacian
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    texture_variance = np.var(laplacian, where=(mask > 0))
    
    # Define thresholds
    darkness_threshold = 60 # Lower value means darker
    texture_threshold = 100 # Higher value means more texture

    is_blocked = (mean_intensity > darkness_threshold) or (texture_variance < texture_threshold)
    
    # Draw status on the image
    result_image = preprocessed_image.copy()
    status_text = "Blocked" if is_blocked else "Not Blocked"
    color = (0, 0, 255) if is_blocked else (0, 255, 0)
    
    # Draw a circle on the image to show the analyzed area
    cv2.circle(result_image, (cx, cy), r_in, color, 2)
    cv2.putText(result_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    
    return is_blocked, result_image

def main():
    """
    Main function to load images from a folder and test the isBlockedHole function.
    Press 'a' or 'd' to cycle through images, 'q' to quit.
    """
    folder_path = r'C:\Project\nozzleScan\NozzleCleanerProject\roi\1'
    if not os.path.isdir(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        sys.exit()

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"Error: No image files found in '{folder_path}'.")
        sys.exit()

    current_index = 0
    while True:
        file_path = os.path.join(folder_path, image_files[current_index])
        
        # Read the image
        roi = cv2.imread(file_path)
        if roi is None:
            print(f"Warning: Failed to read image '{file_path}'. Skipping.")
            current_index = (current_index + 1) % len(image_files)
            continue

        # Get the result from the isBlockedHole function
        is_blocked, result_image = isBlockedHole(roi)

        # Display the result
        display_text = f"{current_index + 1} of {len(image_files)}"
        cv2.putText(result_image, display_text, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        # Move the windows to prevent them from overlapping
        cv2.imshow("Result", result_image)
        cv2.moveWindow("Result", 200, 200)

        # Handle user input
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            current_index = (current_index - 1 + len(image_files)) % len(image_files)
        elif key == ord('d'):
            current_index = (current_index + 1) % len(image_files)
            
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
