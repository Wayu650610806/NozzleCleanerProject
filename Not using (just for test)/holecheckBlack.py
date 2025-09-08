import cv2
import numpy as np
import os
import sys

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

def _hough_from_filtered_contours(thresh_image, filtered_contours):
    """
    Creates a mask from filtered contours and finds circles using Hough Circle Transform.
    
    Args:
        thresh_image (np.ndarray): The thresholded image (used for dimensions).
        filtered_contours (list): A list of filtered contours.
        
    Returns:
        np.ndarray: A NumPy array of circles found, or None if no circles are found.
    """
    # Create a blank black image (mask) to draw the filtered contours on
    mask = np.zeros(thresh_image.shape, dtype=np.uint8)
    
    # Draw the filtered contours onto the blank mask
    cv2.drawContours(mask, filtered_contours, -1, 255, -1)
    
    # Use the mask as the input for Hough Circles, which is much faster
    circles = cv2.HoughCircles(
        mask,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=200,
        param1=10,
        param2=10,
        minRadius=5,
        maxRadius=15
    )
    return circles, mask

def _filter_contours(roi_image):
    """
    Finds and filters contours based on size, aspect ratio, and circularity.
    
    Args:
        roi_image (np.ndarray): The region of interest (ROI) image of a single hole.
    
    Returns:
        tuple: A tuple containing:
               - output_image (np.ndarray): The image with filtered contours drawn.
               - filtered_contours (list): A list of filtered contours.
               - thresh (np.ndarray): The thresholded image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    
    # Apply Adaptive Thresholding with improved parameters
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 39, 0)
    
    # Find contours using RETR_CCOMP to handle inner and outer contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area, aspect ratio, and circularity
    min_area = 250  # Adjust as needed
    max_area = 450 # Adjusted for 100x100 size
    filtered_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
            
        # Filter by aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if aspect_ratio < 0.7 or aspect_ratio > 1.4: # A typical range for circle-like objects
            continue
        
        # Filter by circularity
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < 0.5: # Adjust this value, 1.0 is a perfect circle
            continue
            
        filtered_contours.append(contour)
    
    # For visualization, let's draw the filtered contours on the original image
    output_image = roi_image.copy()
    cv2.drawContours(output_image, filtered_contours, -1, (0, 255, 0), 2)
    
    return output_image, filtered_contours, thresh

def isBlockedHole(roi_image):
    """
    Checks if a hole is blocked using contour filtering.
    
    Args:
        roi_image (np.ndarray): The region of interest (ROI) image of a single hole.
    
    Returns:
        bool: False for now, to be implemented with further logic.
        (True will indicate a blocked hole, False an unblocked hole)
    """
    # Preprocess the image by resizing it
    preprocessed_image = _preprocess_roi(roi_image)
    
    # Use the new helper function to get the contours and visualization image
    processed_image, filtered_contours, _ = _filter_contours(preprocessed_image)
    
    # This part will be updated with Hough Circle and Texture Analysis later
    # For now, we return False
    return False

def main():
    """
    Main function to load images from a folder and test the isBlockedHole function.
    Press 'a' or 'd' to cycle through images, 'q' to quit.
    """
    folder_path = r'C:\Project\nozzleScan\NozzleCleanerProject\roi\3'
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

        # Resize the image for consistent processing
        resized_roi = _preprocess_roi(roi)

        # Get the processed image and contour count
        processed_image, filtered_contours, thresh_image = _filter_contours(resized_roi)
        
        # Find circles using Hough Circle on a mask of the filtered contours
        circles, hough_mask = _hough_from_filtered_contours(thresh_image, filtered_contours)
        hough_image = resized_roi.copy()
        
        # Draw circles if any are found
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Draw the outer circle
                cv2.circle(hough_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Draw the center of the circle
                cv2.circle(hough_image, (i[0], i[1]), 2, (0, 0, 255), 3)

        # Display the result
        display_text = f"{current_index + 1} of {len(image_files)}"
        cv2.putText(processed_image, display_text, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        # Move the windows to prevent them from overlapping
        cv2.imshow("Contour Filtering", processed_image)
        cv2.moveWindow("Contour Filtering", 200, 200)
        cv2.imshow("Threshold Image", thresh_image)
        cv2.moveWindow("Threshold Image", 400, 200)
        cv2.imshow("Hough Circles", hough_image)
        cv2.moveWindow("Hough Circles", 600, 200)
        cv2.imshow("Hough Mask", hough_mask)
        cv2.moveWindow("Hough Mask", 800, 200)
        cv2.imshow("Image", roi)
        cv2.moveWindow("Image", 1000, 200)

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
