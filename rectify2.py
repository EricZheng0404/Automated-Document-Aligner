import cv2
import numpy as np
import os
import sys

# --- Fixed System Parameters (Required for Automation) ---
# Required output size: 550 x 425 pixels (50 DPI for 11x8.5")
OUTPUT_WIDTH = 550
OUTPUT_HEIGHT = 425

# Preprocessing parameters (Adjusted for better noise suppression and background separation)
GAUSSIAN_BLUR_KERNEL = (7, 7) # Increased blur kernel size for stronger noise reduction
# Adaptive Thresholding parameters: chosen to be robust against varied lighting
ADAPTIVE_THRESH_BLOCK_SIZE = 21 # Increased block size for local mean calculation
ADAPTIVE_THRESH_C = 8           # Increased constant subtracted from the mean

# Contour Approximation parameter (Ramer-Douglas-Peucker epsilon ratio)
# This ratio determines how aggressively the contour is simplified. 0.02 is a common starting point.
APPROX_POLY_EPSILON_RATIO = 0.02

def order_points(pts):
    """
    Ensures the four corner points are ordered consistently:
    Top-Left, Top-Right, Bottom-Right, Bottom-Left.
    """
    # Initialize a list of coordinates that will be ordered
    # (top-left, top-right, bottom-right, bottom-left)
    rect = np.zeros((4, 2), dtype="float32")

    # The sum of the coordinates will yield the top-left (smallest sum)
    # and the bottom-right (largest sum)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-Left (TL)
    rect[2] = pts[np.argmax(s)]  # Bottom-Right (BR)

    # The difference between the coordinates (x - y) will yield the
    # top-right (smallest difference) and the bottom-left (largest difference)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Top-Right (TR)
    rect[3] = pts[np.argmax(diff)] # Bottom-Left (BL)

    # Return the ordered coordinates
    return rect

def rectify_document(image_path):
    """
    Processes a single image to detect the document and perform a perspective warp.
    """
    print(f"Processing {os.path.basename(image_path)}...")
    
    # 1. Load the input image
    original = cv2.imread(image_path)
    if original is None:
        print(f"ERROR: Could not load image at {image_path}")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur for noise reduction while preserving edges
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL, 0)

    # 1. Preprocessing and Binarization
    # Use Adaptive Gaussian Thresholding to segment the document
    # cv2.THRESH_BINARY_INV is used to ensure the document (lighter part) is white (255)
    # in the binary image, which helps with contour finding.
    thresh = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        ADAPTIVE_THRESH_BLOCK_SIZE, 
        ADAPTIVE_THRESH_C
    )

    # 2. Feature and Contour Extraction
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("WARNING: No contours found.")
        return None
        
    # Sort the contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # 3. Corner Detection / Localization (Robust Search)
    screen_contour = None
    
    # Iterate over the top 10 largest contours to find the one that is a quadrilateral
    for c in contours[:10]: 
        # Ramer-Douglas-Peucker algorithm for curve approximation
        perimeter = cv2.arcLength(c, True)
        # Calculate epsilon based on a percentage of the perimeter
        epsilon = APPROX_POLY_EPSILON_RATIO * perimeter
        approx = cv2.approxPolyDP(c, epsilon, True)

        # We look for a quadrilateral (4 corners)
        if len(approx) == 4:
            screen_contour = approx
            break

    if screen_contour is None:
        print("WARNING: Could not find a quadrilateral (4-corner) document contour. Skipping.")
        return None

    # Reshape and re-order the corner points
    source_pts = screen_contour.reshape(4, 2)
    source_pts = order_points(source_pts)

    # 4. Geometric Rectification
    # Define the destination points for the 550x425 output image
    dst_pts = np.array([
        [0, 0],                              # Top-Left
        [OUTPUT_WIDTH - 1, 0],               # Top-Right
        [OUTPUT_WIDTH - 1, OUTPUT_HEIGHT - 1], # Bottom-Right
        [0, OUTPUT_HEIGHT - 1]               # Bottom-Left
    ], dtype="float32")

    # Compute the perspective transform matrix (Homography H)
    H = cv2.getPerspectiveTransform(source_pts, dst_pts)

    # Apply the perspective transform to the original color image
    rectified = cv2.warpPerspective(original, H, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    
    print("Rectification successful.")
    return rectified

def main():
    # Check for the required command-line argument (input folder path)
    if len(sys.argv) < 2:
        print("Usage: python3 rectify.py <path_to_input_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = "output" # Fixed output folder name

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Process all image files in the input folder
    for filename in os.listdir(input_folder):
        # Only process known image file extensions
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            
            # The output filename must be outputN.jpg for inputN.jpg/.png
            # We strip the original extension and add the new one
            name_part = os.path.splitext(filename)[0]
            # Use replace with count=1 to only replace the first 'input' found in the filename
            output_filename = name_part.replace("input", "output", 1) + ".jpg" 
            output_path = os.path.join(output_folder, output_filename)
            
            # Process the image
            result_image = rectify_document(input_path)
            
            # Save the result
            if result_image is not None:
                cv2.imwrite(output_path, result_image)
                print(f"Saved rectified image to {output_path}")
            else:
                print(f"Failed to process {filename}. No output saved.")
            
            print("-" * 20)

if __name__ == "__main__":
    # Ensure OpenCV and NumPy are available
    try:
        main()
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print("Please ensure you have OpenCV and NumPy installed (e.g., pip install opencv-python numpy).")
