import os
import cv2  # OpenCV for image processing
import numpy as np

# Path to the folder with brain scan images
input_folder = "C://Users//Trust_pc_dz//Documents//IMED//test"  # Replace with your folder path
output_folder = "C://Users//Trust_pc_dz//Documents//IMED//CROPPED_DATASET"
# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate over each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".png") or filename.endswith(".jpg"):  # Add other formats if needed
        image_path = os.path.join(input_folder, filename)
        
        # Read the image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply edge detection (Canny)
        edges = cv2.Canny(image, threshold1=30, threshold2=100)
        
        # Find contours from the edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the bounding box of the largest contour, which should correspond to the brain area
        if contours:
            # Combine all contours for a global bounding box
            all_contours = np.vstack(contours)
            x, y, w, h = cv2.boundingRect(all_contours)
            
            # Crop the image to the bounding box
            cropped_image = image[y:y+h, x:x+w]
            
            # Save the cropped image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, cropped_image)
            print(f"Cropped image saved at: {output_path}")
        else:
            print(f"No contours found in {filename}, skipping.")
