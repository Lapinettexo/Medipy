import os
import cv2
import numpy as np
from skimage.measure import shannon_entropy

class ImageEnhancer:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.quality_threshold = 100  # Ajustez cette valeur selon les tests
        self.entropy_threshold = 6.0    # Threshold for detecting low quality


    def remove_background(self):
        for filename in os.listdir(self.input_folder):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                image_path = os.path.join(self.input_folder, filename)
                
                image_noir = self._noir_background(image_path)
                
                if image_noir is not None:
                    cv2.imwrite(image_path, image_noir)
                    print(f"Background removed and saved in place: {image_path}")
                else:
                    print(f"No contours found in {filename}, skipping.")

    def crop_to_content(self):
        for filename in os.listdir(self.input_folder):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                image_path = os.path.join(self.input_folder, filename)
                
                cropped_image = self._crop_to_content(image_path)
                
                if cropped_image is not None:
                    cv2.imwrite(image_path, cropped_image)
                    print(f"Image cropped to content and saved in place: {image_path}")
                else:
                    print(f"No contours found in {filename}, skipping.")

    def _noir_background(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Could not read {image_path}. Skipping.")
            return None

        _, thresh = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)
        
        # Find contours to detect the main region of interest
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Create a mask for the largest contour
            mask = np.zeros_like(image)
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=1)

            # Apply the mask to keep only the main region and remove noise
            image_noir = cv2.bitwise_and(image, mask)
            return image_noir
        else:
            return image  

    def _crop_to_content(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Could not read {image_path}. Skipping.")
            return None

        edges = cv2.Canny(image, threshold1=30, threshold2=100)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Combine all contours for a global bounding box
            all_contours = np.vstack(contours)
            x, y, w, h = cv2.boundingRect(all_contours)
            
            # Crop the image based on the bounding box
            cropped_image = image[y:y+h, x:x+w]
            return cropped_image
        else:
            return None


    def remove_low_quality_images(self):
        """Remove images that appear low-quality based on sharpness."""
        for filename in os.listdir(self.input_folder):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                image_path = os.path.join(self.input_folder, filename)
                
                # Check if the image meets quality standards
                if self._is_low_quality(image_path):
                    os.remove(image_path)
                    print(f"Removed low-quality image: {image_path}")
                    print("--------------------")
                else:
                    print(f"Image {filename} meets quality standards.")
                    print("--------------------")

    def _is_low_quality(self, image_path):
        """Determine if an image is low quality based on its sharpness."""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Could not read {image_path}. Skipping.")
            return False

        # Calculate the Laplacian variance
        #laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        #print(f"Image {image_path} has Laplacian variance: {laplacian_var}")

        entropy = shannon_entropy(image)
        print(f"Image {image_path} has entropy: {entropy}")
        
        # Return True if variance is below the threshold (indicating low quality)
        return entropy > self.entropy_threshold
