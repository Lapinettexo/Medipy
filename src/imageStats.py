from PIL import Image
import numpy as np

class DataCollector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = Image.open(image_path).convert('L')
    
    def imageSplitter(self):
        width, height = self.image.size
        mid_x, mid_y = width // 2, height // 2
        
        parts = [
            self.image.crop((0, 0, mid_x, mid_y)),            # Top-left
            self.image.crop((mid_x, 0, width, mid_y)),        # Top-right
            self.image.crop((0, mid_y, mid_x, height)),       # Bottom-left
            self.image.crop((mid_x, mid_y, width, height))    # Bottom-right
        ]
        return parts
    
    def calculate_pixel_frequency(self, part):
        frequencies = []
        # Convert the image to a numpy array for easy pixel manipulation
        img_array = np.array(part)
        
        # Flatten the array (convert 2D to 1D) to get all pixel values
        pixels = img_array.flatten()
        
        # Calculate the frequency of each pixel value (0-255)
        frequency = np.bincount(pixels, minlength=256)  # Ensure 256 bins for all grayscale values
        #frequencies.append(frequency.tolist()) 
        
        # Return the frequency as a vector (list of 256 values)
        return frequency.tolist()