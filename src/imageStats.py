from PIL import Image
import numpy as np
from utility import check_image_exists, writeFrequencyIntoJSON
import os

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
    
    
    
    def get_frequency_for_all_images(self, dataset_folder):
        images = [f for f in os.listdir(dataset_folder) if f.endswith('.jpg') or f.endswith('.png')]

        if not images:
            print(f"No images found in '{dataset_folder}'")
            return

        all_image_frequencies = {}
        print("start training")
        image_count = 0

        for image_file in images:
            image_path = os.path.join(dataset_folder, image_file)

            if not check_image_exists(image_path):
                print(f"Error: The image at '{image_path}' does not exist.")
                continue  # Skip to the next image
            
            image_count += 1
            data = DataCollector(image_path)
            
            # Split the image into parts
            parts = data.imageSplitter()

            # Dictionary to store pixel frequencies for this image
            image_frequencies = {}

            for i, part in enumerate(parts):
                
                # Calculate pixel frequencies for the part
                pixel_frequencies = data.calculate_pixel_frequency(part)
                
                # Store the frequencies for this part
                image_frequencies[f"part_{i+1}"] = pixel_frequencies

            # Add the frequencies for this image to the main dictionary
            all_image_frequencies[image_file] = image_frequencies

        #print(all_image_frequencies)
        print(f"Total images processed: {image_count}")

        #writeFrequencyIntoJSON(dataset_folder, all_image_frequencies)
        print("end training")
        save_folder = "C://Users//Trust_pc_dz//Documents//IMED//DATASET"
        writeFrequencyIntoJSON(save_folder, all_image_frequencies, "pituitary_tumor_frequencies")
        print("data written in JSON file.")

