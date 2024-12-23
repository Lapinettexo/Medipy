from PIL import Image
import numpy as np
from utility import check_image_exists, writeFrequencyIntoJSON
import os
from scipy.stats import entropy
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr

class DataCollector:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.images = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]

    def imageSplitter(self, image, mode="4"):
        width, height = image.size
        parts = []
        
        if mode == "4":
            mid_x, mid_y = width // 2, height // 2
            parts = [
                image.crop((0, 0, mid_x, mid_y)),          # Top-left
                image.crop((mid_x, 0, width, mid_y)),      # Top-right
                image.crop((0, mid_y, mid_x, height)),     # Bottom-left
                image.crop((mid_x, mid_y, width, height))  # Bottom-right
            ]
        elif mode == "2h":
            mid_y = height // 2
            parts = [
                image.crop((0, 0, width, mid_y)),          # Top half
                image.crop((0, mid_y, width, height))      # Bottom half
            ]
        elif mode == "2v":
            mid_x = width // 2
            parts = [
                image.crop((0, 0, mid_x, height)),         # Left half
                image.crop((mid_x, 0, width, height))      # Right half
            ]
        elif mode == "1":
            parts = [image]  # Return the whole image as a single part
        else:
            print(f"Unknown split mode '{mode}'")
            
        return parts
    
    def dynamicImageSplitter(self, image, num_rows=2, num_cols=2):
        width, height = image.size
        parts = []
        
        # Calculate the width and height of each part
        part_width = width // num_cols
        part_height = height // num_rows

        # Generate the sub-images by iterating through the grid
        for row in range(num_rows):
            for col in range(num_cols):
                left = col * part_width
                upper = row * part_height
                right = left + part_width
                lower = upper + part_height
                parts.append(image.crop((left, upper, right, lower)))

        return parts
    
    def calculate_pixel_frequency(self, part):
        img_array = np.array(part)
        pixels = img_array.flatten()
        frequency = np.bincount(pixels, minlength=256)
        return frequency.tolist()
    
    
    def get_frequency_for_all_images(self):
        if not self.images:
            print(f"No images found in '{self.folder_path}'")
            return

        all_image_frequencies = {}
        print("Start processing images")
        
        for image_file in self.images:
            image_path = os.path.join(self.folder_path, image_file)

            if not check_image_exists(image_path):
                print(f"Error: The image at '{image_path}' does not exist.")
                continue  

            image = Image.open(image_path).convert('L')
            #parts = self.imageSplitter(image, mode=split_mode)
            parts = self.dynamicImageSplitter(image, 1, 16)

            image_frequencies = {}

            for i, part in enumerate(parts):
                pixel_frequencies = self.calculate_pixel_frequency(part)
                image_frequencies[f"part_{i+1}"] = pixel_frequencies

            all_image_frequencies[image_file] = image_frequencies

        print("End processing")

        save_folder = "C://Users//Trust_pc_dz//Documents//IMED//DATASET//Frequencies//16 Data"
        writeFrequencyIntoJSON(save_folder, all_image_frequencies, "Brain_Tumor_16")
        print("Data written in JSON file.")
