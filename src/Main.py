from imageStats import DataCollector
from utility import check_image_exists, writeFrequencyIntoJSON
import os
import json

def main():
    
    dataset_folder = "C://Users//Trust_pc_dz//Documents//IMED//DATASET"

    images = [f for f in os.listdir(dataset_folder) if f.endswith('.jpg') or f.endswith('.png')]

    if not images:
        print(f"No images found in '{dataset_folder}'")
        return

    all_image_frequencies = {}

    for image_file in images:
        image_path = os.path.join(dataset_folder, image_file)

        if not check_image_exists(image_path):
            print(f"Error: The image at '{image_path}' does not exist.")
            continue  # Skip to the next image
        
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

    writeFrequencyIntoJSON(dataset_folder, all_image_frequencies)

if __name__ == "__main__":
    main()
