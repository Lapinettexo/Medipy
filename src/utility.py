import os
import json
import re
import numpy as np
import shutil
import cv2
import matplotlib.pyplot as plt

def check_image_exists(path):
    return os.path.exists(path)

def repl_func(match):
    return " ".join(match.group().split())

def writeFrequencyIntoJSON(save_path, image_frequencies, save_name):
    json_path = os.path.join(save_path, f"{save_name}.json")
    json_object = json.dumps(image_frequencies, indent=4)

    json_object2 = re.sub(r"(?<=\[)[^\[\]]+(?=\])", repl_func, json_object)

    with open(json_path, 'w') as json_file:
        json_file.write(json_object2)


def getJSONdata(path):
    with open(path, 'r') as f:
        pixel_data = json.load(f)

    return pixel_data

def combine_frequencies(data, N):
        binned_data = {}
        
        for image, parts in data.items():
            binned_data[image] = {}
            
            for part, frequencies in parts.items():
                # Bin the frequencies: sum every N values together
                binned_frequencies = [
                    sum(frequencies[i:i+N]) for i in range(0, len(frequencies), N)
                ]
                
                binned_data[image][part] = binned_frequencies
        
        return binned_data

def move_images_to_parent(parent_folder):
    # Récupère tous les sous-dossiers
    subfolders = [os.path.join(parent_folder, subfolder) for subfolder in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, subfolder))]

    # Parcourt chaque sous-dossier
    for subfolder in subfolders:
        for file_name in os.listdir(subfolder):
            source_path = os.path.join(subfolder, file_name)
            if os.path.isfile(source_path):  # Vérifie si c'est un fichier
                # Génère un nouveau nom de fichier s'il y a des conflits
                destination_path = os.path.join(parent_folder, file_name)
                base_name, ext = os.path.splitext(file_name)
                counter = 1
                while os.path.exists(destination_path):
                    destination_path = os.path.join(parent_folder, f"{base_name}_{counter}{ext}")
                    counter += 1
                # Déplace le fichier
                shutil.move(source_path, destination_path)
    print("Les images ont été déplacées avec succès.")


def plot_histograms(image_paths):
    """
    Calculate and display histograms as bar charts for n images.

    Parameters:
        image_paths (list): A list of paths to image files.
    """
    if not image_paths:
        print("No image paths provided.")
        return

    # Determine the number of images
    n = len(image_paths)
    
    # Create a figure with n subplots
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    
    # Handle the case of a single image (axes won't be a list)
    if n == 1:
        axes = [axes]
    
    for idx, (image_path, ax) in enumerate(zip(image_paths, axes)):
        # Read the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"Unable to read image: {image_path}")
            continue
        
        # Calculate histogram
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        
        # Normalize the histogram for better visibility (optional)
        hist = hist.ravel()
        
        # Plot histogram as bars
        ax.bar(range(256), hist, color='gray', width=1.0)
        ax.set_title(f"Histogram {idx + 1}")
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Frequency")
        ax.grid(False)  # Turn off grid for a cleaner bar chart
    
    plt.tight_layout()
    plt.show()