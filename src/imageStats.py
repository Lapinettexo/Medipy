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
        else:
            print(f"Unknown split mode '{mode}'")
        
        return parts
    
    def calculate_pixel_frequency(self, part):
        img_array = np.array(part)
        pixels = img_array.flatten()
        frequency = np.bincount(pixels, minlength=256)
        return frequency.tolist()
    
    def compare_sides_symmetry(self, image, metric="euclidean"):
        """
        Compare les fréquences des pixels des moitiés gauche et droite de l'image pour détecter des asymétries.
        """
        left_part, right_part = self.imageSplitter(image, mode="2v")
        
        left_frequency = self.calculate_pixel_frequency(left_part)
        right_frequency = self.calculate_pixel_frequency(right_part)
        
        if metric == "euclidean":
            # Utiliser la distance euclidienne
            asymmetry_score = euclidean(left_frequency, right_frequency)
        elif metric == "correlation":
            # Utiliser le score de corrélation de Pearson
            asymmetry_score, _ = pearsonr(left_frequency, right_frequency)
            # On prend 1 - la corrélation pour qu'un score plus élevé indique plus d'asymétrie
            asymmetry_score = 1 - asymmetry_score
        elif metric == "KL-divergence":
            # Si la divergence KL est toujours souhaitée, on peut la calculer ici.
            asymmetry_score = entropy(left_frequency, right_frequency)
        else:
            raise ValueError(f"Unknown metric '{metric}'")
        
        return asymmetry_score
    
    def analyze_symmetry_in_folder(self, threshold=3000):
        """
        Analyse les images dans le dossier pour la symétrie et signale celles avec des asymétries dépassant le seuil.
        """
        flagged_images = []
        print("Start analyzing symmetry in images")

        for image_file in self.images:
            image_path = os.path.join(self.folder_path, image_file)

            if not check_image_exists(image_path):
                print(f"Error: The image at '{image_path}' does not exist.")
                continue

            image = Image.open(image_path).convert('L')
            asymmetry_score = self.compare_sides_symmetry(image)
            
            
            if asymmetry_score > threshold:
                print(f"Asymmetry score for {image_file}: {asymmetry_score}")
                flagged_images.append(image_file)
        
        print("End analyzing symmetry")
        return flagged_images

    def get_frequency_for_all_images(self, split_mode):
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
            parts = self.imageSplitter(image, mode=split_mode)

            image_frequencies = {}

            for i, part in enumerate(parts):
                pixel_frequencies = self.calculate_pixel_frequency(part)
                image_frequencies[f"part_{i+1}"] = pixel_frequencies

            all_image_frequencies[image_file] = image_frequencies

        print("End processing")

        save_folder = "C://Users//Trust_pc_dz//Documents//IMED//DATASET//Frequencies"
        writeFrequencyIntoJSON(save_folder, all_image_frequencies, "NoTumor_2v")
        print("Data written in JSON file.")
