from PIL import Image
import numpy as np
from utility import check_image_exists, writeFrequencyIntoJSON
import os
from scipy.stats import entropy
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr

class Comparer:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.images = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]

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