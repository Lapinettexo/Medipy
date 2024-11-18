import os
import json
import re
import numpy as np
import shutil

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

