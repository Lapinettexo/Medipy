import os
import json
import re
import cv2
import numpy as np

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
                
                # Store the binned frequencies
                binned_data[image][part] = binned_frequencies
        
        return binned_data

def noir_background(image_path):
    # Lire l'image en niveaux de gris
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Appliquer un flou gaussien pour réduire le bruit et aider à la détection des contours
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Appliquer un seuillage pour distinguer le cerveau du fond
    _, thresh = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY)
    
    # Trouver les contours pour détecter la région du cerveau
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Créer un masque vide et y dessiner le contour le plus grand (le cerveau)
        mask = np.zeros_like(img)
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=1)

        # Appliquer le masque sur l'image originale pour garder uniquement le cerveau
        img_processed = cv2.bitwise_and(img, mask)
    else:
        img_processed = img
    
    return img_processed

def traiter_images_dossier(dossier_images):
    # Parcourir toutes les images dans le dossier
    for filename in os.listdir(dossier_images):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(dossier_images, filename)
            img_noircie = noir_background(image_path)
            
            # Écraser l'image d'origine avec l'image traitée
            cv2.imwrite(image_path, img_noircie)
            #print(f"Image traitée et enregistrée : {image_path}")

