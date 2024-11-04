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
                
                binned_data[image][part] = binned_frequencies
        
        return binned_data

