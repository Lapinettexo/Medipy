import os
import json
import re

def check_image_exists(path):
    return os.path.exists(path)

def repl_func(match):
    return " ".join(match.group().split())

def writeFrequencyIntoJSON(save_path, image_frequencies):
    json_path = os.path.join(save_path, "pixel_frequencies.json")
    json_object = json.dumps(image_frequencies, indent=4)

    json_object2 = re.sub(r"(?<=\[)[^\[\]]+(?=\])", repl_func, json_object)

    with open(json_path, 'w') as json_file:
        json_file.write(json_object2)

