from imageStats import DataCollector
from imageEnhance import ImageEnhancer
from utility import getJSONdata, writeFrequencyIntoJSON, combine_frequencies
import os
import json

def main():
    
    """dataset_folder = "./Data"


    json_path = "./Frequencies/no_tumor_frequencies.json"
    pixel_data = getJSONdata(json_path)

    binned_pixel_data = combine_frequencies(pixel_data, 5)

    output_path = "./Frequencies"
    writeFrequencyIntoJSON(output_path, binned_pixel_data, "binned5_no_tumor_frequencies")"""

    #dossier_images =  r"C:\Users\Trust_pc_dz\Documents\IMED\DATASET\archive (1)\Brain Tumor Data Set\Brain Tumor clean - Copie"
    #dossier_images =  r"C:\Users\Trust_pc_dz\Documents\IMED\DATASET\test"
    #dossier_images =  r"C:\Users\Trust_pc_dz\Documents\IMED\DATASET\Clean Data\No tumor - Copie"
    #dossier_sortie = "C://Users//Trust_pc_dz//Documents//IMED//DATASET//clean background"

    #enhancer = ImageEnhancer(dossier_images)

    #enhancer.remove_low_quality_images()
    #data_collector = DataCollector("C://Users//Trust_pc_dz//Documents//IMED//DATASET//Clean Data//No tumor clean")
    #data_collector.get_frequency_for_all_images(split_mode="2v")


if __name__ == "__main__":
    main()
