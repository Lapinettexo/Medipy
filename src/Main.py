from imageStats import DataCollector
from utility import getJSONdata, writeFrequencyIntoJSON, combine_frequencies
import os
import json

def main():
    
    dataset_folder = "C://Users//Trust_pc_dz//Documents//IMED//DATASET//archive (1)//data_base//Training//pituitary_tumor"

    #collector = DataCollector()  
    
    #collector.get_frequency_for_all_images(dataset_folder)
    json_path = "C://Users//Trust_pc_dz//Documents//IMED//DATASET//no_tumor_frequencies.json"
    pixel_data = getJSONdata(json_path)

    binned_pixel_data = combine_frequencies(pixel_data, 3)

    output_path = "C://Users//Trust_pc_dz//Documents//IMED//DATASET"
    writeFrequencyIntoJSON(output_path, binned_pixel_data, "binned_no_tumor_frequencies")


if __name__ == "__main__":
    main()
