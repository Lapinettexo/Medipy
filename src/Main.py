from imageStats import DataCollector
from utility import getJSONdata, writeFrequencyIntoJSON, combine_frequencies
import os
import json

def main():
    
    dataset_folder = "./Data"


    json_path = "./Frequencies/no_tumor_frequencies.json"
    pixel_data = getJSONdata(json_path)

    binned_pixel_data = combine_frequencies(pixel_data, 5)

    output_path = "./Frequencies"
    writeFrequencyIntoJSON(output_path, binned_pixel_data, "binned5_no_tumor_frequencies")


if __name__ == "__main__":
    main()
