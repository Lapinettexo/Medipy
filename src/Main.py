from imageStats import DataCollector
from imageEnhance import ImageEnhancer
from utility import getJSONdata, writeFrequencyIntoJSON, combine_frequencies, move_images_to_parent
import os
import json

def main():
    
    data_collector = DataCollector("C://Users//Trust_pc_dz//Documents//IMED//DATASET//Clean Data//No tumor clean")
    data_collector.get_frequency_for_all_images(split_mode="1")
    #move_images_to_parent("C://Users//Trust_pc_dz//Documents//IMED//DATASET//Clean Data//Brain Tumor clean")



if __name__ == "__main__":
    main()
