from imageStats import DataCollector
from imageEnhance import ImageEnhancer
from utility import getJSONdata, writeFrequencyIntoJSON, combine_frequencies, move_images_to_parent, plot_histograms
import os
import json

def main():
    
    data_collector = DataCollector("C://Users//Trust_pc_dz//Documents//IMED//DATASET//Clean Data//Brain Tumor clean")
    data_collector.get_frequency_for_all_images()
    #move_images_to_parent("C://Users//Trust_pc_dz//Documents//IMED//DATASET//Clean Data//Brain Tumor clean")

    #image_files = ["./img/glioma.jpg", "./img/meningioma.jpg", "./img/pituitary.jpg", "./img/noTumor.jpg", "./img/lightNoTumor.jpg"]
    #image_files = [ "./img/notumor4.jpg", "./img/notumor4light.jpg"]
    #plot_histograms(image_files)



if __name__ == "__main__":
    main()
