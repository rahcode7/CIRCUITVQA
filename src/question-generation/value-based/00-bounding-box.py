"""
  Align bounding boxes of symbol and values

  STEP 1Utilise Google Vision API 
  STEP 2Compute distances and align objects

"""
import xml.etree.ElementTree as ET
import pandas as pd 
from google.cloud import vision
import math
import ast
from scipy.spatial import distance
import numpy as np
import os
from  utils.preprocessing  import xml_processor,starts_digit,cloud_value_bndbox,euclidean_distance,class_cleaner
from pathlib import Path
from icecream import ic 

OUTPUT_PATH = "datasets/questions/value/"
DATA_PATH  = "datasets/"

cnt = 0 
if __name__ == "__main__":
    
    # bounding box from local
    for dir in ['dev']:   
    #for dir in ['test']:
        cnt=0
        for file in os.listdir(os.path.join(DATA_PATH,dir,'xml')):
            ic(file)
            xml_path =  os.path.join(DATA_PATH,dir,'xml',file)       
            symbol_df = xml_processor(xml_path)
            symbol_df = class_cleaner(symbol_df,xml_path)
            dataset_name = xml_path.split('/')[-1][0:2]
            if dataset_name == "d2": # Sample from d1 for testing # NO D4 as gates only, d2 and d3 done, d5 remaining 
                image_name = file.replace(".xml",".jpg")
                image_path = os.path.join(DATA_PATH,dir,'images',image_name) 
                image_name = image_path.split("/")[-1:][0].replace(".jpg","")
                print(image_path)
        
                symbol_df[["xmin", "xmax","ymin","ymax"]] = symbol_df[["xmin", "xmax","ymin","ymax"]].apply(pd.to_numeric)

                cnt +=1
                print(cnt)

                # Check if output already created
                google_file_path = OUTPUT_PATH + "google-bbox/" + 'all/' + image_name + ".csv"
                google_file = Path(google_file_path)
                if google_file.is_file():
                    print("File exists")
                    continue

                #For getting the values from a new image call google vision
                val_df = cloud_value_bndbox(image_path)
                val_df.to_csv(google_file_path,index=None)
                print("File created")

             