"""
  Align bounding boxes of symbol and values
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
GOOGLE_PATH  = "datasets/questions/value/google-bbox"

cnt = 0 
if __name__ == "__main__":
    # bounding box from local
    for dir in ['train','test','dev']:
        for file in os.listdir(os.path.join(DATA_PATH,dir,"images")):
            file_path = os.path.join(DATA_PATH,file)
            file = file.split(".jpg")[0] + ".csv"
            google_file_path = os.path.join(GOOGLE_PATH,'all',file) 
            image_name = file.replace(".csv","")    
            cnt +=1
            ic(cnt,file,google_file_path,image_name)

            if not os.path.isfile(google_file_path):
                  continue

            val_df = pd.read_csv(google_file_path,encoding="utf-8")

            # # Delete rows which are empty
            #print(val_df)
            val_df['val'] =val_df['val'].astype(str)
            val_df['lval'] = val_df['val'].apply(lambda x: len(x))
            val_df['isdig'] = val_df['val'].apply(lambda x: starts_digit(str(x)))
            #print(val_df.shape)
            
            # # Less than 3 characters and contains a number 
            val_df = val_df[(val_df['lval']<=4) & (val_df['isdig']==True)].reset_index()
            val_df.index = list(val_df.index)
            print(val_df.shape)

            # # IF NO DIGIT OR VALUE FOUND, WRITE EMPTY DISTANCE FILE 
            if (val_df['val'] == False).all():
                print("Yes all false")
                cols = val_df['val']
                dist_df = pd.DataFrame(columns=cols)
                dist_df.to_csv(OUTPUT_PATH + "dist-bbox/" + 'all/' + image_name + ".csv",index=None)
                continue
            else:
                ## Computer euclidean distance across 4 data points for each symbol-value pair
                dists_main = []
                min_vals = []
                #print(symbol_df)

                xml_file = image_name + ".xml"
                xml_path =  os.path.join(DATA_PATH,dir,'xml',xml_file) 
                print(xml_path)     
                symbol_df = xml_processor(xml_path)
                symbol_df = class_cleaner(symbol_df,xml_path)
                symbol_df[["xmin", "xmax","ymin","ymax"]] = symbol_df[["xmin", "xmax","ymin","ymax"]].apply(pd.to_numeric) 
                ic(symbol_df)

                for i,row in symbol_df.iterrows():
                    #print(row['s'])
                    c1 = np.array((row['xmin'],row['ymin']))
                    c2 = np.array((row['xmax'],row['ymin']))
                    c3 = np.array((row['xmin'],row['ymax']))
                    c4 = np.array((row['xmax'],row['ymax']))
                    dists = []
                    for j,row_val in val_df.iterrows():
                        #print type((row_val['c1']))
                        row_val['c1'] = ast.literal_eval(row_val['c1'])
                        row_val['c2'] = ast.literal_eval(row_val['c2'])
                        row_val['c3'] = ast.literal_eval(row_val['c3'])
                        row_val['c4'] = ast.literal_eval(row_val['c4'])
                        #print(row_val['c1'],row_val['c2'],row_val['c3'],row_val['c4'])

                        d1 = euclidean_distance(c1,row_val['c1']) 
                        d2 = euclidean_distance(c2,row_val['c2']) 
                        d3 = euclidean_distance(c3,row_val['c3']) 
                        d4 = euclidean_distance(c4,row_val['c4'])
                        
                        #print(d1,d2,d3,d4)
                        dists.append(d1+d2+d3+d4)
                    

                    dists_main.append(dists)
                    min_dist_idx = np.argmin(np.array(dists))
                    #print("min_index ",min_dist_idx)
                    cols = val_df['val']
                    #print(val_df.iloc[min_dist_idx])
                    min_val = val_df['val'].iloc[min_dist_idx]
                    min_vals.append(min_val)
                

                dist_df = pd.DataFrame(dists_main,columns=cols)
                ic(symbol_df)
                dist_df['s'] = symbol_df['s']
                dist_df['xmin'] = symbol_df['xmin']
                dist_df['xmax'] = symbol_df['xmax']
                dist_df['ymin'] = symbol_df['ymin']
                dist_df['ymax'] = symbol_df['ymax']
                dist_df['minval'] = min_vals
                dist_df['image_path'] = image_name
                ic(dist_df)

                #dist_df.to_csv("../../datasets/questions/val/train/autockt_-323_png.rf.a0afbaa7e18cd9ddb05aff68ecbcb38a.csv",index=None)
                #dist_df.to_csv(OUTPUT_PATH + "dist-bbox/" + dir + '/' + image_name + ".csv")
                dist_df.to_csv(OUTPUT_PATH + "dist-bbox/" + 'all/' + image_name + ".csv")

                    

