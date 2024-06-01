import pandas as pd
import os 
from pathlib import Path
from icecream import ic 
import numpy as np 
import json
from tqdm import tqdm
import re
import pybboxes as pbx
from PIL import Image 
tqdm.pandas()
pd.options.mode.chained_assignment = None  # default='warn'

DATA_PATH = "datasets/questions/"
INPUT_PATH = "datasets/model-inputs"
CLASS_DICT = "src/utils/class_dict.json"
OUTPUT_PATH = "datasets/questions/all"


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH + "all/"+"master.csv")
    
    # # Test sample
    # df = df[df['file']=='d3_7_jpg.rf.31bf0c5e0778dba314bbf2fe362e1d6e']
    # ic(df)

    with open(CLASS_DICT,encoding='utf8') as f :
        class_dict = json.load(f)
    ic(class_dict)

    #df['bbox'] = ""
    df['bbox_segment'] = ""
    df['bbox_len'] = 0 



    for i,row in tqdm(df.iterrows(),total=df.shape[0]):
        file_name = row['file'] + '.txt'
        image_name = row['file'] + '.jpg'
        folder_name = row['splittype']


        file_path = os.path.join(INPUT_PATH,"labels",folder_name,file_name)
        image_path = os.path.join(INPUT_PATH,"images",folder_name,image_name)
        img = Image.open(image_path) 
        # ic(file_path,image_path)

        #items = []
        prompts = []
        if  os.path.exists(file_path):
            # read file 
            with open(file_path,'r') as f:
                lines = [line.rstrip() for line in f]
                #ic(lines)
                for line in lines:
                    #ic(line)
                    # line = f.readline()
                    l = line.split()
                    #ic(l)
                    id = str(int(l[0]))

                    # for each line 
                    # Get id -> symbol
                    symbol = class_dict[id]
                    #ic(symbol)
                    if symbol in ['text','junction','terminal']:
                        continue 

                    # add bbox segment
                    yolo_bbox = tuple([float(s) for s in l[1:]])
                    #print(yolo_bbox)

                    #yolo_bbox1 = (0.286972, 0.647157, 0.404930, 0.371237)
                    #yolo_bbox2 = (0.681338, 0.366221, 0.454225, 0.418060)

                    W = img.width 
                    H = img.height 
                    #print(W,H,yolo_bbox)
                    (x,y,w,h) = pbx.convert_bbox(yolo_bbox, from_type="yolo", to_type="voc", image_size=(W, H))
                    ic(x,y,w,h)

                    bbox_segment = ""
                    # divide image into 9 segment 

                    if x < (w / 3):
                        if y < (h/3):
                            bbox_segment = "lower left"
                        elif y > (h/3) and y < (2 * h/3):
                            bbox_segment = "lower middle"
                        else:
                            bbox_segment = "right"
                    elif x > (w/3) and x < (2 * w/3):
                        if y < (h/3):
                            bbox_segment = "left"
                        elif y > (h/3) and y < (2 * h/3):
                            bbox_segment = "middle"
                        else:
                            bbox_segment = "lower right"
                    else:
                        if y < (h/3):
                            bbox_segment = "upper left"
                        elif y > (h/3) and y < (2 * h/3):
                            bbox_segment = "upper middle"
                        else:
                            bbox_segment = "upper right"

                    items = (symbol,bbox_segment)
                    #ic(items)

                    # prompt input
                    # prompt = " Class is " + items[0] +  ". X coordinate " + str(round(float(items[1][0]),3)) + " Y coordinate " +  str(round(float(items[1][1]),3)) + " Width " + str(round(float(items[1][2]),3)) +  " Height " + str(round(float(items[1][3]),3)) + " . "
                    prompt = items[0] + " " + items[1] + " . "
                    #ic(prompt)
                    prompts.append(prompt)

        #ic(prompts)
        #ic(len(prompts),len(set(prompts)))            
        prompts = list(set(prompts))
        prompts = " ".join(prompts)

        #ic(prompts)
        df['bbox_segment'].loc[i] = prompts
        #df['bbox_len'].loc[i] = len(prompts.split(" "))

    ic(df.head(5))

    df.to_csv(os.path.join(OUTPUT_PATH,"master_bbox_segment.csv"),index=False)
    df.to_json(os.path.join(OUTPUT_PATH,"master_bbox_segment.json"),orient="records")