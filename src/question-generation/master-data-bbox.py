import pandas as pd
import os 
from pathlib import Path
from icecream import ic 
import numpy as np 
import json
from tqdm import tqdm
import re
tqdm.pandas()

DATA_PATH = "datasets/questions/"
BBOX_PATH = "datasets/model-inputs/labels"
CLASS_DICT = "src/utils/class_dict.json"
OUTPUT_PATH = "datasets/questions/all"
if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH + "all/"+"master.csv")

    with open(CLASS_DICT,encoding='utf8') as f :
        class_dict = json.load(f)
    ic(class_dict)

    df['bbox'] = ""
    df['bbox_len'] = 0 

    for i,row in tqdm(df.iterrows(),total=df.shape[0]):
        file_name = row['file'] + '.txt'
        folder_name = row['splittype']

        file_path = os.path.join(BBOX_PATH,folder_name,file_name)
        ic(file_path)

        #items = []
        prompts = ""
        if  os.path.exists(file_path):
            # read file 
            with open(file_path,'r') as f:
                lines = [line.rstrip() for line in f]

                for line in lines:
                # line = f.readline()
                    l = line.split()
                    id = str(int(l[0]))

                    # for each line 
                    # Get id -> symbol
                    symbol = class_dict[id]


                    if symbol in ['text','junction']:
                        continue 

                    # add bbox
                    bbox = l[1:]
                    items = (symbol,bbox)
                    #ic(items)

                    # prompt input
                    prompt = "Class is " + items[0] +  ". X coordinate " + str(round(float(items[1][0]),3)) + " Y coordinate " +  str(round(float(items[1][1]),3)) + " Width " + str(round(float(items[1][2]),3)) +  " Height " + str(round(float(items[1][3]),3)) + "."
                    #ic(prompt)
                    
                    prompts += prompt
                    
        ic(prompts)
        df['bbox'].loc[i] = prompts
        df['bbox_len'].loc[i] = len(prompts.split(" "))

    ic(df.head(5))

    df.to_csv(os.path.join(OUTPUT_PATH,"master_bbox.csv"))
