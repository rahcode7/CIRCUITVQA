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

CLASS_DICT = "circuit-QGA-1Dec/src/utils/class_dict.json" #circuit-QGA-1Dec/src/utils
OUTPUT_PATH = "datasets/questions/all"


## App 1
if __name__ == "__main__":
    df_bbox = pd.read_csv(DATA_PATH + "all/"+"master_bbox.csv")


    df_all = pd.read_csv(DATA_PATH + "all/"+"master_adv_ocr.csv")
    #df_all = pd.read_csv(DATA_PATH + "all/"+"master_adv.csv") 


    df_bboxseg = pd.read_csv(DATA_PATH + "all/"+"master_bbox_segment.csv")
    df_bboxseg.columns = [x.lower() for x in df_bboxseg.columns]

    # df_bboxseg_all = df_bboxseg.merge(df_all,on=[
    # 'splittype','file','question','answer','qtype','symbol'
    # ],how='inner')


    # Concat 
    df_all.sort_values(by=['splittype','file','question','answer','qtype','symbol'], inplace=True)
    df_bboxseg.sort_values(by=['splittype','file','question','answer','qtype','symbol'], inplace=True)
    df_bbox.sort_values(by=['splittype','file','question','answer','qtype','symbol'], inplace=True)


    
    df_bboxseg.drop(['splittype','file','question','answer','qtype','symbol','bbox_len'], axis=1, inplace=True)
    df_bbox.drop(['splittype','file','question','answer','qtype','symbol','bbox_len'], axis=1, inplace=True)
    
    ic(df_bbox.columns,df_bboxseg.columns,df_all.columns)


    df_bboxseg_all = pd.concat([df_all,df_bboxseg], ignore_index=True,axis=1)
    ic(df_bboxseg_all.head(5))

    df_bboxseg_all = pd.concat([df_bboxseg_all,df_bbox], ignore_index=True,axis=1)

    ic(df_bboxseg_all.head(5))
   
    l1 = list(df_all.columns.tolist())
    l2 = list(df_bboxseg.columns.tolist())
    l3 = list(df_bbox.columns.tolist())
    l1.extend(l2)
    l1.extend(l3)

    df_bboxseg_all.columns = l1 

    ic(df_bboxseg_all.columns)

    ic(df_all.shape,df_bboxseg.shape,df_bboxseg_all.shape)


    # df_bboxseg_all.to_csv(os.path.join(OUTPUT_PATH,"master_bbox_segment_adv_ocr.csv"),index=False)
    # df_bboxseg_all.to_json(os.path.join(OUTPUT_PATH,"master_bbox_segment_adv_ocr.json"),orient="records")

    df_bboxseg_all.to_csv(os.path.join(OUTPUT_PATH,"master_bbox_and_segment_adv_ocr.csv"),index=False)
    df_bboxseg_all.to_json(os.path.join(OUTPUT_PATH,"master_bbox_and_segment_adv_ocr.json"),orient="records")


    df_sample = df_bboxseg_all.sample(n=400)

    # df_sample.to_csv(os.path.join(OUTPUT_PATH,"master_bbox_segment_adv_ocr_400.csv"),index=False)
    # df_sample.to_json(os.path.join(OUTPUT_PATH,"master_bbox_segment_adv_ocr_400.json"),orient="records")

    df_sample.to_csv(os.path.join(OUTPUT_PATH,"master_bbox_and_segment_adv_ocr_400.csv"),index=False)
    df_sample.to_json(os.path.join(OUTPUT_PATH,"master_bbox_and_segment_adv_ocr_400.json"),orient="records")

