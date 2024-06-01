# title	annotations	yt_url
# Balancing Business & Lifestyle "chapter annotations\nchapter name: Create a list\nvisual captions: In the video, we'll see then [Music] \nIMAGE:https://raw.githubusercontent.com/ceyxasm/vid-frames-dv1-p7/master/RzcHE7KJmoc/RzcHE7KJmoc/1.png\n

# tab separated
# How to measure for a new washing machine | Currys PC World  "chapter annotations\nchapter name: Intro\nvisual captions"  https://www.youtube.com/watch?v=dSUhX8JTC3I

import json
import shutil 
import pandas as pd 
import os 
from icecream import ic 
import numpy as np
import argparse

 # Add desc to question 
def desc(q,s,d):
    #ic(q,s,d)
    if d[-1]==".":
        #return("The following question is about an electrical component : " + s + "  .It cab be described as follows: " +  d + " Now, please provide an answer to the given question. " + q)
        return("Use the following description of the electrical component to answer the question: " +  d + " Now, respond to this question: " + q)
    else:
        return("Use the following description of the electrical component to answer the question: " +  d + ". Now, respond to this question: " + q)

def ocr(q,o):
    return("Use the following OCR output to answer the question: " +  o + ". Now, respond to this question: " + q)

def bbox(q,b):
    return("Use the following bounding box output comprising of the components and their coordinates in the image: " +  b + "Now, respond to this question: " + q)

def bbox_segment(q,b):
    return("Use the following bounding box output comprising of the components and their corresponding positions in the image : " +  b + "Now, respond to this question: " + q)



if __name__ == "__main__":

    pd.set_option('display.max_columns', None)  # or 1000
    pd.set_option('display.max_rows', None)  # or 1000
    pd.set_option('display.max_colwidth', None)  # or 199

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--q_path', help='directory')
    parser.add_argument('--op_path', help='directory')
    parser.add_argument('--exp_name', help='directory')
    parser.add_argument('--hosted_url', help='directory')
    
    args = parser.parse_args()
    EXP = args.exp_name 
    URL = args.hosted_url

    #OP_PATH = "models-hf/gpt4v/datasets/desc"
    # Read master.json to get question,file for generation

    #Q_PATH = "datasets/questions/all" # 
    Q_PATH = args.q_path
    OP_PATH = args.op_path # "models-hf/gpt4v/datasets/desc"

    #FILE_NAME = "master_adv.json"
    df = pd.read_json(Q_PATH)
    df = df[df['splittype']=='test']

    ic(df.head(2))
    
    if EXP == 'desc':   
        #ic(df.desc.isna().sum())         
        df['question'] = df.apply(lambda x : desc(x.question,x.symbol,x.desc) if x.desc is not None else x.question,axis=1)
    elif EXP == 'ocr' or EXP == 'ocr-post':
        df['question'] = df.apply(lambda x : ocr(x.question,x.ocr) if x.ocr!="" else x.question,axis=1)
    elif EXP == 'bbox' or EXP == 'bbox_yolo':
        df['question'] = df.apply(lambda x : bbox(x.question,x.bbox) if x.bbox!="" else x.question,axis=1)
    elif EXP == 'bbox_segment' or EXP == 'bbox_segment_yolo':
        df['question'] = df.apply(lambda x : bbox_segment(x.question,x.bbox_segment) if x.bbox_segment!="" else x.question,axis=1)
    else: # BASE 
        pass 


    df = df.reset_index()
    #ic(df.head(4))
    df['id'] = df.index + 1
    df['image_url']  = df['file'].apply(lambda x: URL + x + ".jpg")
    #print(df['image_url'].head(2))
    #ic(df.head(4))

    df = df[['id','file','question','image_url']]

    #ic(df['file'].nunique())

    NUMBER_OF_SPLITS = 24

    if os.path.exists(OP_PATH):
        shutil.rmtree(OP_PATH)
    os.mkdir(OP_PATH)
    for i, new_df in enumerate(np.array_split(df,NUMBER_OF_SPLITS)):
        file_name = f"master{i}.txt"
        ic(file_name)
        with open(os.path.join(OP_PATH,file_name),"w") as fo:
            fo.write(new_df.to_csv(sep="\t",index=None))