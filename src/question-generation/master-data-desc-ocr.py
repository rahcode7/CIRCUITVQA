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

def get_desc(q,symbol_desc):
    q = re.split(r'[?\s]\s*?',q)
    #ic(q)
    for w in q:
        if w == "":
            continue
        if w[-1] == 's':
            w = w[:-1]
        #ic(w)
        if w in symbol_desc.keys():
            #ic("f",w)
            return symbol_desc[w]
    return None

def remove_consecutive(string, char):
        new_string = ""
        prev_char = ""
        for curr_char in string:
            if curr_char == char and curr_char == prev_char:
                continue
            new_string += curr_char
            prev_char = curr_char
        return new_string

def process_ocr(s):
    new_s = ""  
    for i in range(0,len(s)-1):
        if i==0 and s[i] in chars_keep:
            new_s+=s[i]
        elif s[i] in chars_keep and s[i-1].isdigit():
            new_s+=s[i]
        elif s[i].isdigit() or s[i]==",":
            new_s+=s[i]
        else:
            continue
    
    return remove_consecutive(new_s,",").rstrip(',')

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH + "all/"+"master.csv")

    ic(df.head(10))
    ic(df['answer'].value_counts())

    ######################### ADD DESCRIPTION FROM SYMBOLS
    with open(DATA_PATH + "all/" + "symbols_desc.json",encoding='utf8') as f :
        symbol_desc = json.load(f)
    ic(symbol_desc)

    symbol_desc = {k.lower(): v for k,v in symbol_desc.items()}
    ic(symbol_desc.keys())


    symbol_desc_sorted = {}
    for k in sorted(symbol_desc, key=len, reverse=True):
        symbol_desc_sorted[k] = symbol_desc[k]

    ic("Description processing")
    df['desc'] = df['question'].progress_apply(lambda q : get_desc(q,symbol_desc_sorted))
    # df.to_csv(DATA_PATH +"/all/"+"master-desc.csv",index=None)

    # df.to_json(DATA_PATH+"/all/"+"master_desc.json",orient="records")

    ######################### Add OCR inputs from GOOGLE VISION 
    

    GOOGLE_FILE_PATH = "datasets/questions/value/google-bbox/all"

    # For each image,check in dir and if it exists open and get 1st column c, all values, append as a a list in the df['ocr_vals'] column
    df['ocr'] = ""
    for i,row in tqdm(df.iterrows(),total=df.shape[0]):
        file_name = row['file'] + '.csv'
        file_path = os.path.join(GOOGLE_FILE_PATH,file_name)
        if  os.path.exists(file_path):
            val_df = pd.read_csv(file_path)
            val_df['val'] = val_df['val'].astype(str)
            #ic(val_df)
        df['ocr'].loc[i] = ",".join(val_df['val'].tolist())

        
    # Detect language - filter english only, remove braces etc

    df.to_csv(DATA_PATH +"all/"+"master_adv.csv",index=None)
    df.to_json(DATA_PATH+"all/"+"master_adv.json",orient="records",force_ascii=False)

    ##################################     OCR post processing 
    # things to remove words - NN,relay,heon,-OOV
    # characters to keep - ohm,V,F
    chars_keep = ['Ω','H','A','F','V','W','k','K',".","к","M"]
    pattern = r'[0-9.HAFVW]'
    
    ic("Ocr post processing")
    df['ocr'] = df['ocr'].progress_apply(lambda s: process_ocr(s))



    
    df.to_csv(DATA_PATH +"all/"+"master_adv_ocr.csv",index=None)
    df.to_json(DATA_PATH+"all/"+"master_adv_ocr.json",orient="records",force_ascii=False)     



    
  




