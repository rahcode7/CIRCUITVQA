"""
    Generate master questions dataset using individual question set
"""

import pandas as pd
import os 
from pathlib import Path
from icecream import ic 
import numpy as np 

DATA_PATH = "datasets/questions/"

if __name__ == "__main__":

    #dist_df_all = pd.DataFrame(columns=['splittype', 'file', 'question', 'answer', 'qtype'])

    # COUNT DATASET
    cnt_df = pd.read_csv(DATA_PATH + "/count/Q-count.csv",encoding='ascii')
    ic(cnt_df.shape)


    # POS DATASET
    pos_df = pd.read_csv(DATA_PATH + "/position/Q-position.csv")
    ic(pos_df.columns)
    

    # VAL LABELLED DATASET
    val_df = pd.read_csv(DATA_PATH + "/value/Q-value-labelled-final.csv")
    

    val_df = val_df[['splittype','file','question','qtype','answer_label','answer_label_f']]
    ic(val_df.info())
    

    # Subset d1 only for higher accuracy
    ONLY_D1 = True # False
    #ONLY_D1 = False


    if ONLY_D1:        
        #val_df.rename(columns={'answer_label_f':'answer'},inplace=True)
        val_df['dataset'] = val_df['file'].apply(lambda x: x.split("/")[-1][0:2])
        val_df = val_df[(val_df.dataset == "d1") | (val_df.answer_label.notnull())]
        ic(val_df['dataset'].value_counts())

        val_df = val_df[['splittype','file','question','qtype','answer_label_f']]
        val_df.rename(columns={'answer_label_f':'answer'},inplace=True)
        ic(val_df.shape)
        
    else:
        val_df = val_df[['splittype','file','question','qtype','answer_label_f']]
        val_df.rename(columns={'answer_label_f':'answer'},inplace=True)
        ic(val_df.shape)

    # Count complex LABELLED DATASET
    complex_df = pd.read_csv("datasets/questions/count-complex/labelled/Q-count-complex-labelled-final.csv")


    # Jnc based datasets
    #datasets/questions/junction/Q-junction.csv
    jn_df = pd.read_csv(DATA_PATH + "/junction/Q-junction.csv")

    lst = [cnt_df,pos_df,val_df,jn_df,complex_df]

    master_df = pd.concat(lst)
    ic(master_df.info())

    #master_df[master_df.splittype=='dev']
    # CHANGE splittype "DEV" TO "VAL"
    #master_df.loc[master_df['splittype']=='dev',['splittype']] == 'val'
    master_df['splittype'] = np.where(master_df['splittype'] == 'dev', 'val',master_df['splittype'])
    
    master_df['answer'] = master_df['answer'].astype('str').str.lower()
    
    ic(master_df['splittype'].value_counts())
    ic(master_df['qtype'].value_counts())
    master_df.to_csv(DATA_PATH+"/all/"+"master.csv",index=None)
    master_df.to_json(DATA_PATH+"/all/"+"master.json",orient="records")
    ic("Qs written")




