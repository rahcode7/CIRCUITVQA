import math
import shutil
import pandas as pd
from icecream import ic 
import ast
import os 
import argparse
import glob
from tqdm import tqdm 
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--prediction_dir', help='directory')
    parser.add_argument('--exp_name', help='directory')
    
    args = parser.parse_args()

    RESULTS_DIR = args.prediction_dir
    EXP_NAME = args.exp_name 


    # Read all 23 files and join them first
    path = RESULTS_DIR + '/' + EXP_NAME + '-all'
   
    all_files = glob.glob(os.path.join(path , "*.tsv"))
    ic(all_files)
    li = []
    for filename in all_files:
        small_df = pd.read_csv(filename, sep='\t', header=0)
        li.append(small_df)
    #ic(li)
    df = pd.concat(li, axis=0, ignore_index=True)
    ic(df.shape[0])
    ic(df.info())


    df['prediction'] = ""
    for i,rows in tqdm(df.iterrows(),total=df.shape[0]):
        res = rows['Result.OutputResult']
        #ic(rows,res)
        if isinstance(res,str):
            res = res[res.index("{"):res.index("}")+1]
            res_d = ast.literal_eval(res)
            res = res[res.index("{"):res.index("}")]
            res_d['prediction']
            df.at[i,'prediction'] = res_d['prediction'] 
        elif math.isnan(res):
            df.at[i,'prediction'] = ""
        else:
            break
        #ic(res,res_d['prediction'])
    ic(df.head(3))
    df= df[['id','file','question','prediction']]
    #ic(df.info())
    df['id']=df['id']-1
    #df.index = df['id'] 

    #df.set_index('id',inplace=True)
    ic(df.head(3))

    # Combine with master.json
    Q_PATH = "datasets/questions/all"
    OP_PATH = "gpt4v/datasets/"
    FILE_NAME = "master_bbox_segment.json"
    df_master  = pd.read_json(os.path.join(Q_PATH,FILE_NAME))
    df_master = df_master[df_master['splittype']=='test'] # .reset_index()  
    df_master = df_master.reset_index()
    #df_master.index = df_master.index.values
    ic(df_master.head(3)) 

    df_master['id'] = df_master.index + 1

    df_all = pd.merge(df_master[['id','answer','qtype']],df,on=['id'])
    ic(df_all.head(10))
    

    OP_PATH = RESULTS_DIR + '/' + EXP_NAME
    if os.path.exists(OP_PATH):
        shutil.rmtree(OP_PATH)
        
    os.mkdir(OP_PATH)
    df_all.to_csv(os.path.join(OP_PATH,'predictions.csv'), index=None) # , lines=True)
    df_all.to_json(os.path.join(OP_PATH,'predictions.json'), orient='records', lines=True)






