import pandas as pd 
import sys 
import os 
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import precision_recall_fscore_support as score

from icecream import ic 
import re 
import os 
import numpy as np 
from tqdm import tqdm 
import ast
import json 

if __name__ == "__main__":
    #MODEL="InstructBLIP" # 
    MODEL='gpt4v'
    PREDICTION_FILE="predictions-final.csv"
    #exp_list = ['base','ocr-pre','ocr-post','desc','bbox','bbox-segment','bbox-yolo','bbox-segment-yolo']
    #exp_list = ['ocr','ocr-post','bbox_yolo','bbox_segment_yolo',
    exp_list = ['bbox','bbox_segment']
    # MODEL='BLIP'  
    # PREDICTION_FILE="predictions.csv"
    # exp_list = ['base','base-lr','desc','ocr-pre','ocr-post','wce','bbox','bbox-segment','bbox-yolo','bbox-segment-yolo'] # BLIP

    # MODEL='PIX' 
    # PREDICTION_FILE="predictions.csv"
    # exp_list = ['bbox-segment-yolo'] #,'desc','ocr-pre','ocr-post','wce','bbox','bbox-segment','bbox-yolo']

    # MODEL='GIT'  
    # PREDICTION_FILE="predictions.csv"
    # exp_list = ['base','base-lr','desc','ocr-pre','ocr-post','wce','bbox','bbox-segment','bbox-yolo','bbox-segment-yolo'] # GIT

    # MODEL='BLIP' # BLIP
    # PREDICTION_FILE="predictions.csv"
    # exp_list = ['base-lr','desc','ocr-pre','ocr-post','wce','bbox','bbox-segment','all','bbox-yolo','bbox-segment-yolo'] # BLIP


    ROOT_DIR = 'models-' + MODEL  + '-hf'
    SIZE = '384' # 576 
    
    hs_vqa_list,hs_cnt_list,hs_ood_list,hs_id_list = [],[],[],[]
    op_dir = 'results-ddp/'  + SIZE + 'a'

    for EXP in exp_list:
        ic(MODEL,EXP)
        #ic(EXP)
        
        results_dir = 'results-ddp/'  + SIZE + 'a/'+ EXP
        df = pd.read_csv(os.path.join((ROOT_DIR),results_dir,PREDICTION_FILE))
        #ic(df.head(2))
        
        # #if MODEL in ['BLIP','GIT','PIX']
        df['prediction'] = df.apply(lambda row : str(row.prediction).split(".")[0] if row.qtype in ['count-complex','count']  else str(row.prediction),axis=1)
        #df['prediction'] = df.apply(lambda row : str(row.prediction).split(".")[0] if row.qtype in ['count-complex','count']  else str(row.prediction),axis=1)

        
        df['answer'] = df.apply(lambda row : str(row.answer).split(".")[0] if row.qtype in ['count-complex','count']  else str(row.answer),axis=1)
        #ic(df.info())

        # df['cnt_over']  = np.where(df['prediction'] > df['answer'],df['prediction']-df['answer'],0)
        
        #df = df[df['qtype'].isin(['count','count-complex','value','junction'])] # 'count','count-complex'

        with open('src/utils/class_dict.json') as f: 
            data = f.read()
        class_dict = json.loads(data)
        class_list = list(class_dict.values())
        #ic(class_list)

        oodomain = ['circle','square','A','B','D','F','triangle','carlin','nano','peizo-keeper','trigger','Snake Snake Detector','wire','line'] 
        oodomain_numbers = [str(i) for i in range(50)]
        idomain = class_list  # + ['yes','no']
        
        #total_cnt = 0 
        df['cnt_over'] = 0
        
        cnt_p = 0 
        #cnt_cp=0
        for j,row in tqdm(df.iterrows(),total=df.shape[0]):
            cnt_ood,cnt_id,cnt_over = 0,0,0
            #ic(row)

            if row['qtype'] in ['position','junction']:
                #total+=1
                
                if MODEL == 'LLaVA':
                    l = row['raw_prediction'].lower().split()
                else:
                    l = str(row['prediction']).lower().split()
                    #ic(l)

                for item in oodomain:
                    if item in l:
                        cnt_ood +=1 

                #for item in l: 
                if l[0] in oodomain_numbers:              
                    #ic(row['answer'],l,item,cnt_ood)
                    cnt_ood+=1


                #for item in idomain:
                #if item in l and item != str(row['answer']):
                if l[0] in idomain:
                    #ic(l[0],l,row['answer'])
                    cnt_id +=1 
                    #ic(row['answer'],item,l,cnt_ood)

                #ic(row['qtype'],row['answer'],row['raw_prediction'],cnt_ood,cnt_id)

            elif row['qtype'] in ['count','count-complex']:
                #total_cnt+=1
                
                #ic(row['prediction'],row['answer'])
                #if row['prediction'] in ['yes','no','xor','or','resistor','not','gnd','nor',"['1","['1ωa']","['1η']","['2v']","['100']","['1ohm', '1ohm']", "['1ohm', '3ohm']","['25v']","['1ohm', '2ohm', '3ohm']"]:

                l = [str(i) for i in range(0,100)]
                if row['prediction'] not in l:  
                    p = 0
                else:
                    p = int(row['prediction'])
                a = int(row['answer'])
                if p > a:
                    #if row['qtype'] == 'count':
                    cnt_over  = (p - a) / p 
                    #ic(p,a,cnt_over)
                    #ic(p,a,cnt_over)

                    # else:
                    #     cnt_cp += p
                    #cnt_p = p

            elif row['qtype'] == 'value': # Ocr
               #total+=1 
                cnt_over = 0 

                if MODEL == 'LLaVA':
                    lp = len(list(ast.literal_eval(row['prediction'])))
                    la = len(list(ast.literal_eval(row['answer'])))
                else:
                    lp = len(row['prediction'][1:-1].split(','))
                    la = len(row['answer'][1:-1].split(","))
                #     lp = len(list(row['prediction']))
                #     la = len(list(row['answer']))
                    #ic(lp,la)
                if lp > la:
                    cnt_over = (lp - la) / lp
                    #ic(row['prediction'],row['answer'],lp,la,cnt_over)

            df.at[j,'cnt_over'] = cnt_over 
            df.at[j,'cnt_ood'] = cnt_ood
            df.at[j,'cnt_id'] = cnt_id
            #df.at[j,'cnt_p'] = cnt_p 

        # HS score overall
        cnt_over_total = df['cnt_over'].sum() # [df['qtype'].isin(['count','count-complex','value'])].sum()
        
        hs_cnt = cnt_over_total/df.shape[0]

        ic(hs_cnt,cnt_over_total,cnt_p,df['cnt_over'].groupby(df['qtype']).sum())
       

        ood_total = df['cnt_ood'].sum() # [df['qtype'].isin(['position','junction'])].sum()
        hs_ood = ood_total / df.shape[0]  # tota number of predictions
        ic(hs_ood,ood_total,df['cnt_ood'].sum(),df.shape[0])


        id_total = df['cnt_id'].sum() # [df['qtype'].isin(['position','junction'])].sum()
        hs_id = id_total / df.shape[0] 
        ic(hs_id,id_total,df['cnt_id'].sum(),df.shape[0])

        # hs_value = df['cnt_over'][df['qtype']=='value'].sum()/df[df['qtype']=='value'].shape[0]
        # ic(hs_value,df['cnt_over'][df['qtype']=='value'].sum(),df[df['qtype']=='value'].shape[0])

        # hs_obj = df['cnt_over'][df['qtype']=='junction'].sum()/df[df['qtype']=='junction'].shape[0]
        # ic(hs_obj,df['cnt_over'][df['qtype']=='junction'].sum(),df[df['qtype']=='junction'].shape[0])

        ## hs-overall
        hs_vqa = (hs_cnt + hs_ood + hs_id)/3
        hs_cnt_list.append(hs_cnt)
        hs_ood_list.append(hs_ood)
        hs_id_list.append(hs_id)
        hs_vqa_list.append(hs_vqa)

    
    res_df = pd.DataFrame(zip(exp_list,hs_vqa_list,hs_cnt_list,hs_ood_list,hs_id_list),columns=['exp-name','hs-vqa','hs-count','hs-oodomain','hs-idomain'])
    res_df['model'] = MODEL
    ic(res_df)
    res_df.to_csv(os.path.join(ROOT_DIR,op_dir,"hallucination-scores.csv"),index=None)



        
