import pandas as pd 
import sys 
import os 
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import precision_recall_fscore_support as score

from icecream import ic 
import re 
import os 

if __name__ == "__main__":
    #MODEL="LLaVA"
    # MODEL='InstructBlip'
    # PREDICTION_FILE="predictions-final.csv"
    # exp_list = ['base','desc','ocr-pre','ocr-post','bbox','bbox-segment','bbox-yolo','bbox-segment-yolo'] 


    MODEL='GPT4V'
    SIZE = '384'
    ROOT_DIR = 'models-' + MODEL  + '-hf'
    PREDICTION_FILE="predictions-final.csv"
    #exp_list = #['base','desc'
    #exp_list =  ['ocr','ocr-post','bbox_yolo','bbox_segment_yolo'] #'bbox','bbox-segment'
    exp_list = ['bbox','bbox_segment']
    # ROOT_DIR = '  models-' + MODEL  + '-hf'
    # SIZE = '384' # 576 
    

    #SIZE = '576'
    # LLAVA
    #ROOT_DIR = "datasets/results/llava/384a/base"


    # EXP = '576base'
    #exp_list = ['base-lr','desc','ocr-pre','ocr-post','bbox','bbox-segment']  # 384 BLIP

    #exp_list = ['base-lr','desc','ocr-pre','ocr-post','wce','bbox','bbox-segment','all'] # PIX
    #exp_list = ['base','base-lr','desc','ocr-pre','ocr-post','wce','bbox','bbox-segment','all'] # GIT
        
    # MODEL='PIX'
    # PREDICTION_FILE="predictions.csv"
    # SIZE = '384'
    # ROOT_DIR = 'models-' + MODEL  + '-hf'
    # exp_list = ['bbox-segment-yolo'] # ,'base-lr','desc','ocr-pre','ocr-post','wce','bbox','bbox-segment','all','bbox-yolo','bbox-segment-yolo'] # BLIP

   
    for EXP in exp_list:
        ic(EXP)
        results_dir = 'results-ddp/'  + SIZE + 'a/'+ EXP
        
        df = pd.read_csv(os.path.join((ROOT_DIR),results_dir,PREDICTION_FILE))
        df['prediction'] = df['prediction'].astype('str')
        df['answer'] = df['answer'].astype('str')

        df['answer'] = df.apply(lambda row : row.answer.split(".")[0] if row.qtype =='count-complex'  else row.answer,axis=1)

        #if MODEL in ['BLIP','GIT','PIX']:
        df['prediction'] = df.apply(lambda row : row.prediction.split(".")[0] if row.qtype =='count-complex'  else row.prediction,axis=1)
        df['prediction'] = df.apply(lambda row : row.prediction.split(".")[0] if row.qtype =='count'  else row.prediction,axis=1)
        # df = df[df.qtype=='count-complex']
        ic(df.shape)
        ic(df.info())
        ic(df[df.qtype=='count-complex'].head(10))

        #ic(df.info())
        target_names = list(df['answer'].unique())
        #ic(target_names)

        # Rectify predictions 
        # if space between text 
        df['prediction'] = df['prediction'].apply(lambda x: re.sub(" ","",x))
        #ic(df[df.qtype=='count-complex'].head(10))


        #report = classification_report(df['answer'], df['prediction'], target_names=target_names,output_dict=True)
        report = classification_report(df['answer'], df['prediction'],output_dict=True)
        
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(ROOT_DIR,results_dir,'classification_report.csv'))

        # Qtype wise classification report
        qtypes = list(df['qtype'].unique())

        summary_df = pd.DataFrame(columns=['qtype','precision','recall','fscore','average','size','model','model_category'])

        for qt in qtypes:
            sub_df = df[df['qtype']==qt]
            qt_report = classification_report(sub_df['answer'], sub_df['prediction'],output_dict=True)
            #precision, recall, fscore, support = score(sub_df['answer'], sub_df['prediction'])
            report_df = pd.DataFrame(qt_report).transpose()
            report_df.to_csv(os.path.join(ROOT_DIR,results_dir,'cls_report_' + qt + '.csv'))
    
            averaging = ['micro','macro','weighted']

            for avg in averaging:
                precision = precision_score(sub_df['answer'], sub_df['prediction'],average=avg )
                recall = recall_score(sub_df['answer'], sub_df['prediction'],average=avg )
                fscore = f1_score(sub_df['answer'], sub_df['prediction'],average=avg )
                score_df = pd.DataFrame([qt,precision, recall,fscore,avg]).transpose()
                score_df.columns=['qtype','precision','recall','fscore','average']
                score_df['model'] = EXP 
                score_df['size'] = SIZE
                score_df['model_category'] = MODEL
                summary_df = pd.concat([summary_df,score_df])


        # get overall type 
        qt = 'all'
        #sub_df = df

        
        averaging = ['micro','macro','weighted']
        for avg in averaging:
            precision = precision_score(df['answer'], df['prediction'],average=avg )
            recall = recall_score(df['answer'], df['prediction'],average=avg )
            fscore = f1_score(df['answer'], df['prediction'],average=avg )
            score_df = pd.DataFrame([qt,precision, recall,fscore,avg]).transpose()
            score_df.columns=['qtype','precision','recall','fscore','average']
            score_df['model'] =  EXP 
            score_df['size'] = SIZE
            score_df['model_category'] = MODEL
            summary_df = pd.concat([summary_df,score_df])

        ic(summary_df)

        summary_df[['precision','recall','fscore']] = summary_df[['precision','recall','fscore']].astype(float).round(3)
        summary_df.to_csv(os.path.join(ROOT_DIR,results_dir,'summary_report.csv'),index=None)
