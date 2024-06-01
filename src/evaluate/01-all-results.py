import pandas as pd 
from icecream import ic 
import os 



RESULTS_DIR = 'datasets/results'

# Combine all 384
if __name__ == "__main__":


    #MODELS = ['InstructBlip'] # ,'GIT'] # ,'PIX'],LLaVA'
    MODELS = ''
    SIZE = '384'

    df = pd.DataFrame(columns=['qtype','precision','recall','fscore','average','size','model','model_category'])

    for model in MODELS:
        ic(model)
        # ROOT_DIR = 'models-' + model + '-hf/results-ddp/384a/'
        ROOT_DIR = 'models-' + model + '-hf/results-ddp/' + SIZE  + 'a'

        for f in os.listdir(ROOT_DIR):

            if f=='.DS_Store'or f=='hallucination-scores.csv':
                continue
            ic(f)
            model_df = pd.read_csv(os.path.join(ROOT_DIR,f,'summary_report.csv'))
            #ic(model_df.head(4))
            df = pd.concat([df,model_df])

        ic(df.shape)
        file_name = model + '-' + SIZE + '-final-summary-report.csv'

        df.to_csv(os.path.join(RESULTS_DIR,file_name),index=None)
