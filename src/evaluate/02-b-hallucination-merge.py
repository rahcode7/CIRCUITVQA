import pandas as pd 
from icecream import ic 
import os 

if __name__ == "__main__":
    MODELS =["LLaVA","BLIP","GIT","PIX","InstructBLIP"] #,"GPT4V"]  
    SIZE='384'

    RESULTS_DIR = 'datasets/results'

    df = pd.DataFrame(columns=['exp-name','hs-vqa','hs-count','hs-oodomain','hs-idomain','model'])

    for model in MODELS:
        ic(model)
        # ROOT_DIR = 'models-' + model + '-hf/results-ddp/384a/'
        ROOT_DIR = 'models-' + model + '-hf/results-ddp/' + SIZE  + 'a'

        model_df = pd.read_csv(os.path.join(ROOT_DIR,"hallucination-scores.csv"))
        #df['model'] = model 
            #ic(model_df.head(4))
        df = pd.concat([df,model_df])
        ic(df.shape)


    file_name = 'final-HS-report.csv'
    df.to_csv(os.path.join(RESULTS_DIR,file_name),index=None)
