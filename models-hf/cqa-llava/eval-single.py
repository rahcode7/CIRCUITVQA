from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import pandas as pd 
import os 
from icecream import ic 
from tqdm import tqdm
import time 
import argparse
pd.options.mode.chained_assignment = None
import json 
model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--question_dir', help='q directory')
    parser.add_argument('--image_dir', help='imagedirectory')
    parser.add_argument('--results_dir', help='Results directory')
    parser.add_argument('--exp_name', help='Exp name')


    args = parser.parse_args()

    QUESTIONS_PATH = args.question_dir
    IMAGE_DIR =args.image_dir
    OUTPUT_PATH = args.results_dir 
    EXP = args.exp_name

    df = pd.read_json(QUESTIONS_PATH)
    ic(df.head(5))
    

    df = df[df.splittype=="test"]
    #ic(df['symbol'].isna().sum() )
    
    df['symbol'] = df['symbol'].apply(lambda x : " " if x is None else x)
   
    #ic(df['symbol'].isna().sum())

    image_list = df['file'].unique().tolist()

    #col_list = ['splittype', 'file','question', 'answer','qtype','symbol','prediction']
    #pred_df = pd.DataFrame(columns=col_list)
    if EXP == 'desc':  
        df = df[df['qtype'].isin(['count','count-complex'])]
        df['desc'] = df['desc'].apply(lambda x : " " if x is None else x)
    elif EXP == "bbox-segment":
        df['bbox_segment'] = df['bbox_segment'].apply(lambda x : " " if x is None else x)
    else:
        pass

    cnt=0
    index = 0 

    ic(os.path.join(OUTPUT_PATH,'predictions.json'))
    for image_file in tqdm(image_list):
        # cnt +=1
        # if cnt>5:break
        q_df = df[df['file']==image_file] 
        
        file_name = image_file
        image_name = image_file + '.jpg'
        image_path = os.path.join(IMAGE_DIR,'model-inputs','test',image_name)
        
        for _,row in q_df.iterrows():
            with open(os.path.join(OUTPUT_PATH,'predictions.json'),'a+') as f:
                #ic(row)
                if row['qtype'] == 'junction':
                    file_name = image_file
                    image_path = os.path.join(IMAGE_DIR,'model-inputs-jn','test',image_name)
                
                q = row['question']
                answer = row['answer']

                if EXP == 'bbox-segment':
                    context = "Here are the bounding box segment of each component of the given image given in the a pair of component name and segment name. "  + row['bbox_segment']
                    prompt = context +  " Now, Given the image, answer the following question : "
                    #ic(prompt)
                elif EXP == 'base':
                    prompt = "Given the image, answer the following question : "
                
                elif EXP == 'desc':      
                    #ic(row)  
                    if row['symbol'] == "" or row['desc'] == "":
                        context = "The question is about the circuit : " + row['symbol'] + " .Its definition is as following :" + row['desc']
                        prompt = context + " .Now, Given the image, answer the following question : "
                    else: # use base prompt
                        #ic("using base")
                        prompt = "Given the image, answer the following question : "
                elif EXP == 'ocr-pre' or EXP == 'ocr-post':# Ocr - here is the ocr content . You can use it to answer the following question. now, given the image
                    context = "Here is the ocr information " + row['ocr'][0:200] + " You can use it to answer the following question. "
                    prompt = context + " Now, Given the image, answer the following question : "
                elif EXP == 'bbox':

                    context = "Here are the bounding box coordinates of each component of the given image given in the a pair of component name and coordinates "  + row['bbox_segment']
                    prompt = context +  " Now, Given the image, answer the following question : "
                else:
                    pass 

                if "?" not in q:
                    prompt += q + " ? "
                else:
                    prompt += q + " "

                #ic(prompt)

                
                # Desc - the question is about the symbol resistor. Its definition is (add defn). now, given the image ..
                args = type('Args', (), {
                "model_path": model_path,
                "model_base": None,
                "model_name": get_model_name_from_path(model_path),
                "query": prompt,
                "conv_mode": None,
                "image_file": image_path,
                "sep": ",",
                "temperature": 0,
                "top_p": None,
                "num_beams": 1,
                "max_new_tokens": 512,
                "tokenizer":tokenizer,
                "model":model,
                "image_processor": image_processor,
                "context_len":context_len
                })()
                pred = str(eval_model(args))
                #ic(prompt,pred,index)
        
                # pred_df.loc[index] = [row['splittype'],row['file'],row['question'],row['answer'],row['qtype'],row['symbol'],pred]
                # index+=1

                # Write row as dict and in json
                d = {'splittype' : row['splittype'], 'file': row['file'],'question'  : row['question'], 'answer':row['answer'],
                    'symbol'    : row['symbol'],   'qtype' : row['qtype'],'prediction':pred
                    }
                
                #ic(d)
                json.dump(d,f)
                f.write('\n')
                f.close()
                
                #ic("written json")

    #pred_df.to_csv(os.path.join(OUTPUT_PATH,'predictions.csv'), index=None)

    end_time = time.time()
    elapsed_time = end_time - start_time
    ic("Time taken",elapsed_time) # ,pred_df.shape[0])  




            


