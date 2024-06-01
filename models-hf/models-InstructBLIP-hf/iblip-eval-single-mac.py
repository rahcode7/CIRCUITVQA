from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests
import pandas as pd 
import os 
from icecream import ic 
from tqdm import tqdm
import time 
import argparse
pd.options.mode.chained_assignment = None
import json 

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


    model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
    #model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b",load_in_4bit=True, torch_dtype=torch.float16)
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # uncomment for MAC, comment for MS
    model.to(device) 
        
    df = pd.read_json(QUESTIONS_PATH)
    ic(df.head(5))

    df = df[df.splittype=="test"]
    
    df['symbol'] = df['symbol'].apply(lambda x : " " if x is None else x)

    image_list = df['file'].unique().tolist()

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

    cnt = 0 
    for image_file in tqdm(image_list):
        q_df = df[df['file']==image_file] 
        
        file_name = image_file
        image_name = image_file + '.jpg'
        image_path = os.path.join(IMAGE_DIR,'model-inputs','test',image_name)
        image = Image.open(image_path).convert("RGB")
        for _,row in q_df.iterrows():
            #cnt+=1 
            # if cnt <= 160:
            #     continue
            # else:
            with open(os.path.join(OUTPUT_PATH,'predictions.json'),'a+') as f:
                #ic(row)
                if row['qtype'] == 'junction':
                    file_name = image_file
                    image_path = os.path.join(IMAGE_DIR,'model-inputs-jn','test',image_name)
                    image = Image.open(image_path).convert("RGB")
                    
                q = row['question']
                answer = row['answer']
                prompt = ""

                if EXP == 'bbox-segment':
                    context = "Here are the bounding box segment of each component of the given image given in the a pair of component name and segment name. "  + row['bbox_segment']
                    prompt = context + " "
                    #ic(prompt)
                elif EXP == 'base':
                    pass
                    #prompt = "Given the image, answer the following question."
                elif EXP == 'desc':      
                    #ic(row)  
                    if row['desc']:
                        context = "The question is about the circuit : " + row['symbol'] + " .Its definition is as following :" + row['desc']
                        prompt = context + " "
                    else: # use base prompt
                        #ic("using base")
                        #prompt = "Given the image, answer the following question : "
                        pass
                elif EXP == 'ocr-pre' or EXP == 'ocr-post':# Ocr - here is the ocr content . You can use it to answer the following question. now, given the image
                    context = "Here is the ocr information " + row['ocr'][0:200] + " You can use it to answer the following question. "
                    prompt = context + " "
                elif EXP == 'bbox':
                    context = "Here are the bounding box coordinates of each component of the given image given in the a pair of component name and coordinates "  + row['bbox'][0:200]
                    prompt = context + " "
                else:
                    print("select 1 model first")

                if "?" not in q:
                    prompt += q + " ? "
                else:
                    prompt += q.split("?")[0] + " ? "
                
                #inputs = processor(images=image, text=prompt, return_tensors="pt").to(device=device, dtype=torch.float16)
                inputs = processor(images=image, text=prompt, return_tensors="pt").to(device=device) # MAC               
                ic(prompt)
                
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=5,
                    max_length=256,
                    min_length=1,
                    top_p=0.9,
                    repetition_penalty=1.5,
                    length_penalty=1.0,
                    temperature=0,
                )
                pred = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
               
                #ic(prompt,pred,index,answer)
        
                # Write row as dict and in json
                d = {'splittype' : row['splittype'], 'file': row['file'],'question'  : row['question'], 'answer':row['answer'],
                    'symbol'    : row['symbol'],   'qtype' : row['qtype'],'prediction':pred
                    }
                
                #ic(d)
                json.dump(d,f)
                f.write('\n')
                f.close()
            
    end_time = time.time()
    elapsed_time = end_time - start_time
    ic("Time taken",elapsed_time) # ,pred_df.shape[0])  


