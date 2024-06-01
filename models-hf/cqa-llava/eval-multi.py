from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import pandas as pd 
import os 
from icecream import ic 
from tqdm import tqdm
import time 
import argparse



#### Code 2
model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

if __name__ == "__main__":
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--question_dir', help='directory')
    parser.add_argument('--datasets_dir', help='Check directory')
    parser.add_argument('--results_dir', help='Results directory')

    args = parser.parse_args()

    QUESTIONS_PATH = args.question_dir
    IMAGE_DIR =args.datasets_dir
    OUTPUT_PATH = args.results_dir 

    df = pd.read_json(QUESTIONS_PATH)
    ic(df.head(5))
    
    df = df[df.splittype=="test"]
    image_list = df['file'].unique().tolist()

    col_list = ['file','splittype', 'question', 'answer', 'prediction','qtype']
    pred_df = pd.DataFrame(columns=col_list)

    cnt=0
    for image_file in tqdm(image_list):
        cnt +=1
        if cnt>10:break
        #ic(cnt)
        
        q_df = df[df['file']==image_file] 
        
        file_name = image_file
        image_file = image_file + '.jpg'
        image_path = os.path.join(IMAGE_DIR,image_file)

        # Split data into subset of 5 questions per batch
        n = 5
        # Split the DataFrame into chunks
        list_df = [q_df[i:i+n] for i in range(0, len(q_df), n)]

        ic(list_df)

        for sub_df in list_df:
            questions = sub_df["question"].tolist()
            answers = sub_df["answer"].tolist()

            prompt = ""
            for q in questions:
                if "?" not in q:
                    prompt += q + "? "
                else:
                    prompt += q + " "
            
            #prompt += "Answer in format Answer 1 to Answer and return answers " + str(len(answers)) # + "separate answers"
            prompt += "Answer in a list format with answer 1 to answer " + str(len(answers)) + "and each answer as one item " # + "separate answers"
            # prompt += "Now Answer in the format - Answer items starting with text answer and answer id and so on for each answer"

            # ic(prompt)

            # args = type('Args', (), {
            # "model_path": model_path,
            # "model_base": None,
            # "model_name": get_model_name_from_path(model_path),
            # "query": prompt,
            # "conv_mode": None,
            # "image_file": image_path,
            # "sep": ",",
            # "temperature": 0.2,
            # "top_p": None,
            # "num_beams": 1,
            # "max_new_tokens": 512
            # })()

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

            ic(pred)
            pred_list = []
            for i in range(1,len(answers)+1):
                if i == len(answers)+1:
                    pred_list.append(pred[pred.find(str(i-1)+"."):])
                else:
                    l = str(i) + "."
                    r = str(i+1) + "."
                    pred_list.append(pred[pred.find(l):pred.find(r)])

            ic(pred_list,len(pred_list),sub_df.shape)
            sub_df['prediction'] = pred_list

            pred_df = pd.concat([pred_df, sub_df])

        pred_df.to_csv(os.path.join(OUTPUT_PATH,OUTPUT_FOLDER) + '/predictions.csv', index=None)

    end_time = time.time()
    elapsed_time = end_time - start_time
    ic("Time taken",elapsed_time)  




            


