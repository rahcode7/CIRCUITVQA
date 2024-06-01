
import pandas as pd
import os 
from pathlib import Path
from icecream import ic 
import numpy as np 
import json
from transformers import AutoProcessor, BlipForQuestionAnswering
from tqdm import tqdm
from transformers import BertTokenizer
from collections import Counter
tqdm.pandas()

DATA_PATH = "datasets/questions/"

if __name__ == "__main__":
    # Qs
    df = pd.read_csv(DATA_PATH + "/all/"+"master.csv")
    ic(df.head(10))
    ic(df['answer'].value_counts())
    #answer_df = pd.DataFrame(df['answer'].value_counts()).reset_index()

    answer_df = df 
    ic(answer_df.info())
    #processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

    processor = AutoProcessor.from_pretrained("google/pix2struct-base",local_files_only=True)
    tokenizer = processor.tokenizer 
    ic(tokenizer.vocab_size)
    t5_vocab = tokenizer.get_vocab()

    with open('datasets/questions/all/t5_vocab.json','w') as fp:
        json.dump(t5_vocab,fp)
        
    ic(tokenizer.sep_token_id)
    vocab_size = processor.tokenizer.vocab_size
    ic(vocab_size,len(tokenizer))

    #with open('')
    # vocab_size = len(processor.tokenizer.word_index) + 1
    # ic(vocab_size)

    #labels = processor(text=answer_df['answer'], return_tensors="pt", padding=True).input_ids
    
    answer_df['tokens'] = answer_df['answer'].progress_apply(lambda x : processor(text=str(x), padding=True).input_ids)  #.input_ids[:,1:][0].tolist())
    answer_df.to_csv("datasets/questions/all/answer_tokens-pix.csv",index=None)
    ic(answer_df.head(5))

    token_counter = Counter()
    for line in list(answer_df['tokens']):
        #ic(line)
        for token in line:
            #ic(token)
            token_counter[token]+=1
            #token_counter.update([token])

    tc  = dict(Counter(token_counter))
    

    with open('datasets/questions/all/token_counter-pix.json', 'w') as fp:
        json.dump(tc, fp)

    ic(token_counter)
    ic(len(token_counter))
    vocabulary = list(token_counter.keys())
    
    # token_counts = [token_counter[token] for token in vocabulary]
    # ic(token_counts)


    class_weights = [0] * vocab_size 
    # ic(answer_df.shape[0])            
    for element,count in token_counter.items():
        class_weights[element] = round(1/count,6)

    ic(len(class_weights))
    #     if element in [101,102]:
    #         class_weights[element] = round(1/answer_df.shape[0],6)
    #     else:
    #         class_weights[element] = round(1/count,6)
    
    # class_weights[101] = round(1/answer_df.shape[0],6)
    
    #ic(class_weights[0:110])

    

    #answer_df['weights']= 100/answer_df['count']
    # ic(answer_df.head(20)) 
    # ic(answer_df.tail(20))

    # answer_df['tokens'] = answer_df['tokens'].apply(str)

    #d = pd.Series(answer_df.weights.values,index=answer_df.tokens).to_dict()
    # ic(d)
    with open('datasets/questions/all/class_weights-pix.json', 'w') as fp:
        json.dump(class_weights, fp)


