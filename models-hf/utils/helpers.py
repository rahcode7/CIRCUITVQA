import os 
import torch
import random 
import numpy as np 
from icecream import ic 

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def get_questions(DESC_FLAG,OCR_FLAG,BBOX_FLAG,BBOX_SEGMENT_FLAG,tokenizer,questions,ocr,desc,bbox,bbox_segment):

    questions_new = []
    if BBOX_FLAG and not OCR_FLAG and not BBOX_SEGMENT_FLAG and not DESC_FLAG:
        #print("True")
        for q,b in zip(questions,bbox):
            questions_new.append(str(b)[0:100] + " " + q)
    
    elif BBOX_SEGMENT_FLAG and not OCR_FLAG and not DESC_FLAG and not BBOX_FLAG:
        for q,b in zip(questions,bbox_segment):
            questions_new.append(str(b)[0:200] + " " + q)

        #return questions_new,tokenizer

    elif DESC_FLAG and OCR_FLAG and not BBOX_SEGMENT_FLAG and not BBOX_FLAG: 
        special_tokens_dict = {'additional_special_tokens': ['[OCR]','[DESC]']}
        #ic(special_tokens_dict)
        tokenizer.add_special_tokens(special_tokens_dict)

        for q,o,d in zip(questions,ocr,desc):
            if o and d:
                questions_new.append(str(o)[0:100] + " [OCR] " + d + " [DESC] " + q)
            elif d and not o :
                questions_new.append(str(d) + " [DESC] " + q)
            elif o and not d :
                questions_new.append(str(o)[0:100] + " [OCR] " + q)
            else:
                questions_new.append(q)

    elif DESC_FLAG and not OCR_FLAG and not BBOX_SEGMENT_FLAG and not BBOX_FLAG:# and not desc: 
        special_tokens_dict = {'additional_special_tokens': ['[DESC]']}
        #ic(special_tokens_dict)
        tokenizer.add_special_tokens(special_tokens_dict)
        
        for q,d in zip(questions,desc):
            questions_new.append(str(d)[0:200] + " [DESC] " + q)
    
    elif OCR_FLAG and not DESC_FLAG and not BBOX_SEGMENT_FLAG and not BBOX_FLAG: #and not ocr:
        special_tokens_dict = {'additional_special_tokens': ['[OCR]']}
        tokenizer.add_special_tokens(special_tokens_dict)   

        for q,o in zip(questions,ocr):
            questions_new.append(str(o)[0:100] + " [OCR] " + q)

    elif BBOX_FLAG and BBOX_SEGMENT_FLAG and OCR_FLAG and DESC_FLAG:
        special_tokens_dict = {'additional_special_tokens': ['[OCR]','[DESC]']}
        tokenizer.add_special_tokens(special_tokens_dict)   
        
        for q,o,d,b,bs in zip(questions,ocr,desc,bbox,bbox_segment):
            questions_new.append(str(o)[0:100] + " [OCR] " + str(d) + " [DESC] " + str(bs)[0:100] + str(b)[0:100] + " " + q)

    else:
        questions_new = questions

    return questions_new,tokenizer