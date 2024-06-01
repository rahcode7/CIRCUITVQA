# INFERENCE
import requests
from transformers import AutoProcessor, Pix2StructForConditionalGeneration,Pix2StructConfig,Pix2StructImageProcessor
from datasets import VQACircuitDataset, vqa_collate_fn
from torch.utils.data import DataLoader
import torch
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.helpers import get_questions
import wandb
from tqdm import tqdm
import os
from icecream import ic
import pandas as pd
import os 
import argparse
import shutil
import site
import json 

ROOT_DIR = os.getcwd()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--test_dir', help='Test  directory')
    parser.add_argument('--test_batch_size',type=int, help='Check directory')
    parser.add_argument('--checkpoint_dir', help='Check directory')
    parser.add_argument('--results_dir', help='Results directory')
    parser.add_argument('--question_dir', help='Q directory')
    parser.add_argument('--machine_type', help='dp or ddp or single or cpu')
    parser.add_argument('--max_patches',type=int,default=1024,help='image size height and width')
    parser.add_argument('--image_size',type=int,default=384,help=' image size height and width')
    parser.add_argument('--ocr',type=int,default=0,help="ocr prefix")
    parser.add_argument('--desc',type=int,default=0,help="desc prefix")
    parser.add_argument('--bbox',type=int,default=0,help="bbox prefix")
    parser.add_argument('--bbox_segment',type=int,default=0,help="bbox prefix")
    parser.add_argument('--wce',type=int,default=0,help="wce")
    parser.add_argument('--lr',type=int,default=0)


    args = parser.parse_args()

    CHECKPOINTS_DIR = args.checkpoint_dir
    TEST_IMAGE_DIR =args.test_dir
    OUTPUT_PATH = args.results_dir 
    Q_PATH = args.question_dir
    TEST_BATCH_SIZE=args.test_batch_size
    MACHINE_TYPE=args.machine_type
    IMAGE_SIZE = args.image_size
    MAX_PATCHES = args.max_patches
    

    WCE_FLAG,LR_FLAG,OCR_FLAG,DESC_FLAG,BBOX_FLAG,BBOX_SEGMENT_FLAG = False,False,False,False,False,False
    
    if args.wce == 1:WCE_FLAG=True
    if args.lr == 1: LR_FLAG=True 
    if args.ocr == 1:OCR_FLAG=True
    if args.desc == 1:DESC_FLAG=True
    if args.bbox == 1:BBOX_FLAG=True
    if args.bbox_segment == 1:BBOX_SEGMENT_FLAG=True

    # Create results folder
    if not os.path.isdir(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    SITE_PACKAGES_PATH = site.getsitepackages()[0]
    src = 'models-hf/pix-files/image_processing_pix2struct.py'
    dest = os.path.join(SITE_PACKAGES_PATH,'transformers/models/pix2struct','image_processing_pix2struct.py')
    shutil.copy(src,dest)

    if WCE_FLAG:
        src = 'models-hf/pix-files/modeling_pix2struct.py'
        dest = os.path.join(SITE_PACKAGES_PATH,'transformers/models/pix2struct')
        shutil.copy(src,dest)

        src = 'models-hf/pix-files/configuration_pix2struct.py'
        dest = os.path.join(SITE_PACKAGES_PATH,'transformers/models/pix2struct')
        shutil.copy(src,dest)

        ic("copied wce files")
        class_file = os.path.join(os.getcwd(), 'models-hf/pix-files/class_weights-pix.json')
        with open(class_file, 'r') as fp:
            class_weights = json.load(fp)
        #ic(len(class_weights))

        configuration = Pix2StructConfig(text_config={'class_weights':class_weights},vision_config={'num_hidden_layers' : 12 ,"hidden_size":768,'seq_len':4096})
        #ic(len(tokenizer))
        model =  Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-base",config=configuration,ignore_mismatched_sizes=True) #,local_files_only=True)
        # model.resize_token_embeddings(len(tokenizer))
        
    else:
        # Get original files back 
        ic("copied original files")
        src = 'models-hf/pix-files/original/modeling_pix2struct.py'
        dest = os.path.join(SITE_PACKAGES_PATH,'transformers/models/pix2struct')
        shutil.copy(src,dest)

        ic("copied original files")
        src = 'models-hf/pix-files/original/configuration_pix2struct.py'
        dest = os.path.join(SITE_PACKAGES_PATH,'transformers/models/pix2struct')
        shutil.copy(src,dest)

        configuration = Pix2StructConfig(vision_config={'num_hidden_layers' : 12 ,"hidden_size":768,'seq_len':4096})
        model =  Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-base",config=configuration,ignore_mismatched_sizes=True) #,local_files_only=True)


    test_dataset = VQACircuitDataset(TEST_IMAGE_DIR, Q_PATH, split='test',ocr=OCR_FLAG,desc=DESC_FLAG,bbox=BBOX_FLAG,bbox_segment=BBOX_SEGMENT_FLAG)
    test_dataloader = DataLoader(
        test_dataset, shuffle=True, batch_size=TEST_BATCH_SIZE, collate_fn=vqa_collate_fn)

    text_processor = AutoProcessor.from_pretrained("google/pix2struct-base")
    tokenizer = text_processor.tokenizer
    
    ## Use as a bug fix for token size mismatch for ocr,desc codes
    # special_tokens_dict = {'additional_special_tokens': ['[OCR]']} 
    # tokenizer.add_special_tokens(special_tokens_dict)

    vqa_processor = Pix2StructImageProcessor(is_vqa=True,max_patches=MAX_PATCHES) #,patch_size={'height':32,'width':32})


    device = "cuda" if torch.cuda.is_available() else "cpu"
    ic(device)
    if MACHINE_TYPE=='dp':
        model = DataParallel(model)
        model.module.resize_token_embeddings(len(tokenizer))
    elif MACHINE_TYPE == 'ddp':
        model = DDP(model)
        model.module.resize_token_embeddings(len(tokenizer))
    else:
        model.resize_token_embeddings(len(tokenizer))

    ic(len(tokenizer))

    model.to(device)
    
    starting_name = 'checkpoint'
    file_list = os.listdir(CHECKPOINTS_DIR)
    ic(file_list)

    checkpoint_files = [filename for filename in file_list if filename.startswith(starting_name)]
    checkpoint_files.sort()

    checkpoint_file = checkpoint_files[-1]
    ic(checkpoint_file)

    checkpoint = torch.load(os.path.join(CHECKPOINTS_DIR,checkpoint_file))
    ic("checkpoint file loaded",checkpoint_file)

    # ic(checkpoint)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # ic(model)
    total = 0
    accuracy_batch = 0.0

    
    col_list = ['file','splittype', 'question', 'answer', 'prediction','qtype']
    pred_df = pd.DataFrame(columns=col_list)

    # model.resize_token_embeddings(len(tokenizer))
    for idx, (file_name,images, questions, answers,qtype,desc,ocr,bbox,bbox_segment) in enumerate(tqdm(test_dataloader)):
        total += len(answers)
        

        if DESC_FLAG or OCR_FLAG or BBOX_FLAG or BBOX_SEGMENT_FLAG:
            questions,tokenizer = get_questions(DESC_FLAG,OCR_FLAG,BBOX_FLAG,BBOX_SEGMENT_FLAG,tokenizer,questions,ocr,desc,bbox,bbox_segment)
        
        inputs = vqa_processor.preprocess(images=images, header_text=questions,return_tensors="pt", add_special_tokens=True)
        labels = text_processor(text=answers, return_tensors="pt", padding=True,add_special_tokens=True).input_ids

        inputs = inputs.to(device)
        labels = labels.to(device)

        #outputs = model(**inputs,labels=labels)

        if MACHINE_TYPE == 'dp': 
            preds = model.module.generate(**inputs,max_new_tokens=100)
        else:
            preds = model.generate(**inputs,max_new_tokens=100)

        predicted = text_processor.batch_decode(preds, skip_special_tokens=True)
        accuracy_batch += len([i for i,j in zip(predicted, answers) if i == j])

        # store answers in json or csv
        data = [file_name,questions, answers, predicted,qtype]
        #ic(data)
        df = pd.DataFrame(data).transpose()
        df.columns = ['file','question', 'answer', 'prediction','qtype']

        df['splittype'] = 'test'
        pred_df = pd.concat([pred_df, df])

    # #  Compute Accuracy
    test_accuracy = (100 * accuracy_batch / total)
    ic(test_accuracy, total)

    pred_df.to_csv(OUTPUT_PATH + '/predictions.csv', index=None)
