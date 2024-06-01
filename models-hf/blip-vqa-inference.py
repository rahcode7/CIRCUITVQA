# INFERENCE
import requests
from transformers import AutoProcessor, BlipForQuestionAnswering,BlipImageProcessor
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
import json 
from accelerate import load_checkpoint_and_dispatch
from transformers import BlipConfig

ROOT_DIR = os.getcwd()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--test_dir', help='Test  directory')
    parser.add_argument('--test_batch_size',type=int, help='Check directory')
    parser.add_argument('--checkpoint_dir', help='Check directory')
    parser.add_argument('--results_dir', help='Results directory')
    parser.add_argument('--question_dir', help='Q directory')
    parser.add_argument('--machine_type', help='dp or ddp or single or cpu')
    parser.add_argument('--image_size',type=int,default=384,help=' image size height and width')
    parser.add_argument('--ocr',type=int,default=0,help="ocr prefix")
    parser.add_argument('--desc',type=int,default=0,help="desc prefix")
    parser.add_argument('--bbox',type=int,default=0,help="bbox prefix")
    parser.add_argument('--bbox_segment',type=int,default=0,help="bbox prefix")
    parser.add_argument('--wce',type=int,default=0,help="wce")
    parser.add_argument('--lr',type=int,default=0,help="lr")

    args = parser.parse_args()

    CHECKPOINTS_DIR = args.checkpoint_dir
    TEST_IMAGE_DIR =args.test_dir
    OUTPUT_PATH = args.results_dir 
    Q_PATH = args.question_dir
    TEST_BATCH_SIZE=args.test_batch_size
    MACHINE_TYPE=args.machine_type
    IMAGE_SIZE = args.image_size
    
    WCE_FLAG,LR_FLAG,OCR_FLAG,DESC_FLAG,BBOX_FLAG,BBOX_SEGMENT_FLAG = False,False,False,False,False,False
    
    if args.ocr == 1:WCE_FLAG=True
    if args.lr == 1: LR_FLAG=True 
    if args.ocr == 1:OCR_FLAG=True
    if args.desc == 1:DESC_FLAG=True
    if args.bbox == 1:BBOX_FLAG=True
    if args.bbox_segment == 1:BBOX_SEGMENT_FLAG=True
    if args.wce == 1:WCE_FLAG=True

    # Create results folder
    if not os.path.isdir(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    test_dataset = VQACircuitDataset(TEST_IMAGE_DIR, Q_PATH, split='test',ocr=OCR_FLAG,desc=DESC_FLAG,bbox=BBOX_FLAG,bbox_segment=BBOX_SEGMENT_FLAG)
    test_dataloader = DataLoader(
        test_dataset, shuffle=True, batch_size=TEST_BATCH_SIZE, collate_fn=vqa_collate_fn)

    if WCE_FLAG:
        class_file = os.path.join(os.getcwd(), 'datasets/class_weights.json')
        with open(class_file, 'r') as fp:
            class_weights = json.load(fp)

        configuration = BlipConfig(class_weights=class_weights)
        model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base",config=configuration,ignore_mismatched_sizes=True)
    else:  
        model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base",ignore_mismatched_sizes=True)

    processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
    tokenizer = processor.tokenizer

    ic("model loaded")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cuda"
    ic(device)
    

    if MACHINE_TYPE=='dp':
        model = DataParallel(model)
        #model.module.resize_token_embeddings(len(tokenizer))
    elif MACHINE_TYPE == 'ddp':
        model = DDP(model)
    else:
        pass

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


    # checkpoint = torch.load(os.path.join(
    #     CHECKPOINTS_DIR, 'checkpoint_%02d.pth' % EPOCHS))

    # ic(checkpoint)
    model.load_state_dict(checkpoint['model'],strict=False)
    model.eval()

    # ic(model)
    total = 0
    accuracy_batch = 0.0

    col_list = ['file','splittype', 'question', 'answer', 'prediction','qtype']
    pred_df = pd.DataFrame(columns=col_list)

    for idx, (file_name,images, questions, answers,qtype,desc,ocr,bbox,bbox_segment) in enumerate(tqdm(test_dataloader)):
        total += len(answers)
        inputs = processor(images=images, text=questions,
                           return_tensors="pt", padding=True)
        

        if DESC_FLAG or OCR_FLAG or BBOX_FLAG or BBOX_SEGMENT_FLAG:
            questions_new,tokenizer = get_questions(DESC_FLAG,OCR_FLAG,BBOX_FLAG,BBOX_SEGMENT_FLAG,tokenizer,questions,ocr,desc,bbox,bbox_segment)
            enc = tokenizer.batch_encode_plus(questions_new,return_tensors="pt",padding=True)
            inputs["input_ids"] = enc["input_ids"]
            inputs["attention_mask"] = enc["attention_mask"]

        if IMAGE_SIZE!=384:
                #ic(inputs['pixel_values'].shape)
                im_processor =  BlipImageProcessor(image_size=IMAGE_SIZE)
                inputs_im = im_processor(images=images,size={"height":IMAGE_SIZE,"width":IMAGE_SIZE},return_tensors="pt")
                inputs['pixel_values'] = inputs_im['pixel_values']
                #ic(inputs['pixel_values'].shape)

        
        labels = processor(text=answers, return_tensors="pt",
                           padding=True).input_ids
        inputs["labels"] = labels

        inputs = inputs.to(device)
        labels = labels.to(device)

        if MACHINE_TYPE in  ['dp','ddp']:
            outputs = model.module.generate(**inputs,max_new_tokens=100)
        else:
            outputs = model.generate(**inputs,max_new_tokens=100)
    
        
        predicted = processor.batch_decode(outputs, skip_special_tokens=True)
        accuracy_batch += len([i for i,j in zip(predicted, answers) if i == j])

        # store answers in json or csv
        data = [file_name,questions, answers, predicted,qtype]
        ic(data)
        df = pd.DataFrame(data).transpose()
        df.columns = ['file','question', 'answer', 'prediction','qtype']

        df['splittype'] = 'test'
        pred_df = pd.concat([pred_df, df])

    # #  Compute Accuracy
    test_accuracy = (100 * accuracy_batch / total)
    ic(test_accuracy, total)

    pred_df.to_csv(OUTPUT_PATH + '/predictions.csv', index=None)
