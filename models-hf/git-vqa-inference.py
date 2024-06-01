# INFERENCE
import requests
#from transformers import AutoProcessor, BlipForQuestionAnswering
from transformers import AutoProcessor,AutoModelForCausalLM
from datasets_git import VQACircuitDataset, vqa_collate_fn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel
import torch
import wandb
from tqdm import tqdm
import os
from transformers import GitConfig, GitModel
from utils.helpers import set_seed,get_questions
from icecream import ic
import pandas as pd
import os 
import argparse
import json 
import shutil
import site

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
    IMAGE_SIZE = args.image_size
    MACHINE_TYPE=args.machine_type

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

    test_dataset = VQACircuitDataset(TEST_IMAGE_DIR, Q_PATH,split='test',ocr=OCR_FLAG,desc=DESC_FLAG,bbox=BBOX_FLAG,bbox_segment=BBOX_SEGMENT_FLAG)
    test_dataloader = DataLoader(
    test_dataset, shuffle=True, batch_size=TEST_BATCH_SIZE, collate_fn=vqa_collate_fn)

    #model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")
    # configuration = GitConfig(vision_config={'image_size':IMAGE_SIZE})
    # ic(configuration)
    # model = AutoModelForCausalLM.from_pretrained("microsoft/git-base",config=configuration,ignore_mismatched_sizes=True)


    SITE_PACKAGES_PATH = site.getsitepackages()[0]
    src = 'models-hf/git-files/clip/image_processing_clip.py'
    dest = os.path.join(SITE_PACKAGES_PATH,'transformers/models/clip')
    shutil.copy(src,dest)
    ic("Base file copied to transformers lib")

    if WCE_FLAG:
        # Copy wce files
        src = 'models-hf/git-files/wce/configuration_git.py'
        dest = os.path.join(SITE_PACKAGES_PATH,'transformers/models/git')
        shutil.copy(src,dest)

        src = 'models-hf/git-files/wce/modeling_git_wce.py'
        shutil.copy(src,dest)

        ic("copied wce files")
        class_file = os.path.join(os.getcwd(), 'datasets/class_weights.json')
        with open(class_file, 'r') as fp:
            class_weights = json.load(fp)

        configuration = GitConfig(vision_config={'image_size':IMAGE_SIZE},class_weights=class_weights)
        model = AutoModelForCausalLM.from_pretrained("microsoft/git-base",config=configuration,ignore_mismatched_sizes=True)

    else:
        # Get original file of wce back if changes
        src = 'models-hf/git-files/original/configuration_git.py'
        dest = os.path.join(SITE_PACKAGES_PATH,'transformers/models/git','configuration_git.py')
        shutil.copy(src,dest)
        

        src = 'models-hf/git-files/original/modeling_git.py'
        dest = os.path.join(SITE_PACKAGES_PATH,'transformers/models/git','modeling_git.py')
        shutil.copy(src,dest)

        ic("copied original files of git back")
        configuration = GitConfig(vision_config={'image_size':IMAGE_SIZE})
        model = AutoModelForCausalLM.from_pretrained("microsoft/git-base",config=configuration,ignore_mismatched_sizes=True)
   
    processor = AutoProcessor.from_pretrained("microsoft/git-base")
    tokenizer = processor.tokenizer

    special_tokens_dict = {'additional_special_tokens': ['[OCR]','[DESC]']}
    tokenizer.add_special_tokens(special_tokens_dict)
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ic(device)


    starting_name = 'checkpoint'
    file_list = os.listdir(CHECKPOINTS_DIR)
    ic(file_list)

    checkpoint_files = [filename for filename in file_list if filename.startswith(starting_name)]
    checkpoint_files.sort()

    checkpoint_file = checkpoint_files[-1]
    ic(checkpoint_file)

    if MACHINE_TYPE=='dp':
        model = DataParallel(model)
        model.module.resize_token_embeddings(len(tokenizer))

    elif MACHINE_TYPE == 'ddp':
        model = DDP(model)
    else:
        pass 

    if device == "cpu":
        checkpoint = torch.load(os.path.join(CHECKPOINTS_DIR,checkpoint_file),map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(os.path.join(CHECKPOINTS_DIR,checkpoint_file))

    model.to(device)
    ic("checkpoint file loaded",checkpoint_file)

    # ic(checkpoint)
    model.load_state_dict(checkpoint['model'],strict=False,)
    model.eval()

    # ic(model)
    total = 0
    accuracy_batch = 0.0

    col_list = ['file','splittype', 'question', 'answer', 'prediction','qtype']
    pred_df = pd.DataFrame(columns=col_list)

    for idx, (file_name,images, questions, answers,question_answers,qtype,desc,ocr,bbox,bbox_segment) in enumerate(tqdm(test_dataloader)):
        #ic(questions)

        total += len(answers)
        inputs = processor(images=images,size=(IMAGE_SIZE,IMAGE_SIZE),do_resize=True,do_center_crop=False,return_tensors="pt").to(device)   
        
        pixel_values = inputs.pixel_values.to(device)

        # special_tokens_dict = {'additional_special_tokens': ['[OCR]','[DESC]']}
        # tokenizer.add_special_tokens(special_tokens_dict)
        
        if DESC_FLAG or OCR_FLAG or BBOX_FLAG or BBOX_SEGMENT_FLAG:
            #ic("inside ocr")
            qustions,tokenizer = get_questions(DESC_FLAG,OCR_FLAG,BBOX_FLAG,BBOX_SEGMENT_FLAG,tokenizer,questions,ocr,desc,bbox,bbox_segment)
            # Append batch answers 
            question_answers = [q + " " + a for q,a in zip(questions,answers)]

        input_ids = processor(text=question_answers, add_special_tokens=True,padding=True,return_tensors="pt") # .input_ids.to(device)
        input_ids["input_ids"] = input_ids["input_ids"].to(device)
        input_ids["attention_mask"] = input_ids["attention_mask"].to(device)
        model.module.resize_token_embeddings(len(tokenizer))

        #Pass the 101 token and no 102 token for inference
        #ic(input_ids,input_ids.shape)
        input_ids_qa = processor(text=questions, add_special_tokens=False,padding=True,return_tensors="pt").input_ids
        add_cls_token = processor.tokenizer.cls_token_id
        #ic(input_ids_qa)
        

        cls_tensor = torch.tensor([[add_cls_token]]* input_ids_qa.shape[0],dtype=input_ids_qa.dtype)
        input_ids_qa = torch.cat((cls_tensor,input_ids_qa),dim=1).to(device)
        #ic(input_ids,input_ids.shape)

        input_ids_qa = input_ids_qa.to(device)

        if MACHINE_TYPE in ['dp','ddp']:
            preds = model.module.generate(pixel_values=pixel_values,input_ids=input_ids_qa, max_new_tokens=100)
        else:
            preds = model.generate(pixel_values=pixel_values,input_ids=input_ids_qa, max_new_tokens=100)

        #ic(preds)
        predicted_caption = processor.batch_decode(preds, skip_special_tokens=True)
        #ic(predicted_caption)

        predicted  = [item.split("?")[-1].strip() for item in predicted_caption] 
        accuracy_batch += len([i for i,j in zip(predicted, answers) if i == j])

        #ic(predicted,answers,accuracy_batch,total,accuracy_batch/total)
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


    if WCE_FLAG:
        src = 'models-hf/git-files/original/configuration_git.py'
        dest = os.path.join(SITE_PACKAGES_PATH,'transformers/models/git','configuration_git.py')
        shutil.copy(src,dest)

        src = 'models-hf/git-files/original/modeling_git.py'
        dest = os.path.join(SITE_PACKAGES_PATH,'transformers/models/git','modeling_git.py')
        shutil.copy(src,dest)

    ic("Inference Completed")
