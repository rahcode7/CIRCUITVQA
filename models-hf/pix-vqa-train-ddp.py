
import requests
from transformers import AutoProcessor, Pix2StructForConditionalGeneration,Pix2StructConfig,Pix2StructImageProcessor
from datasets import VQACircuitDataset, vqa_collate_fn
from torch.utils.data import DataLoader
import torch
import wandb
from tqdm import tqdm
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import argparse
from utils.helpers import set_seed,get_questions
from icecream import ic
import torch.nn as nn 
import logging
import time 
import random 
import gc
from accelerate import Accelerator
import bitsandbytes as bnb
import json 
import shutil
import site
import logging
from transformers import get_cosine_schedule_with_warmup



def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    ic("prev checkpoint loaded",checkpoint['epoch'])
    return model, optimizer, checkpoint['epoch']

def train(epoch,model,train_dataloader,vqa_processor,text_processor,tokenizer,optimizer,scheduler,device,accelerator,IMAGE_SIZE,DESC_FLAG,OCR_FLAG,BBOX_FLAG,BBOX_SEGMENT_FLAG,LR_FLAG,NUM_GPUS,MACHINE_TYPE,ACCUMULATION_STEPS):
    model.train()
     
    train_losses = []
    train_corrects = []
    train_size = [] 

    for idx, (file_name,images, questions, answers,qtype,desc,ocr,bbox,bbox_segment) in enumerate(tqdm(train_dataloader)):
        with accelerator.accumulate(model):
            
            if DESC_FLAG or OCR_FLAG or BBOX_FLAG or BBOX_SEGMENT_FLAG:
                questions,tokenizer = get_questions(DESC_FLAG,OCR_FLAG,BBOX_FLAG,BBOX_SEGMENT_FLAG,tokenizer,questions,ocr,desc,bbox,bbox_segment)

            inputs = vqa_processor.preprocess(images=images, header_text=questions,return_tensors="pt", add_special_tokens=True)

            labels = text_processor(text=answers, return_tensors="pt", padding=True,add_special_tokens=True).input_ids
            #ic(labels)
            optimizer.zero_grad()
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            #ic(train_loss)
            # if MACHINE_TYPE == 'dp':
            outputs = model.module(**inputs,labels=labels)
            loss = outputs.loss
            train_losses.append(accelerator.gather(outputs.loss))
    
    
            preds = model.module.generate(**inputs,max_new_tokens=100)

            predicted = text_processor.batch_decode(preds, skip_special_tokens=True)
            

            # ic(answers,predicted)
            
            train_batch_corrects = len([i for i,
                                        j in zip(predicted, answers) if i == j])
            train_batch_corrects = torch.tensor(train_batch_corrects).to(device)
            gathered_tensor = accelerator.gather(train_batch_corrects)
            train_corrects.append(gathered_tensor)

            label_size = torch.tensor(len(answers)).to(device)
            gathered_sizes = accelerator.gather(label_size)
            train_size.append(gathered_sizes)
            

            accelerator.backward(loss)

            optimizer.step()

            optimizer.zero_grad()
    
    # Call every epoch
    if LR_FLAG:
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print("Epoch %d: AdamW lr %.10f -> %.10f" % (epoch, before_lr, after_lr))

    total_train_size = torch.sum(torch.cat(train_size)).item()
    total_corrects = torch.sum(torch.cat(train_corrects)).item()
    
    train_loss = torch.sum(torch.cat(train_losses)).item() / len(train_dataloader)
    train_accuracy = total_corrects / total_train_size

    ic(train_loss,train_accuracy,total_corrects,len(train_dataloader),total_train_size)

    return train_loss,train_accuracy


def evaluate(model,val_dataloader,vqa_processor,text_processor,tokenizer,device,accelerator,IMAGE_SIZE,DESC_FLAG,OCR_FLAG,BBOX_FLAG,BBOX_SEGMENT_FLAG,NUM_GPUS,MACHINE_TYPE):

    val_losses = []
    val_corrects = []
    val_size =[]


    with torch.no_grad():
        model.eval()
        for idx, (file_name,images, questions, answers,qtype,desc,ocr,bbox,bbox_segment) in enumerate(tqdm(val_dataloader)):


            if DESC_FLAG or OCR_FLAG:
                questions,tokenizer = get_questions(DESC_FLAG,OCR_FLAG,BBOX_FLAG,BBOX_SEGMENT_FLAG,tokenizer,questions,ocr,desc,bbox,bbox_segment)

            inputs = vqa_processor.preprocess(images=images, header_text=questions,return_tensors="pt", add_special_tokens=True)
            labels = text_processor(text=answers, return_tensors="pt", padding=True,add_special_tokens=True).input_ids

            inputs = inputs.to(device)
            labels = labels.to(device)


            outputs = model.module(**inputs,labels=labels)
            val_losses.append(accelerator.gather(outputs.loss))         

            preds = model.module.generate(**inputs,max_new_tokens=100)
            predicted = text_processor.batch_decode(
                preds, skip_special_tokens=True)
            
            val_batch_corrects  = len([i for i,j in zip(predicted, answers) if i == j])
            val_batch_corrects =  torch.tensor(val_batch_corrects).to(device)
            val_corrects.append(accelerator.gather(val_batch_corrects))

            label_size = torch.tensor(len(answers)).to(device)
            #ic(label_size)
            gathered_sizes = accelerator.gather(label_size)
            val_size.append(gathered_sizes)
            

    total_val_size = torch.sum(torch.cat(val_size)).item()
    total_corrects = torch.sum(torch.cat(val_corrects)).item()

    val_loss = torch.sum(torch.cat(val_losses)).item() / len(val_dataloader)
    val_accuracy = total_corrects / total_val_size

    ic(device, val_accuracy,val_loss, total_corrects,len(val_dataloader),total_val_size)

    return val_loss,val_accuracy    

def run_ddp_accelerate(args):

    EPOCHS = args.num_epochs
    TRAIN_BATCH_SIZE = args.train_batch_size
    VAL_BATCH_SIZE = args.val_batch_size
    TRAIN_DIR = args.train_dir
    VAL_DIR = args.val_dir
    LEARNING_RATE = args.learning_rate
    CHECKPOINT_DIR = args.checkpoint_dir
    EXPERIMENT_NAME = args.experiment_name
    LOCAL_RANK = args.local_rank
    Q_PATH = args.question_dir
    NUM_GPUS = args.ngpus
    WANDB_STATUS = args.wandb_status
    IMAGE_SIZE = args.image_size
    MACHINE_TYPE=args.machine_type
    MAX_PATCHES=args.max_patches
    ic(MACHINE_TYPE,MAX_PATCHES)

    ic(NUM_GPUS)
    ACCUMULATION_STEPS=args.accumulation_steps

    WCE_FLAG,LR_FLAG,OCR_FLAG,DESC_FLAG,BBOX_FLAG,BBOX_SEGMENT_FLAG = False,False,False,False,False,False
    
    if args.wce == 1:WCE_FLAG=True
    if args.lr == 1: LR_FLAG=True 
    if args.ocr == 1:OCR_FLAG=True
    if args.desc == 1:DESC_FLAG=True
    if args.bbox == 1:BBOX_FLAG=True
    if args.bbox_segment == 1:BBOX_SEGMENT_FLAG=True

    ic(WCE_FLAG,LR_FLAG,OCR_FLAG,DESC_FLAG,BBOX_FLAG,BBOX_SEGMENT_FLAG)

    accelerator = Accelerator(gradient_accumulation_steps=ACCUMULATION_STEPS)


  
    # Record start time
    start_time = time.time()
    log_file = os.path.join(CHECKPOINT_DIR,'statistics.log')
    ic(log_file)
    
    if not os.path.exists(log_file):
        open(log_file, 'w').close()
        print("log file created")
    else:
        print("log file exists")


    logging.basicConfig(filename=log_file,level=logging.INFO,filemode="a")
    logger = logging.getLogger("mylogger")

    ngpus = args.ngpus

    ic(TRAIN_BATCH_SIZE, VAL_BATCH_SIZE)

    if accelerator.is_main_process:
        wandb.init(project='CircuitPIX',
                config={
                    'learning_rate': LEARNING_RATE,
                    'epochs': EPOCHS,
                    'train_batch_size': TRAIN_BATCH_SIZE,
                    'val_batch_size': VAL_BATCH_SIZE
                }
                    ,mode=WANDB_STATUS)
        wandb.run.name = EXPERIMENT_NAME


    ic(args.is_distributed)


    text_processor = AutoProcessor.from_pretrained("google/pix2struct-base")
    tokenizer = text_processor.tokenizer
    vqa_processor = Pix2StructImageProcessor(is_vqa=True,max_patches=MAX_PATCHES) #,patch_size={'height':32,'width':32})
    #text_processor.resize_token    

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

    # ic(model.config)


    train_dataset = VQACircuitDataset(TRAIN_DIR, Q_PATH, split='train',ocr=OCR_FLAG,desc=DESC_FLAG,bbox=BBOX_FLAG,bbox_segment=BBOX_SEGMENT_FLAG)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=TRAIN_BATCH_SIZE,drop_last=True, collate_fn= vqa_collate_fn,pin_memory=False)

    val_dataset =  VQACircuitDataset(VAL_DIR, Q_PATH, split='val',ocr=OCR_FLAG,desc=DESC_FLAG,bbox=BBOX_FLAG,bbox_segment=BBOX_SEGMENT_FLAG)
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=VAL_BATCH_SIZE,drop_last=True,collate_fn= vqa_collate_fn,pin_memory=False)

    ic("data loaded")



    #optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    
    
    device = accelerator.device
    ic(device)
    # loss_function = torch.nn.CrossEntropyLoss()

    if LR_FLAG:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=3, num_training_steps=10)
    else:
        scheduler = ""

    
    model.to(device)
    ic(next(model.parameters()).is_cuda)
    #model.train()
   
    best_accuracy = [0, 0.0]

    # Restart from latest checkpoint if available 

    starting_name = 'checkpoint'
    stats_name = 'statistics'

    file_list = os.listdir(CHECKPOINT_DIR)
    ic(file_list)
    START = 0

    if file_list:
        checkpoint_files = [filename for filename in file_list if filename.startswith(starting_name)]
        ic(checkpoint_files)
        # Get latest checkpoint
        if checkpoint_files:
            checkpoint_files.sort()
            LATEST_CHECKPOINT=checkpoint_files[-1]
            START = int(checkpoint_files[-1].split(".pth")[0][-1:])+1
            ic("Previous checkpoint found",LATEST_CHECKPOINT,START)
        
     
    # Check log file for best validation accuracy
    prev_accuracy = []
    prev_epoch = []
    stats = os.path.join(CHECKPOINT_DIR,'statistics.log')

    if os.path.exists(stats):
        with open(stats) as f:
            lines = f.readlines()
            #ic(lines)
            for line in lines:
                if 'Val accuracy' in line:
                    eid =line.index('Epoch') + 6
                    aid =line.index('Val accuracy') + 15

                    prev_epoch.append(int(line[eid]))
                    prev_accuracy.append(float(line[aid:aid+6]))
                    #ic(prev_accuracy,prev_epoch)
    ic(prev_accuracy,prev_epoch)

    if prev_accuracy:
        max_index, max_val = max(enumerate(prev_accuracy),key=lambda x:x[1])
        ic(max_val,max_index)
        best_accuracy = [prev_epoch[max_index],max_val]
        START = prev_epoch[-1] +1
        ic("Previous best accuracy found",best_accuracy[1],"best epoch",best_accuracy[0])
        chk_file = os.path.join(CHECKPOINT_DIR,'checkpoint_0' + str(max_index) + '.pth')
        ic(chk_file)
        model, optimizer,_= load_ckp(chk_file, model, optimizer)
    else:
        # Remove any checkpoints 
        for fname in os.listdir(CHECKPOINT_DIR):
            if fname.startswith("checkpoint"):
                os.remove(os.path.join(CHECKPOINT_DIR,fname))
                ic("removed checkpoint",fname)
            START=0

    ic("Running from epoch",START,"to epoch",EPOCHS)


    
    
    model.resize_token_embeddings(len(tokenizer))
    ic(len(tokenizer))

    train_dataloader, val_dataloader, model, optimizer,scheduler = accelerator.prepare(
        train_dataloader,val_dataloader, model, optimizer,scheduler
    )
    

    

    for epoch in range(START,EPOCHS):
        start_time_epoch = time.time()
        ic(epoch)

        
        train_loss,train_accuracy = train(epoch,model,train_dataloader,vqa_processor,text_processor,tokenizer,optimizer,scheduler,device,accelerator,IMAGE_SIZE,DESC_FLAG,OCR_FLAG,BBOX_FLAG,BBOX_SEGMENT_FLAG,LR_FLAG,NUM_GPUS,MACHINE_TYPE,ACCUMULATION_STEPS)
        val_loss, val_accuracy = evaluate(model,val_dataloader,vqa_processor,text_processor,tokenizer,device,accelerator,IMAGE_SIZE,DESC_FLAG,OCR_FLAG,BBOX_FLAG,BBOX_SEGMENT_FLAG,NUM_GPUS,MACHINE_TYPE)

        logger.info(f'Epoch {epoch} Train loss : {train_loss} Train accuracy : {train_accuracy}')
        logger.info(f'Epoch {epoch} Val loss : {val_loss} Val accuracy : {val_accuracy}')

        ic(epoch,train_loss,val_loss,train_accuracy,val_accuracy)
     
        #wandb.log({'epoch': epoch+1, 'train loss': train_loss})
        if accelerator.is_main_process:
            ic("True")
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'epoch': epoch+1
            })   
        
            # Save model per epoch
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 
                'epoch': epoch
            }
        
            if epoch == 0:   
                ic("Saving 0 epoch checkpoint")
                torch.save(save_obj, os.path.join(
                    CHECKPOINT_DIR, 'checkpoint_%02d.pth' % epoch))
                best_accuracy[0] = epoch 
                best_accuracy[1] = float(val_accuracy)

            elif val_accuracy > best_accuracy[1] and epoch > 0:
                os.remove(os.path.join(CHECKPOINT_DIR,
                        'checkpoint_%02d.pth' % best_accuracy[0]))
                ic("old checkpoint removed")
                best_accuracy[0] = epoch
                best_accuracy[1] = float(val_accuracy)
                torch.save(save_obj, os.path.join(
                    CHECKPOINT_DIR, 'checkpoint_%02d.pth' % epoch))
            else:
                pass

        # Record end time
        end_time_epoch = time.time()
        run_time_epoch = end_time_epoch - start_time_epoch
        logger.info(f'Epoch {epoch} run time: {run_time_epoch:.2f} seconds')
        
        
    if WCE_FLAG:
        src = 'models-hf/pix-files/original/modeling_pix2struct.py'
        dest = os.path.join(SITE_PACKAGES_PATH,'transformers/models/pix2struct','modeling_pix2struct.py')
        shutil.copy(src,dest)

    # Record end time
    end_time = time.time()
    run_time = end_time - start_time
    logger.info(f'Script run time: {run_time:.2f} seconds')



    ic("Everything Completed")


if __name__ == "__main__":

    wandb.login(key='ce18e8ae96d72cd78a7a54de441e9657bc0a913d')

    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=1,
                        type=int, help='number of epochs')
    parser.add_argument('--learning_rate', default=2e-5,
                        type=float, help='Learning rate')
    parser.add_argument('--train_batch_size', default=2,
                        type=int, help='train batch size')
    parser.add_argument('--val_batch_size', default=2,
                        type=int, help='val batch size')
    parser.add_argument('--test_batch_size', default=2,
                        type=int, help='test batch size')
    parser.add_argument('--train_dir', help='train directory')
    parser.add_argument('--val_dir', help='Val directory')
    parser.add_argument('--checkpoint_dir', help='Val  directory')
    parser.add_argument('--experiment_name', help='exp name')
    parser.add_argument('--is_distributed', type=int,
                        default=0, help="DDP enabled")
    # parser.add_argument('--world_size',help='World size') # Enable for ddp
    parser.add_argument('--local_rank', type=int, default=-1,
                        metavar='N', help='Local process rank.')
    parser.add_argument('--question_dir', help='Q directory')
    parser.add_argument('--ngpus',type=int,default=0, help='Q directory')
    parser.add_argument('--machine_type',default='single',help='dp or ddp or single or cpu')
    parser.add_argument('--wandb_status',default='online', help='wandb set to online for online sync else disabled')
    parser.add_argument('--max_patches',type=int,default=1024,help='image size height and width')
    parser.add_argument('--image_size',type=int,default=384,help=' image size height and width')
    parser.add_argument('--ocr',type=int,default=0,help="ocr prefix")
    parser.add_argument('--desc',type=int,default=0,help="desc prefix")
    parser.add_argument('--bbox',type=int,default=0,help="bbox prefix")
    parser.add_argument('--bbox_segment',type=int,default=0,help="bbox prefix")
    parser.add_argument('--accumulation_steps',type=int,default=4,help="acc steps")
    parser.add_argument('--wce',type=int,default=0,help="wce")
    parser.add_argument('--lr',type=int,default=0)




    args = parser.parse_args()

    run_ddp_accelerate(args)

