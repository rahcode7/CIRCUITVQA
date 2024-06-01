
import requests
from transformers import AutoProcessor, BlipForQuestionAnswering,BlipImageProcessor
from  datasets  import VQACircuitDataset, vqa_collate_fn
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


def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    ic("prev checkpoint loaded",checkpoint['epoch'])
    return model, optimizer, checkpoint['epoch']

def train(model,train_dataloader,processor,im_processor,tokenizer,optimizer,device,IMAGE_SIZE,DESC_FLAG,OCR_FLAG,BBOX_FLAG,BBOX_SEGMENT_FLAG,NUM_GPUS,MACHINE_TYPE):
    model.train()
    num_batches = 0
    total = 0 
    train_loss = 0.0
    accuracy_batch = 0 

    #batch_time = []
    for idx, (files,images, questions, answers,qtype,desc,ocr,bbox,bbox_segment) in enumerate(tqdm(train_dataloader)):
        total += len(answers)
        num_batches += 1
        #ic(idx, images, questions, answers)

        inputs = processor(images=images, text=questions,return_tensors="pt", padding=True)  

        if DESC_FLAG or OCR_FLAG or BBOX_FLAG or BBOX_SEGMENT_FLAG:

            questions_new,tokenizer = get_questions(DESC_FLAG,OCR_FLAG,BBOX_FLAG,BBOX_SEGMENT_FLAG,tokenizer,questions,ocr,desc,bbox,bbox_segment)
            #ic(questions_new)
            enc = tokenizer.batch_encode_plus(questions_new,return_tensors="pt",padding=True)
            inputs["input_ids"] = enc["input_ids"]
            inputs["attention_mask"] = enc["attention_mask"]
            
        
        if IMAGE_SIZE!=384:
            #ic(inputs['pixel_values'].shape)
            
            inputs_im = im_processor(images=images,size={"height":IMAGE_SIZE,"width":IMAGE_SIZE},return_tensors="pt")
            inputs['pixel_values'] = inputs_im['pixel_values']
            #ic(inputs['pixel_values'].shape)
        
        labels = processor(text=answers, return_tensors="pt", padding=True).input_ids
        inputs["labels"] = labels

        optimizer.zero_grad()
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(**inputs)

        loss = outputs.loss

        #ic(train_loss)
        if MACHINE_TYPE == 'dp':
            train_loss += loss.sum().item()
            preds = model.module.generate(**inputs,max_new_tokens=100)
        else:
            #ic(loss,MACHINE_TYPE)
            train_loss += loss.item()
            preds = model.generate(**inputs,max_new_tokens=100)

        predicted = processor.batch_decode(preds, skip_special_tokens=True)

        #ic(questions,answers,predicted)
        #ic(predicted)
        accuracy_batch += len([i for i,
                                    j in zip(predicted, answers) if i == j])
        #ic(loss.sum().item())

        if MACHINE_TYPE == 'dp':
            loss.sum().backward()
        else:
            loss.backward()

        optimizer.step()

    train_loss = train_loss / (num_batches*NUM_GPUS)
    train_accuracy = (100 * accuracy_batch / total)

    return train_loss,train_accuracy


def evaluate(model,val_dataloader,processor,im_processor,tokenizer,device,IMAGE_SIZE,DESC_FLAG,OCR_FLAG,BBOX_FLAG,BBOX_SEGMENT_FLAG,NUM_GPUS,MACHINE_TYPE):
    ic(BBOX_FLAG)
    model.eval()
    num_batches = 0
    total = 0 
    val_loss = 0.0
    accuracy_batch = 0

    with torch.no_grad():
        for idx, (files,images, questions, answers,qtype,desc,ocr,bbox,bbox_segment) in enumerate(tqdm(val_dataloader)):
            num_batches += 1
            total += len(answers)
                
            inputs = processor(images=images, text=questions,return_tensors="pt", padding=True)  

            if DESC_FLAG or OCR_FLAG or BBOX_FLAG or BBOX_SEGMENT_FLAG:
                #ic("Yes",DESC_FLAG,OCR_FLAG,BBOX_FLAG,BBOX_SEGMENT_FLAG)
                questions_new,tokenizer = get_questions(DESC_FLAG,OCR_FLAG,BBOX_FLAG,BBOX_SEGMENT_FLAG,tokenizer,questions,ocr,desc,bbox,bbox_segment)
                #ic(questions_new)
                enc = tokenizer.batch_encode_plus(questions_new,return_tensors="pt",padding=True)
                inputs["input_ids"] = enc["input_ids"]
                inputs["attention_mask"] = enc["attention_mask"]
                
            inputs = processor(images=images, text=questions,return_tensors="pt", padding=True)

            if IMAGE_SIZE!=384:
                inputs_im = im_processor(images=images,size={"height":IMAGE_SIZE,"width":IMAGE_SIZE},return_tensors="pt")
                inputs['pixel_values'] = inputs_im['pixel_values']
                #ic(inputs['pixel_values'].shape)


            labels = processor(
                text=answers, return_tensors="pt", padding=True).input_ids
            inputs["labels"] = labels
            inputs = inputs.to(device)
            #labels = labels.to(device)

            outputs = model(**inputs)
            outputs = outputs
            loss = outputs.loss            

            if MACHINE_TYPE == 'dp':
                #ic(loss,MACHINE_TYPE)
                val_loss += loss.sum().item()
                preds = model.module.generate(**inputs,max_new_tokens=100)
            else:
                #ic(loss,MACHINE_TYPE)
                val_loss += loss.item()
                preds = model.generate(**inputs,max_new_tokens=100)

            
            predicted = processor.batch_decode(
                preds, skip_special_tokens=True)
            #ic(questions,answers,predicted)
            accuracy_batch += len([i for i,
                                    j in zip(predicted, answers) if i == j])
                

    val_loss = val_loss / (num_batches*NUM_GPUS)
    val_accuracy = (100 * accuracy_batch / total)

    return val_loss,val_accuracy


if __name__ == "__main__":

    wandb.login(key='ce18e8ae96d72cd78a7a54de441e9657bc0a913d')
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=1,
                        type=int, help='number of epochs')
    parser.add_argument('--learning_rate', default=2e-5,
                        type=int, help='Learning rate')
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
    parser.add_argument('--image_size',type=int,default=384,help=' image size height and width')
    parser.add_argument('--ocr',type=int,default=0,help="ocr prefix")
    parser.add_argument('--desc',type=int,default=0,help="desc prefix")
    parser.add_argument('--bbox',type=int,default=0,help="bbox prefix")
    parser.add_argument('--bbox_segment',type=int,default=0,help="bbox prefix")

    args = parser.parse_args()

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
    
    OCR_FLAG=False
    if args.ocr == 1:
        OCR_FLAG=True

    DESC_FLAG=False
    if args.desc == 1:
        DESC_FLAG=True

    BBOX_FLAG=False
    if args.bbox == 1:
        BBOX_FLAG=True

    BBOX_SEGMENT_FLAG=False
    if args.bbox_segment == 1:
        BBOX_SEGMENT_FLAG=True

    ic(OCR_FLAG,DESC_FLAG)
    ic(MACHINE_TYPE)


    ic(NUM_GPUS,os.environ["CUDA_VISIBLE_DEVICES"])

    ic(os.environ["CUDA_VISIBLE_DEVICES"])
    gpus_avail = torch.cuda.device_count()
    if gpus_avail < NUM_GPUS:
        NUM_GPUS = gpus_avail
        ic("Only these gpus available",NUM_GPUS)
    

    device_ids = []
    for i in range(NUM_GPUS):
        device_ids.append(i)

    # Record start time
    start_time = time.time()
    log_file = os.path.join(CHECKPOINT_DIR,'statistics.log')
    ic(log_file)
    
    if not os.path.exists(log_file):
        open(log_file, 'w').close()
        print("log file created")
    else:
        print("log file exists")


    logging.basicConfig(filename=log_file,level=logging.INFO,filemode="a") # #,
    logger = logging.getLogger("mylogger")

    ngpus = args.ngpus

    ic(TRAIN_BATCH_SIZE, VAL_BATCH_SIZE)

    wandb.init(project='CircuitBLIP',
            config={
                'learning_rate': LEARNING_RATE,
                'epochs': EPOCHS,
                'train_batch_size': TRAIN_BATCH_SIZE,
                'val_batch_size': VAL_BATCH_SIZE
            }
                ,mode=WANDB_STATUS)
    wandb.run.name = EXPERIMENT_NAME

    ic(args.is_distributed)

    # model = BlipForQuestionAnswering.from_pretrained(
    #     "Salesforce/blip-vqa-base")
    #model.config.to_json_file("config.json")

    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base",ignore_mismatched_sizes=True,local_files_only=False) # ,config=config_file,ignore_mismatched_sizes=True)
    processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
    im_processor =  BlipImageProcessor(image_size=IMAGE_SIZE)
    tokenizer = processor.tokenizer


    print(model.config)
    torch.cuda.empty_cache()

    #if DESC_FLAG or OCR_FLAG:
    train_dataset = VQACircuitDataset(TRAIN_DIR, Q_PATH, split='train',ocr=OCR_FLAG,desc=DESC_FLAG,bbox=BBOX_FLAG,bbox_segment=BBOX_SEGMENT_FLAG)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=TRAIN_BATCH_SIZE, collate_fn= vqa_collate_fn,pin_memory=False)

    val_dataset =  VQACircuitDataset(VAL_DIR, Q_PATH, split='val',ocr=OCR_FLAG,desc=DESC_FLAG,bbox=BBOX_FLAG,bbox_segment=BBOX_SEGMENT_FLAG)
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=VAL_BATCH_SIZE, collate_fn= vqa_collate_fn,pin_memory=False)

    ic("data loaded")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ic(device)
    loss_function = torch.nn.CrossEntropyLoss()

    
    if MACHINE_TYPE=='dp':
        model = nn.DataParallel(model,device_ids=device_ids)


    model.to(device)
    ic(next(model.parameters()).is_cuda)
    model.train()

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


    ic(DESC_FLAG,OCR_FLAG,BBOX_FLAG,BBOX_SEGMENT_FLAG)

    for epoch in range(START,EPOCHS):
    #for epoch in range(EPOCHS):
        start_time_epoch = time.time()
        ic(epoch)
        train_loss,train_accuracy = train(model,train_dataloader,processor,im_processor,tokenizer,optimizer,device,IMAGE_SIZE,DESC_FLAG,OCR_FLAG,BBOX_FLAG,BBOX_SEGMENT_FLAG,NUM_GPUS,MACHINE_TYPE)
        val_loss, val_accuracy = evaluate(model,val_dataloader,processor,im_processor,tokenizer,device,IMAGE_SIZE,DESC_FLAG,OCR_FLAG,BBOX_FLAG,BBOX_SEGMENT_FLAG,NUM_GPUS,MACHINE_TYPE)

        logger.info(f'Epoch {epoch} Train loss : {train_loss} Train accuracy : {train_accuracy}')
        logger.info(f'Epoch {epoch} Val loss : {val_loss} Val accuracy : {val_accuracy}')

        ic(train_loss,train_accuracy,val_loss,val_accuracy)
     

            
        wandb.log({'epoch': epoch+1, 'train loss': train_loss})
        wandb.log({'epoch': epoch+1, 'train accuracy': train_accuracy})
        wandb.log({'epoch': epoch+1, 'val loss': val_loss})
        wandb.log({'epoch': epoch+1, 'val accuracy': val_accuracy})


        # Save model per epoch
        save_obj = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }
        
        if epoch == 0:   
            ic("Saving 0 epoch checkpoint")
            torch.save(save_obj, os.path.join(CHECKPOINT_DIR, 'checkpoint_%02d.pth' % epoch))
            best_accuracy[0] = epoch 
            best_accuracy[1] = float(val_accuracy)

        elif val_accuracy > best_accuracy[1] and epoch > 0:
            os.remove(os.path.join(CHECKPOINT_DIR,
                    'checkpoint_%02d.pth' % best_accuracy[0]))
            ic("old checkpoint removed")
            best_accuracy[0] = epoch
            best_accuracy[1] = float(val_accuracy)
            torch.save(save_obj, os.path.join(CHECKPOINT_DIR, 'checkpoint_%02d.pth' % epoch))
        else:
            pass

        # Record end time
        end_time_epoch = time.time()
        #ic(end_time_epoch)
        #torch.cuda.current_stream().synchronize()
        #ic(end_time_epoch)
        end_time_epoch = time.time()
        run_time_epoch = end_time_epoch - start_time_epoch
        logger.info(f'Epoch {epoch} run time: {run_time_epoch:.2f} seconds')
        
        

    # Record end time
    end_time = time.time()
    run_time = end_time - start_time
    logger.info(f'Script run time: {run_time:.2f} seconds')

    ic("Everything Completed")
