
import requests
from transformers import AutoProcessor, BlipForQuestionAnswering,BlipImageProcessor
from datasets import VQACircuitDataset,vqa_collate_fn
from torch.utils.data import DataLoader
import torch
import wandb
from tqdm import tqdm
import os 
import torch.distributed as dist
import argparse
from icecream import ic 
from datetime import timedelta
import torch.multiprocessing as mp
import logging
import time 
from utils.helpers import set_seed,get_questions
from accelerate import Accelerator
from icecream import ic 
from transformers import Adafactor
import bitsandbytes as bnb
import json 
from transformers import BlipConfig,BlipTextConfig
import shutil
import site


def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    ic("prev checkpoint loaded",checkpoint['epoch'])
    return model, optimizer, checkpoint['epoch']

def train(epoch,model,train_dataloader,processor,im_processor,tokenizer,optimizer,scheduler,device,accelerator,IMAGE_SIZE,DESC_FLAG,OCR_FLAG,BBOX_FLAG,BBOX_SEGMENT_FLAG,LR_FLAG,NUM_GPUS,MACHINE_TYPE,ACCUMULATION_STEPS):
    model.train()

    train_losses = []
    train_corrects = []
    train_size = [] 

    for idx, (files,images, questions, answers,qtype,desc,ocr,bbox,bbox_segment) in enumerate(tqdm(train_dataloader)):
        #ic(questions,answers)
        with accelerator.accumulate(model):
            inputs = processor(images=images, text=questions,return_tensors="pt", padding=True)

            if DESC_FLAG or OCR_FLAG or BBOX_FLAG or BBOX_SEGMENT_FLAG:

                questions_new,tokenizer = get_questions(DESC_FLAG,OCR_FLAG,BBOX_FLAG,BBOX_SEGMENT_FLAG,tokenizer,questions,ocr,desc,bbox,bbox_segment)
                #ic(questions_new)
                enc = tokenizer.batch_encode_plus(questions_new,return_tensors="pt",padding=True)
                inputs["input_ids"] = enc["input_ids"]
                inputs["attention_mask"] = enc["attention_mask"]

                #ic(questions_new)
            if IMAGE_SIZE!=384:
                    #ic(inputs['pixel_values'].shape)
                    im_processor =  BlipImageProcessor()
                    inputs_im = im_processor(images=images,size={"height":IMAGE_SIZE,"width":IMAGE_SIZE},return_tensors="pt")
                    inputs['pixel_values'] = inputs_im['pixel_values']

            labels = processor(text=answers, return_tensors="pt",padding=True).input_ids
            inputs["labels"] = labels
            

            #inputs = inputs(requires_grad=True)
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model.module(**inputs)
            loss = outputs.loss
            
            ## DDP code
            train_losses.append(accelerator.gather(outputs.loss))
            
            #inputs = inputs(requires_grad=True)
            preds = model.module.generate(**inputs,max_new_tokens=100)
            predicted = processor.batch_decode(preds, skip_special_tokens=True)
            train_batch_corrects = len([i for i,
                                            j in zip(predicted, answers) if i == j])
            #ic(train_batch_corrects)
            train_batch_corrects = torch.tensor(train_batch_corrects).to(device)
            #ic(predicted,outputs.loss,train_batch_corrects)

            gathered_tensor = accelerator.gather(train_batch_corrects)
            train_corrects.append(gathered_tensor)

            # answers_batch = accelerator.gather(answers)
            # ic(answers_batch)
            # train_size += gathered_tensor.size(dim=1)

            label_size = torch.tensor(len(labels)).to(device)
            
            gathered_sizes = accelerator.gather(label_size)
            train_size.append(gathered_sizes)

            #ic(answers,len(labels),predicted,outputs.loss,gathered_sizes,train_size)

            #train_corrects.append(train_batch_corrects)
            # train_corrects.append(accelerator.gather_for_metrics(torch.tensor([train_batch_corrects])))


            accelerator.backward(loss)
            # Gradient accumulation 
            #if (idx + 1)% ACCUMULATION_STEPS == 0:
            optimizer.step()
            
            #scheduler.step()
            optimizer.zero_grad()


    # Call every epoch
    if LR_FLAG:
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print("Epoch %d: AdamW lr %.10f -> %.10f" % (epoch, before_lr, after_lr))

    total_train_size = torch.sum(torch.cat(train_size)).item()
    total_corrects = torch.sum(torch.cat(train_corrects)).item()

    ic(device,total_corrects,total_train_size)
    train_loss = torch.sum(torch.cat(train_losses)).item() / len(train_dataloader)

    #train_accuracy = torch.sum(torch.cat(train_corrects)) / len(train_dataloader.dataset)
    train_accuracy =  total_corrects / total_train_size

    ic(train_loss,train_accuracy, torch.sum(torch.cat(train_corrects)))
    return train_loss,train_accuracy

def evaluate(model,val_dataloader,processor,im_processor,tokenizer,device,accelerator,IMAGE_SIZE,DESC_FLAG,OCR_FLAG,BBOX_FLAG,BBOX_SEGMENT_FLAG,NUM_GPUS,MACHINE_TYPE):
    val_losses = []
    val_corrects = []
    val_size =[]
    # if accelerator.is_main_process:
    #     ic("True")
    with torch.no_grad(): 
        model.eval()
        for idx,(files,images, questions, answers,qtype,desc,ocr,bbox,bbox_segment) in enumerate(tqdm(val_dataloader)):
            inputs = processor(images=images, text=questions, return_tensors="pt",padding=True)

            if DESC_FLAG or OCR_FLAG or BBOX_FLAG or BBOX_SEGMENT_FLAG:
                #ic("Yes",DESC_FLAG,OCR_FLAG,BBOX_FLAG,BBOX_SEGMENT_FLAG)
                questions_new,tokenizer = get_questions(DESC_FLAG,OCR_FLAG,BBOX_FLAG,BBOX_SEGMENT_FLAG,tokenizer,questions,ocr,desc,bbox,bbox_segment)
                #ic(questions_new)
                enc = tokenizer.batch_encode_plus(questions_new,return_tensors="pt",padding=True)
                inputs["input_ids"] = enc["input_ids"]
                inputs["attention_mask"] = enc["attention_mask"]
            
            if IMAGE_SIZE!=384:
                im_processor =  BlipImageProcessor()
                inputs_im = im_processor(images=images,size={"height":IMAGE_SIZE,"width":IMAGE_SIZE},return_tensors="pt")
                inputs['pixel_values'] = inputs_im['pixel_values']
            labels = processor(text=answers, return_tensors="pt",padding=True).input_ids
            inputs["labels"] = labels

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model.module(**inputs)
            

            ### DDP 
            val_losses.append(accelerator.gather(outputs.loss))
            preds  = model.module.generate(**inputs,max_new_tokens=100)

            predicted = processor.batch_decode(preds, skip_special_tokens=True)
            #ic(predicted)
            val_batch_corrects  = len([i for i, j in zip(predicted,answers) if i == j])

            val_batch_corrects =  torch.tensor(val_batch_corrects).to(device)
            val_corrects.append(accelerator.gather(val_batch_corrects))
            
            label_size = torch.tensor(len(labels)).to(device)
            #ic(label_size)
            gathered_sizes = accelerator.gather(label_size)
            val_size.append(gathered_sizes)

            #ic(answers,len(labels),predicted,outputs.loss,gathered_sizes,val_size)


            #ic(predicted,answers,len(answers),val_corrects,accelerator.gather(val_batch_corrects))
    
    total_val_size = torch.sum(torch.cat(val_size)).item()
    total_corrects = torch.sum(torch.cat(val_corrects)).item()
    ic(total_corrects,device,len(val_dataloader),total_val_size)

    val_loss = torch.sum(torch.cat(val_losses)).item() / len(val_dataloader)
    val_accuracy = total_corrects / total_val_size
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
    ACCUMULATION_STEPS=args.accumulation_steps
    
    WCE_FLAG,LR_FLAG,OCR_FLAG,DESC_FLAG,BBOX_FLAG,BBOX_SEGMENT_FLAG = False,False,False,False,False,False
    
    if args.wce == 1:WCE_FLAG=True
    if args.lr == 1: LR_FLAG=True 
    if args.ocr == 1:OCR_FLAG=True
    if args.desc == 1:DESC_FLAG=True
    if args.bbox == 1:BBOX_FLAG=True
    if args.bbox_segment == 1:BBOX_SEGMENT_FLAG=True


    ic(WCE_FLAG,LR_FLAG,OCR_FLAG,DESC_FLAG,BBOX_FLAG,BBOX_SEGMENT_FLAG,LEARNING_RATE,type(LEARNING_RATE))

    accelerator = Accelerator(gradient_accumulation_steps=ACCUMULATION_STEPS)

    # Record start time
    start_time = time.time()

   
    log_file = os.path.join(CHECKPOINT_DIR,'statistics.log')

    if not os.path.exists(log_file):
        open(log_file, 'w+').close()
        print("log file created")
    else:
        print("log file exists")

    logging.basicConfig(filename=log_file,level=logging.INFO,filemode="a") # #,
    logger = logging.getLogger("mylogger")
    
    if accelerator.is_main_process:
        wandb.init(project='CircuitBLIP',
                config = {
                'learning_rate':LEARNING_RATE,
                'epochs':EPOCHS,
                'train_batch_size': TRAIN_BATCH_SIZE,
                'val_batch_size':VAL_BATCH_SIZE
                },mode=WANDB_STATUS)
        wandb.run.name = EXPERIMENT_NAME


    SITE_PACKAGES_PATH = site.getsitepackages()[0]
    processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
    im_processor =  BlipImageProcessor(image_size=IMAGE_SIZE)
    tokenizer = processor.tokenizer
    
    # dest = os.path.join(SITE_PACKAGES_PATH,'transformers/models/blip')

    if IMAGE_SIZE==384:
        src = 'models-hf/blip-files/modeling_blip_384.py'
        dest = os.path.join(SITE_PACKAGES_PATH,'transformers/models/blip','modeling_blip.py')
        shutil.copy(src,dest)
    elif IMAGE_SIZE==576:
        ic("image size 576 file copied")
        src = 'models-hf/blip-files/modeling_blip_576.py'
        dest = os.path.join(SITE_PACKAGES_PATH,'transformers/models/blip','modeling_blip.py')
        shutil.copy(src,dest)
    else:
        src = 'models-hf/blip-files/original/modeling_blip.py'
        dest = os.path.join(SITE_PACKAGES_PATH,'transformers/models/blip','modeling_blip.py')
        shutil.copy(src,dest)

    if WCE_FLAG:
        # Copy wce files
        src = 'models-hf/blip-files/configuration_blip.py'
        dest = os.path.join(SITE_PACKAGES_PATH,'transformers/models/blip','configuration_blip.py')
        shutil.copy(src,dest)
        
        src = 'models-hf/blip-files/modeling_blip_text.py'
        dest = os.path.join(SITE_PACKAGES_PATH,'transformers/models/blip','modeling_blip_text.py')
        shutil.copy(src,dest)
        ic("copied wce files")

        class_file = os.path.join(os.getcwd(), 'datasets/class_weights.json')
        with open(class_file, 'r') as fp:
            class_weights_list = json.load(fp)

        # configuration = BlipConfig()
        #configuration = BlipConfig(text_config=None,vision_config=None,class_weights=class_weights)
        #text_configuration= BlipTextConfig(class_weights=class_weights_list)
        configuration = BlipConfig(text_config={'class_weights':class_weights_list})


        model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base",config=configuration,ignore_mismatched_sizes=True)

    else:
        # pass
        #Get original file of wce back if changes
        ic("copying config original")
        src = 'models-hf/blip-files/original/configuration_blip.py'
        dest = os.path.join(SITE_PACKAGES_PATH,'transformers/models/blip','configuration_blip.py')
        shutil.copy(src,dest)

        
        src = 'models-hf/blip-files/original/modeling_blip_text.py'
        dest = os.path.join(SITE_PACKAGES_PATH,'transformers/models/blip','modeling_blip_text.py')
        shutil.copy(src,dest)

        configuration = BlipConfig()
        model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base",ignore_mismatched_sizes=True)

        # model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base",ignore_mismatched_sizes=True)

    if accelerator.is_main_process:
        wandb.watch(model)


    train_dataset = VQACircuitDataset(TRAIN_DIR, Q_PATH, split='train',ocr=OCR_FLAG,desc=DESC_FLAG,bbox=BBOX_FLAG,bbox_segment=BBOX_SEGMENT_FLAG)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=TRAIN_BATCH_SIZE, drop_last=True,collate_fn= vqa_collate_fn,pin_memory=False)

    val_dataset =  VQACircuitDataset(VAL_DIR, Q_PATH, split='val',ocr=OCR_FLAG,desc=DESC_FLAG,bbox=BBOX_FLAG,bbox_segment=BBOX_SEGMENT_FLAG)
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=VAL_BATCH_SIZE,drop_last=True, collate_fn= vqa_collate_fn,pin_memory=False)


    print("data loaded")
    device = accelerator.device
    ic(device)


    #optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    # optimizer = Adafactor(model.parameters(),lr=LEARNING_RATE,weight_decay=0.05,relative_step=False)


    if LR_FLAG:
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=3, num_training_steps=10)
    else:
        scheduler = ""
        

    starting_name = 'checkpoint'
    stats_name = 'statistics'
    prev_accuracy = []
    prev_epoch = []
    best_accuracy = [0, 0.0]

    file_list = os.listdir(CHECKPOINT_DIR)
    ic(file_list)
    START = 0

    if file_list:
        checkpoint_files = [filename for filename in file_list if filename.startswith(starting_name)]
        #stat_files = [filename for filename in file_list if filename.startswith(stats_name)]
        ic(checkpoint_files)
        # Get latest checkpoint
        if checkpoint_files:
            checkpoint_files.sort()
            LATEST_CHECKPOINT=checkpoint_files[-1]
            START = int(checkpoint_files[-1].split(".pth")[0][-1:])+1
            ic("Previous checkpoint found",LATEST_CHECKPOINT,START)   
    
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

    
    train_dataloader, val_dataloader, model, optimizer,scheduler = accelerator.prepare(
        train_dataloader,val_dataloader, model, optimizer,scheduler
    )

    for epoch in range(START,EPOCHS):
        start_time_epoch = time.time()

        train_loss,train_accuracy = train(epoch,model,train_dataloader,processor,im_processor,tokenizer,optimizer,scheduler,device,accelerator,IMAGE_SIZE,DESC_FLAG,OCR_FLAG,BBOX_FLAG,BBOX_SEGMENT_FLAG,LR_FLAG,NUM_GPUS,MACHINE_TYPE,ACCUMULATION_STEPS)
        val_loss, val_accuracy = evaluate(model,val_dataloader,processor,im_processor,tokenizer,device,accelerator,IMAGE_SIZE,DESC_FLAG,OCR_FLAG,BBOX_FLAG,BBOX_SEGMENT_FLAG,NUM_GPUS,MACHINE_TYPE)
             
        logger.info(f'Epoch {epoch} Train loss : {train_loss} Train accuracy : {train_accuracy}')
        logger.info(f'Epoch {epoch} Val loss : {val_loss} Val accuracy : {val_accuracy}')
        
        ic(epoch,train_loss,val_loss,train_accuracy,val_accuracy)
        
        if accelerator.is_main_process:
            ic("True")
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'epoch': epoch+1
            })      

            #Save model per epoch
            save_obj = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        #'config': model.config,
                        'epoch': epoch,
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

    wandb.finish()
    ic("Completed training")


    if WCE_FLAG:
        src = 'models-hf/blip-files/original/configuration_blip.py'
        dest = os.path.join(SITE_PACKAGES_PATH,'transformers/models/blip','configuration_blip.py')
        shutil.copy(src,dest)

        src = 'models-hf/blip-files/original/modeling_blip_text.py'
        dest = os.path.join(SITE_PACKAGES_PATH,'transformers/models/blip','modeling_blip_text.py')

        shutil.copy(src,dest)
    # Record end time
    end_time = time.time()
    # Calculate script run time
    run_time = end_time - start_time
    # Log the script run time
    logger.info(f'Script run time: {run_time:.2f} seconds')


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
    parser.add_argument('--accumulation_steps',type=int,default=0,help="acc steps")
    parser.add_argument('--wce',type=int,default=0,help="wce")
    parser.add_argument('--lr',type=int,default=0)


    args = parser.parse_args()    

    run_ddp_accelerate(args)
    