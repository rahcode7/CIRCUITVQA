
# Circuit-VQA: A Visual Question Answering Dataset for Electrical Circuit Images (Accepted at ECML-PKDD,2024)

![Alt text](src/visualizations/circuit-image.png?raw=true")


## License
All content and the datasets is licensed under [CC-BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) unless
otherwise stated.

## Downloads

#### Question-Answer Pairs
https://drive.google.com/file/d/10kqGKCNAIG3VpOUvj4Ym6AH49ZaLvC_1/view?usp=sharing


#### Images Only
https://drive.google.com/file/d/1qH6jfWym0Wjg9KKCPQto7MZr0VWM9QpF/view?usp=sharing


## Section 1: Dataset Preparation Guide


#### 1. Prepare master dataset of images and metadata
##### Step 1 Unify all 5 datasets

```
python src/data-prep/02-data-prep-master.py 
```

##### Step 2 Identify and remove duplicate images 

```
python src/data-prep/03-duplicate-identify.py 
python src/data-prep/03-duplicate-remove.py
```

##### Step 3 Split datasets
```
python src/data-prep/04-split-dataset.py
```

##### Step 4 Map classes
```
python src/data-prep/05-class-mapping.py
```

#### 2. Prepare Questions-Answers for various question types
##### Prepare count based questions
```
python src/question-generation/count-based/Q-count.py
```

##### Prepare spatial count based questions
```
python src/question-generation/count-based/Q-count-complex.py
```

##### Prepare junction based questions
```
python src/question-generation/junction-based/Q-junction.py
```

##### Prepare position based questions
```
python src/question-generation/junction-based/Q-position.py
```

##### Prepare value based questions
```
python src/question-generation/value-based/00-bounding-box.py
python src/question-generation/value-based/01-dist-calc.py
python src/question-generation/value-based/02-Q-value-based.py
```

#### 3. Prepare master VQA datasets
##### Prepare master VQA dataset
```
python src/question-generation/master-data.py
```

##### Prepare master VQA dataset for OCR and Description experiments
```
python src/question-generation/master-data-desc-ocr.py
```


##### Prepare master VQA dataset for Bounindg box experiments
```
python src/question-generation/master-data-bbox.py
```

##### Prepare master VQA dataset for Bounindg box segments experiments
```
python src/question-generation/master-data-bbox-segment.py
```

##### Prepare class weights for weighted cross entropy experiments
```
python src/question-generation/master-data-classweights.py
```


## Section 2 : Run Generative - Fine tuning and instruction tuned models 


#### Install conda environment
```
conda create --name cqa-size python=3.8.17  
conda activate cqa-size
conda install pip  
pip install -r requirements.txt   

```

###### Transformers versions
For BLIP and GIT - 4.30.2
For Pix2Struct - 4.36.0


## Download images and questions
#### Images Resized (384a)
```
tar -xvf 384a.tar -C datasets
```

#### Download Question datasets
```
mv master.json datasets/questions/all
mv master_adv.json datasets/questions/all
mv master_adv_ocr.json datasets/questions/all
mv master_bbox.json datasets/questions/all
mv master_bbox_segment.json datasets/questions/all
mv master_bbox_segment_adv_ocr.json datasets/questions/all
```

### Distributed set up 
```
pip install bitsandbytes scipy accelerate
```
### Fine Tuned generative models 

#### PIX (Distributed) - 384


#### Experiment 1 PIX WCE 

```

MODEL='pix'  # git,blip
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='wce' # wce,size,focal,desc,ocr,ocr-post,ocr-desc,size-768,post-576,base-384,post-384,base-384
RUN_TYPE='train' # train,inference
DATE='19Jan'
SIZE='384'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$SIZE-$DATE"
echo $CHECKPOINT
DATASET_SIZE='384a'
NUM_GPUS=4

rm -rf $CHECKPOINT
mkdir $CHECKPOINT

export NUM_NODES=1
export EPOCHS=10
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set your gpu ids

accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 16 --val_batch_size 16 --train_dir datasets/$DATASET_SIZE \
    --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  \
    --question_dir datasets/questions/all/master.json  --experiment_name ms-$MODEL-$EXP_NAME \
    --ngpus 4 --machine_type $MACHINE_TYPE --wandb_status online --max_patches 512 --accumulation_steps 4 --lr 1 --wce 1
```

#### Experiment 2 PIX OCR PRE
```
MODEL='pix'  # git,blip
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='ocr-pre' # wce,size,focal,desc,ocr,ocr-post,ocr-desc,size-768,post-576,base-384,post-384,base-384
RUN_TYPE='train' # train,inference
DATE='24Jan'
SIZE='384'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$SIZE-$DATE"
echo $CHECKPOINT
DATASET_SIZE='384a'
NUM_GPUS=4

rm -rf $CHECKPOINT
mkdir $CHECKPOINT

export NUM_NODES=1
export EPOCHS=10
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set your gpu ids

accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 16 --val_batch_size 16 --train_dir datasets/$DATASET_SIZE \
    --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  \
    --question_dir datasets/questions/all/master_adv.json  --experiment_name ms-$MODEL-$EXP_NAME \
    --ngpus 4 --machine_type $MACHINE_TYPE --wandb_status online --max_patches 512 --accumulation_steps 4 --wce 1 --ocr 1 
```


#### Experiment 3 PIX OCR POST
```
MODEL='pix'  # git,blip
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='ocr-post' # wce,size,focal,desc,ocr,ocr-post,ocr-desc,size-768,post-576,base-384,post-384,base-384
RUN_TYPE='train' # train,inference
DATE='28Jan'
SIZE='384'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$SIZE-$DATE"
echo $CHECKPOINT
DATASET_SIZE='384a'
NUM_GPUS=4

rm -rf $CHECKPOINT
mkdir $CHECKPOINT

export NUM_NODES=1
export EPOCHS=10
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set your gpu ids

accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 16 --val_batch_size 16 --train_dir datasets/$DATASET_SIZE \
    --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  \
    --question_dir datasets/questions/all/master_adv_ocr.json  --experiment_name ms-$MODEL-$EXP_NAME \
    --ngpus 4 --machine_type $MACHINE_TYPE --wandb_status online --max_patches 512 --accumulation_steps 4 --wce 1 --ocr 1
```

#### Experiment 4 PIX DESC
```
MODEL='pix'  # git,blip
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='desc' # wce,size,focal,desc,ocr,ocr-post,ocr-desc,size-768,post-576,base-384,post-384,base-384
RUN_TYPE='train' # train,inference
DATE='24Jan'
SIZE='384'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$SIZE-$DATE"
echo $CHECKPOINT
DATASET_SIZE='384a'
NUM_GPUS=4

rm -rf $CHECKPOINT
mkdir $CHECKPOINT

export NUM_NODES=1
export EPOCHS=10
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set your gpu ids

accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 16 --val_batch_size 16 --train_dir datasets/$DATASET_SIZE \
    --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  \
    --question_dir datasets/questions/all/master_adv_ocr.json  --experiment_name ms-$MODEL-$EXP_NAME \
    --ngpus 4 --machine_type $MACHINE_TYPE --wandb_status online --max_patches 512 --accumulation_steps 4 --wce 1 --desc 1
```

#### Experiment 5 PIX BBOX
```
MODEL='pix'  # git,blip
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='bbox' # wce,size,focal,desc,ocr,ocr-post,ocr-desc,size-768,post-576,base-384,post-384,base-384
RUN_TYPE='train' # train,inference
DATE='24Jan'
SIZE='384'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$SIZE-$DATE"
echo $CHECKPOINT
DATASET_SIZE='384a'
NUM_GPUS=4

rm -rf $CHECKPOINT
mkdir $CHECKPOINT

export NUM_NODES=1
export EPOCHS=10
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set your gpu ids

accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 16 --val_batch_size 16 --train_dir datasets/$DATASET_SIZE \
    --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  \
    --question_dir datasets/questions/all/master_bbox.json  --experiment_name ms-$MODEL-$EXP_NAME \
    --ngpus 4 --machine_type $MACHINE_TYPE --wandb_status online --max_patches 512 --accumulation_steps 4 --wce 1 --bbox 1
```

#### Experiment 6 PIX BBOX SEGMENT
```
MODEL='pix'  # git,blip
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='bbox-segment' # wce,size,focal,desc,ocr,ocr-post,ocr-desc,size-768,post-576,base-384,post-384,base-384
RUN_TYPE='train' # train,inference
DATE='24Jan'
SIZE='384'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$SIZE-$DATE"
echo $CHECKPOINT
DATASET_SIZE='384a'
NUM_GPUS=4

rm -rf $CHECKPOINT
mkdir $CHECKPOINT

export NUM_NODES=1
export EPOCHS=10
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set your gpu ids

accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 16 --val_batch_size 16 --train_dir datasets/$DATASET_SIZE \
    --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  \
    --question_dir datasets/questions/all/master_bbox_segment.json  --experiment_name ms-$MODEL-$EXP_NAME \
    --ngpus 4 --machine_type $MACHINE_TYPE --wandb_status online --max_patches 512 --accumulation_steps 4 --wce 1 --bbox_segment 1

```

### Instruction Fine Tuned models

### 1. GPT4 Experiments

#### Step 1 Input data prep for the model


```
python models-hf/gpt4v/data-prep.py --q_path datasets/questions/all/master_adv.json --op_path models-hf/gpt4v/datasets/ocr --exp_name ocr --hosted_url "https://xxxx.github.io/"
python models-hf/gpt4v/data-prep.py --q_path datasets/questions/all/master_adv_ocr.json --op_path models-hf/gpt4v/datasets/ocr-post --exp_name ocr-post
python models-hf/gpt4v/data-prep.py --q_path datasets/questions/all/master_bbox.json --op_path models-hf/gpt4v/datasets/bbox --exp_name bbox
python models-hf/gpt4v/data-prep.py --q_path datasets/questions/all/master_bbox_segment.json --op_path models-hf/gpt4v/datasets/bbox_segment --exp_name bbox_segment
python models-hf/gpt4v/data-prep.py --q_path datasets/questions/all/master_bbox_yolo.json --op_path models-hf/gpt4v/datasets/bbox_yolo --exp_name bbox_yolo
python models-hf/gpt4v/data-prep.py --q_path datasets/questions/all/master_bbox_segment_yolo.json --op_path models-hf/gpt4v/datasets/bbox_segment_yolo --exp_name bbox_segment_yolo
```


#### Step 2  Post processing of the model outputs

##### Prepare predictions.json file by merging multiple outputs file
```
python models-hf/gpt4v/post-process-0.py --prediction_dir models-gpt4v-hf/results-ddp/384a  --exp_name ocr
python models-hf/gpt4v/post-process-0.py --prediction_dir models-gpt4v-hf/results-ddp/384a  --exp_name ocr-post
python models-hf/gpt4v/post-process-0.py --prediction_dir models-gpt4v-hf/results-ddp/384a  --exp_name bbox
python models-hf/gpt4v/post-process-0.py --prediction_dir models-gpt4v-hf/results-ddp/384a  --exp_name bbox_segment
python models-hf/gpt4v/post-process-0.py --prediction_dir models-gpt4v-hf/results-ddp/384a  --exp_name bbox_yolo
python models-hf/gpt4v/post-process-0.py --prediction_dir models-gpt4v-hf/results-ddp/384a  --exp_name bbox_segment_yolo


```


##### Step 3 repare predictions-final.json file by merging multiple outputs file
```
python models-hf/gpt4v/post-process.py --prediction_dir models-gpt4v-hf/results-ddp/384a/ocr  --exp_name ocr
python models-hf/gpt4v/post-process.py --prediction_dir models-gpt4v-hf/results-ddp/384a/ocr-post  --exp_name ocr-post
python models-hf/gpt4v/post-process.py --prediction_dir models-gpt4v-hf/results-ddp/384a/bbox  --exp_name bbox
python models-hf/gpt4v/post-process.py --prediction_dir models-gpt4v-hf/results-ddp/384a/bbox_segment  --exp_name bbox_segment
python models-hf/gpt4v/post-process.py --prediction_dir models-gpt4v-hf/results-ddp/384a/bbox_yolo  --exp_name bbox_yolo
python models-hf/gpt4v/post-process.py --prediction_dir models-gpt4v-hf/results-ddp/384a/bbox_segment_yolo  --exp_name bbox_segment_yolo

```




### 2. LLaVA Experiments

#### Get LLaVA
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA

conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e . -->


##### Run LLaVA Base (Zero-shot)
```
MODEL='llava'  
EXP_NAME='base' # wce,size,focal,desc,ocr,ocr-post,ocr-desc,size-768,post-576,base-384,post-384,base-384
RUN_TYPE='inference' 
DATE='3Feb'
DATASET_SIZE='384a'
NUM_GPUS=1
export NUM_NODES=1
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0
python LLaVA/LLaVa-mac/cqa-llava/eval-single.py --question_dir datasets/questions/all/master.json  \
    --image_dir datasets/$DATASET_SIZE --results_dir models-LLAVA-hf/results-ddp/$DATASET_SIZE/$EXP_NAME --exp_name $EXP_NAME
```


### InstructBLIP Experiments

##### Step 1 Get packages

```
pip install bitsandbytes
cd circuitQA
git pull
conda activate <circuitQAenvironment>
mkdir datasets/results/InstructBLIP
```

##### Step 2 Run experiments
##### BASE model 
```
MODEL='InstructBLIP'
EXP_NAME='base'
DATASET_SIZE='384a'

# MS
export CUDA_VISIBLE_DEVICES=0
python models-hf/models-InstructBLIP-hf/iblip-eval-single.py --question_dir datasets/questions/all/master.json  \
    --image_dir datasets/$DATASET_SIZE --results_dir datasets/results/InstructBLIP/$EXP_NAME --exp_name $EXP_NAME

# Ada
python models-hf/models-InstructBLIP-hf/iblip-eval-single-mac.py --question_dir ../datasets/questions/all/master_adv_ocr.json  \
    --image_dir ../datasets/$DATASET_SIZE --results_dir ../datasets/results/InstructBLIP/$EXP_NAME --exp_name $EXP_NAME

```

##### DESC model 
```
MODEL='InstructBLIP'
EXP_NAME='desc'
DATASET_SIZE='384a'
export CUDA_VISIBLE_DEVICES=1
python models-hf/models-InstructBLIP-hf/iblip-eval-single.py --question_dir datasets/questions/all/master_adv_ocr.json  \
    --image_dir datasets/$DATASET_SIZE --results_dir datasets/results/InstructBLIP/$EXP_NAME --exp_name $EXP_NAME
```

##### OCR PRE model 
```
MODEL='InstructBLIP'
EXP_NAME='ocr-pre'
DATASET_SIZE='384a'

python models-hf/models-InstructBLIP-hf/iblip-eval-single.py --question_dir datasets/questions/all/master_adv.json  \
    --image_dir datasets/$DATASET_SIZE --results_dir datasets/results/InstructBLIP/$EXP_NAME --exp_name $EXP_NAME
```

##### OCR POST model 
```
MODEL='InstructBLIP'
EXP_NAME='ocr-post'
DATASET_SIZE='384a'

python models-hf/models-InstructBLIP-hf/iblip-eval-single.py --question_dir datasets/questions/all/master_adv_ocr.json  \
    --image_dir datasets/$DATASET_SIZE --results_dir datasets/results/InstructBLIP/$EXP_NAME --exp_name $EXP_NAME
``` 

##### BBOX-YOLO model 
```
MODEL='InstructBLIP'
EXP_NAME='bbox-yolo'
DATASET_SIZE='384a'

python models-hf/models-InstructBLIP-hf/iblip-eval-single.py --question_dir datasets/questions/all/master_bbox_yolo.json  \
    --image_dir datasets/$DATASET_SIZE --results_dir datasets/results/InstructBLIP/$EXP_NAME --exp_name $EXP_NAME
```

##### BBOX-Segment-YOLO  model 
```
MODEL='InstructBLIP'
EXP_NAME='bbox-segment-yolo'
DATASET_SIZE='384a'

python models-hf/models-InstructBLIP-hf/iblip-eval-single.py --question_dir datasets/questions/all/master_bbox_segment_yolo.json  \
    --image_dir datasets/$DATASET_SIZE --results_dir datasets/results/InstructBLIP/$EXP_NAME --exp_name $EXP_NAME

```

##### BBOX-ORACLE model 
```
MODEL='InstructBLIP'
EXP_NAME='bbox'
DATASET_SIZE='384a'

python models-hf/models-InstructBLIP-hf/iblip-eval-single.py --question_dir datasets/questions/all/master_bbox.json  \
    --image_dir datasets/$DATASET_SIZE --results_dir datasets/results/InstructBLIP/$EXP_NAME --exp_name $EXP_NAME
```

##### BBOX-Segment-ORACLE model 
```
MODEL='InstructBLIP'
EXP_NAME='bbox-segment'
DATASET_SIZE='384a'

python models-hf/models-InstructBLIP-hf/iblip-eval-single.py --question_dir datasets/questions/all/master_bbox_segment.json  \
    --image_dir datasets/$DATASET_SIZE --results_dir datasets/results/InstructBLIP/$EXP_NAME --exp_name $EXP_NAME

```


###### Step 2 Post process
```
python circuit-QGA/models-hf/models-InstructBLIP-hf/post-process.py --prediction_dir models-InstructBLIP-hf/results-ddp/384a --exp_name base
python circuit-QGA/models-hf/models-InstructBLIP-hf/post-process.py --prediction_dir models-InstructBLIP-hf/results-ddp/384a --exp_name desc
```


## Section 3 Compute Accuracy and Hallucination scores

#### Step 4.1   Accuracy calculations
```
python src/evaluate/00-evaluate-pred.py 
```

#### Step 4.2 Calculate Hallucination scores (HVQA) 


```
python src/evaluate/02-a-hallucination.py 
```

## License
All content and the datasets is licensed under [CC-BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) unless
otherwise stated.


