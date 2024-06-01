## LLaVA (MS)

#### Step 1Get LLaVA
```

cd circuit-QGA/models-hf

git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA

conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e . 

git pull
pip install -e .

```

#### Step 2 Get CircuitQA LLaVA codes
```
cd ~/circuit-QGA # # main cqa repo 
git pull
cp -r models-hf/cqa-llava models-hf/LLaVA
cp models-hf/cqa-llava/model/builder.py models-hf/LLaVA/llava/model/builder.py
cp models-hf/cqa-llava/eval/run_llava.py models-hf/LLaVA/llava/eval/run_llava.py
```


#### Step 3  Run CircuitQA LLaVA - Experiment1 Base
```
cd ~/circuit-QGA     # main cqa repo 
MODEL='llava'  # git,blip
EXP_NAME='base' # wce,size,focal,desc,ocr,ocr-post,ocr-desc,size-768,post-576,base-384,post-384,base-384
RUN_TYPE='inference' # train,inference
DATE='4Feb'
DATASET_SIZE='384a'
NUM_GPUS=4

export NUM_NODES=1
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2,3

mkdir datasets/results/$MODEL
mkdir datasets/results/$MODEL/$DATASET_SIZE/
mkdir datasets/results/$MODEL/$DATASET_SIZE/$EXP_NAME

python models-hf/LLaVA/cqa-llava/eval-single.py --question_dir datasets/questions/all/master.json  \
--image_dir datasets/$DATASET_SIZE --results_dir datasets/results/$MODEL/$DATASET_SIZE/$EXP_NAME
```


SIZE='384a'
export RESULTS_DIR_LLAVA=models-llava/results/$SIZE
export LOCAL_SCRIPTS=models-hf


scp -r $LOCAL_SCRIPTS/cqa-llava LLaVA/cqa-llava


cd LLaVA
cp cqa-llava/model/builder.py llava/model/builder.py
cp cqa-llava/eval/run_llava.py llava/model/run_llava.py 


conda activate llava
sbatch LLaVA/cqa-llava/script.sh


#SBATCH --output=runs/llava/llava-base.txt
cat runs/llava/llava-base.txt
vi datasets/results/llava/384a/desc/predictions.json



squeue -u $USER

### For MAC
conda activate llava

python LLaVA/LLaVa-mac/cqa-llava/eval-single.py --question_dir datasets/questions/all/master.json \
--image_dir datasets/$DATASET_SIZE --results_dir datasets/results/$MODEL/$DATASET_SIZE/$EXP_NAME

For error - MPS does not support cumsum op with int64 input - Answer - get nightly version
conda install pytorch -c pytorch-nightly

For error - transformer has same config
git pull
pip install -e .  -->