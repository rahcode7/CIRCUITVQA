import pandas as pd 
import os
import shutil
from icecream import ic

INPUT_PATH_MAIN = "datasets/questions/count-complex/"
INPUT_FILE = INPUT_PATH_MAIN+ '/Q-count-complex.csv'

MASTER_IMAGE_PATH = "datasets/"

OUTPUT_PATH = "datasets/model-inputs/"

if __name__ == "__main__":
    ip_datasets = ['train','test','dev']
    op_datasets = ['train','test','val']

    if os.path.isdir(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
        os.mkdir(OUTPUT_PATH)
        print("folder created")

    if not os.path.isdir(OUTPUT_PATH + '/val'):
        shutil.copytree(INPUT_PATH_MAIN + 'dev/images',OUTPUT_PATH + 'val', dirs_exist_ok=True) 

    if not os.path.isdir(OUTPUT_PATH+ '/train'):
        os.mkdir(OUTPUT_PATH+'/train')
        ic(OUTPUT_PATH+'/train')
        shutil.copytree(INPUT_PATH_MAIN + 'train/images' ,OUTPUT_PATH + 'train', dirs_exist_ok=True)  
    
    if not os.path.isdir(OUTPUT_PATH+ '/test'):
        os.mkdir(OUTPUT_PATH+'/test')
        shutil.copytree(INPUT_PATH_MAIN + 'test/images',OUTPUT_PATH + 'test', dirs_exist_ok=True)  

    # Then bring unanottated images
    for ip,op in zip(ip_datasets,op_datasets):
        for c,filename in enumerate(os.listdir(os.path.join(MASTER_IMAGE_PATH,ip,'images'))):
            src = os.path.join(MASTER_IMAGE_PATH,ip,'images',filename)
            dest = os.path.join(OUTPUT_PATH,op,filename)
            ic(src,dest)
            if not os.path.exists(dest):
                shutil.copy(src,dest)
            else:
                ic(src,dest)

