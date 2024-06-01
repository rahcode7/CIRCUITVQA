import json 
import os 
import pandas as pd
from icecream import ic 
import re
from itertools import chain
from PIL import Image 
import shutil
from pathlib import Path
# Read master data files and update labels-final with new labels

def class_map_dataset():

    output  = "labels-final"
    class_mapping_file = "src/utils/class_mappings.json"
    data_path = "datasets/master"

    class_dict = json.load(open(class_mapping_file))

    cnt = 0 

    file = "d1_electrical_15_png.rf.a1e04b7d3495e5778cad5b20c5044262.txt"

    #for file in os.listdir(os.path.join(data_path,'labels')):
    cnt +=1 

    print(file)
    file_path = data_path +  '/labels/' + file
    file_path_op = data_path +  '/labels-final/' + file
    
    image_name = file.replace(".txt","")
    file_image_path = data_path +  '/images/' + image_name + '.jpg'

    dataset_name = file.split('.')[0][0:2]
    ic(file_path,image_name,dataset_name)
    
    if dataset_name!='d2':
        # Read file line by line 
        with open(file_path) as f, open(file_path_op,'w+') as fnew:

            ic(cnt,file_path,file_path_op)

            # new_lines = [[]]
            # new_line = []
            for line in f:
                l = line.split()
                #ic(l)
                k = int(l[0])

                # Gives item at k index
                item_name = class_dict[dataset_name][0][k]
                item_index = class_dict[dataset_name][1][k]

                ic(item_name,item_index,k)
                # New list of index + bbox
                new_line = [[str(item_index)]]
                new_line.append(l[1:5])

                ic(new_line)
                #new_line = list(chain(*new_line))
                new_line = list(chain.from_iterable(new_line))
                ic(new_line)
                #ic(new_line)
                #new_lines.append(new_line)
                #ic(new_lines)

                #fnew.write(f"{new_line}\n")
                #fnew.writelines("%s\n" % item for item in new_line)
                fnew.write(" ".join(new_line))
                fnew.write("\n")

        fnew.close()


    # Else dataset d2
    else:
        with open(file_path) as f, open(file_path_op,'w+') as fnew:
            ic(dataset_name,cnt,file_path,file_path_op)

            img = Image.open(file_image_path) 
            width = img.width 
            height = img.height 
            ic(width,height)

            for line in f:
                l = line.split()
                #ic(l)
                
                symbol,xmin,ymin,xmax,ymax = l[0],int(l[1]),int(l[2]),int(l[3]),int(l[4])
                # width = xmax - xmin 
                # height = ymax - ymin 


                x = (xmin + xmax) / 2 / width 
                y = (ymin + ymax) / 2 / height 

                w = (xmax - xmin) / width 
                h = (ymax - ymin) / height

                for k,val in enumerate(class_dict[dataset_name][0]):
                    if symbol == val:
                        #ic(k,val)
                        item_index = class_dict[dataset_name][1][k]

                new_line = [[str(item_index)]]
                new_line.append([str(x),str(y),str(w),str(h)])

                #ic(new_line)
                #new_line = list(chain(*new_line))
                new_line = list(chain.from_iterable(new_line))
            
                #ic(new_line)

                fnew.write(" ".join(new_line))
                fnew.write("\n")

        fnew.close()


"""
Copies new yolo files from master folder to final model_inputs sub folders - train,test,val
"""
def copy_master_splits(split_path,src_path):
    
    # Write to train,test,val
    cnt=0 
    for dir in ['train','test','val']:

        # Delete folder if it exists
        dirpath = Path(os.path.join(split_path,dir))
        if dirpath.exists() and dirpath.is_dir():
            shutil.rmtree(dirpath)

        for file in os.listdir(os.path.join(split_path,'images',dir)):
            print(file)
            file_name = file.replace(".jpg",".txt")
            file_path = os.path.join(split_path,dir,file_name)

            #ic(file_name,file_path)

            # copy text file if it exist in new folder
            dest = f"{split_path}/labels/{dir}/{file_name}"
            src =  f"{src_path}/labels-final/{file_name}"
            ic(src,dest)
            
            shutil.copy(src, dest) 
            cnt+=1
            ic(cnt)

if __name__ == "__main__":

    # Create updated txt files with updated class mappings
    class_map_dataset()
    
    split_path = "datasets/model-inputs"
    src_path = "datasets/master"
    
    copy_master_splits(split_path,src_path)













