"""To unify all 5 datasets"""

# Read files, create mapping of file names and new id for each image
import os 
import shutil
import yaml
import pandas as pd
import numpy as np
import json 
from utils.preprocessing import xml_processor,class_cleaner

data_path = "datasets/raw/"
output_path = "datasets/"
  

if __name__ == "__main__":

    datasets = ['d2-kaggle','CGHD-supplement.v6i.yolov8','Circuit Recognition Electronics.v6i.yolov8','Circuit Recognition.v9i.yolov8','CGHD-full-supplemented.v13-dataset-b.yolov8']
    datasetsx = ['d2-kaggle','CGHD-supplement.v6i.voc','Circuit Recognition Electronics.v6i.voc','Circuit Recognition.v9i.voc (1)','CGHD-full-supplemented.v13-dataset-b.voc']
    #for dataset in datasers
    df = pd.read_csv("../datasets/processed/dataset.csv")
    
    if not os.path.exists(os.path.join(output_path,'master')):
        os.mkdir(output_path + 'master')
    if not os.path.exists(os.path.join(output_path,'master','images')):
        os.mkdir(output_path + 'master' + '/images')
    if not os.path.exists(os.path.join(output_path,'master','labels')):
        os.mkdir(output_path + 'master' + '/labels')
    if not os.path.exists(os.path.join(output_path,'master','xml')):
        os.mkdir(output_path + 'master' + '/xml')


    # For each dataset
    for i,dataset in enumerate(datasets):
        print(dataset)

        if dataset=="d2-kaggle":
            index = df.index[df.dataset==dataset].tolist()[0]
            internal_name = df.at[index,"internal_name"]

            for f in os.listdir(data_path+dataset):
                d = os.path.join(data_path+dataset, f)
                if os.path.isdir(d):
                    data_folder = f 
                    #print(data_folder)

                    # extra loop for images
                    s = "images"
                    for c,filename in enumerate(os.listdir(data_path+ dataset + '/' + data_folder +'/' + s)):
                        dest = f"{output_path}master/{s}/{internal_name}_{filename}"
                        src =  f"{data_path}/{dataset}/{data_folder}/{s}/{filename}"
                        #print(dest)
                        #print(src)
                        shutil.copy(src, dest)  

                    # xml
                    s = "annotations"
                    for c,filename in enumerate(os.listdir(data_path+ dataset + '/' + data_folder +'/' + s)):
                        dest = f"{output_path}master/xml/{internal_name}_{filename}"
                        src =  f"{data_path}/{dataset}/{data_folder}/{s}/{filename}"
                        #print(dest)
                        #print(src)
                        shutil.copy(src, dest)  

                    # labels generate labels files by parsing xml annotations
                    s = "labels"
                    pull_folder = "annotations"
                    for c,filename in enumerate(os.listdir(data_path+ dataset + '/' + data_folder +'/' + pull_folder)):
                        dest = f"{output_path}master/{s}/{internal_name}_{filename}"
                        src =  f"{data_path}/{dataset}/{data_folder}/{pull_folder}/{filename}"
                        #print(dest)
                        #print(src)
                        dest = dest.replace(".xml",".txt")
                        symbol_df = xml_processor(src)
                        symbol_df = class_cleaner(symbol_df,filename)

                        with open("  datasets/raw/d2-kaggle/classes.json") as classes_file:
                            d = json.loads(classes_file.read())
                        class_list  = list(d.keys())
                        class_vals = list(d.values())
                        
                        #symbol_df['s'] = symbol_df['s'].apply(lambda s: d[s])
                        #print(symbol_df)

                        #print(symbol_df)
                        np.savetxt(dest, symbol_df.values, fmt='%s')
                        #shutil.copy(src, dest)  
        else:   
            # Get index
            index = df.index[df.dataset==dataset].tolist()[0]
            internal_name = df.at[index,"internal_name"]
            print(internal_name)
            for f in os.listdir(data_path+dataset):
                d = os.path.join(data_path+dataset, f)
                if os.path.isdir(d):
                    print("folder running",f)
                    data_folder = f 

                    # sort folders first
                    # Images 
                    s = "images"
                    for c,filename in enumerate(os.listdir(data_path+ dataset + '/' + data_folder +'/' + s)):
                        print(filename)
                        #dest = f"{output_path}master/{s}/{str(count_i)}_{internal_name}.jpg"
                        
                        dest = f"{output_path}master/{s}/{internal_name}_{filename}"
                        src =  f"{data_path}/{dataset}/{data_folder}/{s}/{filename}"
                        #print(dest)
                        #print(src)
                        #os.rename(src,dest)
                        shutil.copy(src, dest)
                        #count_i+=1

                    # Labels
                    s = "labels"
                    for c,filename in enumerate(os.listdir(data_path+ dataset + '/' + data_folder +'/' + s)):
                        #dest = f"{output_path}master/{s}/{str(count_l)}_{internal_name}.txt"
                        # print(dest)
                        # print(src)
                        dest = f"{output_path}master/{s}/{internal_name}_{filename}"
                        src =  f"{data_path}{dataset}/{data_folder}/{s}/{filename}"
                        #  datasets/raw/CGHD-supplement.v6i.yolov8/train/images
                        #os.rename(src,dest)
                        shutil.copy(src, dest)
                        #count_l+=1

                    # XML
                    s = "xml"
                    print(datasetsx[i])
                    for c,filename in enumerate(os.listdir(data_path+ datasetsx[i] + '/' + data_folder +'/' )):
                        if '.xml' in filename:
                            #dest = f"{output_path}master/{s}/{str(count_x)}_{internal_name}.xml"
                            dest = f"{output_path}master/{s}/{internal_name}_{filename}"
                            # print(dest)
                            # print(src)
                            src =  f"{data_path}{datasetsx[i]}/{data_folder}/{filename}"
                            #print(src)
                            #  datasets/raw/CGHD-supplement.v6i.yolov8/train/images
                            #os.rename(src,dest)
                            shutil.copy(src, dest)
                            #count_x+=1

            # Classes - Parse yaml and store classes list 
            with open(data_path + dataset + '/' + "data.yaml", 'r') as stream:
                yaml_dict  = yaml.safe_load(stream)

            #print(type(yaml_dict['names']))
            #print(yaml_dict['roboflow']['url'])

            df['class_list'] =  df['class_list'].astype('object')

            # Get index of dataset then insert
            if not dataset=='Circuit Recognition.v9i.yolov8':
                df.at[index,'class_list'] =  yaml_dict['names']
            else:
                df.at[index,'class_list'] =  ['acv','arr','capacitor','ammeter','inductor','inductor2','resistor','voltmeter']

            df.at[index,'url'] =  yaml_dict['roboflow']['url']
            df.at[index,'classes'] =  yaml_dict['nc']
            df.at[index,'project'] =  yaml_dict['roboflow']['project']
            df.at[index,'version'] =  yaml_dict['roboflow']['version']
    
    # Write files
    df.to_csv('../datasets/processed/dataset.csv',index=None)

    