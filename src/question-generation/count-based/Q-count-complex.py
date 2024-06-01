"""
Count based

How many resisters are on the left of the ammeter ?
How many gates are connected to the and gates directly ?

"""
import cv2 
import csv
import pandas as pd 
import os 
import ast 
import cv2
from utils.preprocessing import xml_processor,randomize_qs,class_cleaner
from  utils.templates import complex_count_templates
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys 
import math
import random
from collections import defaultdict
from shapely.geometry import Polygon, LineString
from PIL import Image  
import PIL  
import matplotlib.pyplot as plt
from icecream import ic 

np.set_printoptions(sys.maxsize)

INPUT_PATH = "datasets/"
OUTPUT_PATH_MAIN = "datasets/questions/count-complex/"

OUTPUT_FILE = OUTPUT_PATH_MAIN + '/Q-count-complex.csv'
MAX_SAMPLES_PER_Q = 6
MIN_SAMPLES_PER_Q = 2


if __name__ == "__main__":
    datasets = ['train','test','dev']
    SAMPLE_LIMITS = [300,100,100]
    NUM_SAMPLES = [140,40,20] 

    random.seed(42)


    if os.path.isfile(OUTPUT_FILE):
       os.remove(OUTPUT_FILE) 
       ic("File removed")

    for ds,SAMPLE_LIMIT,SAMPLES in zip(datasets,SAMPLE_LIMITS,NUM_SAMPLES):
        
        XML_PATH = os.path.join(INPUT_PATH,ds,'xml')

        print(ds)
        if not os.path.exists(os.path.join(OUTPUT_PATH_MAIN,ds)):
            os.mkdir(OUTPUT_PATH_MAIN + ds)
        OUTPUT_PATH = OUTPUT_PATH_MAIN + ds + '/'

        if not os.path.exists(os.path.join(OUTPUT_PATH,'images')):
            os.mkdir(OUTPUT_PATH+ 'images')
        OUTPUT_PATH = OUTPUT_PATH + '/images'
        print(OUTPUT_PATH)

        ##### For each dataset,sample files 
        dataset_dict = {}
        dataset_dict["d1"] = []
        dataset_dict["d2"] = []
        dataset_dict["d3"] = []
        dataset_dict["d4"] = []
        dataset_dict["d5"] = []

        for c,filename in enumerate(os.listdir(os.path.join(INPUT_PATH,ds,'images'))):
            dataset_name = filename.split('/')[-1][0:2]
            dataset_dict[dataset_name].append(filename)


        dataset_dict_sample = defaultdict()
        for dataset_name in dataset_dict.keys():
            dataset_dict_sample[dataset_name] = set()
       
       # Keep a set of lists of images
        
        k = int(SAMPLE_LIMIT/5)
        for index,(dataset_name,v) in enumerate(dataset_dict.items()):
            ic(dataset_name)
            size = len(dataset_dict[dataset_name])
            j = 0 
            random.seed(42)
            num_list = random.sample(range(0, size),size)
            #ic(len(num_list),size)
            for i,num in enumerate(num_list):
                if j >=k:
                    break
                #ic(dataset_dict[dataset_name][num],dataset_dict_sample[dataset_name])
                if dataset_dict[dataset_name][num] not in dataset_dict_sample[dataset_name]:
                    #ic(dataset_dict[dataset_name][num])
                    j+=1
                    dataset_dict_sample[dataset_name].add(dataset_dict[dataset_name][num])
                else:
                    continue

        #ic(dataset_dict_sample)
        #ic(k,ds,len(dataset_dict_sample['d1']),len(dataset_dict_sample['d2']),len(dataset_dict_sample['d3']),len(dataset_dict_sample['d4']),len(dataset_dict_sample['d5']))

        # Write qlist of all to csv
        if ds=='train':
            with open(OUTPUT_PATH_MAIN + 'Q-count-complex.csv', 'a') as file:
                writer = csv.writer(file)
                header = ['splittype','file','question','answer','qtype','symbol']
                writer.writerow(header)

        for dataset_name in dataset_dict_sample.keys():
            cnt=0
            l = list(dataset_dict_sample[dataset_name])
            l.sort()
            for index,filename in enumerate(l):
                ic(dataset_name,filename)

                file_path = INPUT_PATH + ds + '/images/' +  filename  
                xmlfile = XML_PATH + '/' + file_path.split('/')[-1] 
                xmlfile = xmlfile.replace('.JPG','.xml')
                xmlfile = xmlfile.replace('.jpg','.xml')
                xmlfile = xmlfile.replace('.jpeg','.xml')
                
                op_filename =  OUTPUT_PATH + '/' +  file_path.split('/')[-1]
                # Add labels to gates and save image
                image = cv2.imread(file_path)

                symbol_df = xml_processor(xmlfile)
                symbol_df = class_cleaner(symbol_df,file_path)

                symbol_df = symbol_df[~symbol_df['s'].isin(['__background__', 'text','unknown','junction'])]
                if symbol_df.empty or symbol_df.shape[0]==2:
                    continue
                cnt+=1
                # if cnt >=2:
                #     break

                font = cv2.FONT_HERSHEY_SIMPLEX
                #print(image.shape)
                fontScale = min(image.shape[0], image.shape[1]) / 1000.0
                color = (255,0,255)

                # if dataset_name =="d1":
                #     fontScale =1
                # else:
                #     fontScale = min(image.shape[0], image.shape[1]) / 1000.0

                # fontscale=1    
                thickness = 1
                
                # unique gates
                unique_gates = symbol_df['s'].unique()
                #print(unique_gates)

                #gates_qlist = []
                qlist = set()
                symbol_df = symbol_df.astype({"xmin": int, "xmax": int, "ymin":int,"ymax":int})
                i = 1
                for index,row in symbol_df.iterrows():
                    if dataset_name == "d4":
                        text = "Gate" + str(i)
                    else:
                        text = "C" + str(i)
                    #print(int(row['xmin'])+int(row['xmax']))
                    #coordinates = (int((row['xmin']+row['xmax'])/2),int((row['ymin']+row['ymax'])/2))
                    coordinates = (row['xmin'],row['ymin'])
                    #ic(i,text,coordinates,row['xmin'],row['xmax'],row['ymin'],row['ymax'])
                    cv2.putText(image, text, coordinates, font, fontScale, color, thickness)
                    i+=1
                    for q in complex_count_templates[dataset_name]:
                        #print(q)
                        if "How many YY" in q:
                            for gate in unique_gates:
                                if gate == row['s']:
                                    continue
                                else:
                                    #print(gate,text,q)
                                    q = q.replace("XX", text)
                                    q = q.replace("YY", gate)
                                qlist.add(q)
                        else:
                            q = q.replace("XX", text)
                            qlist.add(q)             
                #cv2.destroyAllWindows()


                cv2.imwrite(op_filename,image)
                qlist = list(qlist)
                qlist.sort()
                qlist_size = len(qlist)
                #ic(qlist,len(qlist))

                if qlist_size <= 3:
                    continue
                elif qlist_size > 3 and qlist_size <=10:
                    random.seed(42)
                    qlist_sample = random.sample(qlist, MIN_SAMPLES_PER_Q)
                elif qlist_size > 10 and qlist_size <=20:
                    random.seed(42)
                    qlist_sample = random.sample(qlist, MIN_SAMPLES_PER_Q+2)
                else:
                    random.seed(42)
                    qlist_sample = random.sample(qlist, MAX_SAMPLES_PER_Q)
                
                # Create list of lists
                header = ['splittype','file','question','answer','qtype','symbol']
                qlist_main = []
                for q in qlist_sample:
                    qlist_main.append([ds,filename,q,'','count-complex',''])

                ic(filename,len(qlist_main))
                
                with open(OUTPUT_FILE, 'a') as file:
                    writer = csv.writer(file)
                    writer.writerows(qlist_main)



