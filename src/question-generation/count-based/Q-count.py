"""
Read images and text and generate questions based on count
"""

import os 
import pandas as pd
import random
import ast 
import itertools
from utils.templates import count_templates
from utils.preprocessing import randomize_qs as randomize_qcounts
from icecream import ic 

#classes = ['acv', 'arr', 'c', 'i', 'l', 'l-', 'r', 'v']
classes = ['acv','arr','capacitor','ammeter','inductor','inductor2','resistor','voltmeter']


# Return a random count template question

if __name__ == "__main__":

    #data_path = "datasets/Circuit Recognition.v9i.yolov8"
    data_path = "datasets/"
    q_path = "datasets/questions/count/"

    # Iterate train,test,dev 
    count_qlist= [] 
    for dir in ['train','test','dev']:
        print(dir)
        for file in os.listdir(os.path.join(data_path,dir,'labels')):
            print(file)
            file_path = data_path + dir + '/labels/' + file
            image_name = file.replace(".txt","")
            #print(file)

            # Get classes 
            df = pd.read_csv(data_path + "/processed/" "dataset.csv")
            df['class_list'] = df['class_list'].apply(lambda s: list(ast.literal_eval(s)))
            classes = df[df.internal_name==file.split('.')[0][0:2]]['class_list']
            classes = list(itertools.chain(*classes))
            dataset_name = file.split('.')[0][0:2]

            # get labels
            count_dict = {}

            with open(file_path) as f:
                for line in f:
                    k = line.split(None,1)[0]
                    if dataset_name == "d2":
                        if k in ['__background__', 'text','unknown','junction','crossover','terminal']: # exclude these
                            #print("excluding...")
                            continue
                        elif k == 'voltage.dc':
                            count_dict['voltage-dc'] = count_dict.get('voltage-dc', 0) + 1
                        elif k == 'voltage.ac':
                            count_dict['voltage-ac'] = count_dict.get('voltage-ac', 0) + 1
                        elif k == 'capacitor.unpolarized':
                            count_dict['capacitor-unpolarized'] = count_dict.get('capacitor-unpolarized', 0) + 1
                        else:
                            count_dict[k] = count_dict.get(k, 0) + 1
                    else:
                        k = int(line[0])
                        count_dict[classes[k]] = count_dict.get(classes[k], 0) + 1
            #print(count_dict)

            for k,v in count_dict.items():
                # q = "How many " + k + "s does the given circuit have ?"    # Add random questions
                q = randomize_qcounts(count_templates)
                q = q.replace("XX", k)
                a = v
                count_qlist.append([dir,image_name,q,a,'count',k])

df = pd.DataFrame(count_qlist,columns =['splittype','file', 'question', 'answer','qtype','symbol'])
ic(df['file'].nunique())

df = df[~df.symbol.isin(['junction','terminal','text','inductor2'])]

df.to_csv(q_path +  "Q-count.csv",index=None)
ic(df.shape)
