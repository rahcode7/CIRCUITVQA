#from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import numpy as np
import os 
import pandas as pd
import itertools
import ast
import shutil
from sklearn.model_selection import train_test_split
from icecream import ic 
import collections
import json 
import logging

TRAIN_SIZE = 0.7
TEST_SIZE = 0.2
DEV_SIZE = 0.1

"""
  X : list of image ids
  
"""
def random_split(X,split_size):
   train, test = train_test_split(X, train_size=split_size)
   print(train[0:5])
   return train,test 

if __name__ == "__main__":

   data_path = 'datasets/master/'
   mid_path = "datasets/processed/"
   # Add images to list
   X = []
   for file in os.listdir(os.path.join(data_path,'images/')):
      #file_path = data_path + '/labels/' + file
      #X.append(file.split('.txt')[0])
      X.append(file)

   print(X[0:5])
   print([item for item, count in collections.Counter(X).items() if count > 1])

   ic(len(X))
   X = list(set(X))
   ic(len(set(X)))


   # SPECIFY SPLIT Randomize to 70-20-10 split 
   #print(len(X))
   train,test = random_split(X,TRAIN_SIZE)
   test,dev = random_split(test,TEST_SIZE/(DEV_SIZE+TEST_SIZE))

   # Save statistics
   
   print(len(train),len(test),len(dev))
   
   f = open(mid_path + "stats-datasplits.txt",'w+')
   f.write("Training images :" + str(len(train)) + "\n") 
   f.write("Test images :" + str(len(test))  + "\n")
   f.write("Val images :" + str(len(dev)) + "\n")
   f.write("Total images :" + str(len(test) + len(train) + len(dev)) + "\n")

   f.close()

   ic(list(set(train).intersection(test)))
   ic(list(set(test).intersection(dev)))

   #Create files for train
   output_path  =  "datasets/"
   

   # Prepare train test dev folders with images and txt from master dataset using the above split
   sets = [train,test,dev]
   
   ids = {}
   ids['train'] = train
   ids['test'] = test
   ids['dev'] = dev 

   with open(mid_path + 'splits.json', 'w') as f:
      json.dump(ids, f)   

   for dataset in sets:
      if dataset == train:
         dataset_folder = 'train'
      elif dataset == test:
         dataset_folder = 'test'
      else:
         dataset_folder = 'dev'
      print(dataset_folder)


      if not os.path.exists(os.path.join(output_path,dataset_folder)):
         os.mkdir(output_path + dataset_folder)
      if not os.path.exists(os.path.join(output_path,dataset_folder,'images')):
         os.mkdir(output_path + dataset_folder + '/images')
      if not os.path.exists(os.path.join(output_path,dataset_folder,'labels')):
         os.mkdir(output_path + dataset_folder + '/labels')
      if not os.path.exists(os.path.join(output_path,dataset_folder,'xml')):
         os.mkdir(output_path + dataset_folder + '/xml')
   
      #print(id)
      #cnt=0
      for id in dataset:
         print(id)
         
         # cnt+=1
         # if cnt >=10:
         #    break

         s = "images"
         filename = id 
         dest = f"{output_path}{dataset_folder}/{s}/{filename}"
         src =  f"{data_path}{s}/{filename}"
         shutil.copy(src, dest)

         s = "labels"
         filename_l = filename.split(".jpg")[0] + '.txt'
         dest = f"{output_path}{dataset_folder}/{s}/{filename_l}"
         src =  f"{data_path}{s}/{filename_l}"
         shutil.copy(src, dest)

         s = "xml"
         filename_x = filename.split(".jpg")[0] + '.xml'
         dest = f"{output_path}{dataset_folder}/{s}/{filename_x}"
         src =  f"{data_path}{s}/{filename_x}"
         shutil.copy(src, dest)

 


