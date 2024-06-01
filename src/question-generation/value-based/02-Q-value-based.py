"""
   Generate question based on symbol,value pairs per images
"""

# 
import pandas as pd 
import os 
import random
import ast 
import itertools
from utils.templates import value_templates,totalvalue_templates
from utils.preprocessing import randomize_qs
from icecream import ic 

#  class_dict = {'acv':'acv','arr':'arr','c':'capacitor','i':'ammeter','i':'inductor','l':'inductor2','r':'resistor','v':'voltmeter'}

DATA_PATH = "datasets/questions/value/dist-bbox/all/"
OUTPUT_PATH = "datasets/questions/value/"
MAIN_DATA_PATH = "datasets/"

# Pull directly from main data
if __name__ == "__main__":
   qlist_all = []
   qs_all_df = pd.DataFrame(columns =['splittype','file', 'question', 'answer','qtype'])

   for dir in ['train','test','dev']:
      print(dir)
      for file in os.listdir(os.path.join(MAIN_DATA_PATH,dir,"images")):
            file = file.split(".jpg")[0] + ".csv"
            file_path = os.path.join(DATA_PATH,file)
            image_name = file.replace(".csv","")

            ic(file,file_path,image_name)
            if os.path.isfile(file_path):
               #ic(os.stat(file_path).st_size)
               if os.stat(file_path).st_size <= 1:
                  ic("file empty")
                  continue
               else:
                  # Read bounding boxes from diff folder
                  df = pd.read_csv(file_path)
                  #ic(df.columns)
                  #ic(df)
                  if df.empty:
                     continue

                  df_group = df.groupby('s')['minval'].apply(list).reset_index(name='answer')
                  print(df_group)
                  qlist = []
                  for i,row in df_group.iterrows():
                     #if row['s'] in ['c','r']:
                     q = randomize_qs(value_templates)
                     a = row['answer']
                     if len(a)==1:
                        q = q.replace("XX",row['s'])
                        q = q.replace(" are "," is ")
                     else:
                        q = q.replace("XX",row['s']+'s')

                     qlist.append([dir,image_name,q,a,'value'])
                  qs_df = pd.DataFrame(qlist,columns =['splittype','file', 'question', 'answer','qtype'])  
                  print(qs_df)
                  qs_all_df = pd.concat([qs_all_df, qs_df], ignore_index=True)
            else:
               ic("File doesn't exist")
               continue
   print(qs_all_df.shape)
   qs_all_df.to_csv(OUTPUT_PATH + 'Q-value-new.csv',index=None)


#DATA_PATH = "datasets/questions/value/dist-bbox/"
#OUTPUT_PATH = "datasets/questions/value/"
# if __name__ == "__main__":
#    qlist_all = []
#    qs_all_df = pd.DataFrame(columns =['splittype','file', 'question', 'answer','qtype'])

#    for dir in ['train','test','dev']:
#       print(dir)
#       for file in os.listdir(os.path.join(DATA_PATH,dir)):
#             print(file)
#             file_path = os.path.join(DATA_PATH,dir,file)
#             image_name = file.replace(".csv","")
#             df = pd.read_csv(file_path)
#             #df = pd.read_csv("../../datasets/questions/value/train/autockt_-323_png.rf.a0afbaa7e18cd9ddb05aff68ecbcb38a.csv")
#             #print(df)
#             if df.empty:
#                continue

#             df_group = df.groupby('s')['minval'].apply(list).reset_index(name='answer')
#             print(df_group)
#             qlist = []
#             for i,row in df_group.iterrows():
#                #if row['s'] in ['c','r']:
#                q = randomize_qs(value_templates)
#                a = row['answer']
#                if len(a)==1:
#                   q = q.replace("XX",row['s'])
#                   q = q.replace(" are "," is ")
#                else:
#                   q = q.replace("XX",row['s']+'s')

#                qlist.append([dir,image_name,q,a,'value'])
#             qs_df = pd.DataFrame(qlist,columns =['splittype','file', 'question', 'answer','qtype'])  
#             print(qs_df)
#             qs_all_df = pd.concat([qs_all_df, qs_df], ignore_index=True)
#    #print(qs_all_df)
#    print(qs_all_df.shape)
#    qs_all_df.to_csv(OUTPUT_PATH + 'Q-value.csv',index=None)