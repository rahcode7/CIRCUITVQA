"""

With xmax maximum, generate Q - Which is the rightmost gate ?
With xmim mimimum, generate Q - Which is the leftmost gate ?
"""

import cv2 
import pandas as pd 
import os 
import ast 
from utils.preprocessing import xml_processor,randomize_qs,class_cleaner
from utils.templates import symbol_position

#IMAGE_PATH = "datasets/train/images/img248_png.rf.1a79b5b566dfca51b0067d21ed0b4991.jpg"
#XML_PATH = "datasets/train/xml/img248_png.rf.1a79b5b566dfca51b0067d21ed0b4991.xml"


DATA_PATH = "datasets/"
OUTPUT_PATH = "datasets/questions/position"

if __name__ == "__main__":
    # Query bounding boxes 
    qlist_all = []
    qs_all_df = pd.DataFrame(columns =['splittype','file', 'question', 'answer','qtype'])
   
    for dir in ['train','test','dev']:
        print(dir)
        for file in os.listdir(os.path.join(DATA_PATH,dir,'xml')):
            #print(file)
            file_path = DATA_PATH + dir + '/' + 'xml/' + file
            file_path
            images_path = DATA_PATH + dir + '/' + 'images/' + file 
            images_path = images_path.replace(".xml", ".jpg")
            image_name = file.replace(".xml","")
        
            df = xml_processor(file_path) #change to file_path
            df = class_cleaner(df,file_path)
            #print(df)
            qlist = []

            if df.empty:
                continue 

            # left
            df['xmin'] = df['xmin'].astype(str).astype(int)
            #print(df.info())
            index = df['xmin'].idxmin(axis=0)

            if df[df['xmin']==df['xmin'].min()].shape[0]==1:  #Create Q if only 1 answer available 
                row = df.iloc[index]
                q = randomize_qs(symbol_position)
                q = q.replace("XX","left")
                a = row['s']
                qlist.append([dir,image_name,q,a,'position'])


            # right
            df['xmax'] = df['xmax'].astype(str).astype(int)
            if df[df['xmax']==df['xmax'].max()].shape[0]==1:
                index = df['xmax'].idxmax(axis=0)
                row = df.iloc[index]
                
                q = randomize_qs(symbol_position)
                q = q.replace("XX","right")
                a = row["s"]
                qlist.append([dir,image_name,q,a,'position'])
                
            # top
            df['ymax'] = df['ymax'].astype(str).astype(int)
            if df[df['ymax']==df['ymax'].max()].shape[0]==1:
                index = df['ymax'].idxmax(axis=0)
                row = df.iloc[index]
                
                q = randomize_qs(symbol_position)
                q = q.replace("XX","top")
                a = row["s"]
                qlist.append([dir,image_name,q,a,'position'])


            # bottom
            df['ymin'] = df['ymin'].astype(str).astype(int)
            #print(df.info())   
            index = df['ymin'].idxmin(axis=0)

            if df[df['ymin']==df['ymin'].min()].shape[0]==1:  #Create Q if only 1 answer available 
                row = df.iloc[index]
                q = randomize_qs(symbol_position)
                q = q.replace("XX","bottom")
                a = row['s']
                qlist.append([dir,image_name,q,a,'position'])


            qs_df = pd.DataFrame(qlist,columns =['splittype','file', 'question', 'answer','qtype'])  
            qs_all_df = pd.concat([qs_all_df, qs_df], ignore_index=True)
    print(qs_all_df)

    
    qs_all_df.to_csv(OUTPUT_PATH +'/Q-position.csv',index=None)
    






