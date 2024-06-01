import cv2 
import csv
import pandas as pd 
import os
import numpy as np
from icecream import ic 
from  utils.preprocessing  import xml_processor,starts_digit,cloud_value_bndbox,euclidean_distance
from utils.templates import junction_templates
from utils.symbols_desc import symbol_dict
import random 
import shutil

def class_cleaner(df,file):
    #print(file)
    dataset_name = file.split('/')[-1][0:2]
    #print(dataset_name)
    # d2 kaggle dataset specific - Exclude few classes
    if dataset_name == "d2":
        #print(df.shape)
        df = df[~df['s'].isin(['text','__background__','unknown','terminal'])].reset_index(drop=True)
        #print(df.shape)

        # replace 
        df['s'] = df['s'].replace(['voltage.dc'], 'voltage-dc')
        df['s'] = df['s'].replace(['voltage.ac'], 'voltage-ac')
        df['s'] = df['s'].replace(['capacitor.unpolarized'], 'capacitor-unpolarized')
        #df['s'] = df['s'].replace(['capacitor.polarized'], 'capacitor-polarized')
        #df['s'] = df['s'].replace(['transistor.photo'], 'transistor-photo')
        

    # d1 mappings a:ammeter
    elif dataset_name == "d1":
        df['s'] = df['s'].replace(['r'], 'resistor')
        df['s'] = df['s'].replace(['c'], 'capacitor')
        df['s'] = df['s'].replace(['i'], 'ammeter')
        df['s'] = df['s'].replace(['v'], 'voltmeter')
        df['s'] = df['s'].replace(['l'], 'inductor')
        df['s'] = df['s'].replace(['l-'], 'inductor')
        #print(df)

    elif dataset_name == "d3":
        df = df[~df['s'].isin(['text'])].reset_index(drop=True)

    return df

if __name__ == "__main__":

    INPUT_PATH = "datasets/model-inputs"
    OUTPUT_PATH_MAIN = "datasets/questions/junction/"
    OUTPUT_FILE = OUTPUT_PATH_MAIN + 'Q-junction.csv'

    if os.path.exists(os.path.join(OUTPUT_FILE)):
            os.remove(OUTPUT_FILE)
            ic("file removed")

    with open(OUTPUT_FILE, 'w') as file:
        writer = csv.writer(file)
        header = ['splittype','file','question','answer','qtype','symbol']
        writer.writerow(header)

    datasets = ['train','test','dev']

    cntr = 0 
    for ds in datasets:
        # Op directory

        # Check image folder    
        if not os.path.isdir(os.path.join(OUTPUT_PATH_MAIN,ds)):
            os.mkdir(os.path.join(OUTPUT_PATH_MAIN,ds))
        else:
            shutil.rmtree(os.path.join(OUTPUT_PATH_MAIN,ds))
            os.mkdir(os.path.join(OUTPUT_PATH_MAIN,ds))


        for filename in os.listdir(os.path.join(INPUT_PATH,ds,"images")):
            cntr+=1
            # if cntr>10:
            #     break
            ic(cntr)

            dataset_name = filename.split('/')[-1][0:2]
            #ic(dataset_name)
            if dataset_name in ['d1','d4']:
                ic("continue")
                continue

            op_filename = os.path.join(OUTPUT_PATH_MAIN,ds,filename)

            xml_file = filename.split(".jpg")[0] + ".xml"
            xml_path = os.path.join(INPUT_PATH,ds,"xml",xml_file)
            ic(dataset_name,ds,filename,xml_file,op_filename)

            image =  cv2.imread(os.path.join(INPUT_PATH,ds,"images",filename), cv2.IMREAD_GRAYSCALE)

            # xml_path = os.path.join(INPUT_PATH,"xml","d2_C74_D2_P4.xml")
            # image = cv2.imread(os.path.join(INPUT_PATH,"images","d2_C74_D2_P4.jpg"), cv2.IMREAD_GRAYSCALE)
            
            # xml_path = os.path.join(INPUT_PATH,"xml","d2_C75_D1_P1.xml")
            # image = cv2.imread(os.path.join(INPUT_PATH,"images","d2_C75_D1_P1.jpg"), cv2.IMREAD_GRAYSCALE)
            

            #filename = "d2_C51_D1_P2.jpg"
            #xml_path = os.path.join(INPUT_PATH,"xml","d2_C51_D1_P2.xml")
            #image = cv2.imread(os.path.join(INPUT_PATH,"images","d2_C51_D1_P2.jpg"), cv2.IMREAD_GRAYSCALE)

            # Put junctions ids in the images 
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = min(image.shape[0], image.shape[1]) / 1000.0
            color = (0,0,255)
            thickness = 2


            # keep one with smallest        
            symbol_df = xml_processor(xml_path)
            symbol_df = class_cleaner(symbol_df,xml_path)
            symbol_df.info()

            # 
            #ic(symbol_df)

            junc_df = symbol_df[symbol_df.s=='junction'].reset_index(drop=True)
            item_df = symbol_df[symbol_df['s']!='junction'].reset_index(drop=True)
            #ic(junc_df,item_df)

            if junc_df.shape[0] < 2:
                ic("junc_df <2")
                continue

    # Plot junctions
    # color = (0, 255, 0)  # Green color (BGR format)
    # thickness = 2   
    # for i,row in junc_df.iterrows():
    #     x1,x2,y1,y2 = int(row['xmin']),int(row['xmax']),int(row['ymin']),int(row['ymax'])
    #     #ic(x1,x2,y1,y2)
    #     cv2.rectangle(image,(x1, y1), (x2, y2), color, thickness)
    # # cv2.imshow("rectangles",image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
        
            # Check items between junction
            triplet_list = []
            for i,low in junc_df.iterrows():
                #ic(type(low))
                for j,high in junc_df.iterrows():
                    if i!=j:
                        # for the junction pair vertically, find if some item exists
                        for k,irow in item_df.iterrows():
                            
                            if irow['ymin'] > low['ymin'] and irow['ymax'] < high['ymin'] and (int(low['xmin']) <= int(irow['xmax']) <=  int(low['xmax'])+60)  :
                                d1 = euclidean_distance((low['xmin'],low['ymin']),(irow['xmin'],irow['xmax'])) 
                                d2 = euclidean_distance((irow['xmax'],irow['ymin']),(high['xmax'],high['ymax'])) 
                                triplet_list.append([list(low),list(high),list(irow),d1+d2])

                            # if irow['xmin'] > low['xmin'] and irow['xmax'] < high['xmin'] and (low['ymin'] <= irow['ymax'] <= low['ymax']+50):
                            #     ic(irow)
                            #     d1 = euclidean_distance((low['xmin'],low['ymin']),(irow['xmin'],irow['xmax'])) 
                            #     d2 = euclidean_distance((irow['xmax'],irow['ymin']),(high['xmax'],high['ymax'])) 
                            #     triplet_list.append([list(low),list(high),list(irow),d1+d2]) 
                            #     #ic(i,j,k)

                        # for the junction pair horizontally, find if some item exists
                    #for k,irow in item_df.iterrows():

            # ic(triplet_list)

            # keep smallest 
            all_min_triplets = []
            for k,irow in item_df.iterrows():
                min_dist = 100000000
                min_triplet = [[]]
                for item in triplet_list:
                    if irow['s'] == item[2][0] and irow['xmin']==item[2][2] and irow['xmax']==item[2][1] and irow['ymin'] == item[2][4]  and irow['ymax']==item[2][3]:
                        # distance item[3]
                        if item[3] < min_dist:
                            #ic(item[3],min_dist)
                            min_triplet = item
                            min_dist = item[3]
                    else:
                        continue 

                if min_triplet!=[[]]:
                    all_min_triplets.append(min_triplet)
            
            #ic(all_min_triplets)

            # Draw these items found b/w junctions (Optional)
            color = (0, 255, 0)  # Green color (BGR format)
            thickness = 2   
            cnt= 0     
    
            # Annotate junction in image
            for index,row in junc_df.iterrows():
                text = "J" + str(index+1)
                coordinates = (row['xmin'],row['ymin'])
                cv2.putText(image, text, coordinates, font, fontScale, color, thickness)

            #cv2.imshow("jn",image)
            #cv2.waitKey(0)
            cv2.imwrite(op_filename,image)

            # cv2.imshow("rectangles",image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # Generate junction Questions
            #unique_components = list(item_df['s'].unique())
            unique_components = list(symbol_dict.keys())
            unique_components = [x.lower() for x in unique_components] 

            q_list = []
            random.seed(42)
            for item in all_min_triplets:
                component = item[2][0]
                j1 = item[0]
                j2 = item[1]
                junction1_id = -1
                for index,row in junc_df.iterrows():
                    if j1[2] == row['xmin'] and j1[1] == row['xmax'] and j1[4] == row['ymin'] and j1[3] == row['ymax']:
                        junction1_id = index+1

                junction2_id = -1
                for index,row in junc_df.iterrows():
                    if j2[2] == row['xmin'] and j2[1] == row['xmax'] and j2[4] == row['ymin'] and j2[3] == row['ymax']:
                        junction2_id = index+1  
                
                #random.seed(42)
                q = random.choice(junction_templates)
                #ic(q)
                q = q.replace("XX",component)
                q = q.replace("YY",str(junction1_id))
                q = q.replace("ZZ",str(junction2_id))
                answer = "Yes"
                q_list.append([q,answer])
                

                q2 = random.choice(junction_templates)

                unique_components_except = [i for i in unique_components if i!=component]
                component2 = random.choice(unique_components_except)
                #ic(q,component,q2,unique_components_except)

                q2 = q2.replace("XX",component2)
                q2 = q2.replace("YY",str(junction1_id))
                q2 = q2.replace("ZZ",str(junction2_id))
                answer2 = "No"
                #ic(q2,answer2)

                q_list.append([q2,answer2])

                #ic(q_list)
                header = ['splittype','file','question','answer','qtype','symbol']
            
            qlist_main = []
            for index,item in enumerate(q_list):
                qlist_main.append(['train',filename, item[0],item[1],'junction'])

            #ic(qlist_main)

            with open(OUTPUT_FILE, 'a') as f:
                writer = csv.writer(f)
                writer.writerows(qlist_main)






















