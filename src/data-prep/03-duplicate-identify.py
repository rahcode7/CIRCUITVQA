# pip install imagededup
from imagededup.methods import PHash
from imagededup.utils import plot_duplicates
from collections import defaultdict
import os 
import json 
from tqdm import tqdm 
from icecream import ic

DATA_PATH = "datasets/master/"
OUTPUT_PATH = "datasets/processed"

if __name__ == '__main__':
    phasher = PHash()

    # Get hashes
    encodings_dict = defaultdict()
    encoding = phasher.encode_images(image_dir=os.path.join(DATA_PATH,'images'))

    duplicates = phasher.find_duplicates(encoding_map=encoding,scores=True,max_distance_threshold=0)
    print(duplicates)

    # count pairs
    cnt=0
    for k,v in duplicates.items():
        if len(v)!=0:
            cnt+=1
    print(cnt)


    # Sort dict albhabetically
    duplicates = dict(sorted(duplicates.items()))


    #plot_duplicates(image_dir=IMAGE_PATH,duplicate_map=duplicates,filename='d2_C33_D2_P1.jpg')
    with open(OUTPUT_PATH + "/duplicate_images.json", "w") as file:
        json.dump(duplicates , file)


    with open(OUTPUT_PATH + "/duplicate_images.json", "r") as file:
        duplicates = json.load(file)
    print(duplicates)

    
    unique_dict = defaultdict()
    cnt=0
    for i,(k,v) in enumerate(tqdm(duplicates.items())):
        cnt+=1
        #ic(cnt)
        if not v:
            continue
        else:
            v = v[0][0]
            for i,(k2,v2) in enumerate(duplicates.items()):
                if not v2:
                    continue
                elif k in unique_dict.values():
                    ku = [ku for ku, vu in unique_dict.items() if vu == k]
                    #ic(ku)
                    if ku[0] == v:
                        continue
                else:
                    v2 = v2[0][0]
                    #ic(k,v,k2,v2)  
                    if k == v2 and v == k2:
                        unique_dict[k] = v 
                    elif k==k2 and v==v2:
                        continue
                    elif k==k2 and v!=v2:
                        unique_dict[k].append(v2)
                    # elif not v:
                    #     continue
                    else:
                        continue
                        #unique_dict[k] = v


    with open(OUTPUT_PATH + "/duplicate_unique.json", "w") as file:
        json.dump(unique_dict , file)               
                

   
            
   



