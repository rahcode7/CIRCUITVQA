# For master dataset folder, remove duplicate files
import os 
import json 
from tqdm import tqdm 
from icecream import ic

DATA_PATH = "datasets/master/"
DUP_PATH = "datasets/processed"



def file_count(DATA_PATH):
    _, _, files = next(os.walk(DATA_PATH))
    return len(files)

if __name__ == "__main__":
    # dups
    with open(DUP_PATH + "/duplicate_unique.json", "r") as file:
        duplicates = json.load(file)
    #(duplicates)

    ic(file_count(DATA_PATH+'/images'))
    ic(file_count(DATA_PATH+'/xml'))
    ic(file_count(DATA_PATH+'/labels'))


    # Iterate dict, and remove values
    for i,(k,file_remove) in enumerate(tqdm(duplicates.items())):
        os.remove(DATA_PATH + '/images/' +  file_remove)
        os.remove(DATA_PATH + '/xml/' +  file_remove.split(".jpg")[0] + ".xml")
        os.remove(DATA_PATH + '/labels/' +  file_remove.split(".jpg")[0] + ".txt")

    ic(file_count(DATA_PATH+'/images'))
    ic(file_count(DATA_PATH+'/xml'))
    ic(file_count(DATA_PATH+'/labels'))
