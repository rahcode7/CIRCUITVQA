from torch.utils.data import Dataset,DataLoader
import os 
from PIL import Image
import json 
from icecream import ic 

class VQACircuitDataset(Dataset):
    def __init__(self,root_dir,Q_PATH,split,ocr=False,desc=False) -> None:
        with open(Q_PATH, 'r',encoding='utf8') as f:
            self.qa = json.load(f)

        self.split = split  
        self.qa = [i for i in self.qa if i['splittype']==split]
        self.root_dir = root_dir
        self.ocr = ocr 
        self.desc = desc
        
    def __len__(self):
        return len(self.qa)

    def __getitem__(self,idx):
        file_name = self.qa[idx]["file"] + '.jpg'
        question = self.qa[idx]["question"]
        answer = self.qa[idx]["answer"]
        qtype = self.qa[idx]["qtype"]

        if self.ocr:
            ocr = self.qa[idx]["ocr"]
        else:
            ocr = ""

        if self.desc:
            desc = self.qa[idx]["desc"]
        else:
            desc = ""
        
        if self.bbox_flag:
            bbox = self.qa[idx]["bbox"]
        else:
            bbox = ""
    
        if self.bbox_flag:
            bbox_segment = self.qa[idx]["bbox_segment"]
        else:
            bbox_segment = ""

        if qtype == "junction":
            # try:    
            image_path = os.path.join(self.root_dir,'model-inputs-jn',self.split,file_name)
            image = Image.open(image_path).convert('RGB')   
            # except:
            #     return None
        else:
            image_path = os.path.join(self.root_dir,'model-inputs',self.split,file_name)
            image = Image.open(image_path).convert('RGB')   

        return file_name,image,question,answer,qtype,desc,ocr,bbox,bbox_segment
    
def vqa_collate_fn(batch):
    #len_batch = len(batch)
    file_list,image_list, question_list, answer_list,qtype_list,desc_list,ocr_list,bbox_list,bbox_segment_list = [],[], [], [],[],[],[],[],[]
    #ic(batch)

    batch = list(filter (lambda x:x is not None, batch))
    #ic(batch)
    for file_name,image, question, answer,qtype,desc,ocr,bbox,bbox_segment in batch:
        file_list.append(file_name)
        image_list.append(image)
        question_list.append(question)    
        answer_list.append(str(answer))
        qtype_list.append(qtype)
        desc_list.append(desc)
        ocr_list.append(ocr)
        bbox_list.append(bbox)
        bbox_segment_list.append(bbox_segment)

    return file_list,image_list,question_list,answer_list,qtype_list,desc_list,ocr_list,bbox_list,bbox_segment_list
