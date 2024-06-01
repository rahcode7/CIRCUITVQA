import xml.etree.ElementTree as ET
import pandas as pd 
import random 
from google.cloud import vision
from google.oauth2 import service_account
from pathlib import Path
from icecream import ic 
#credentials = service_account.Credentials.from_service_account_file('service_acc_key.json')


def xml_processor(file):
    tree = ET.parse(file)
    root = tree.getroot()
    #print(root.tag)

    flag = 0
    #columns = ['s','junk','xmin','xmax','ymin','ymax','text']
    columns = ['s','xmax','xmin','ymax','ymin']
    rows = []
    for obj in root.iter('object'):
        for child in obj:
            t = []
            if child.tag == 'name':
                l = [child.text]
                #print(l)
            #print(child.tag, child.text)
            if child.tag == 'bndbox':
                for c in child.iter():
                    #ic(c)
                    if c.tag == 'xmin':
                        t.append(('xmin',c.text))
                    if c.tag == 'xmax':
                        t.append(('xmax',c.text))
                    if c.tag == 'ymin':
                        t.append(('ymin',c.text))
                    if c.tag == 'ymax':
                        t.append(('ymax',c.text))
                    #ic(t)
                t.sort(key=lambda a: a[0])
                
                #ic(t)
                for item in t:
                    l.append(item[1])
                #ic(l)
                rows.append(l)


    df = pd.DataFrame(rows,columns=columns)
    #del df['junk']
    #print(df)

    return df 

def class_cleaner(df,file):
    #print(file)
    dataset_name = file.split('/')[-1][0:2]
    #print(dataset_name)
    # d2 kaggle dataset specific - Exclude few classes
    if dataset_name == "d2":
        #print(df.shape)
        df = df[~df['s'].isin(['text','__background__','unknown','junction','terminal'])].reset_index(drop=True)
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
        df = df[~df['s'].isin(['text','junction'])].reset_index(drop=True)

    return df


def randomize_qs(templates):  
    x = len(templates)
    rindex = random.randrange(0,x)
    return templates[rindex]

#def randomize_qcounts(count_templates):
#     x = len(count_templates)
#     rindex = random.randrange(0,x)
#     return count_templates[rindex]
 
def detect_text(path):
    """Detects text in the file."""
    #client = vision.ImageAnnotatorClient(credentials=credentials)
    client = vision.ImageAnnotatorClient()

    path = Path(path)

    if path.is_file():
        with open(path, "rb") as image_file:
            content = image_file.read()

        print("Sending image", path)
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        texts = response.text_annotations
        print(texts)
        return(texts)
    else:
        return ""
    # for text in texts:
    #     print(f'\n"{text.description}"')

    #     vertices = [
    #         f"({vertex.x},{vertex.y})" for vertex in text.bounding_poly.vertices
    #     ]

    #     print("bounds: {}".format(",".join(vertices)))

    # if response.error.message:
    #     raise Exception(
    #         "{}\nFor more info on error messages, check: "
    #         "https://cloud.google.com/apis/design/errors".format(response.error.message)
    #     )
    #texts = ""
    


def starts_digit(s):
    #for c in s:
    if s[0].isdigit():
        return True
    return False

# Format response into a dataframe of values and bounding boxes 
def cloud_value_bndbox(image_file):
    texts  = detect_text(image_file)
    if texts == "":
        columns = ['val','c1','c2','c3','c4']
        print("Empty returned from google vision")
        val_df = pd.DataFrame(columns=columns)
        return val_df
    else:
        columns = ['val','c1','c2','c3','c4']
        rows = []
        for t in texts:
            l = [str(t.description)]
            if '\n' in str(t.description):
                continue
            else:
                for vertex in t.bounding_poly.vertices:
                    l.append((vertex.x,vertex.y))
                rows.append(l)
        val_df = pd.DataFrame(rows,columns=columns)
        return val_df

def euclidean_distance(d1,d2):
    from scipy.spatial.distance import euclidean
    dst = euclidean([d1[0],d1[1]], [d2[0],d2[1]])
    
    return float(dst)
