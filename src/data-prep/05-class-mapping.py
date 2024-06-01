

# Combine all yaml, create new mapping ids, map to new ids 
import os 
import pandas as pd
import numpy as np
import json 
from icecream import ic
import collections
import ast
#from utils.preprocessing import xml_processor,class_cleaner

# yaml_path = "  datasets/raw/CGHD-supplement.v6i.yolov8/data.yaml"
# data_path = "  datasets/master/"
# output_path = "  datasets/"
dataset_df = pd.read_csv("datasets/processed/dataset.csv")
op = "src/utils"
# for all yaml files 
# with open(data_path + dataset + '/' + "data.yaml", 'r') as stream:
#     yaml_dict  = yaml.safe_load(stream)

#  df = df[~df['s'].isin(['text','__background__','unknown','junction','terminal'])].reset_index(drop=True)
#         #print(df.shape)

#         # replace 
#         df['s'] = df['s'].replace(['voltage.dc'], 'voltage-dc')
#         df['s'] = df['s'].replace(['voltage.ac'], 'voltage-ac')
#         df['s'] = df['s'].replace(['capacitor.unpolarized'], 'capacitor-unpolarized')
#         #df['s'] = df['s'].replace(['capacitor.polarized'], 'capacitor-polarized')
#         #df['s'] = df['s'].replace(['transistor.photo'], 'transistor-photo')
        
#     # d1 mappings a:ammeter
#     elif dataset_name == "d1":
#         df['s'] = df['s'].replace(['r'], 'resistor')
#         df['s'] = df['s'].replace(['c'], 'capacitor')
#         df['s'] = df['s'].replace(['i'], 'ammeter')
#         df['s'] = df['s'].replace(['v'], 'voltmeter')
#         df['s'] = df['s'].replace(['l'], 'inductor')
#         df['s'] = df['s'].replace(['l-'], 'inductor')
#         #print(df)

#     elif dataset_name == "d3":
#         df = df[~df['s'].isin(['text','junction'])].reset_index(drop=True)

# For each dataset file in master update labels file with new ids being assigned 


# 
class_newids_all = ['and', 'nand', 'nor', 'not', 'or', 'xnor', 'xor','acv', 'arr', 'capacitor', 'ammeter', 'inductor', 'inductor2', 'resistor', 'voltmeter',
 'capacitor-unpolarized','crossover', 'current-source', 'diode', 'gnd', 'junction', 'multi-cell-battery',
 'single-cell-battery', 'terminal', 'text', 'voltage-dc', 'voltage-dc_ac','integrated_circuit', 'lamp', 'operational_amplifier', 'optocoupler', 'socket', 'speaker', 'switch', 'thyristor', 'transformer', 'transistor', 'triac', 'varistor', 'voltage-ac', 'voltage-dc', 'vss',
 '__background__', 'text', 'junction', 'terminal', 'gnd', 'vss', 'voltage.dc', 'voltage.ac', 'voltage.battery', 'resistor', 'resistor.adjustable', 'resistor.photo', 'capacitor.unpolarized', 'capacitor.polarized', 'capacitor.adjustable', 'inductor', 'inductor.ferrite', 'inductor.coupled', 'transformer', 'diode', 'diode.light_emitting', 'diode.thyrector', 'diode.zener', 'diac', 'triac', 'thyristor', 'varistor', 'transistor.bjt', 'transistor.fet', 'transistor.photo', 'operational_amplifier', 'operational_amplifier.schmitt_trigger', 'optocoupler', 'integrated_circuit', 'integrated_circuit.ne555', 'integrated_circuit.voltage_regulator', 'xor', 'and', 'or', 'not', 'nand', 'nor', 'probe', 'probe.current', 'probe.voltage', 'switch', 'relay', 'socket', 'fuse', 'speaker', 'motor', 'lamp', 'microphone', 'antenna', 'crystal', 'mechanical', 'magnetic', 'optical', 'block', 'unknown'
] 

# 

class_newids = []
[class_newids.append(symbol) for symbol in class_newids_all if symbol not in class_newids]
sorted(class_newids, key=str.lower)


class_newids = ['__background__', 'acv', 'ammeter', 'and', 'antenna', 'arr', 'block', 'capacitor', 'capacitor-unpolarized', 
 'capacitor.adjustable', 'capacitor-polarized', 'crossover', 'crystal', 'current-source',
   'diac', 'diode', 'diode.light_emitting', 'diode.thyrector', 'diode.zener',
     'fuse', 'gnd', 'inductor', 'inductor.coupled', 'inductor.ferrite', 'inductor2',
       'integrated_circuit', 'integrated_circuit.ne555', 'integrated_circuit.voltage_regulator',
        'junction', 'lamp', 'magnetic', 'mechanical', 'microphone', 'motor', 'multi-cell-battery', 'nand',
         'nor', 'not', 'operational_amplifier', 'operational_amplifier.schmitt_trigger', 'optical',
           'optocoupler', 'or', 'probe', 'probe.current', 'probe.voltage', 'relay', 'resistor', 'resistor.adjustable',
             'resistor.photo', 'single-cell-battery', 'socket', 'speaker', 'switch', 'terminal', 'text',
               'thyristor', 'transformer', 'transistor', 'transistor-photo', 'transistor.bjt',
                 'transistor.fet', 'triac', 'unknown', 'varistor', 'voltage-ac', 'voltage-dc', 'voltage-dc_ac',
                    'voltage.battery','voltmeter', 'vss', 'xnor', 'xor']


class_newdict = {k:v for k,v in enumerate(class_newids)}
with open(os.path.join(op,'class_dict.json'), 'w') as fp:
    json.dump(class_newdict, fp)



# New class_dict file 
ic(dataset_df.info())
dataset_df['internal_name'] = dataset_df['internal_name'].astype('str')
   
class_dict = dict(zip(dataset_df['internal_name'], dataset_df['class_list']))
ic(class_dict)

class_dict_main = collections.defaultdict(dict)
for i,(ds,class_list) in enumerate(class_dict.items()):
    ic(ds,class_list)
    class_dict_main[ds] = [[],[]]
    for item in ast.literal_eval(class_list):
        class_dict_main[ds][0].append(item) 
        #ic(class_dict_main,item)
        
        for i in range(len(class_newids)):
            if class_newids[i] == item:
                class_dict_main[ds][1].append(i)
            elif item == 'voltage.dc' and class_newids[i] == 'voltage-dc':
                class_dict_main[ds][1].append(i) 
            elif item == 'voltage.ac' and class_newids[i] == 'voltage-ac':
                class_dict_main[ds][1].append(i) 
            elif item == 'capacitor.unpolarized' and class_newids[i] == 'capacitor-unpolarized':
                class_dict_main[ds][1].append(i) 
            elif item == 'capacitor.polarized' and class_newids[i] == 'capacitor-polarized':
                class_dict_main[ds][1].append(i) 
            elif item == 'transistor.photo' and class_newids[i] == 'transistor-photo':
                class_dict_main[ds][1].append(i) 
            else:
                continue
            
            # else:
            #     class_dict_main[ds][1].append(-1) 
    ic(len(class_dict_main[ds][0]),len(class_dict_main[ds][1]))
        
ic(class_dict_main)
# save dict
import json
with open(os.path.join(op,'class_mappings.json'), 'w') as fp:
    json.dump(class_dict_main, fp)


