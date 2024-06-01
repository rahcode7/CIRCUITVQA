import pandas as pd 
from tqdm import tqdm
import re
import ast 
import numpy
from re import search
from icecream import ic 
from utils.symbols_desc import symbol_dict
import math 
import re
tqdm.pandas()

  # For various components,append suffixes
def process(answer,question):
    print(answer)

    if search('resistor',question):
        symbol = 'Ω'
    elif search('inductor',question):
        symbol = 'H'
    elif search('ammeter',question):
        symbol = 'A'
    elif search('voltmeter',question) or search('voltage-dc',question):
        symbol = 'V'
    elif search('capacitor',question):
        symbol = 'F'
    elif search('speaker',question):
        symbol = 'W'
    else:
        symbol = ""

    # 1. Convert to a list 
    if "cos" in answer or "sin" in answer:
        a_list = answer
        return a_list 
    elif answer[0] == '[':
        if "hz" in answer.lower() or "f" in answer.lower():
            a_list = answer
            return a_list
        else:
            a_list = ast.literal_eval(answer)
    else:
        a_list = answer.split(",")
    
    a_list = list(filter(None, a_list))
    #print(type(a_list))
    #print(answer,a_list)

    num_list = []
    mul_list = []
    label_list = [] 
    for a in a_list:
        #print(a)
        # Extract num
        num = ''
        mul = ''
        label = ''
            
        # if a num 
        if isinstance(a, int) or isinstance(a,float):
            num = a
            mul = "1"
            #label = ""
        else:
            # for each char 
            for c in a:
                #print(c)
                if c.isdigit():
                    num += c
                if c == ".":
                    num += c

                if c == 'k' or c == 'K':
                    mul = '1000'
                    label = c                    
                elif c == "M":
                    mul = "1000000"
                    label = c
                elif c == "u": # micro
                    mul = "0.000001"
                    label = c
                elif c == "p" or c == "P":
                    mul = "0.000000000001"
                    label = c
                elif c == "n" or c == "N":
                    mul = "0.000000001"
                    label = c
                elif c == "u":
                    mul = "0.001"
                    label = c
                else:
                    mul = "1"
                    label = ""

        ic(num,type(num))
        if isinstance(num,str):
            num = re.sub('[^A-Za-z0-9]+', '', num)
            ic(num)
        if "." in str(num):
            num = float(num)
        elif num=="":
            pass
        else: #isinstance(num,int):
            num = int(num)
       
        num_list.append(num)
        mul_list.append(mul)
        label_list.append(label)

    print(num_list,mul_list,label_list)
    
    # sort them 
    val_list = []
    for n,m in zip(num_list,mul_list):
        if n == "":
            continue
        val_list.append(n * float(m))
    print(val_list)
    # return indices of minimum to max of the list
    sort_index = numpy.argsort(val_list)
    #print(sort_index)
    a_list = [a_list[i] for i in sort_index]
    print("FINAL",a_list)

    


    # 3. add suffix at end based on q
    #print(symbol)
    f_list = []
    for a in a_list:
        #print(a)
        if not search(symbol,str(a)):
            f_list.append(str(a) + symbol)
        else:
            f_list.append(str(a))
    #print(f_list)
    return f_list


def find_answer_class(question,answer_classes):
    for item in answer_classes:
        for word in question.split(" "):
            if item in word:
                return item 

if __name__ == "__main__":

    # Get labelled data
    df = pd.read_csv("datasets/questions/value/Q-value-labelled.csv")   
    ic(df.head(4))

    # subset non empty
    df = df[~df['answer_label'].isna()]
    df.to_csv("datasets/questions/value/Q-value-labelled-processed.csv",index=None)
    print(df.shape)

    print(df['answer_label'].values)
    df['answer_label_f'] = df.progress_apply(lambda x: process(x.answer_label,x.question),axis=1)
    ic(df.head(4))



    # Read Questions file 
    val_df = pd.read_csv("datasets/questions/value/Q-value.csv")
    #ic(val_df.columns)

    # Add answer class col
    #ic(symbol_dict.keys())
    symbols = list(symbol_dict.keys())
    symbols = [i.lower() for i in symbols]
    symbols.sort(key=len,reverse=True)
    #ic(symbols)

    val_df['answer_class'] = val_df['question'].apply(lambda x : find_answer_class(x,symbols))
    #ic(val_df[['question','answer_class']].head(10))

    df['answer_class'] = df['question'].apply(lambda x : find_answer_class(x,symbols))
    ic(df[['question','answer_class']].head(10))


    # join 2 tables based on answer class
    main_df = pd.merge(val_df,df,on=["file","answer_class"],how="left")
    main_df = main_df.drop(columns=['splittype_y','question_y','qtype_y','answer_y'])
    #main_df.rename(columns = {'': 'X', 'b': 'Y'})
    main_df.columns = ['splittype', 'file', 'question', 'answer', 'qtype',
                            'answer_class','answer_label', 'comments',
                            'answer_label_f']
    ic(main_df.head(4),main_df.columns)
    ic(main_df['file'].nunique(),main_df.shape)
    ic(main_df['answer_label'].notnull().sum())

    # Process google answer as well
    for i,row in main_df.iterrows():
        #ic(row['answer_label'],len(str(row['answer_label'])))
        #if len(str(row['answer_label'])) == 0:
        #if isinstance(row['answer_label'],str):

        if row['answer_label']!=row['answer_label']:
            print(row)
            answer_label_f = process(row['answer'],row['question'])
            #main_df.iloc[i,df.columns.get_loc('answer_label_f')] = answer_label_f 
            main_df.at[i,'answer_label_f'] = answer_label_f

    main_df.to_csv("datasets/questions/value/Q-value-labelled-final.csv",index=None)
