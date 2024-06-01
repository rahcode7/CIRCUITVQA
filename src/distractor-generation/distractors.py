
import pandas as pd 
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils.symbols_desc import symbol_dict
import random
pd.options.mode.chained_assignment = None 
import os 
from icecream import ic 

model = SentenceTransformer('all-MiniLM-L6-v2')

OUTPUT_PATH = "datasets/questions/"

# Top K answers candidates
K=50
TEST = 200

# Num of distractors
TOP = 3
BOTTOM = 1

def sent_embeddings(sentence):
    return model.encode(sentence)

if __name__ == "__main__":

    Q_PATH = "  datasets/questions/position/Q-position.csv"
    main_df = pd.read_csv(Q_PATH)
    #print(main_df.head(10))

    qtype = "position"
    ##### Type 1  subset symbol based qs
    df = main_df.loc[main_df['qtype'].isin(['position'])]
    print(df.shape)

    #print(df.info()) 
    #df = df.head(TEST) # Only for testing, uncomment it for prod

    # get symbol embeddings for chatgpt sentences
    symbol_emb = {}
    for i,(k,v) in enumerate(symbol_dict.items()):
        #print(k,v)
        symbol_emb[k.lower()] = model.encode(v.lower())
    #print(symbol_emb.keys())
    print(len(symbol_emb))

    # Generate embeddings
    # Get nearest 50 questions for each
    embeddings = model.encode(df['question'])
    print(embeddings.shape)
    matrix = []
    for val in embeddings:
        matrix.append(cosine_similarity([val],embeddings)[0])
    #print(matrix.shape)

    # get top k+1 indexes for each
    topmatrix = []
    for arr in matrix:
        topmatrix.append(arr.argsort()[-K:][::-1].tolist())
    #print(len(topmatrix))

    # Get top questions for each question and their answers, store these answers
    # Assign similar answer
    df["answer_similar"] = ""
    df["distractor1"] = ""
    df["distractor2"] = ""
    df["distractor3"] = ""
    df["distractor4"] = ""

    for i,item in enumerate(topmatrix):
        similar_answers  = set(df['answer'].iloc[item].tolist()) # unique answers
        df['answer_similar'].iloc[i] = similar_answers
        answer =  df['answer'].iloc[i]
        print(i,answer,similar_answers)

        # remove same answers 
        similar_answers_filtered = [i for i in similar_answers if i != answer]
        #print(similar_answers_filtered)

        #df['answer_similar'].iloc[i] = similar_answers_filtered

        ####  remove with similarity > 0.90 - NO, keep top 3 from most similar, 1 from lowest
        cosines = []
        for item in similar_answers_filtered:
            #print(symbol_emb[item])
            cosines.append(cosine_similarity([symbol_emb[answer]],[symbol_emb[item]])[0][0])
        #print(cosines)
        

        # Get top answers
        top_indexes = sorted(range(len(cosines)), key=lambda i: cosines[i])[-TOP:]
        #print(top_indexes)

        top_answers = [similar_answers_filtered[i] for i in top_indexes]
        print(top_answers)

        # Get bottom answers 
        bottom_indexes = sorted(range(len(cosines)), key=lambda i: cosines[i])[0] 
        #print(bottom_indexes)
        bottom_answers = similar_answers_filtered[bottom_indexes] 
        #print(bottom_answers)
       
        df['distractor1'].iloc[i] = top_answers[0]
        df['distractor2'].iloc[i] = top_answers[1]
        df['distractor3'].iloc[i] = top_answers[2]

        df['distractor4'].iloc[i] = bottom_answers
        print(df.iloc[i])

    del df['answer_similar']
    #print(df)
    df.to_csv(OUTPUT_PATH+qtype+ '/Q-position-distractor.csv',index=None)


    # ##### Type 2  subset qs type
    # #main_df = pd.read_csv("datasets/questions/questions-all.csv") # delete when master dataset
    DATA_PATH = "datasets/questions/"
    
    q_count_type  = ['count'] # Add more Q types

    for qtype in q_count_type:

        main_df = pd.read_csv(DATA_PATH+ qtype+"/Q-" + qtype + ".csv")
        value_df = main_df[main_df['qtype']==qtype]
    
        ic(value_df.shape)
        value_df = value_df[~value_df.symbol.isin(['junction','terminal','text'])]
        ic(value_df.shape)

        
        value_df["distractor1"],value_df["distractor2"],value_df["distractor3"],value_df["distractor4"] = "","","",""
        # get range for each symbol
        
        # Create dictionary of symbol : max,min values for each symbols based on subset of same symbol qtypes
        symbol_range_dict = {}
        for key in symbol_dict:
            key = key.lower()
            #print(key)
            
            #question_df = value_df[value_df["question"].str.contains(key)]
            question_df = value_df[value_df["symbol"]==key].reset_index()
            ic(key,question_df.shape[0])

            if question_df.shape[0]!=0:
                max_val = question_df["answer"].max()
                min_val = question_df["answer"].min()
                #print("max value,min val")
                #print(max_val,min_val)
            else:
                max_val,min_val = 0,0

            symbol_range_dict[key] = (min_val,max_val)

        #ic(symbol_range_dict)
        #ic(value_df)

        for i,row in value_df.iterrows():
            t = symbol_range_dict[row['symbol']]
            min,max = t[0],t[1]


            answer = row['answer']
            ic(t,i,answer,min,max)
            distractors = []
            for j in range(1000):
                if max <= 5:
                    num = random.randrange(1,6)
                else:
                    num = random.randrange(min,max)
                #ic(num)
                if num not in distractors and num != answer:
                    distractors.append(num)
                if len(distractors)==4:
                    break 
                
            #ic(distractors)
            value_df['distractor1'].iloc[i] = distractors[0]
            value_df['distractor2'].iloc[i] = distractors[1]
            value_df['distractor3'].iloc[i] = distractors[2]
            value_df['distractor4'].iloc[i] = distractors[3]
            
        value_df.to_csv(os.path.join(OUTPUT_PATH,qtype)+'/Q-'+qtype+'-distractor.csv',index=None)










    