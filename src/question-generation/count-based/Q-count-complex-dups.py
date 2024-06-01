# """ 
# Script to flag duplicate images
# """
# import json 
# from icecream import ic
# import pandas as pd



# if __name__ == "__main__":
#     qs_df = pd.read_csv("datasets/questions/count-complex/Q-count-complex.csv")
#     ic(qs_df.info())


#     with open(datasets/questions/others/duplicate_images.json', 'r') as f:
#         dup_dict = json.load(f)

#     #ic(dup_dict)

#     qs_df['duplicateof'] = ""
#     for i,row in qs_df.iterrows():
#         f = row['splittype'] + '_' + row['file']
#         print(f)
#         try:
#             if len(dup_dict[f]):
#                 #print("yes")
#                 qs_df['duplicateof'].iloc[i] = dup_dict[f]
#             else:
#                 qs_df['duplicateof'].iloc[i]  = []
#         except:
#             qs_df['duplicateof'].iloc[i]  = []
        
#     ic(qs_df.head(10))
#     qs_df.to_csv(" datasets/questions/count-complex/Q-count-complex-dup.csv",index=None)

