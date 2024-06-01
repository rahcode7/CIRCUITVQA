from icecream import ic 
import pandas as pd 
import os
import json
import uuid
from tqdm import tqdm


if __name__ == "__main__":

    # Load train,test or dev 
    datasets = ['train','val']
    output_folder = 'LLaVA/LLaVA-finetune/dataset-cqa'
    QUESTIONS_PATH = 'datasets/questions/all/master.json'
    main_df = pd.read_json(QUESTIONS_PATH)
    ic(main_df.head(5))

    for dataset in datasets:
        df = main_df[main_df.splittype==dataset]

        # For each question, append to json list
        json_data_list = []
        for _,row in tqdm(df.iterrows(),total=df.shape[0]):
            #ic(row)
            unique_id = row['file']
            formatted_answers = row['answer']

            json_data = {
                "id": unique_id,
                "image": f"{unique_id}.jpg",
                "conversations": [
                    {
                        "from": "human",
                        "value": row['question']
                    },
                    {
                        "from": "gpt",
                        "value": formatted_answers
                    }
                ]
            }

            json_data_list.append(json_data)
        

        dataset_folder = os.path.join(output_folder, dataset)
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)


        json_output_path = os.path.join(dataset_folder,'dataset.json')
        with open(json_output_path, 'w') as json_file:
            json.dump(json_data_list, json_file, indent=4)
