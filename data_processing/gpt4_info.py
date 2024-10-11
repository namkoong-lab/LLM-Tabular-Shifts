import json 
import pandas as pd 
import numpy as np
import pickle
from sklearn.decomposition import PCA
from embed import embed_e5


ALL_STATES= ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 
            'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']


# load gpt4's response
task_name = 'income'
prompt_method = 'gpt4'
root_dir = '/shared/share_mala/llm-dro/domain_info'
path = root_dir + "/{}-{}.pkl".format(task_name, prompt_method)

with open(path, 'rb') as f:
    gpt4_info = pickle.load(f)

instruct = {}
instruct["income"] = "Classify whether US working adults' yearly income is above $50000."
instruct["pubcov"] = "Classify whether a low-income individual, not eligible for Medicare, has coverage from public health insurance."
instruct["mobility"] = "Classify whether a young adult moved addresses in the last year."



def get_gpt4_embedding(task):
    prompt_list = []
    for state in ALL_STATES:
        prompt_list.append('Instruct: {}\nQuery: {}'.format(instruct[task], gpt4_info[state]))
    
    df = pd.DataFrame(prompt_list, columns=['input'])
    embedding = embed_e5(df)
    np.save(f"/shared/share_mala/llm-dro/domain_info/gpt4/{task}.npy", embedding)
    print(embedding.shape)
    
if __name__ == "__main__":
    get_gpt4_embedding("income")