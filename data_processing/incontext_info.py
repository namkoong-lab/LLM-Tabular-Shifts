import json 
import pandas as pd 
import numpy as np
import pickle
from sklearn.decomposition import PCA
from embed import embed_e5

from whyshift import get_data, degradation_decomp, fetch_model, risk_region
from whyshift.folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage, ACSMobility, ACSTravelTime

from dataset import *
from preprocess import *
from serialize import *
from embed import *


ALL_STATES= ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 
            'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']

# get incontext text
def get_incontext_text(task_name, state, year = 2018, root_dir = None, num_examples = 10):
    if task_name == 'income':
        task = ACSIncome
        task.task_name = task_name
        task.state  = state
        task.year = year
        task.data_name = 'ACSIncome-{}-{}'.format(state, year)
        task.root_dir = root_dir
    elif task_name == 'mobility':
        task = ACSMobility
        task.task_name = task_name
        task.state  = state
        task.year = year
        task.data_name = 'ACSMobility-{}-{}'.format(state, year)
        task.root_dir = root_dir
    elif task_name == 'pubcov':
        task = ACSPublicCoverage
        task.task_name = task_name
        task.state  = state
        task.year = year
        task.data_name = 'ACSPubCov-{}-{}'.format(state, year)
        task.root_dir = root_dir
    else:
        raise NotImplementedError
    
    # raw dataset
    raw_df = get_raw_df(task.task_name, task.state, root_dir = task.root_dir, year=task.year)
    # serialize table
    serialize_df = serialize_table(task.data_name, raw_df, target = task.target, group = task.group, root_dir = task.root_dir)

    # Set the random seed for reproducibility
    np.random.seed(42)
    # Randomly select n rows from the DataFrame
    random_indices = np.random.choice(serialize_df.index, size=num_examples, replace=False)
    random_rows = serialize_df.loc[random_indices]
    # combine the feature and labels
    answer_dict = {False: 'No.', True: 'Yes.'}
    current_text = 'Here are some examples of the data: \n'
    for i, row in random_rows.iterrows():
        # get label from the target col
        current_text += f"{row['feature']} \nAnswer: {answer_dict[row[task.target]]}\n\n"
    return current_text

    
# get incontext text
import numpy as np
import pandas as pd

# derive natural language input
instruct = {}
instruct["income"] = "Classify whether US working adults' yearly income is above $50000."
instruct["pubcov"] = "Classify whether a low-income individual, not eligible for Medicare, has coverage from public health insurance."
instruct["mobility"] = "Classify whether a young adult moved addresses in the last year."


# get incontext embedding
def get_incontext_embedding(task='income', num_examples = 32):
    # get root dir
    root_dir = f'/shared/share_mala/llm-dro/{task}/'
    # get incontext natural language
    incontext_info = dict()
    for state in ALL_STATES:
        incontext_info[state] = get_incontext_text(task_name, state, root_dir = root_dir, num_examples = num_examples)
    # rewrite incontext natural language using e5 template
    prompt_list = []
    for state in ALL_STATES:
        prompt_list.append('Instruct: {}\nQuery: {}'.format(instruct[task], incontext_info[state]))
    # convert into dataframe
    df = pd.DataFrame(prompt_list, columns=['input'])
    # convert into embedding
    embedding = embed_e5(df)
    # save embedding
    save_dir = f"/shared/share_mala/llm-dro/domain_info/incontext{num_examples}/"
    if not os.path.exists(save_dir ):
        os.makedirs(save_dir )
    np.save(save_dir + f"/{task}.npy", embedding)
    print(embedding.shape)

if __name__ == "__main__":
    get_incontext_embedding(task="income", num_examples=8)
    get_incontext_embedding(task="income", num_examples=16)
    get_incontext_embedding(task="income", num_examples=32)

    get_incontext_embedding(task="mobility", num_examples=8)
    get_incontext_embedding(task="mobility", num_examples=16)
    get_incontext_embedding(task="mobility", num_examples=32)

    get_incontext_embedding(task="pubcov", num_examples=8)
    get_incontext_embedding(task="pubcov", num_examples=16)
    get_incontext_embedding(task="pubcov", num_examples=32)