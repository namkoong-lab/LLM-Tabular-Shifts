import json 
import random 
import os 
import numpy as np 
import itertools

from whyshift import get_data
from whyshift.folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage, ACSMobility, ACSTravelTime
from sklearn.linear_model import LogisticRegression
import numpy as np 
import pandas as pd

from src.sklearn_interface import * 
from src.mlp import * 
from src.mlp_e5 import *
from src.mlp_concat import * 
from src.fairness import *
from src.subsampling import * 

s = 'AL,AK,AZ,AR,CA,CO,CT,DE,FL,GA,HI,ID,IL,IN,IA,KS,KY,LA,ME,MD,MA,MI,MN,MS,MO,MT,NE,NV,NH,NJ,NM,NY,NC,ND,OH,OK,OR,PA,RI,SC,SD,TN,TX,UT,VT,VA,WA,WV,WI,WY,PR'
all_states = s.split(',')
state_to_idx = {state: idx for idx, state in enumerate(all_states)}

# get raw data 
def get_raw_data(task_name, embedding_method, prompt_method = None, 
                 state = 'CA', save_dir = '/shared/share_mala/llm-dro/', year = 2018):
    if embedding_method == 'one_hot':
        X, y = get_onehot_data(task_name, state, False, f"{save_dir}/{task_name}", year)
    elif embedding_method == 'e5':
        X, y = get_e5_data(task_name, prompt_method, state, f"{save_dir}/{task_name}", year)
    elif embedding_method == 'concat':
        X, y = get_concat_data(task_name, state, f"{save_dir}/{task_name}", year)
    else:
        raise NotImplementedError
    return X, y

def get_e5_data(task_name, prompt_method, state, root_dir, year=2018):
    # find data path
    if task_name == 'income':
        task = ACSIncome
        path = f'{root_dir}/embed/ACSIncome-{state}-{year}-{prompt_method}.pkl'
    elif task_name == 'pubcov':
        task = ACSPublicCoverage
        path = f'{root_dir}/embed/ACSPubCov-{state}-{year}-{prompt_method}.pkl'
    elif task_name == 'mobility':
        task = ACSMobility
        path = f'{root_dir}/embed/ACSMobility-{state}-{year}-{prompt_method}.pkl'
    else:
        raise NotImplementedError
    # load data 
    data = pd.read_pickle(path)
    X = data["embedding"].values.tolist()
    X = np.array(X).astype(float)
    y = data[task.target].values.tolist()
    y = np.array(y).reshape(-1)
    y = y.astype(np.int64)
    return X, y

def get_onehot_data(task_name, state, preprocessing, path, year):
    X, y, _ = get_data(task_name, state, preprocessing, path, year)
    return X, y

def get_concat_data(task_name, state, root_dir, year=2018):
    # find data path
    if task_name == 'income':
        path = f'{root_dir}/embed/ACSIncome-{state}-{year}-domainlabel.pkl'
        task = ACSIncome
    elif task_name == 'pubcov':
        task = ACSPublicCoverage
        path = f'{root_dir}/embed/ACSPubCov-{state}-{year}-domainlabel.pkl'
    elif task_name == 'mobility':
        task = ACSMobility
        path = f'{root_dir}/embed/ACSMobility-{state}-{year}-domainlabel.pkl'
    else:
        raise NotImplementedError
    # load data 
    data = pd.read_pickle(path)
    X = data["embedding"].values.tolist()
    X = np.array(X).astype(float)
    y = data[task.target].values.tolist()
    y = np.array(y).reshape(-1)
    y = y.astype(np.int64)
    # get state id 
    state_idx = np.array(state_to_idx[state])
    state_idx = np.full((X.shape[0], 1), state_idx, dtype=int)  # Create an array of state indices
    X = np.concatenate((X, state_idx), axis = 1)                # concate X with state idx
    return X, y


## sample data and config

def sample_config(model, seed=0, id=0):
    # set up seed
    np.random.seed(seed)
    random.seed(seed)
    # load all configs
    dir_path = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(dir_path, f'hps/{model}.json')
    model_config = json.load(open(json_path))
    value_lists = model_config.values()
    param_names = list(model_config.keys())
    # select a specific config by id/randomly
    all_combinations = list(itertools.product(*value_lists))
    if len(all_combinations)> 200:
        idx_list = np.random.permutation(len(all_combinations))
    else: 
        idx_list = list(range(len(all_combinations)))
    dict_combinations = [dict(zip(param_names, combo)) for combo in all_combinations]
    return dict_combinations[min(len(dict_combinations)-1, idx_list[id])]

def sample_refit_config(model, task_name, seed=0, id=0):
    # set up seed
    np.random.seed(seed)
    random.seed(seed)
    # load all configs
    dir_path = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(dir_path, f'hps/{model}.json')
    refit_configs = json.load(open(json_path))
    # load refit config for specific task
    assert task_name in refit_configs, f"{task_name} not supported in refit hps config"
    refit_config = refit_configs[task_name]
    # sample config
    value_lists = refit_config.values()
    param_names = list(refit_config.keys())
    # select a specific config by id/randomly
    all_combinations = list(itertools.product(*value_lists))
    if len(all_combinations)> 200:
        idx_list = np.random.permutation(len(all_combinations))
    else: 
        idx_list = list(range(len(all_combinations)))
    dict_combinations = [dict(zip(param_names, combo)) for combo in all_combinations]
    return dict_combinations[min(len(dict_combinations)-1, idx_list[id])]


def sample_data(X, y, num=20000, test=False, seed=0, add_intercept = False):
    np.random.seed(seed)
    random.seed(seed)
    # shuffle data
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)
    # add intercept if required
    if add_intercept:
        X_intercept = np.ones((n, 1))
        X = np.hstack((X, X_intercept))
    # for source state, split into train and validation
    if not test:    
        train_num = 0
        if n > 2*num:
            train_num = num
        else:
            train_num = n//2
        trainX = X[indices[:train_num]]
        trainy = y[indices[:train_num]]
        valX = X[indices[train_num:2*train_num]]
        valy = y[indices[train_num:2*train_num]]
        return trainX, trainy, valX, valy
    else: # for target state, return all data
        return X[indices[:num]], y[indices[:num]]

# sample validation data (balanced among positive and negative samples)  
def sample_val_data(X, y, val_num=32, seed=0, add_intercept=False):
    np.random.seed(seed)
    random.seed(seed)
    # Add intercept if required
    if add_intercept:
        X_intercept = np.ones((X.shape[0], 1))
        X = np.hstack((X, X_intercept))
    # Separate indices by class, make sure positive samples = negative samples
    indices_0 = np.where(y == 0)[0]
    indices_1 = np.where(y == 1)[0]
    # Shuffle indices
    np.random.shuffle(indices_0)
    np.random.shuffle(indices_1)
    # Determine the number of samples for each class
    val_num_per_class = min(len(indices_0), len(indices_1), val_num // 2)
    # Sample indices
    sampled_indices_0 = indices_0[:val_num_per_class]
    sampled_indices_1 = indices_1[:val_num_per_class]
    # Combine and shuffle the sampled indices
    sampled_indices = np.hstack((sampled_indices_0, sampled_indices_1))
    np.random.shuffle(sampled_indices)
    # Select the sampled data
    valX = X[sampled_indices]
    valy = y[sampled_indices]
    return valX, valy, sampled_indices

# sample refit data (uniformly from target distribution)
def sample_refit_data(X, y, refit_num=32, seed=0, add_intercept=False):
    np.random.seed(seed)
    random.seed(seed) 
    # Add intercept if required
    if add_intercept:
        X_intercept = np.ones((X.shape[0], 1))
        X = np.hstack((X, X_intercept))
    # Shuffle the dataset
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    # Sample indices
    sampled_indices = indices[:refit_num]
    # Select the sampled data
    refitX = X[sampled_indices]
    refity = y[sampled_indices]
    return refitX, refity, sampled_indices

## fetch model
def fetch_model(model, is_regression, input_dim=77, initial_embedding_method='wiki', training_method='pca'):
    if model in ['lr', 'rf', 'gbm', 'lightgbm', 'svm', 'xgb']:
        return Sklearn_models(model)
    elif model == 'mlp':
        return MLPClassifier(input_dim=input_dim)
    elif model == 'mlp_e5':
        return MLPe5Classifier(input_dim=input_dim)
    elif model == 'mlp_concat':
        return MLPconcatClassifier(input_dim=input_dim, initial_embedding_method=initial_embedding_method, training_method=training_method)
    elif model == 'mlp_chi_dro':
        return chi_DRO(input_dim=input_dim)
    elif model == 'mlp_cvar_dro':
        return cvar_DRO(input_dim=input_dim)
    elif model == 'mlp_cvar_doro':
        return cvar_DORO(input_dim=input_dim)
    elif model == 'mlp_chi_doro':
        return chi_DORO(input_dim=input_dim)
    elif model == 'fairness_in':
        return FairInprocess()
    elif model == 'fairness_post':
        return FairPostprocess()
    elif model == 'subsampling':
        return SubSample()
    else:
        raise NotImplementedError(f"Algorithm {model} not implemented yet!")

if __name__ == "__main__":
    sample_config("unified_dro_l2",1)