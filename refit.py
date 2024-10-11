import numpy as np 
import pandas as pd

import os 
import sys
import json 
import argparse
import torch 
import gc 
from joblib import Parallel, delayed

from utils import (
    get_raw_data, 
    sample_config, 
    sample_refit_config, 
    fetch_model, 
    sample_data, 
    sample_val_data, 
    sample_refit_data
)
from src.add_lora_layer import *

# Define several constants -- Should not be changed
ALL_STATES= ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
             'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 
             'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 
             'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 
             'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']
VAL_NUM_LIST = [16, 32, 64, 128]   # number of validation samples
YEAR = 2018
SEED = 0
NUM_REFIT_EXPERIMETNS = 50  # number of refit experiments for each base model

def load_refit_val_test_data(args):
    '''
    Return refit, val, test data
    '''
    # data arguments
    task_name = args.task
    year = args.year
    embedding_method = args.embedding
    prompt_method = args.prompt
    refit_num = args.refit_num
    seed = args.seed 
    # target state
    target_state = args.target
    
    # load raw data for target state
    data_dir = '/shared/share_mala/llm-dro/'
    X, y = get_raw_data(task_name, embedding_method, prompt_method, 
                        target_state, data_dir, year)
    
    # sample refit data (uniformly from the target state)
    refitX, refity, _ = sample_refit_data(X, y, 
                                          refit_num = refit_num, 
                                          seed=seed, add_intercept = False)  
    # sample val data (balanced for each class)
    val_dict = {}
    val_num_list = VAL_NUM_LIST
    for val_num in val_num_list:
        valX, valy, _ = sample_val_data(X, y, val_num=val_num, seed=seed)
        val_dict[val_num] = (valX, valy)
    # test data
    testX, testy = sample_data(X, y, test=True, seed=seed)  
    return refitX, refity, val_dict, testX, testy

def refit_result_exists(args):
    # train arguments
    task_name = args.task
    embedding_method = args.embedding
    prompt_method = args.prompt
    initial_embedding_method = args.initial_embedding_method
    training_method = args.training_method
    model_name = args.model
    experiment_id = args.id

    source_state = args.source
    source_state_str = "-".join(source_state)
    # refit arguments
    target_state = args.target
    refit_method = args.refit_method
    refit_num = args.refit_num
    refit_id = args.refit_id
    target_state = args.target

    # find result path
    save_dir = '/shared/share_mala/llm-dro/'
    if embedding_method == 'concat':
        if refit_method == training_method:     # if refit method is the same as training method
            os.makedirs(f'{save_dir}/refit_results/{task_name}/{embedding_method}/{initial_embedding_method}/{training_method}/{model_name}/refit_{refit_num}/source_{source_state_str}/target_{target_state}/', exist_ok=True)            
            path = f'{save_dir}/refit_results/{task_name}/{embedding_method}/{initial_embedding_method}/{training_method}/{model_name}/refit_{refit_num}/source_{source_state_str}/target_{target_state}/experiment{experiment_id}_refit{refit_id}.json'
        else:                                   # if refit method is different from training method
            os.makedirs(f'{save_dir}/refit_results/{task_name}/{embedding_method}/{initial_embedding_method}/{training_method}-{refit_method}/{model_name}/refit_{refit_num}/source_{source_state_str}/target_{target_state}/', exist_ok=True)            
            path = f'{save_dir}/refit_results/{task_name}/{embedding_method}/{initial_embedding_method}/{training_method}-{refit_method}/{model_name}/refit_{refit_num}/source_{source_state_str}/target_{target_state}/experiment{experiment_id}_refit{refit_id}.json'
    elif embedding_method == 'e5':
        os.makedirs(f'{save_dir}/refit_results/{task_name}/{prompt_method}/{refit_method}/{model_name}/refit_{refit_num}/source_{source_state_str}/target_{target_state}/', exist_ok=True)            
        path = f'{save_dir}/refit_results/{task_name}/{prompt_method}/{refit_method}/{model_name}/refit_{refit_num}/source_{source_state_str}/target_{target_state}/experiment{experiment_id}_refit{refit_id}.json'
    elif embedding_method == 'one_hot':
        os.makedirs(f'{save_dir}/refit_results/{task_name}/{embedding_method}/{refit_method}/{model_name}/refit_{refit_num}/source_{source_state_str}/target_{target_state}/', exist_ok=True)            
        path = f'{save_dir}/refit_results/{task_name}/{embedding_method}/{refit_method}/{model_name}//refit_{refit_num}/source_{source_state_str}/target_{target_state}/experiment{experiment_id}_refit{refit_id}.json'
    else:
        raise NotImplementedError

    # check if the experiment has been done
    if os.path.exists(path):
        print(f"Refit {task_name}-{model_name}-Source {source_state_str}-Target {target_state}-Experiment ID {experiment_id}-Refit ID {refit_id} already exists")
        return True
    else:
        return False

def safe_refit(refitX, refity, val_dict, testX, testy, args):
    try:
        refit(refitX, refity, val_dict, testX, testy, args)
    except Exception as e:
        print(f"Error in experiment: {e}")

def refit(refitX, refity, val_dict, testX, testy, args):
    '''
    refit and test models on target states
    '''
    # train arguments
    task_name = args.task
    year = args.year
    embedding_method = args.embedding
    prompt_method = args.prompt
    initial_embedding_method = args.initial_embedding_method
    training_method = args.training_method
    model_name = args.model
    experiment_id = args.id

    source_state = args.source
    source_state_str = "-".join(source_state)
    # refit arguments
    target_state = args.target
    seed = args.seed
    refit_method = args.refit_method
    refit_num = args.refit_num
    refit_id = args.refit_id
    is_regression = args.is_regression
    gpu_id = args.gpu_id

    # set up gpu
    if 'mlp' in model_name:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
    
    # find trained model dir
    save_dir = '/shared/share_mala/llm-dro/'
    if embedding_method == 'concat':
        model_dir = f'{save_dir}/save_models/{task_name}/{source_state_str}/{embedding_method}/{initial_embedding_method}/{training_method}/'    
    elif embedding_method == 'e5':
        model_dir = f'{save_dir}/save_models/{task_name}/{source_state_str}/{embedding_method}/{prompt_method}/'
    elif embedding_method == 'one_hot':
        model_dir = f'{save_dir}/save_models/{task_name}/{source_state_str}/{embedding_method}/'
    else:
        raise NotImplementedError
    
    # save hyperparamters
    result_record = {}    
    result_record["model"] = model_name
    result_record["source_state"] = source_state_str
    result_record["year"] = str(year)
    result_record["embedding"] = embedding_method
    if embedding_method != 'one_hot':
        result_record["prompt"] = prompt_method
    
    result_record["refit_num"] = str(refit_num)
    result_record["experiment_id"] = experiment_id
    result_record["target_state"] = target_state
    result_record["refit_id"] = str(refit_id)
    
    # load trained model and hyperparameters
    if embedding_method == 'concat':
        model = fetch_model('mlp_concat', is_regression, 
                            refitX.shape[1]-1, 
                            initial_embedding_method=initial_embedding_method, 
                            training_method=training_method)
        config = sample_config(f'mlp_concat_{training_method}', seed, experiment_id)
    elif embedding_method == 'e5':
        model = fetch_model('mlp_e5', is_regression, refitX.shape[1])
        config = sample_config('mlp_e5', seed, experiment_id)
    elif embedding_method == 'one_hot':
        model = fetch_model(model_name, is_regression, refitX.shape[1])
        config = sample_config(model_name, seed, experiment_id)
    else:
        raise NotImplementedError

    if 'mlp' in model_name:
        config["device"] = gpu_id
    result_record["config"] = config 
    result_record['initial_embedding_method'] = initial_embedding_method
    result_record['training_method'] = training_method
    result_record['refit_method'] = refit_method     
    try: 
        model.load(experiment_id, model_dir)
    except:
        raise ValueError(f"Model {model_name}_{experiment_id} not found in {model_dir}")

    # load refit hyperparameters
    if refit_method == 'freeze_embedding':
        refit_config = sample_refit_config(f'refit_mlp_freeze_embedding', task_name, seed, refit_id)
    elif refit_method == 'nn':
        refit_config = sample_refit_config('refit_mlp_nn', task_name, seed, refit_id)
    elif refit_method == 'lora':
        refit_config = sample_refit_config(f'refit_mlp_lora', task_name, seed, refit_id)
    else:
        raise NotImplementedError
    result_record['refit_config'] = refit_config
    model.update_refit_config(refit_config)     # update refit config (Do NOT re-initialize model)
    print(refit_config)

    # load lora rank if refit method is lora
    if refit_method == 'lora':
        lora_rank = refit_config['lora_rank']
        
    # refit and test model on target states
    # load trained model
    model.load(experiment_id, model_dir)
    # add lora layer if refit method is lora
    if refit_method == 'lora':
        apply_lora_to_linear_layers(model.model, rank=lora_rank)
    # update refit method
    model.refit_method = refit_method
    model.model.refit_method = refit_method
    # refit model 
    model.refit(refitX, refity)
    
    # model validation: use samples from the target state
    val_num_list = VAL_NUM_LIST
    val_result_acc = {}
    val_result_f1 = {}
    for val_num in val_num_list:
        # load validation data
        valx, valy = val_dict[val_num]
        # save accuracy and f1 score
        acc, f1 = model.score(valx, valy)
        val_result_acc[val_num] = acc 
        val_result_f1[val_num] = f1
    # save validation results
    result_record["val_result_acc"] = val_result_acc
    result_record["val_result_f1"] = val_result_f1

    # test model
    model.model.eval()
    # save accuracy and f1 score
    acc, f1 = model.score(testX, testy)
    test_result_acc = acc 
    test_result_f1 = f1
    # save test results
    result_record["test_result_acc"] = test_result_acc
    result_record["test_result_f1"] = test_result_f1
    if 'mlp' in model_name:
        result_record["config"]["device"] = gpu_id
    
    # find result path
    save_dir = '/shared/share_mala/llm-dro/'
    if embedding_method == 'concat':
        if refit_method == training_method:     # if refit method is the same as training method
            os.makedirs(f'{save_dir}/refit_results/{task_name}/{embedding_method}/{initial_embedding_method}/{training_method}/{model_name}/refit_{refit_num}/source_{source_state_str}/target_{target_state}/', exist_ok=True)            
            path = f'{save_dir}/refit_results/{task_name}/{embedding_method}/{initial_embedding_method}/{training_method}/{model_name}/refit_{refit_num}/source_{source_state_str}/target_{target_state}/experiment{experiment_id}_refit{refit_id}.json'
        else:                                   # if refit method is different from training method
            os.makedirs(f'{save_dir}/refit_results/{task_name}/{embedding_method}/{initial_embedding_method}/{training_method}-{refit_method}/{model_name}/refit_{refit_num}/source_{source_state_str}/target_{target_state}/', exist_ok=True)            
            path = f'{save_dir}/refit_results/{task_name}/{embedding_method}/{initial_embedding_method}/{training_method}-{refit_method}/{model_name}/refit_{refit_num}/source_{source_state_str}/target_{target_state}/experiment{experiment_id}_refit{refit_id}.json'
    elif embedding_method == 'e5':
        os.makedirs(f'{save_dir}/refit_results/{task_name}/{prompt_method}/{refit_method}/{model_name}/refit_{refit_num}/source_{source_state_str}/target_{target_state}/', exist_ok=True)            
        path = f'{save_dir}/refit_results/{task_name}/{prompt_method}/{refit_method}/{model_name}/refit_{refit_num}/source_{source_state_str}/target_{target_state}/experiment{experiment_id}_refit{refit_id}.json'
    elif embedding_method == 'one_hot':
        os.makedirs(f'{save_dir}/refit_results/{task_name}/{embedding_method}/{refit_method}/{model_name}/refit_{refit_num}/source_{source_state_str}/target_{target_state}/', exist_ok=True)            
        path = f'{save_dir}/refit_results/{task_name}/{embedding_method}/{refit_method}/{model_name}/refit_{refit_num}/source_{source_state_str}/target_{target_state}/experiment{experiment_id}_refit{refit_id}.json'
    else:
        raise NotImplementedError
    
    # save result
    with open(path, 'w') as f:
        json.dump(result_record, f)
    del model 
    torch.cuda.empty_cache()
    gc.collect() 
    print(f"Refit {task_name}-{model_name}-Source {source_state_str}-Target {target_state}-Experiment ID {experiment_id}-Refit ID {refit_id} finished")
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DRO-bench')
    # training arguments
    parser.add_argument('--embedding', default='e5', 
                        help='embedding method, e.g., one-hot, e5')
    parser.add_argument('--prompt', default='None', 
                        help='if we add any prompt info to LLM embedding, e.g., wiki, gpt4')
    parser.add_argument('--initial_embedding_method', default='wiki', type=str, 
                        help='initial embedding of contextual info, e.g., wiki, gpt4')
    parser.add_argument('--model', 
                        help='model, e.g., DRO, lr, mlpm etc')
    parser.add_argument('--training_method', default='freeze_embedding', type=str, 
                        help='training method, e.g., nn, freeze_embedding')
    parser.add_argument('--task', default='income', 
                        help='task name, e.g., income, pubcov, mobility')
    parser.add_argument('--source', type=str, nargs='+',  
                        help='source state list, e.g., CA PR')
    parser.add_argument('--num', type=int, nargs='+', 
                        help='number of samples of each source state, e.g., 1000 1000')
    # target adaptation arguments
    parser.add_argument('--id', default=0, type=int, 
                        help='id of experiment/training config of the base model')
    parser.add_argument('--refit_method', default='nn', type=str, 
                        help='refit method, e.g., nn, freeze_embedding, lora')
    parser.add_argument('--refit_num', default=32, type=int, 
                        help='num of refit samples, e.g., 32, 64, 128')
    parser.add_argument('--target', type=str, 
                        help='target state, e.g., CA PR')
    args = parser.parse_args()
    
    # parallel training: choose a random gpu
    num_gpus = torch.cuda.device_count()
    args.gpu_id = np.random.randint(0, num_gpus)
    # set seed
    args.seed = SEED
    args.is_regression = 0
    args.year = YEAR

    # load training and testing data (once for a series of experiments)
    refitX, refity, val_dict, testX, testy = load_refit_val_test_data(args)
    
    # number of refit configs
    num_refit_experiments = NUM_REFIT_EXPERIMETNS 
    if args.refit_method =='lora':
        num_configs = 15
    elif args.refit_method == 'freeze_embedding':
        num_configs = 12
    elif args.refit_method == 'nn':
        num_configs = 18
    else:
        raise NotImplementedError

    # select config list 
    if num_configs >= num_refit_experiments:
        np.random.seed(SEED)
        refit_id_list = np.random.choice(num_configs, num_refit_experiments, replace=False)
    else:
        refit_id_list = np.arange(num_configs)
    
    # parallel computing
    for refit_id in refit_id_list:
        args.refit_id = refit_id
        # check if the experiment has been done
        if refit_result_exists(args) == False:
            safe_refit(refitX, refity, val_dict, testX, testy, args)
  