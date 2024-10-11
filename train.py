import numpy as np 
import pandas as pd

import os 
import json 
import argparse
import torch 
import gc 
from joblib import Parallel, delayed

from utils import (
    get_raw_data, 
    sample_config, 
    fetch_model, 
    sample_data, 
    sample_val_data
)

# Define several constants -- Should not be changed
ALL_STATES= ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
             'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 
             'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 
             'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 
             'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']
VAL_NUM_LIST = [16, 32, 64, 128]   # number of validation samples
YEAR = 2018
SEED = 0
NUM_EXPERIMENTS = 200 # number of experiments for each algorithm

def load_train_val_test_data(args):
    '''
    Return train, val, test data
    '''
    task_name = args.task
    year = YEAR
    source_state_list = args.source
    num_list = args.num
    embedding_method = args.embedding   
    prompt_method = args.prompt
    seed = args.seed
    target_state_list = ALL_STATES
    data_dir = '/shared/share_mala/llm-dro/'
    val_num_list = VAL_NUM_LIST

    # load training/validation data
    train_dict = {}
    val_dict = {}
    test_dict = {}
    for val_num in val_num_list:
        val_dict[val_num] = {}
    # load training data
    for idx, state in enumerate(source_state_list):
        X, y = get_raw_data(task_name, embedding_method, prompt_method, 
                            state, data_dir, year)
        # sample training/test data for the source state
        trainx, trainy, testx, testy = sample_data(X, y, num=num_list[idx], 
                                                   test=False, seed=seed)
        train_dict[state] = [trainx, trainy]
        test_dict[state] = [testx, testy]
    trainx = np.concatenate([train_dict[state][0] for state in train_dict.keys()], axis=0)
    trainy = np.concatenate([train_dict[state][1] for state in train_dict.keys()], axis=0)
    # load validation and testing data
    for idx, state in enumerate(target_state_list):
        if state in source_state_list:          # load validation data if source state
            for val_num in val_num_list:        # validation
                val_dict[val_num][state] = test_dict[state]
        else: # load test data if target state
            X, y = get_raw_data(task_name, embedding_method, prompt_method, 
                                state, data_dir, year)
            # validation data 
            for val_num in val_num_list:
                valx, valy, _ = sample_val_data(X, y, val_num=val_num, seed=seed)
                val_dict[val_num][state] = [valx, valy]
            # test data
            testx, testy = sample_data(X, y, test=True, seed=seed)  
            test_dict[state] = [testx, testy]
    return trainx, trainy, val_dict, test_dict

def result_exists(args):
    # load args
    task_name = args.task
    source_state_list = args.source
    embedding_method = args.embedding   
    prompt_method = args.prompt
    initial_embedding_method = args.initial_embedding_method
    training_method = args.training_method
    model_name = args.model
    experiment_id = args.id
    
    source_state_str = "-".join(source_state_list)
    # find result path
    save_dir = '/shared/share_mala/llm-dro/'
    if embedding_method == 'one_hot':
        os.makedirs(f'{save_dir}/results/{task_name}/{embedding_method}/{source_state_str}/{model_name}', exist_ok=True)            
        path = f'{save_dir}/results/{task_name}/{embedding_method}/{source_state_str}/{model_name}/{experiment_id}.json'
    elif embedding_method == 'e5':
        os.makedirs(f'{save_dir}/results/{task_name}/{prompt_method}/{source_state_str}/{model_name}', exist_ok=True)            
        path = f'{save_dir}/results/{task_name}/{prompt_method}/{source_state_str}/{model_name}/{experiment_id}.json'
    elif embedding_method == 'concat':
        os.makedirs(f'{save_dir}/results/{task_name}/{embedding_method}/{initial_embedding_method}/{training_method}/{source_state_str}/{model_name}', exist_ok=True)            
        path = f'{save_dir}/results/{task_name}/{embedding_method}/{initial_embedding_method}/{training_method}/{source_state_str}/{model_name}/{experiment_id}.json'
    else:
        raise NotImplementedError

    # check if the experiment has been done
    if os.path.exists(path):
        print(f"Experiment {task_name}-{source_state_str}-{model_name}-ID {experiment_id} already exists")
        return True
    else:
        return False

def safe_experiment(trainx, trainy, val_dict, test_dict, args):
    try:
        experiment(trainx, trainy, val_dict, test_dict, args)
    except Exception as e:
        print(f"Error in experiment: {e}")

def experiment(trainx, trainy, val_dict, test_dict, args):
    '''
    fit and save models only, no validation or test
    experiment_id: experiment id to random sample configs
    '''
    # load args
    task_name = args.task
    year = args.year
    embedding_method = args.embedding   
    prompt_method = args.prompt
    initial_embedding_method = args.initial_embedding_method
    training_method = args.training_method
    model_name = args.model
    
    source_state_list = args.source
    source_state_str = "-".join(source_state_list)
    # training arguments
    target_state_list = ALL_STATES
    seed = args.seed
    experiment_id = args.id
    gpu_id = args.gpu_id  
    is_regression = args.is_regression  

    # set up gpu
    if 'mlp' in model_name:
        gpu_id = args.gpu_id    
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
    
    # find model dir
    save_dir = '/shared/share_mala/llm-dro/'
    if embedding_method == 'one_hot':
        save_model_dir = f'/shared/share_mala/llm-dro/save_models/{task_name}/{source_state_str}/{embedding_method}/'
    elif embedding_method == 'e5':
        save_model_dir = f'/shared/share_mala/llm-dro/save_models/{task_name}/{source_state_str}/{embedding_method}/{prompt_method}/'
    elif embedding_method == 'concat':
        save_model_dir = f'/shared/share_mala/llm-dro/save_models/{task_name}/{source_state_str}/{embedding_method}/{initial_embedding_method}/{training_method}/'
    else:
        raise NotImplementedError
    
    # find result path
    if embedding_method == 'one_hot':
        os.makedirs(f'{save_dir}/results/{task_name}/{embedding_method}/{source_state_str}/{model_name}', exist_ok=True)            
        path = f'{save_dir}/results/{task_name}/{embedding_method}/{source_state_str}/{model_name}/{experiment_id}.json'
    elif embedding_method == 'e5':
        os.makedirs(f'{save_dir}/results/{task_name}/{prompt_method}/{source_state_str}/{model_name}', exist_ok=True)            
        path = f'{save_dir}/results/{task_name}/{prompt_method}/{source_state_str}/{model_name}/{experiment_id}.json'
    elif embedding_method == 'concat':
        os.makedirs(f'{save_dir}/results/{task_name}/{embedding_method}/{initial_embedding_method}/{training_method}/{source_state_str}/{model_name}', exist_ok=True)            
        path = f'{save_dir}/results/{task_name}/{embedding_method}/{initial_embedding_method}/{training_method}/{source_state_str}/{model_name}/{experiment_id}.json'
    else:
        raise NotImplementedError
    
    # check if the experiment has been done
    if os.path.exists(path):
        return 
    
    print(f"Experiment {task_name}-{source_state_str}-{model_name}-ID {experiment_id} begins")
    
    # save hyperparamters
    result_record = {}    
    result_record["model"] = model_name
    result_record["source_state"] = source_state_str
    result_record["year"] = year
    result_record["embedding"] = embedding_method
    if embedding_method != 'one_hot':
        result_record["prompt"] = prompt_method
    
    # load model and hyperparameters
    if model_name == 'mlp':
        if embedding_method == 'e5':
            model = fetch_model('mlp_e5', is_regression, trainx.shape[1])
            config = sample_config('mlp_e5', seed, experiment_id)
        elif embedding_method == 'concat':
            model = fetch_model('mlp_concat', is_regression, trainx.shape[1]-1, initial_embedding_method=initial_embedding_method, training_method=training_method)
            config = sample_config(f'mlp_concat_{training_method}', seed, experiment_id)
            result_record['initial_embedding_method'] = initial_embedding_method
            result_record['training_method'] = training_method        
        else:
            model = fetch_model(model_name, is_regression, trainx.shape[1])
            config = sample_config(model_name, seed, experiment_id)
    else:
        model = fetch_model(model_name, is_regression, trainx.shape[1])
        config = sample_config(model_name, seed, experiment_id)
    result_record["config"] = config 
    if 'mlp' in model_name:
        config["device"] = gpu_id
    model.update(config)        # update config and initialize model
    print(config)
    # model training
    model.fit(trainx, trainy)
    model.save(experiment_id, save_model_dir)
    
    # model validation: use samples from the target state
    if 'mlp' in model_name:
        model.model.eval()
    val_num_list = VAL_NUM_LIST
    val_result_acc = {}
    val_result_f1 = {}
    for val_num in val_num_list:
        # create a dict to store results
        val_result_acc[val_num] = {}
        val_result_f1[val_num] = {}
        for target_state in target_state_list:
            valx, valy = val_dict[val_num][target_state]
            # save accuracy and f1 score
            acc, f1 = model.score(valx, valy)
            val_result_acc[val_num][target_state] = acc 
            val_result_f1[val_num][target_state] = f1
    # save validation results
    result_record["val_result_acc"] = val_result_acc
    result_record["val_result_f1"] = val_result_f1

    # model testing
    if 'mlp' in model_name:
        model.model.eval()
        
    test_result_acc = {}
    test_result_f1 = {}
    for target_state in target_state_list:
        testx, testy = test_dict[target_state]
        # save accuracy and f1 score
        acc, f1 = model.score(testx, testy)
        test_result_acc[target_state] = acc 
        test_result_f1[target_state] = f1
    # save test results
    result_record["test_result_acc"] = test_result_acc
    result_record["test_result_f1"] = test_result_f1
    if 'mlp' in model_name:
        result_record["config"]["device"] = gpu_id
    
    # save result
    with open(path, 'w') as f:
        json.dump(result_record, f)
    del model 
    torch.cuda.empty_cache()
    gc.collect() 
    print(f"Experiment {task_name}-{source_state_str}-{model_name}-ID {experiment_id} finished!!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DRO-bench')
    parser.add_argument('--embedding', default='e5', help='embedding method, e.g., one-hot, e5')
    parser.add_argument('--prompt', default='None', help='prompt method, e.g., wiki, gpt4')
    parser.add_argument('--model', help='model, e.g., DRO, lr, etc')
    parser.add_argument('--task', help='task, e.g., income, pubcov, mobility')
    parser.add_argument('--source', type=str, nargs='+',  help='source state list, e.g., CA PR')
    parser.add_argument('--num', type=int, nargs='+', help='number of samples of each source state, e.g., 1000 1000')
    parser.add_argument('--gpu', default=6, type=int, help='gpu id')
    parser.add_argument('--id', default=0, type=int, help='experiment id')
    parser.add_argument('--initial_embedding_method', default='wiki', type=str, help='initial embedding method, e.g., wiki, gpt4')
    parser.add_argument('--training_method', default='pca', type=str, help='training method, e.g., pca, nn, freeze_embedding')
    args = parser.parse_args()
    
    # set seed
    args.seed = SEED
    args.is_regression = 0
    args.year = YEAR
    # mlp: execute single experiment
    if args.model == 'mlp':
        # parallel training 
        num_gpus = torch.cuda.device_count()
        args.gpu_id = np.random.randint(0, num_gpus)
        # check if the experiment has been done
        if result_exists(args) == False:
            # load training and testing data (once for a series of experiments)
            trainx, trainy, val_dict, test_dict = load_train_val_test_data(args)
            #print(f"Finish loading data! Train size is {trainx.shape}")
            # start training
            safe_experiment(trainx, trainy, val_dict, test_dict, args)
    else:
        num_experiments = NUM_EXPERIMENTS
        if args.model in ['xgb', 'lightgbm']:
            config_sum = 1944
        elif args.model == 'svm':
            config_sum = 34
        elif args.model == 'rf':
            config_sum = 1280
        elif args.model == 'gbm':
            config_sum = 360
        elif args.model == 'lr':
            config_sum = 23

        # select config list 
        if config_sum >= num_experiments:
            np.random.seed(SEED)
            selected_configs = np.random.choice(config_sum, num_experiments, replace=False)
        else:
            selected_configs = np.arange(config_sum)

        # load training and testing data (once for a series of experiments)
        trainx, trainy, val_dict, test_dict = load_train_val_test_data(args)
        print(f"Finish loading data! Train size is {trainx.shape}")
        # parallel computing
        for experiment_id in selected_configs:
            args.id = experiment_id
            safe_experiment(trainx, trainy, val_dict, test_dict, args)