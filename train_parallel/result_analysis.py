import json
import pickle
import os 
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# Define several constants -- Should not be changed
VAL_NUM_LIST = ['16', '32', '64', '128', 'oracle']
NUM_EXPERIMENTS = 200
SEED = 0
HPS_SELECTION_METHOD_LIST = ['source', 'target_oracle', 'target_16', 'target_32', 'target_64', 'target_128']
RESULT_SUMMARY_DIR = '/shared/share_mala/llm-dro/results/result_summary/'

def load_source_target_results(task_name, train_method, source_state, target_state_list, model_name):
    """
    Load F1 score on source and target states
    """
    # find path to experiment results
    source_state_str = '-'.join(source_state.split(" "))
    # find dir 
    path = f'/shared/share_mala/llm-dro/results/{task_name}/{train_method}/{source_state_str}/{model_name}'
    # source results, val results on target states
    source_acc_results, source_f1_results = [], []
    target_acc_dict, target_f1_dict = dict(), dict()
    # create a dictionary to store the results
    val_num_list = VAL_NUM_LIST    # number of validation samples from target state
    for val_num in val_num_list:
        target_acc_dict[val_num] = dict()
        target_f1_dict[val_num] = dict()
        for target_state in target_state_list:      # for each target state, create a list to store results
            target_acc_dict[val_num][target_state] = []
            target_f1_dict[val_num][target_state] = []

    # select a subset of experiments
    num_experiments = NUM_EXPERIMENTS
    if model_name in ['xgb', 'lightgbm']:
        config_sum = 1944
    elif model_name == 'svm':
        config_sum = 34
    elif model_name == 'rf':
        config_sum = 1280
    elif model_name == 'gbm':
        config_sum = 360
    elif model_name == 'lr':
        config_sum = 23
    else:
        config_sum = 200

    # select config list 
    if config_sum >= num_experiments:
        np.random.seed(SEED)
        selected_configs = np.random.choice(config_sum, num_experiments, replace=False)
    else:
        selected_configs = np.arange(config_sum)

    # Iterate over each result file in the directory
    # each result file corresponds to one experiment id or training hps
    for config_file in os.listdir(path):
        # Extract experiment index from file name
        experiment_id = int(config_file.split('.')[0]) 
        # check if the experiment id is in the selected configs
        if experiment_id not in selected_configs:
            continue
        full_path = os.path.join(path, config_file)
        if os.path.isfile(full_path):  # Make sure it's a file
            with open(full_path, 'r') as file:
                config = json.load(file)
                # load model's test performance on source states
                source_state = source_state_str.split('-')
                source_acc = np.array([value for key, value in config["test_result_acc"].items() if key in source_state])
                source_f1 = np.array([value for key, value in config["test_result_f1"].items() if key in source_state])
                source_acc_avg = source_acc.mean()
                source_f1_avg = source_f1.mean()
                # add experiment id and mean results 
                source_acc_results.append((experiment_id, source_acc_avg, config["config"]))
                source_f1_results.append((experiment_id, source_f1_avg, config["config"]))
                
                # load model's test performance on target states (on test and validation dataset)
                for target_state in target_state_list:
                    if target_state not in source_state:
                        # validation result of target states
                        for val_num in val_num_list[:-1]:
                            target_acc = np.array([value for key, value in config["val_result_acc"][str(val_num)].items() if key in target_state])
                            target_f1 = np.array([value for key, value in config["val_result_f1"][str(val_num)].items() if key in target_state])
                            # add experiment id and mean results 
                            target_acc_dict[val_num][target_state].append((experiment_id, target_acc, config["config"]))
                            target_f1_dict[val_num][target_state].append((experiment_id, target_f1, config["config"]))
                        
                        # oracle(test) result of target states
                        target_acc = np.array([value for key, value in config["test_result_acc"].items() if key in target_state])
                        target_f1 = np.array([value for key, value in config["test_result_f1"].items() if key in target_state])
                        # add experiment id and mean results 
                        target_acc_dict['oracle'][target_state].append((experiment_id, target_acc, config["config"]))
                        target_f1_dict['oracle'][target_state].append((experiment_id, target_f1, config["config"]))
    return source_acc_results, source_f1_results, target_acc_dict, target_f1_dict            

def summarize_best_experiment_results(task_name, train_method, 
                                 source_state, target_state_list, 
                                 model_name, 
                                 hps_selection_method_list = HPS_SELECTION_METHOD_LIST, 
                                 result_summary_dir = RESULT_SUMMARY_DIR):
    
    # convert source state list to str
    source_state_str = '-'.join(source_state.split(" "))
    # check if result dict already exists 
    save_path = f"{result_summary_dir}/{task_name}/{train_method}-{model_name}/{source_state_str}.pkl"
    dir = os.path.dirname(save_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    # check if the file already exists
    #if os.path.exists(save_path):
        #print(f"{task_name}/{train_method}-{model_name}/{source_state_str}.pkl already exists" )
    #    return 
    
    # load experiment results 
    source_acc_results, source_f1_results, target_acc_dict, target_f1_dict = load_source_target_results(task_name, 
                                                                                                        train_method, 
                                                                                                        source_state, 
                                                                                                        target_state_list, 
                                                                                                        model_name)
    # check if hps selection method is valid
    for hps_selection_method in hps_selection_method_list:
        if 'target' in hps_selection_method:
            val_num = hps_selection_method.split('_')[1]
            if val_num not in target_acc_dict:
                raise ValueError(f"Validation number {val_num} is not in the validation number list!")
        elif hps_selection_method not in ['source']:
            raise ValueError(f"Invalid hps selection method: {hps_selection_method}")
    
    # initialize a dictionary to store the results
    result_dict = {}
    
    # for each train method and model, find the best experiment id for each (source, target) pair
    result_dict[train_method] = {}
    result_dict[train_method][model_name] = {}
    
    # for each hps selection method
    for hps_selection_method in hps_selection_method_list:    
        # create a dictionary to store the results for the hps selection method 
        if hps_selection_method not in result_dict[train_method][model_name]:
            result_dict[train_method][model_name][hps_selection_method] = {}
        # create a dictionary to store the results for the source state
        result_dict[train_method][model_name][hps_selection_method][source_state_str] = {}
        try: 
            # method 1: based on the average performance on the source states
            if hps_selection_method == 'source':
                # Find the best experiment ID based on accuracy and F1 score
                best_acc_experiment_id, val_result_acc = max(source_acc_results, key=lambda x: x[1])[:2]
                best_f1_experiment_id, val_result_f1 = max(source_f1_results, key=lambda x: x[1])[:2]
                # save best id and test results for each target state
                for target_state in target_state_list:
                    if target_state not in source_state:
                        # save best id
                        result_dict[train_method][model_name][hps_selection_method][source_state_str][target_state] = {
                                "acc_experiment_id": best_acc_experiment_id,
                                "f1_experiment_id": best_f1_experiment_id}
                        # find test result for experiment id
                        test_result_acc = [result for result in target_f1_dict['oracle'][target_state] if result[0] == best_acc_experiment_id][0]    
                        test_result_f1 = [result for result in target_f1_dict['oracle'][target_state] if result[0] == best_f1_experiment_id][0]    
                        test_result_acc = test_result_acc[1][0]
                        test_result_f1 = test_result_f1[1][0]
                        # save validation result (test performance of source state)
                        result_dict[train_method][model_name][hps_selection_method][source_state_str][target_state]["val_result_acc"] = val_result_acc
                        result_dict[train_method][model_name][hps_selection_method][source_state_str][target_state]["val_result_f1"] = val_result_f1
                        # save target results for the selected experiment id
                        result_dict[train_method][model_name][hps_selection_method][source_state_str][target_state]["test_result_acc"] = test_result_acc
                        result_dict[train_method][model_name][hps_selection_method][source_state_str][target_state]["test_result_f1"] = test_result_f1
            
            # method 2: based on validation/test performance on the target states
            if 'target' in hps_selection_method:
                val_num = hps_selection_method.split('_')[1]
                for target_state in target_state_list:
                    if target_state not in source_state:
                        # Find the best experiment ID based on accuracy and F1 score
                        best_acc_experiment_id, val_result_acc = max(target_acc_dict[val_num][target_state], key=lambda x: x[1])[:2]
                        best_f1_experiment_id, val_result_f1 = max(target_f1_dict[val_num][target_state], key=lambda x: x[1])[:2]
                        # convert list into float
                        val_result_acc = val_result_acc[0]
                        val_result_f1 = val_result_f1[0]
                        # save best id
                        result_dict[train_method][model_name][hps_selection_method][source_state_str][target_state] = {
                                "acc_experiment_id": best_acc_experiment_id,
                                "f1_experiment_id": best_f1_experiment_id}
                        # find test result for experiment id
                        test_result_acc = [result for result in target_f1_dict['oracle'][target_state] if result[0] == best_acc_experiment_id][0]    
                        test_result_f1 = [result for result in target_f1_dict['oracle'][target_state] if result[0] == best_f1_experiment_id][0]    
                        test_result_acc = test_result_acc[1][0]
                        test_result_f1 = test_result_f1[1][0]
                        # save validation result (performance of target state on selected samples)
                        result_dict[train_method][model_name][hps_selection_method][source_state_str][target_state]["val_result_acc"] = val_result_acc
                        result_dict[train_method][model_name][hps_selection_method][source_state_str][target_state]["val_result_f1"] = val_result_f1
                        # save target results for the selected experiment id
                        result_dict[train_method][model_name][hps_selection_method][source_state_str][target_state]["test_result_acc"] = test_result_acc
                        result_dict[train_method][model_name][hps_selection_method][source_state_str][target_state]["test_result_f1"] = test_result_f1
        except:
            print(f"source {source_state_str}, target {target_state}, train method {train_method}, hps selection based on {hps_selection_method} failed!")

    # save results to a json file
    with open(save_path, 'wb') as f:
        pickle.dump(result_dict, f)
    print(f"{task_name}/{train_method}-{model_name}/{source_state_str}.pkl finished" )
    return 


def summarize_best_experiment_results_all_train_methods_source_state(source_state, task_name, train_method_list, target_state_list, hps_selection_method_list):
    # for each train method
    for train_method in set(train_method_list):
        if train_method == 'one_hot':
            concerned_model_name_list = ['lr', 'xgb', 'mlp']
        else:
            concerned_model_name_list = ['mlp']
        # for each model (mlp, lr, xgb)
        for model_name in concerned_model_name_list:
            # load best experiment results
            summarize_best_experiment_results(task_name, train_method, 
                                         source_state, target_state_list, 
                                         model_name, 
                                         hps_selection_method_list)

def load_best_experiment_results(task_name, train_method_list, 
                                 source_state_list, 
                                 hps_selection_method_list = HPS_SELECTION_METHOD_LIST,
                                 result_summary_dir = RESULT_SUMMARY_DIR):
    # initialize result dict
    result_dict = dict()
    
    # for each train method
    for train_method in set(train_method_list):
        # initialize result dict
        if train_method not in result_dict:
            result_dict[train_method] = dict()
        # find model name
        if train_method == 'one_hot':
            concerned_model_name_list = ['lr', 'xgb', 'mlp']
        else:
            concerned_model_name_list = ['mlp']
        # for each model (mlp, lr, xgb)
        for model_name in concerned_model_name_list:
            # initialize result dict
            if model_name not in result_dict[train_method]:
                result_dict[train_method][model_name] = dict()
            # for each source state
            for source_state in source_state_list:
                # convert source state list to str
                source_state_str = '-'.join(source_state.split(" "))
                # find result summary path 
                save_path = f"{result_summary_dir}/{task_name}/{train_method}-{model_name}/{source_state_str}.pkl"
                # load result summary results
                with open(save_path, 'rb') as f:
                    cur_result_dict = pickle.load(f)

                # save results to result dict for each hps selection method
                for hps_selection_method in hps_selection_method_list:
                    if hps_selection_method not in result_dict[train_method][model_name]:
                        result_dict[train_method][model_name][hps_selection_method] = dict()
                    
                    result_dict[train_method][model_name][hps_selection_method][source_state_str] = cur_result_dict[train_method][model_name][hps_selection_method][source_state_str]
    # return result dict
    return result_dict


# summarize results, concat all concerned results
def summarize_target_results(result_dict, train_method_list, model_name_list, 
                             source_state_list, target_state_list, 
                             metric, 
                             hps_selection_method_list = HPS_SELECTION_METHOD_LIST):

    # check if hps selection method is valid
    for hps_selection_method in hps_selection_method_list:
        if 'target' in hps_selection_method:
            val_num = hps_selection_method.split('_')[1]
            if val_num not in VAL_NUM_LIST:
                raise ValueError(f"Validation number {val_num} is not in the validation number list!")
        elif hps_selection_method not in ['source']:
            raise ValueError(f"Invalid hps selection method: {hps_selection_method}")
    
    # save results to target_results
    target_results = {}
    for train_method in train_method_list:
        # for each train method
        target_results[train_method] = {}
        # select models
        if train_method == 'one_hot':
            concerned_model_name_list = model_name_list
        else:
            concerned_model_name_list = ['mlp']
        # iterate over different models
        for model_name in concerned_model_name_list:
            target_results[train_method][model_name] = {}
            for hps_selection_method in hps_selection_method_list:
                target_results[train_method][model_name][hps_selection_method] = {}
                # for each source state
                for source_state in source_state_list:
                    source_state_str = '-'.join(source_state.split(" "))
                    # check if the source state has been added 
                    if source_state_str not in target_results:
                        target_results[train_method][model_name][hps_selection_method][source_state_str] = {}
                    try:
                        # load results for each target state
                        for target_state in target_state_list:
                            if target_state not in source_state:
                                cur_test_result = result_dict[train_method][model_name][hps_selection_method][source_state_str][target_state][f"test_result_{metric}"]
                                target_results[train_method][model_name][hps_selection_method][source_state_str][target_state] = cur_test_result
                    except:
                        print(f"{model_name} on {source_state_str} and {train_method} not trained yet!")
    return target_results

## load refit results
# find the base model used for refit
def find_best_experiment(task_name, train_method, model_name, 
                         source_state_str, target_state,
                         hps_selection_method, hps_selection_metric,
                         result_summary_dir= RESULT_SUMMARY_DIR):
    # load experiment results 
    result_dict = load_best_experiment_results(task_name, [train_method], 
                                                [source_state_str], 
                                                result_summary_dir = result_summary_dir)
    selected_experiment_id = result_dict[train_method][model_name][hps_selection_method][source_state_str][target_state][f"{hps_selection_metric}_experiment_id"]
    return selected_experiment_id

## plot for base models
def compare_base_models_plot(method_list, hps_selection_method_list, test_f1_dict, 
                             sorted_concept_list, K = 500, 
                             ylim=[0.5, 1], 
                             train_method_label_list = ['Classical + xgb', 'Classical + mlp', 'LLM', 'LLM + wiki', 'LLM + wiki + nn',  'LLM + gpt4']):

    ## draw a histogram for each method
    # Colors for each hps selection method
    colors = ['b', 'g', 'r', 'c', 'm', 'orange']

    # Bar width and spacing parameters
    bar_width = 0.15
    spacing = 0.5

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set position of bar on X axis
    positions = []
    for i in range(len(method_list)):
        positions.extend(np.arange(len(hps_selection_method_list)) * bar_width + i * (len(hps_selection_method_list) * bar_width + spacing))


    # Set position of bar on X axis
    for i, method in enumerate(method_list):
        train_method, model_name = method
        # Plot bars for this train method
        for j, hps_selection_method in enumerate(hps_selection_method_list):
            avg_score = np.mean(np.array(test_f1_dict[(train_method, model_name)][hps_selection_method])[sorted_concept_list[:K]])
            se_score = np.std(np.array(test_f1_dict[(train_method, model_name)][hps_selection_method])[sorted_concept_list[:K]]) / np.sqrt(K)
            
            # Calculate the exact position for each bar
            position = i * len(hps_selection_method_list) + j
            ax.bar(positions[position], avg_score, yerr=2*se_score, 
                color=colors[j % len(colors)], width=bar_width, label=hps_selection_method if i == 0 else "")

    # Set y-axis limit
    ax.set_ylim([ylim[0], ylim[1]])  # Adjust this limit based on your data range

    # Add labels, title, and legend
    ax.set_xlabel('Train Methods')
    ax.set_ylabel('Target Test F1 scores')
    #ax.set_title('Performance of Train Methods on Test Methods')
    ax.set_xticks([positions[i * len(hps_selection_method_list) + len(hps_selection_method_list) // 2] for i in range(len(method_list))])
    ax.set_xticklabels(train_method_label_list)
    ax.legend(title='HP Selection Methods', 
            loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(hps_selection_method_list))

    
## table for base models
def compare_base_models_table(method_list, hps_selection_method_list, test_f1_dict, 
                             sorted_concept_list, K = 500):
    ## convert into pandas dataframe
    data = []

    for (train_method, model_name), results_dict in test_f1_dict.items():
        row = {
            'Train Method': train_method,
            'Model Name': model_name
        }
        # Calculate mean and standard error for each hps selection method
        for hps_selection_method in hps_selection_method_list:
            if hps_selection_method in results_dict:
                results = np.array(results_dict[hps_selection_method])[sorted_concept_list[:K]]
                mean_score = np.mean(results) * 100
                ste = np.std(results) / np.sqrt(K) * 100
                row[hps_selection_method] = f"{mean_score:.2f} (+- {ste*2:.2f})"
            else:
                row[hps_selection_method] = "N/A"
        data.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df
