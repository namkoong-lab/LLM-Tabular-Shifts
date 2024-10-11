import os
import json
import numpy as np

from result_analysis import *


def find_best_experiment(task_name, train_method, model_name, 
                         source_state_str, target_state,
                         hps_selection_method, hps_selection_metric,
                         result_summary_dir= '/shared/share_mala/llm-dro/results/result_summary/'):
    # load experiment results 
    result_dict = load_best_experiment_results(task_name, [train_method], 
                                                [source_state_str], 
                                                result_summary_dir = result_summary_dir)
    selected_experiment_id = result_dict[train_method][model_name][hps_selection_method][source_state_str][target_state][f"{hps_selection_metric}_experiment_id"]
    return selected_experiment_id