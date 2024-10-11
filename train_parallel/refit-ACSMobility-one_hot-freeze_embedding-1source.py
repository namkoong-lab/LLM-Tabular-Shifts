import torch
import numpy as np

from parallel_process import parallel_process
from find_best_experiment import *

## ACSMobility
task_name = 'mobility'

# embedding and prompt method
embedding_method = 'one_hot'
prompt_method = 'None'
initial_embedding_method = 'None'
training_method = 'freeze_embedding'

# refit config
refit_method = 'freeze_embedding'
refit_num_list = [16, 32, 64, 128]

# hps/model selection method
model_name = 'mlp'
hps_selection_method = 'target_32'              # how to select the best hps
hps_selection_metric = 'f1'                     # the metric to select the best hps

source_state_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
                     'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 
                     'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 
                     'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 
                     'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']

target_state_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
                     'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 
                     'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 
                     'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 
                     'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']

num_list = ['20000', '20000', '20000', '20000', '20000', '20000', '20000', '20000', '20000', '20000', 
            '20000', '20000', '20000', '20000', '20000', '20000', '20000', '20000', '20000', '20000', 
            '20000', '20000', '20000', '20000', '20000', '20000', '20000', '20000', '20000', '20000',
            '20000', '20000', '20000', '20000', '20000', '20000', '20000', '20000', '20000', '20000',
            '20000', '20000', '20000', '20000', '20000', '20000', '20000', '20000', '20000', '20000', '20000']

# mlp: execute single experiment
command_list = []
# add experiment configs for each source state
for i, source_state in enumerate(source_state_list):
    source_states = source_state.split()
    nums = num_list[i].split()
    # for each target state
    for j, target_state in enumerate(target_state_list):
        # skip the same state
        if target_state != source_state:
            # find the best experiment id
            source_state_str = "-".join(source_state.split(' '))
            if embedding_method == 'concat':
                cur_train_method = f"{embedding_method}/{initial_embedding_method}/{training_method}"
            elif embedding_method == 'e5':
                cur_train_method = prompt_method
            elif embedding_method == 'one_hot':
                cur_train_method = embedding_method
            experiment_id = find_best_experiment(task_name, cur_train_method, model_name,
                                                        source_state_str, target_state,
                                                        hps_selection_method, hps_selection_metric,
                                                        result_summary_dir= '/shared/share_mala/llm-dro/results/result_summary/')
            # conduct refit
            for refit_num in refit_num_list:
                # build the command
                command = ['python', '../refit.py', '--task', task_name, '--model', model_name,
                        '--embedding', embedding_method, '--prompt', prompt_method, 
                        '--initial_embedding', initial_embedding_method, 
                        '--training_method', training_method,
                        '--id', str(experiment_id), 
                        '--target', target_state, 
                        '--refit_method', refit_method, 
                        '--refit_num', refit_num] + ['--source'] + source_states + ['--num'] + nums
                command_list.append(command)

# parallel training
if len(command_list) > 0:
    command_list = np.random.permutation(command_list).tolist()
    num_gpus = torch.cuda.device_count()
    max_concurrent_processes = 4*num_gpus  # Maximum number of concurrent processes
    parallel_process(command_list, max_concurrent_processes=max_concurrent_processes)



