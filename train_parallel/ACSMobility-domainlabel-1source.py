import torch
import numpy as np

from parallel_process import parallel_process

## ACSMobility
task_name = 'mobility'

# embedding and prompt method
embedding_method = 'e5'
prompt_method = 'domainlabel'

# model_name_list = ['lr', 'xgb', 'rf', 'mlp']#, 'gbm', 'subsampling']
model_name_list = ['mlp']
#model_name_list = ['lr','svm','xgb', 'lightgbm', 'rf', 'dwr', 'jtt','suby', 'subg', 'rwy', 'rwg', 'FairPostprocess_exp','FairInprocess_dp', 'FairPostprocess_threshold', 'FairInprocess_eo', 'FairInprocess_error_parity','chi_dro', 'chi_doro','cvar_dro','cvar_doro','group_dro']
source_state_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
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
# add experiment configs
for i, source_state in enumerate(source_state_list):
    if 'mlp' in model_name_list:
        model_name = 'mlp'
        config_sum = 48 
        # Split the source_state string into individual states
        states = source_state.split()
        # Split the num_list string into individual integers
        nums = num_list[i].split()
        command = ['python', '../train.py', '--embedding', embedding_method, '--prompt', prompt_method, 
                   '--model', model_name, '--task', task_name] + ['--source'] + states + ['--num'] + nums
        print(f"Starting process: {command}")
        # add experiment configs
        for experiment_id in range(config_sum):
            # Build the command
            command = ['python', '../train.py', '--embedding', embedding_method, '--prompt', prompt_method, 
                       '--model', model_name, '--task', task_name, 
                       '--id', str(experiment_id)] + ['--source'] + states + ['--num'] + nums
            command_list.append(command)

# parallel training
if len(command_list) > 0:
    command_list = np.random.permutation(command_list).tolist()
    num_gpus = torch.cuda.device_count()
    max_concurrent_processes = 5*num_gpus  # Maximum number of concurrent processes
    parallel_process(command_list, max_concurrent_processes=max_concurrent_processes)

# other models
command_list = []
# add experiment configs
for i, source_state in enumerate(source_state_list):
    for model_name in model_name_list:
        if model_name != 'mlp':    
            # Split the source_state string into individual states
            states = source_state.split()
            # Split the num_list string into individual integers
            nums = num_list[i].split()    
            command_list.append(['python', '../train.py', '--embedding', embedding_method, 
                                 '--prompt', prompt_method, 
                                 '--model', model_name, 
                                 '--task', task_name] + ['--source'] + states + ['--num'] + nums)
# parallel training
if len(command_list) > 0:
    # Generate a random permutation of the command list
    command_list = np.random.permutation(command_list).tolist()
    # parallel computing
    max_concurrent_processes = 20  # Maximum number of concurrent processes
    parallel_process(command_list, max_concurrent_processes=max_concurrent_processes)
    

