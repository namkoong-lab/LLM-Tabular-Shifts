import pickle 

# derive prompt info
state_dict = {
    'AL': 'Alabama','AK': 'Alaska','AZ': 'Arizona','AR': 'Arkansas','CA': 'California','CO': 'Colorado','CT': 'Connecticut',
    'DE': 'Delaware','FL': 'Florida','GA': 'Georgia','HI': 'Hawaii','ID': 'Idaho','IL': 'Illinois','IN': 'Indiana','IA': 'Iowa',
    'KS': 'Kansas','KY': 'Kentucky','LA': 'Louisiana','ME': 'Maine','MD': 'Maryland','MA': 'Massachusetts','MI': 'Michigan',
    'MN': 'Minnesota','MS': 'Mississippi','MO': 'Missouri','MT': 'Montana','NE': 'Nebraska','NV': 'Nevada','NH': 'New Hampshire',
    'NJ': 'New Jersey','NM': 'New Mexico','NY': 'New York','NC': 'North Carolina','ND': 'North Dakota','OH': 'Ohio','OK': 'Oklahoma',
    'OR': 'Oregon','PA': 'Pennsylvania','RI': 'Rhode Island','SC': 'South Carolina','SD': 'South Dakota','TN': 'Tennessee','TX': 'Texas',
    'UT': 'Utah','VT': 'Vermont','VA': 'Virginia','WA': 'Washington','WV': 'West Virginia','WI': 'Wisconsin','WY': 'Wyoming',
    'PR': 'Puerto Rico'
}
    
def get_ACSIncome_prompt(task):
    # instruct prompt
    instruct_prompt = "Classify whether US working adults' yearly income is above $50000 in 2018."
    # query prompt
    # TODO: add domain info from wikipedia
    domain_info = get_domain_info(task)
    state_name = state_dict[task.state]
    domain_label = "The state is {}. ".format(state_name)
    query_prompt = domain_info + domain_label 
    return 'Instruct: {}\nQuery: {}'.format(instruct_prompt, query_prompt)

def get_ACSMobility_prompt(task):
    # instruct prompt
    instruct_prompt = "Classify whether a young adult moved addresses in the last year."
    # query prompt
    # TODO: add domain info from wikipedia
    domain_info = get_domain_info(task)
    state_name = state_dict[task.state]
    domain_label = "The state is {}. ".format(state_name)
    query_prompt = domain_info + domain_label 
    return 'Instruct: {}\nQuery: {}'.format(instruct_prompt, query_prompt)

def get_ACSPubCov_prompt(task):
    # instruct prompt
    instruct_prompt = "Classify whether a low-income individual, not eligible for Medicare, has coverage from public health insurance."
    # query prompt
    # TODO: add domain info from wikipedia
    domain_info = get_domain_info(task)
    state_name = state_dict[task.state]
    domain_label = "The state is {}. ".format(state_name)
    query_prompt = domain_info + domain_label 
    return 'Instruct: {}\nQuery: {}'.format(instruct_prompt, query_prompt)

def get_detailed_prompt(task):
    if task.task_name == 'income':
        return get_ACSIncome_prompt(task)
    elif task.task_name == 'mobility':
        return get_ACSMobility_prompt(task)
    elif task.task_name == 'pubcov':
        return get_ACSPubCov_prompt(task)
    
# domain info 
# TODO: add domain info from wikipedia/gpt
def get_domain_info(task):
    if task.task_name == 'income':
        domain_info = domain_info_dict[task.task_name][task.prompt_method][task.state]
        return domain_info
    elif task.task_name == 'mobility':
        domain_info = domain_info_dict[task.task_name][task.prompt_method][task.state]
        return domain_info
    elif task.task_name == 'pubcov':
        domain_info = domain_info_dict[task.task_name][task.prompt_method][task.state]
        return domain_info
    

domain_info_dict = dict()
domain_info_dir =  '/shared/share_mala/llm-dro/domain_info'
# load domain info for ACSIncome
prompt_method_list = ['domainlabel'] #TODO: add more prompt methods
task_name = 'income'
domain_info_dict['income'] = dict()
for prompt_method in prompt_method_list:
    path = domain_info_dir + "/{}-{}.pkl".format(task_name, prompt_method)
    with open(path, 'rb') as f:
        domain_info_dict['income'][prompt_method] = pickle.load(f)
# load domain info for ACSPubCov
task_name = 'pubcov'
domain_info_dict['pubcov'] = dict()
prompt_method_list = ['domainlabel']
for prompt_method in prompt_method_list:
    path = domain_info_dir + "/{}-{}.pkl".format(task_name, prompt_method)
    with open(path, 'rb') as f:
        domain_info_dict['pubcov'][prompt_method] = pickle.load(f)

# load domain info for ACSMobility
task_name = 'mobility'
domain_info_dict['mobility'] = dict()
for prompt_method in prompt_method_list:
    path = domain_info_dir + "/{}-{}.pkl".format(task_name, prompt_method)
    with open(path, 'rb') as f:
        domain_info_dict['mobility'][prompt_method] = pickle.load(f)

