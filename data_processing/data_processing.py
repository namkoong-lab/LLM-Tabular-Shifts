#!/usr/bin/env python
# coding: utf-8

# In[1]:


from whyshift import get_data, degradation_decomp, fetch_model, risk_region
from whyshift.folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage, ACSMobility, ACSTravelTime

import numpy as np 
import torch 
import random 
import pickle

import tiktoken

from dataset import *
from preprocess import *
from serialize import *
from embed import *

from joblib import Parallel, delayed


# In[2]:


import warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None 


# In[3]:


s = 'AL,AK,AZ,AR,CA,CO,CT,DE,FL,GA,HI,ID,IL,IN,IA,KS,KY,LA,ME,MD,MA,MI,MN,MS,MO,MT,NE,NV,NH,NJ,NM,NY,NC,ND,OH,OK,OR,PA,RI,SC,SD,TN,TX,UT,VT,VA,WA,WV,WI,WY,PR'
all_states = s.split(',')


# In[4]:


def data_processing_ACSIncome(task):
    # raw dataset
    raw_df = get_raw_df(task.task_name, task.state, root_dir = task.root_dir, year=task.year)
    # serialize table
    serialize_df = serialize_table(task.data_name, raw_df, target = task.target, group = task.group, root_dir = task.root_dir)
    # embedding
    # TODO: check how to select data for embedding
    embed_table(task, serialize_df.iloc[:50000], root_dir = task.root_dir)
    print('data processing finished: {}-{}'.format(task.data_name, task.prompt_method))


# In[5]:


def data_processing_ACSPubCov(task):
    # raw dataset
    raw_df = get_raw_df(task.task_name, task.state, root_dir = task.root_dir, year=task.year)
    # serialize table
    serialize_df = serialize_table(task.data_name, raw_df, target = task.target, group = task.group, root_dir = task.root_dir)
    # embedding
    # TODO: check how to select data for embedding
    embed_table(task, serialize_df.iloc[:50000], root_dir = task.root_dir)
    print('data processing finished: {}-{}'.format(task.data_name, task.prompt_method))


# In[6]:


def data_processing_ACSMobility(task):
    # raw dataset
    raw_df = get_raw_df(task.task_name, task.state, root_dir = task.root_dir, year=task.year)
    # serialize table
    serialize_df = serialize_table(task.data_name, raw_df, target = task.target, group = task.group, root_dir = task.root_dir)
    # embedding
    # TODO: check how to select data for embedding
    embed_table(task, serialize_df.iloc[:50000], root_dir = task.root_dir)
    print('data processing finished: {}-{}'.format(task.data_name, task.prompt_method))


# In[7]:


def data_processing(task_name, prompt_method, state, year = 2018, root_dir = None):
    if task_name == 'income':
        task = ACSIncome
        task.task_name = task_name
        task.state  = state
        task.year = year
        task.data_name = 'ACSIncome-{}-{}'.format(state, year)
        task.prompt_method = prompt_method
        task.root_dir = root_dir
        return data_processing_ACSIncome(task)
    elif task_name =='pubcov':
        task = ACSPublicCoverage
        task.task_name = task_name
        task.state = state
        task.year = year
        task.data_name = 'ACSPubCov-{}-{}'.format(state, year)
        task.prompt_method = prompt_method
        task.root_dir = root_dir
        return data_processing_ACSPubCov(task)
    elif task_name == 'mobility':
        task = ACSMobility
        task.task_name = task_name
        task.state = state
        task.year = year
        task.data_name = 'ACSMobility-{}-{}'.format(state, year)
        task.prompt_method = prompt_method
        task.root_dir = root_dir
        return data_processing_ACSMobility(task)


# ### ACSMobility

# In[8]:


task_name = 'mobility'
prompt_method_list = ['domainlabel']

if task_name == 'income':
    root_dir = '/shared/share_mala/llm-dro/income'
elif task_name == 'pubcov':
    root_dir = '/shared/share_mala/llm-dro/pubcov'
elif task_name == 'mobility':
    root_dir = '/shared/share_mala/llm-dro/mobility'


# In[ ]:


for prompt_method in prompt_method_list:
    for state in all_states:
        data_processing(task_name = task_name, prompt_method=prompt_method, state = state, year = 2018, root_dir=root_dir)


# ### ACSPubCov

# In[10]:


task_name = 'pubcov'
prompt_method_list = ['domainlabel']

if task_name == 'income':
    root_dir = '/shared/share_mala/llm-dro/income'
elif task_name == 'pubcov':
    root_dir = '/shared/share_mala/llm-dro/pubcov'
elif task_name == 'mobility':
    root_dir = '/shared/share_mala/llm-dro/mobility'


# In[ ]:


for prompt_method in prompt_method_list:
    for state in all_states:
        data_processing(task_name = task_name, prompt_method=prompt_method, state = state, year = 2018, root_dir=root_dir)


# ### ACSIncome

# In[12]:


task_name = 'income'
prompt_method_list = ['domainlabel']
if task_name == 'income':
    root_dir = '/shared/share_mala/llm-dro/income'
elif task_name == 'pubcov':
    root_dir = '/shared/share_mala/llm-dro/pubcov'
elif task_name == 'mobility':
    root_dir = '/shared/share_mala/llm-dro/mobility'


# In[13]:


for prompt_method in prompt_method_list:
    for state in all_states:
        data_processing(task_name = 'income', prompt_method=prompt_method, state = state, year = 2018, root_dir=root_dir)


# In[ ]:





# In[14]:


## wikipedia info
extra_info = {}
extra_info["CA"] = "California's economy is the largest of any state within the United States, with a $3.6 trillion gross state product (GSP) as of 2022. It is the largest sub-national economy in the world. If California were a sovereign nation, it would rank as the world's fifth-largest economy as of 2022, just ahead of India and the United Kingdom, as well as the 37th most populous. The Greater Los Angeles area and the San Francisco area are the nation's second- and fourth-largest urban economies ($1.0 trillion and $0.6 trillion respectively as of 2020). The San Francisco Bay Area Combined Statistical Area had the nation's highest gross domestic product per capita ($106,757) among large primary statistical areas in 2018, and is home to four of the world's ten largest companies by market capitalization and four of the world's ten richest people. Slightly over 84 percent of the state's residents 25 or older hold a high school degree, the lowest high school education rate of all 50 states."
extra_info["PR"] = "Puerto Rico is classified as a high income economy by the World Bank and International Monetary Fund. It is considered the most competitive economy in Latin America by the World Economic Forum and ranks highly on the Human Development Index. According to World Bank, gross national income per capita in Puerto Rico in 2020 was $21,740. Puerto Rico's economy is mainly driven by manufacturing (primarily pharmaceuticals, textiles, petrochemicals and electronics) followed by services (primarily finance, insurance, real estate and tourism); agriculture represents less than 1\% of GNP. In recent years, it has also become a popular destination for MICE (meetings, incentives, conferencing, exhibitions), with a modern convention center district overlooking the Port of San Juan.Puerto Rico's geography and political status are both determining factors for its economic prosperity, primarily due to its relatively small size; lack of natural resources and subsequent dependence on imports; and vulnerability to U.S. foreign policy and trading restrictions, particularly concerning its shipping industry"
extra_info["TX"] = "As of 2022, Texas had a gross state product (GSP) of $2.4 trillion, the second highest in the U.S. Its GSP is greater than the GDP of Italy, the world's 8th-largest economy. The state ranks 22nd among U.S. states with a median household income of $64,034, while the poverty rate is 14.2%, making Texas the state with 14th highest poverty rate (compared to 13.15% nationally). Texas's economy is the second-largest of any country subdivision globally, behind California.Texas's large population, an abundance of natural resources, thriving cities and leading centers of higher education have contributed to a large and diverse economy. Since oil was discovered, the state's economy has reflected the state of the petroleum industry. In recent times, urban centers of the state have increased in size, containing two-thirds of the population in 2005. The state's economic growth has led to urban sprawl and its associated symptoms.As of May 2020, during the COVID-19 pandemic, the state's unemployment rate was 13 percent.In 2010, Site Selection Magazine ranked Texas as the most business-friendly state, in part because of the state's three-billion-dollar Texas Enterprise Fund. Texas has the highest number of Fortune 500 company headquarters in the United States as of 2022. In 2010, there were 346,000 millionaires in Texas, the second-largest population of millionaires in the nation. In 2018, the number of millionaire households increased to 566,578."
extra_info["SD"] = "The current-dollar gross state product of South Dakota was $39.8 billion as of 2010, the fifth-smallest total state output in the U.S. The per capita personal income was $38,865 in 2010, ranked 25th in the U.S., and 12.5\% of the population was below the poverty line in 2008.CNBC's list of \"Top States for Business for 2010\" has recognized South Dakota as the seventh best state in the nation. In July 2011, the state's unemployment rate was 4.7%.The service industry is the largest economic contributor in South Dakota. This sector includes the retail, finance, and health care industries."

extra_info = {}
extra_info["CA"] = "California's economy is the largest of any state within the United States, with a $3.6 trillion gross state product (GSP) as of 2022. It is the largest sub-national economy in the world. If California were a sovereign nation, it would rank as the world's fifth-largest economy as of 2022, just ahead of India and the United Kingdom, as well as the 37th most populous. The Greater Los Angeles area and the San Francisco area are the nation's second- and fourth-largest urban economies ($1.0 trillion and $0.6 trillion respectively as of 2020). The San Francisco Bay Area Combined Statistical Area had the nation's highest gross domestic product per capita ($106,757) among large primary statistical areas in 2018, and is home to four of the world's ten largest companies by market capitalization and four of the world's ten richest people. Slightly over 84 percent of the state's residents 25 or older hold a high school degree, the lowest high school education rate of all 50 states."
extra_info["PR"] = "Puerto Rico is classified as a high income economy by the World Bank and International Monetary Fund. It is considered the most competitive economy in Latin America by the World Economic Forum and ranks highly on the Human Development Index. According to World Bank, gross national income per capita in Puerto Rico in 2020 was $21,740. Puerto Rico's economy is mainly driven by manufacturing (primarily pharmaceuticals, textiles, petrochemicals and electronics) followed by services (primarily finance, insurance, real estate and tourism); agriculture represents less than 1\% of GNP. In recent years, it has also become a popular destination for MICE (meetings, incentives, conferencing, exhibitions), with a modern convention center district overlooking the Port of San Juan.Puerto Rico's geography and political status are both determining factors for its economic prosperity, primarily due to its relatively small size; lack of natural resources and subsequent dependence on imports; and vulnerability to U.S. foreign policy and trading restrictions, particularly concerning its shipping industry"
extra_info["TX"] = "As of 2022, Texas had a gross state product (GSP) of $2.4 trillion, the second highest in the U.S. Its GSP is greater than the GDP of Italy, the world's 8th-largest economy. The state ranks 22nd among U.S. states with a median household income of $64,034, while the poverty rate is 14.2%, making Texas the state with 14th highest poverty rate (compared to 13.15% nationally). Texas's economy is the second-largest of any country subdivision globally, behind California.Texas's large population, an abundance of natural resources, thriving cities and leading centers of higher education have contributed to a large and diverse economy. Since oil was discovered, the state's economy has reflected the state of the petroleum industry. In recent times, urban centers of the state have increased in size, containing two-thirds of the population in 2005. The state's economic growth has led to urban sprawl and its associated symptoms.As of May 2020, during the COVID-19 pandemic, the state's unemployment rate was 13 percent.In 2010, Site Selection Magazine ranked Texas as the most business-friendly state, in part because of the state's three-billion-dollar Texas Enterprise Fund. Texas has the highest number of Fortune 500 company headquarters in the United States as of 2022. In 2010, there were 346,000 millionaires in Texas, the second-largest population of millionaires in the nation. In 2018, the number of millionaire households increased to 566,578."
extra_info["SD"] = "The current-dollar gross state product of South Dakota was $39.8 billion as of 2010, the fifth-smallest total state output in the U.S. The per capita personal income was $38,865 in 2010, ranked 25th in the U.S., and 12.5\% of the population was below the poverty line in 2008.CNBC's list of \"Top States for Business for 2010\" has recognized South Dakota as the seventh best state in the nation. In July 2011, the state's unemployment rate was 4.7%.The service industry is the largest economic contributor in South Dakota. This sector includes the retail, finance, and health care industries."

