from scipy.optimize import brent
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.autograd import grad
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn import linear_model, ensemble, kernel_approximation, svm
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import xgboost as xgb
from lightgbm import LGBMClassifier
import pickle
import os



class Sklearn_models:
    def __init__(self, method_name):
        self.method_name = method_name

        if method_name == 'rf':
            self.model = RandomForestClassifier()
        elif method_name == 'gbm':
            self.model = GradientBoostingClassifier()
        elif method_name == 'lightgbm':
            self.model = LGBMClassifier()
        elif method_name == 'svm':
            self.model = svm.LinearSVC(max_iter=10000)
        elif method_name == 'xgb':
            self.model = xgb.XGBClassifier()
        elif method_name == 'lr':
            self.model = linear_model.LogisticRegression(solver='sag', n_jobs=5)
    
    def update(self, config):
        if self.method_name == 'rf':
            self.model = RandomForestClassifier(**config)
        elif self.method_name == 'gbm':
            self.model = GradientBoostingClassifier(**config)
        elif self.method_name == 'lightgbm':
            self.model = LGBMClassifier(**config)
        elif self.method_name == 'svm':
            self.model = svm.LinearSVC(**config)
        elif self.method_name == 'xgb':
            self.model = xgb.XGBClassifier(**config)
        elif self.method_name == 'lr':
            config["max_iter"] = 1000
            config["solver"] = 'sag'
            config["n_jobs"] = 5
            self.model = linear_model.LogisticRegression(**config)
    
        
    def fit(self, trainx, trainy, weight=None):
        self.model.fit(trainx, trainy, sample_weight = weight)
    

    def score(self, X, y, weights = None):
        predictions = self.model.predict(X)    
        if weights is not None:
            weights = weights / np.sum(weights)
        else:
            weights = np.zeros(X.shape[0])+1.0/X.shape[0]
        
        pred = (predictions.reshape(-1) == y.reshape(-1))
        acc = np.dot(weights.reshape(-1), pred.reshape(-1))
        f1 = f1_score(y.reshape(-1), predictions.reshape(-1), average='macro')
        
        return acc, f1
    

    def save(self, idx, dir='/shared/share_mala/llm-dro/save_models/'):
        os.makedirs(f'{dir}/{self.method_name}', exist_ok=True) 
        if self.method_name == 'rf':
            with open(f'{dir}rf/{idx}.pkl', 'wb') as file:
                pickle.dump(self.model, file)
        elif self.method_name == 'gbm':
            with open(f'{dir}gbm/{idx}.pkl', 'wb') as file:
                pickle.dump(self.model, file)
        elif self.method_name == 'lightgbm':
            self.model.save_model(f'{dir}lightgbm/{idx}.txt')
        elif self.method_name == 'svm':
            with open(f'{dir}svm/{idx}.pkl', 'wb') as file:
                pickle.dump(self.model, file)
        elif self.method_name == 'xgb':
            self.model.save_model(f"{dir}/xgb/{idx}.json")
        elif self.method_name == 'lr':
            with open(f'{dir}lr/{idx}.pkl', 'wb') as file:
                pickle.dump(self.model, file)
            


    def load(self, idx, dir='/shared/share_mala/llm-dro/save_models/'):
        if self.method_name == 'rf':
            with open(f'{dir}rf/{idx}.pkl', 'rb') as file:
                self.model = pickle.load(file)
        elif self.method_name == 'gbm':
            with open(f'{dir}gbm/{idx}.pkl', 'rb') as file:
                self.model = pickle.load(file)
        elif self.method_name == 'lightgbm':
            self.model = lightgbm.Booster(model_file=f'{dir}lightgbm/{idx}.txt')
        elif self.method_name == 'svm':
            with open(f'{dir}svm/{idx}.pkl', 'rb') as file:
                self.model = pickle.load(file)
        elif self.method_name == 'xgb':
            self.model = xgb.XGBClassifier()
            self.model.load_model(f"{dir}/xgb/{idx}.json")
        elif self.method_name == 'lr':
            with open(f'{dir}lr/{idx}.pkl', 'rb') as file:
                self.model = pickle.load(file)