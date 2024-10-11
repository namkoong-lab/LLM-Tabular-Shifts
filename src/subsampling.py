import numpy as np
import os
from sklearn.metrics import f1_score
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import xgboost as xgb
import pandas as pd 



class SubSample():
    def __init__(self, sensitive_features = [0]):
        self.sensitive_feature = sensitive_features[0]
        self.kind = None
    
    def update(self, config):
        self.base_model = xgb.XGBClassifier(learning_rate=config["learning_rate"], min_split_loss=config["min_split_loss"],
                    max_depth=config["max_depth"], colsample_bytree=config["colsample_bytree"], colsample_bylevel=config["colsample_bylevel"],
                    grow_policy=config["grow_policy"])
        self.kind = config["kind"]
    
    def reload(self, X, y):
        self.df_data = pd.DataFrame(X)
        self.df_data['label'] = y
        
        if self.kind == 'suby':
            if np.mean(y) > 1:
                grp_pos = resample(self.df_data[self.df_data['label'] == 1], replace = True, n_samples = len(self.df_data[self.df_data['label'] == 0]))
                new_data = pd.concat([grp_pos, self.df_data[self.df_data['label'] == 0]])
            else:
                grp_pos = resample(self.df_data[self.df_data['label'] == 0], replace = True, n_samples = len(self.df_data[self.df_data['label'] == 1]))
                new_data = pd.concat([grp_pos, self.df_data[self.df_data['label'] == 1]])
            y = new_data['label'].to_numpy()
            del new_data['label']
            X = np.array(new_data)
            sample_weight = np.ones(len(X))

        elif self.kind == 'subg':
            feature_cnt = self.df_data.groupby(self.sensitive_feature).count()['label']
            min_cnt, min_cnt_grp = np.min(feature_cnt), np.argmin(feature_cnt)

            new_data = self.df_data[self.df_data[self.sensitive_feature] == feature_cnt.index[min_cnt_grp]]
            for i, idx in enumerate(feature_cnt.index):
                if i != min_cnt_grp:
                    grp_data = resample(self.df_data[self.df_data[self.sensitive_feature] == idx], replace = True, n_samples = min_cnt)
                    new_data = pd.concat([new_data, grp_data])    
            y = new_data['label'].to_numpy()
            del new_data['label']
            X = np.array(new_data)
            sample_weight = np.ones(len(X))

        
        elif self.kind == 'rwy':
            up_weight = np.mean(y) / (1 - np.mean(y))
            sample_weight = [1 if y[i] ==  1 else up_weight for i in range(len(y))]
            sample_weight = (sample_weight / np.sum(sample_weight))*X.shape[0]

        elif self.kind == 'rwg':
            feature_cnt = self.df_data.groupby(self.sensitive_feature).count()['label']
            min_cnt, min_cnt_grp = np.min(feature_cnt), np.argmin(feature_cnt)
            sample_weightdict = dict.fromkeys(feature_cnt.index)
            for i, idx in enumerate(feature_cnt.index):
                sample_weightdict[idx] = min_cnt / list(feature_cnt)[i]
            sample_weight = [sample_weightdict[i] for i in self.df_data[self.sensitive_feature]]
            sample_weight = (sample_weight / np.sum(sample_weight))*X.shape[0]
        else:
            raise NotImplementedError

        self.X = X 
        self.y = y 
        self.sample_weight = sample_weight
    
    def fit(self, X, y):
        # balance
        self.reload(X, y)
        # fit 
        fit_params={'sample_weight': self.sample_weight}
        self.base_model.fit(self.X, self.y, **fit_params)
    
    def predict(self, X):
        return self.base_model.predict(X)
    
    def score(self, X, y):
        predicted = self.predict(X)
        correct = (predicted == y).sum()
        total = y.shape[0]
        acc = correct / total
        f1 = self.f1score(X, y)
        return acc, f1

    def f1score(self, X, y):
        predicted = self.predict(X)
        return f1_score(y, predicted, average='macro')
    
    def save(self, idx, dir='/shared/share_mala/llm-dro/save_models/'):
        os.makedirs(f'{dir}/subsampling', exist_ok=True) 
        self.base_model.save_model(f"{dir}/subsampling/{idx}.json")

    def load(self, idx, dir='/shared/share_mala/llm-dro/save_models/'):
        self.base_model = xgb.XGBClassifier()
        self.base_model.load_model(f"{dir}/subsampling/{idx}.json")