import numpy as np
import os
from sklearn.metrics import f1_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
from fairlearn.reductions import DemographicParity, EqualizedOdds, \
    ErrorRateParity, ExponentiatedGradient
from fairlearn.postprocessing import ThresholdOptimizer
from typing import List
import pandas as pd
import warnings
from joblib import dump, load
warnings.filterwarnings("ignore", category=FutureWarning, module="fairlearn")



class CustomExponentiatedGradient(ExponentiatedGradient):
    """Custom class to allow for scikit-learn-compatible interface.

    Specifically, this method takes (and ignores) a sample_weights
    parameter to its .fit() method; otherwise identical to
    fairlearn.ExponentiatedGradient.
    """

    def __init__(self, sensitive_features: List[str], **kwargs):
        super().__init__(**kwargs)
        self.sensitive_features = sensitive_features

    def fit(self, X, y, **kwargs):
        if isinstance(X, pd.DataFrame):
            super().fit(X.values, y.values,
                        sensitive_features=X[self.sensitive_features].values,
                        **kwargs)
        elif isinstance(X, np.ndarray):
            super().fit(X, y, sensitive_features = X[:, self.sensitive_features], **kwargs)
        else:
            raise NotImplementedError

    def predict_proba(self, X):
        """Alias to _pmf_predict(). Note that this tends to return 'hard'
        predictions, which don't perform well for metrics like cross-entropy."""
        return super()._pmf_predict(X)


class FairPostprocess():
    def __init__(self, sensitive_features=[0], base_method='xgb'):
        self.base_method = base_method
        self.sensitive_features = sensitive_features
        if base_method == 'xgb':
            base_model = xgb.XGBClassifier()
        elif base_method == 'gbm':
            base_model = GradientBoostingClassifier()
        else:
            raise NotImplementedError
        self.model = ThresholdOptimizer(estimator=base_model, predict_method='predict')

    def update(self, config):
        base_model = xgb.XGBClassifier(learning_rate=config["learning_rate"], min_split_loss=config["min_split_loss"],
                    max_depth=config["max_depth"], colsample_bytree=config["colsample_bytree"], colsample_bylevel=config["colsample_bylevel"],
                    grow_policy=config["grow_policy"])
        self.kind = config["kind"]
        if config["kind"] == "threshold":
            self.model = ThresholdOptimizer(estimator=base_model, predict_method='predict')
        elif config["kind"] == 'exp':
            self.model = ExponentiatedGradient(estimator=base_model, constraints=DemographicParity(difference_bound=0.02))
        else:
            raise NotImplementedError
    
    def fit(self, X, y, device=None):
        self.model.fit(X, y, sensitive_features=X[:,self.sensitive_features])
        
    def predict(self, X):
        if self.kind == "threshold":
            return self.model.predict(X, sensitive_features=X[:,self.sensitive_features],
                               random_state=0)
        elif self.kind == "exp":
            return self.model.predict(X, random_state=0)
    
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
        os.makedirs(f'{dir}/fairness_post', exist_ok=True) 
        dump(self.model, f'{dir}/fairness_post/{idx}.joblib')
            
    def load(self, idx, dir='/shared/share_mala/llm-dro/save_models/'):
        self.model = load(f'{dir}/fairness_post/{idx}.joblib')

class FairInprocess():
    def __init__(self, sensitive_features = [0], base_method = 'xgb'):
        self.sensitive_features = sensitive_features
        self.base_method = base_method
        if base_method == 'xgb':
            base_model = xgb.XGBClassifier()
        elif base_method == 'gbm':
            base_model = GradientBoostingClassifier()
        else:
            raise NotImplementedError
        self.model = CustomExponentiatedGradient(estimator=base_model,
                                    constraints = EqualizedOdds(),
                                    sensitive_features=self.sensitive_features)
        
        
    def update(self, config):
        base_model = xgb.XGBClassifier(learning_rate=config["learning_rate"], min_split_loss=config["min_split_loss"],
                    max_depth=config["max_depth"], colsample_bytree=config["colsample_bytree"], colsample_bylevel=config["colsample_bylevel"],
                    grow_policy=config["grow_policy"])
        self.kind = config["kind"]
        if config["kind"] == "dp":
            constraint = DemographicParity()
        elif config["kind"] == "eo":
            constraint = EqualizedOdds()
        elif config["kind"] == "error_parity":
            constraint = ErrorRateParity()
        else:
            raise NotImplementedError
        # Model
        self.model = CustomExponentiatedGradient(estimator=base_model,
                                            constraints=constraint,
                                            sensitive_features=self.sensitive_features)
    def fit(self, X, y, device=None):
        self.model.fit(X, y)
        
    def predict(self, X):
        z = self.model.predict_proba(X)
        return np.argmax(z, axis=1)

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
        os.makedirs(f'{dir}/fairness_in', exist_ok=True) 
        dump(self.model, f'{dir}/fairness_in/{idx}.joblib')
            
    def load(self, idx, dir='/shared/share_mala/llm-dro/save_models/'):
        self.model = load(f'{dir}/fairness_in/{idx}.joblib')