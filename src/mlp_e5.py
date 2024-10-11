import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.autograd import grad
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data import random_split

class MLP_e5(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=16, nonlin=nn.ReLU(), dropout_ratio=0.1):
        super().__init__()
        # Input processing layers to reduce dimension to 128
        self.input_process1 = nn.Linear(input_dim, 1024)  # Reduce from 4096 to 1024
        self.input_process2 = nn.Linear(1024, 256)        # Reduce from 1024 to 256
        self.input_process3 = nn.Linear(256, 128)         # Reduce from 256 to 128
        # define layers
        self.dense0 = nn.Linear(128, hidden_dim)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(dropout_ratio)
        self.dense1 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, X, **kwargs):
        X = X.float()
        # Process X through dimensionality reduction
        X = self.nonlin(self.input_process1(X))
        X = self.nonlin(self.input_process2(X))
        X = self.nonlin(self.input_process3(X))
        # pass through network
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = self.output(X)
        return X

class MLPe5Classifier():
    def __init__(self, input_dim=9, num_classes=2, hidden_dim=16, 
                 refit_num=100, refit_lr=0.1, refit_epochs=200, refit_batch_size=128):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = 128
        self.lr = 1e-3
        self.device = torch.device("cuda")
        self.model = MLP_e5(input_dim, num_classes).to(self.device)
        self.train_epochs = 200
        # refit config
        self.refit_num = refit_num
        self.refit_lr = refit_lr
        self.refit_epochs = refit_epochs
        self.refit_batch_size = refit_batch_size

    def update(self, config):
        self.hidden_dim = config["hidden_size"]
        self.lr = config["lr"]
        self.train_epochs = config["train_epochs"]
        gpu_id = config["device"]
        self.device = torch.device(f"cuda:{gpu_id}")
        self.model = MLP_e5(self.input_dim, self.num_classes, hidden_dim=self.hidden_dim, dropout_ratio=config["dropout_ratio"]).to(self.device)

    def update_refit_config(self, config):
        # update refit config
        self.refit_lr = config.get("refit_lr", self.refit_lr)
        self.refit_epochs = config.get("refit_epochs", self.refit_epochs)
        self.refit_batch_size = config.get("refit_batch_size", self.refit_batch_size)
        self.refit_num = config.get("refit_num", self.refit_num)
                
    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = X.astype(np.float64)
            X = torch.tensor(X)
        inputs = X.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.detach().cpu().numpy()

    def predict_proba(self, X):
        if not isinstance(X, torch.Tensor):
            X = X.astype(np.float64)
            X = torch.tensor(X)
        inputs = X.to(self.device)
        outputs = self.model(inputs)
        # print(outputs.shape)
        return outputs.detach().cpu().numpy()

    def score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = X.astype(np.float64)
            X = torch.tensor(X)
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        self.model = self.model.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        total = y.shape[0]
        acc = correct / total 
        f1 = self.f1score(X,y)
        return acc, f1

    def f1score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = X.astype(np.float64)
            X = torch.tensor(X)
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        self.model = self.model.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return f1_score(y.cpu().numpy(), predicted.cpu().numpy(), average='macro')

    def fit(self, X, y):
        # convert to tensor
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)
        # define optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # define dataset and dataloader
        trainset = TensorDataset(X, y)
        trainloader = torch.utils.data.DataLoader(trainset,batch_size=self.batch_size, shuffle=True, num_workers=1)
        # train model
        self.model.train()
        for epoch in tqdm(range(0,self.train_epochs+1)):
            running_loss = 0.0
            epoch_steps = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                epoch_steps += 1
            # if epoch % 25 == 0:
            #     print(f"Epoch {epoch} loss: {running_loss/epoch_steps}")
        return None
    
    def refit(self, X, y):
        # convert to tensor 
        if not isinstance(X, torch.Tensor):
            X = X.astype(np.float64)
            X = torch.tensor(X)
            y = torch.tensor(y)
        # define optimizer for all parameters
        # TODO: update to add other refit methods
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                     lr=self.refit_lr)
        # define dataset and dataloader
        trainset = TensorDataset(X, y)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.refit_batch_size, shuffle=True, num_workers=1)
        # train model
        self.model.train()
        for epoch in tqdm(range(0,self.refit_epochs+1)):
            running_loss = 0.0
            epoch_steps = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                epoch_steps += 1
            #if epoch % 25 == 0:
            #    print(f"Epoch {epoch} loss: {running_loss/epoch_steps}")
        return None

    def save(self, idx, dir='/shared/share_mala/llm-dro/save_models/'):
        self.model.eval()
        os.makedirs(f'{dir}/mlp/', exist_ok=True) 
        torch.save(self.model.cpu(), f"{dir}/mlp/{idx}.pth")
        
    def load(self, idx, dir='/shared/share_mala/llm-dro/save_models/'):
        self.model = torch.load(f"{dir}/mlp/{idx}.pth").to(self.device)


