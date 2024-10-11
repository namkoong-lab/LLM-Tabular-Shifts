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

class PCAEmbedding(nn.Module):
    def __init__(self, base_embedding, num_components):
        super(PCAEmbedding, self).__init__()
        # use register_buffer to ensures they are moved to GPU with the model
        self.register_buffer('base_embedding', base_embedding)
        self.register_buffer('embedding_mean', base_embedding.mean(dim=0, keepdim=True))
        data = self.base_embedding - self.embedding_mean
        # conduct PCA
        u, s, v = torch.pca_lowrank(data, q=num_components)
        self.register_buffer('principal_components', v) # Principal directions
        # derive initial coefficients
        self.coefficients = nn.Parameter(torch.mm(data, self.principal_components))
    def forward(self, indices):
        selected_coeffs = self.coefficients[indices]
        embeddings = torch.matmul(selected_coeffs, self.principal_components.t())
        embeddings = embeddings + self.embedding_mean
        return embeddings

class MLP_concat(nn.Module):
    def __init__(self, input_dim, num_classes, num_embeddings=51, embedding_dim=4096, hidden_dim=64, nonlin=nn.ReLU(), dropout_ratio=0.1, 
                 initial_embedding_method = 'wiki', 
                 training_method = 'pca',
                 refit_method = 'pca', 
                 num_components=5,
                 task_name = 'income'):
        super().__init__()
        self.initial_embedding_method = initial_embedding_method
        self.training_method = training_method
        self.refit_method = refit_method
        self.num_components = num_components
        self.task_name = task_name
        # load intial embedding of domain info
        if initial_embedding_method == 'wiki':
            initial_embedding_path = f"/shared/share_mala/llm-dro/domain_info/wiki/{task_name}.npy"
        elif initial_embedding_method == 'random':      # random initialization
            initial_embedding_path = f"/shared/share_mala/llm-dro/domain_info/random/{task_name}.npy"
        elif initial_embedding_method == 'gpt4':        # ask gpt4 to generate domain info
            initial_embedding_path = f"/shared/share_mala/llm-dro/domain_info/gpt4/{task_name}.npy"
        elif initial_embedding_method == 'incontext':   # use 35 labeled data as domain info
            initial_embedding_path = f"/shared/share_mala/llm-dro/domain_info/incontext/{task_name}.npy"
        elif initial_embedding_method == 'incontext8':   # use 8 labeled data as domain info
            initial_embedding_path = f"/shared/share_mala/llm-dro/domain_info/incontext8/{task_name}.npy"
        elif initial_embedding_method == 'incontext16':   # use 16 labeled data as domain info
            initial_embedding_path = f"/shared/share_mala/llm-dro/domain_info/incontext16/{task_name}.npy"
        elif initial_embedding_method == 'incontext32':   # use 32 labeled data as domain info
            initial_embedding_path = f"/shared/share_mala/llm-dro/domain_info/incontext32/{task_name}.npy"
        else: 
            raise NotImplementedError
        initial_embedding = torch.from_numpy(np.load(initial_embedding_path)).float()  
        
        # add embedding layer
        if training_method == 'pca':
            self.embedding = PCAEmbedding(initial_embedding, num_components)    # intial embedding loaded, see def of PCAEmbedding
        elif training_method == 'nn':
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
            self.embedding.weight.data.copy_(initial_embedding)                 # intial embedding loaded
        elif training_method == 'freeze_embedding':
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
            self.embedding.weight.data.copy_(initial_embedding)                 # intial embedding loaded
            # Freeze embedding parameters
            for param in self.embedding.parameters():
                param.requires_grad = False
        else:
            raise NotImplementedError   
            
        # adjust input_dim 
        total_input_dim = input_dim + embedding_dim             # Now both are 128
        # Input processing layers to reduce dimension to 128
        self.input_process1 = nn.Linear(total_input_dim, 2048)  # Reduce from 4096*2 to 2048
        self.input_process2 = nn.Linear(2048, 512)        # Reduce from 2048 to 512
        self.input_process3 = nn.Linear(512, 128)         # Reduce from 512 to 128
        # define layers
        self.dense0 = nn.Linear(128, hidden_dim)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(dropout_ratio)
        self.dense1 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, X, domain_idx, **kwargs):
        # embed domain idx
        embedded = self.embedding(domain_idx).view(domain_idx.size(0), -1)
        # concatenate with X
        X = torch.cat([embedded, X.float()], dim=1)
        # Process X through dimensionality reduction
        X = self.nonlin(self.input_process1(X))
        X = self.nonlin(self.input_process2(X))
        X = self.nonlin(self.input_process3(X))
        # pass through network
        X = self.nonlin(self.dense0(X.float()))
        X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = self.output(X)
        return X

class MLPconcatClassifier():
    def __init__(self, input_dim=4096, num_classes=2, num_embeddings=51, embedding_dim=4096, 
                 hidden_dim=64, dropout_ratio=0.1, 
                 refit_num = 200, refit_lr = 0.1, refit_epochs = 200, refit_batch_size = 128,
                 initial_embedding_method = 'wiki', 
                 training_method = 'pca',
                 refit_method = 'pca', 
                 num_components = 5, # number of PCA components
                 task_name = 'income'):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_ratio = dropout_ratio
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = 128
        self.lr = 1e-3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLP_concat(input_dim, num_classes, num_embeddings, embedding_dim, hidden_dim, nn.ReLU(), dropout_ratio, 
                                initial_embedding_method=initial_embedding_method, 
                                training_method=training_method,
                                refit_method=refit_method, 
                                num_components=num_components, 
                                task_name=task_name).to(self.device)
        self.train_epochs = 200

        self.initial_embedding_method = initial_embedding_method
        self.training_method = training_method
        self.refit_method = refit_method
        self.num_components = num_components
        self.task_name = task_name

        self.refit_num = refit_num
        self.refit_lr = refit_lr
        self.refit_epochs = refit_epochs
        self.refit_batch_size = refit_batch_size

    def update(self, config):
        # model config
        self.hidden_dim = config.get("hidden_size", self.hidden_dim)
        self.num_embeddings = config.get("num_embeddings", self.num_embeddings)
        self.embedding_dim = config.get("embedding_dim", self.embedding_dim)
        # training config
        self.num_components = config.get("num_components", self.num_components)

        self.batch_size = config.get("batch_size", self.batch_size)
        self.lr = config.get("lr", self.lr)
        self.train_epochs = config.get("train_epochs", self.train_epochs)
        self.dropout_ratio = config.get("dropout_ratio", self.dropout_ratio)
        # update model
        self.model = MLP_concat(self.input_dim, self.num_classes, self.num_embeddings, self.embedding_dim, hidden_dim=self.hidden_dim, dropout_ratio=self.dropout_ratio, 
                                initial_embedding_method=self.initial_embedding_method,
                                training_method=self.training_method,
                                refit_method=self.refit_method,
                                num_components = self.num_components,
                                task_name=self.task_name).to(self.device)
        
    def update_refit_config(self, config):
        # update refit config
        self.refit_lr = config.get("refit_lr", self.refit_lr)
        self.refit_epochs = config.get("refit_epochs", self.refit_epochs)
        self.refit_batch_size = config.get("refit_batch_size", self.refit_batch_size)
        self.refit_num = config.get("refit_num", self.refit_num)
    
    def predict(self, X):
        # split into X and domain_idx
        X, domain_idx = X[:,:-1], X[:,-1]
        # convert to tensor
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float)
        if not isinstance(domain_idx, torch.Tensor):
            domain_idx = torch.tensor(domain_idx, dtype=torch.long)
        else:
            domain_idx = domain_idx.long()  # to feed into embedding layer
        # move to device
        X = X.to(self.device)
        domain_idx = domain_idx.to(self.device)
        # predict
        outputs = self.model(X, domain_idx)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.detach().cpu().numpy()

    def predict_proba(self, X):
        # split into X and domain_idx
        X, domain_idx = X[:,:-1], X[:,-1]
        # convert to tensor
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float)
        if not isinstance(domain_idx, torch.Tensor):
            domain_idx = torch.tensor(domain_idx, dtype=torch.long)
        else:
            domain_idx = domain_idx.long()  # to feed into embedding layer
        # move to device
        X = X.to(self.device)
        domain_idx = domain_idx.to(self.device)
        # predict
        outputs = self.model(X, domain_idx)
        # print(outputs.shape)
        return outputs.detach().cpu().numpy()

    def score(self, X, y):
        # split into X and domain_idx
        X, domain_idx = X[:,:-1], X[:,-1]
        # Convert to tensor
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float)
        if not isinstance(domain_idx, torch.Tensor):
            domain_idx = torch.tensor(domain_idx, dtype=torch.long)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)
        # move to device
        X = X.to(self.device)
        domain_idx = domain_idx.to(self.device)
        labels = y.to(self.device)
        # predict
        self.model = self.model.to(self.device)
        outputs = self.model(X, domain_idx)
        _, predicted = torch.max(outputs.data, 1)
        # accuracy and f1 score
        correct = (predicted == labels).sum().item()
        total = y.shape[0]
        acc = correct / total 
        f1 = f1_score(labels.detach().cpu().numpy(), predicted.detach().cpu().numpy(), average='macro')
        return acc, f1

    def fit(self, X, y):
        # split into X and domain_idx
        X, domain_idx = X[:,:-1], X[:,-1]
        # convert to tensor
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float)
        if not isinstance(domain_idx, torch.Tensor):
            domain_idx = torch.tensor(domain_idx, dtype=torch.long)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)
        
        dataset = TensorDataset(X, domain_idx, y)
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)
        # exclude frozen parameters 
        # (see definition for different training method)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)

        # model training
        self.model.train()
        for epoch in tqdm(range(0,self.train_epochs+1)):
            running_loss = 0.0
            epoch_steps = 0
            for i, data in enumerate(trainloader, 0):
                X, domain_idx, labels = data
                X, domain_idx, labels = X.to(self.device), domain_idx.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X, domain_idx)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                epoch_steps += 1
        #    if epoch % 25 == 0:
        #        print(f"Epoch {epoch} loss: {running_loss/epoch_steps}")
        return None

    def refit(self, X, y):
        # split into X and domain_idx
        X, domain_idx = X[:,:-1], X[:,-1]
        # Convert to tensor
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float)
        if not isinstance(domain_idx, torch.Tensor):
            domain_idx = torch.tensor(domain_idx, dtype=torch.long)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)

        dataset = TensorDataset(X, domain_idx, y)
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=self.refit_batch_size, shuffle=True, num_workers=1)
        # train different params for different refitting methods
        if self.refit_method == 'pca':                # only train pca coefficients
            optimizer = torch.optim.Adam([self.model.embedding.coefficients], lr=self.refit_lr)
        elif self.refit_method == 'nn':                 # only train nn.embedding
            optimizer = torch.optim.Adam(self.model.embedding.parameters(), lr=self.refit_lr)
        elif self.refit_method == 'freeze_embedding':   # only train NN (excluding nn.embedding)
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.refit_lr)
        elif self.refit_method == 'lora':               # only train lora layer
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.refit_lr)
        else:
            raise NotImplementedError

        # model training
        self.model.train()
        for epoch in tqdm(range(self.refit_epochs)):
            epoch_steps = 0
            running_loss = 0.0
            for data in trainloader:
                X, domain_idx, labels = data
                X, domain_idx, labels = X.to(self.device), domain_idx.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X, domain_idx)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                epoch_steps += 1
        #    if epoch % 25 == 0:
        #        print(f"Epoch {epoch} loss: {running_loss/epoch_steps}")
        return None

    def save(self, idx, dir='/shared/share_mala/llm-dro/save_models/'):
        self.model.eval()
        os.makedirs(f'{dir}/mlp/', exist_ok=True) 
        torch.save(self.model.cpu(), f"{dir}/mlp/{idx}.pth")
        
    def load(self, idx, dir='/shared/share_mala/llm-dro/save_models/'):
        self.model = torch.load(f"{dir}/mlp/{idx}.pth").to(self.device)