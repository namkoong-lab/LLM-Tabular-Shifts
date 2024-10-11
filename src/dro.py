import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.autograd import grad
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data import random_split
import torch
import itertools
from torch.nn import CrossEntropyLoss
import math
from scipy import optimize as sopt
from torch.nn import functional as F


GEOMETRIES = ('cvar', 'chi-square')
MIN_REL_DIFFERENCE = 1e-5


def chi_square_value(p, v, reg):
    """Returns <p, v> - reg * chi^2(p, uniform) for Torch tensors"""
    m = p.shape[0]

    with torch.no_grad():
        chi2 = (0.5 / m) * reg * (
                torch.norm(m * p - torch.ones(m, device=v.device), p=2) ** 2)

    return torch.dot(p, v) - chi2


def cvar_value(p, v, reg):
    """Returns <p, v> - reg * KL(p, uniform) for Torch tensors"""
    m = p.shape[0]

    with torch.no_grad():
        idx = torch.nonzero(p)  # where is annoyingly backwards incompatible
        kl = np.log(m) + (p[idx] * torch.log(p[idx])).sum()

    return torch.dot(p, v) - reg * kl


def fenchel_kl_cvar(v, alpha):
    """Returns the empirical mean of the Fenchel dual for KL CVaR"""
    v -= np.log(1 / alpha)
    v1 = v[torch.lt(v, 0)]
    v2 = v[torch.ge(v, 0)]
    w1 = torch.exp(v1) / alpha - 1
    w2 = (v2 + 1) * (1 / alpha) - 1
    return (w1.sum() + w2.sum()) / v.shape[0]


def bisection(eta_min, eta_max, f, tol=1e-2, max_iter=500):
    """Expects f an increasing function and return eta in [eta_min, eta_max] 
    s.t. |f(eta)| <= tol (or the best solution after max_iter iterations"""
    lower = f(eta_min)
    upper = f(eta_max)

    # until the root is between eta_min and eta_max, double the length of the 
    # interval starting at either endpoint.
    while lower > 0 or upper < 0:
        length = eta_max - eta_min
        if lower > 0:
            eta_max = eta_min
            eta_min = eta_min - 2 * length
        if upper < 0:
            eta_min = eta_max
            eta_max = eta_max + 2 * length

        lower = f(eta_min)
        upper = f(eta_max)

    for _ in range(max_iter):
        eta = 0.5 * (eta_min + eta_max)

        v = f(eta)

        if torch.abs(v) <= tol:
            return eta

        if v > 0:
            eta_max = eta
        elif v < 0:
            eta_min = eta

    return 0.5 * (eta_min + eta_max)


def huber_loss(x, delta=1.):
    """ Standard Huber loss of parameter delta

    https://en.wikipedia.org/wiki/Huber_loss

    returns 0.5 * x^2 if |a| <= \delta
            \delta * (|a| - 0.5 * \delta) o.w.
    """
    if torch.abs(x) <= delta:
        return 0.5 * (x ** 2)
    else:
        return delta * (torch.abs(x) - 0.5 * delta)


class RobustLoss(nn.Module):
    """PyTorch module for the batch robust loss estimator"""
    def __init__(self, size, reg, geometry, tol=1e-2,
                 max_iter=500, debugging=False, is_regression=False, device=torch.device("cuda:6")):
        """
        Parameters
        ----------

        size : float
            Size of the uncertainty set (\rho for \chi^2 and \alpha for CVaR)
            Set float('inf') for unconstrained
        reg : float
            Strength of the regularizer, entropy if geometry == 'cvar'
            $\chi^2$ divergence if geometry == 'chi-square'
        geometry : string
            Element of GEOMETRIES
        tol : float, optional
            Tolerance parameter for the bisection
        max_iter : int, optional
            Number of iterations after which to break the bisection

        """
        super().__init__()
        self.size = size
        self.reg = reg
        self.geometry = geometry
        self.tol = tol
        self.max_iter = max_iter
        self.debugging = debugging
        self.is_regression = is_regression
        self.is_erm = size == 0
        self.device = device 
        

        if geometry not in GEOMETRIES:
            raise ValueError('Geometry %s not supported' % geometry)

        if geometry == 'cvar' and self.size > 1:
            raise ValueError(f'alpha should be < 1 for cvar, is {self.size}')

    def best_response(self, v):
        size = self.size
        reg = self.reg
        m = v.shape[0]

        if self.geometry == 'cvar':
            if self.reg > 0:
                if size == 1.0:
                    return torch.ones_like(v) / m

                def p(eta):
                    x = (v - eta) / reg
                    return torch.min(torch.exp(x),
                                     torch.Tensor([1 / size]).type(x.dtype).to(self.device)) / m

                def bisection_target(eta):
                    return 1.0 - p(eta).sum()

                eta_min = reg * torch.logsumexp(v / reg - np.log(m), 0)
                eta_max = v.max()

                if torch.abs(bisection_target(eta_min)) <= self.tol:
                    return p(eta_min)
            else:
                cutoff = int(size * m)
                surplus = 1.0 - cutoff / (size * m)

                p = torch.zeros_like(v).to(self.device)
                idx = torch.argsort(v, descending=True)
                p[idx[:cutoff]] = 1.0 / (size * m)
                if cutoff < m:
                    p[idx[cutoff]] = surplus
                return p

        if self.geometry == 'chi-square':
            if (v.max() - v.min()) / v.max() <= MIN_REL_DIFFERENCE:
                return (torch.ones_like(v) / m).to(self.device)

            if size == float('inf'):
                assert reg > 0

                def p(eta):
                    return (torch.relu(v - eta) / (reg * m)).to(device)

                def bisection_target(eta):
                    return 1.0 - p(eta).sum()

                eta_min = min(v.sum() - reg * m, v.min())
                eta_max = v.max()

            else:
                assert size < float('inf')

                # failsafe for batch sizes small compared to
                # uncertainty set size
                if m <= 1 + 2 * size:
                    out = (v == v.max()).float()
                    out /= out.sum()
                    return out

                if reg == 0:
                    def p(eta):
                        pp = torch.relu(v - eta)
                        return pp / pp.sum()

                    def bisection_target(eta):
                        pp = p(eta)
                        w = m * pp - torch.ones_like(pp)
                        return 0.5 * torch.mean(w ** 2) - size

                    eta_min = -(1.0 / (np.sqrt(2 * size + 1) - 1)) * v.max()
                    eta_max = v.max()
                else:
                    def p(eta):
                        pp = torch.relu(v - eta)

                        opt_lam = max(
                            reg, torch.norm(pp) / np.sqrt(m * (1 + 2 * size))
                        )

                        return pp / (m * opt_lam)

                    def bisection_target(eta):
                        return 1 - p(eta).sum()

                    eta_min = v.min() - 1
                    eta_max = v.max()

        eta_star = bisection(
            eta_min, eta_max, bisection_target,
            tol=self.tol, max_iter=self.max_iter)

        if self.debugging:
            return p(eta_star), eta_star
        return p(eta_star)

    def forward(self, outputs, targets):
        """Value of the robust loss

        Note that the best response is computed without gradients

        Parameters
        ----------

        v : torch.Tensor
            Tensor containing the individual losses on the batch of examples

        Returns
        -------
        loss : torch.float
            Value of the robust loss on the batch of examples
        """

        if self.is_regression:
            v = (outputs.squeeze() - targets) ** 2
        else:
            losstemp = CrossEntropyLoss(reduction="none")
            v = losstemp(outputs, targets.long())

        if self.is_erm:
            return v.mean()
        else:
            with torch.no_grad():
                p = self.best_response(v)

            if self.geometry == 'cvar':
                return cvar_value(p, v, self.reg)
            elif self.geometry == 'chi-square':
                return chi_square_value(p, v, self.reg)

def chi_square_doro_criterion(outputs, targets, alpha, eps):
    batch_size = len(targets)
    criterion = CrossEntropyLoss(reduction="none")
    loss = criterion(outputs, targets, )
    # Chi^2-DORO
    max_l = max(10., int(0.6*loss.max().item()))
    C = math.sqrt(1 + (1 / alpha - 1) ** 2)
    n = int(eps * batch_size)
    rk = torch.argsort(loss, descending=True)
    l0 = loss[rk[n:]]
    foo = lambda eta: C * math.sqrt(
        (F.relu(l0.cpu() - eta) ** 2).mean().item()) + eta
    # print("begin opt search")
    # opt_eta = sopt.brent(foo, brack=(0, max_l),tol=1e-2)
    opt_eta = sopt.brent(foo, brack=(0, max_l))
    
    # print("end opt search")
    loss = C * torch.sqrt((F.relu(l0 - opt_eta) ** 2).mean()) + opt_eta
    return loss


# DORO; adapted from
# https://github.com/RuntianZ/doro/blob/master/wilds-exp/algorithms/doro.py
def cvar_doro_criterion(outputs, targets, eps, alpha):
    batch_size = len(targets)
    criterion = CrossEntropyLoss(reduction="none")
    loss = criterion(outputs, targets)
    gamma = eps + alpha * (1 - eps)
    n1 = int(gamma * batch_size)
    n2 = int(eps * batch_size)
    rk = torch.argsort(loss, descending=True)
    loss = loss[rk[n2:n1]].sum() / alpha / (batch_size - n2)
    return loss

def get_dro_criterion(size, reg, geometry, tol=1e-4,max_iter=1000, is_regression=False):
    return RobustLoss(
            geometry=geometry,
            size=size,
            reg=reg,
            max_iter=max_iter,
            is_regression=is_regression
        )
    

def group_dro_criterion(outputs, targets, sens):
    # TODO: adjust these setups to not binary group??
    assert torch.all(torch.logical_or(sens == 1., sens == 2.)), \
        "only binary groups supported."
    subgroup_losses = []
    n_attrs = sens.shape[1]
    criterion = CrossEntropyLoss(reduction="none")
    elementwise_loss = criterion(outputs, targets)
    # Compute the loss on each subgroup
    for subgroup_idxs in itertools.product(*[(1, 2)] * n_attrs):
        subgroup_idxs = torch.Tensor(subgroup_idxs).to(sens.device)
        mask = torch.all(sens == subgroup_idxs, dim=1)
        subgroup_loss = elementwise_loss[mask].sum() / mask.sum()
        subgroup_losses.append(subgroup_loss)
    return torch.stack(subgroup_losses)



class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, num_units=16, nonlin=nn.ReLU(), dropout_ratio=0.1):
        super().__init__()

        self.dense0 = nn.Linear(input_dim, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(dropout_ratio)
        self.dense1 = nn.Linear(num_units, num_units)
        self.output = nn.Linear(num_units, num_classes)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X.float()))
        X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = self.output(X)
        return X


class MLPClassifier():
    def __init__(self, input_dim=9, num_classes=2):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = 128
        self.lr = 1e-3
        self.device = torch.device("cuda")
        self.model = MLP(input_dim, num_classes).to(self.device)
        self.train_epochs = 200

    def update(self, config):
        self.lr = config["lr"]
        self.train_epochs = config["train_epochs"]
        self.device = config["device"]
        self.model = MLP(self.input_dim, self.num_classes, num_units=config["hidden_size"], dropout_ratio=config["dropout_ratio"]).to(self.device)
        
    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = X.astype(np.float)
            X = torch.tensor(X)
        inputs = X.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.detach().cpu().numpy()

    def predict_proba(self, X):
        if not isinstance(X, torch.Tensor):
            X = X.astype(np.float)
            X = torch.tensor(X)
        inputs = X.to(self.device)
        outputs = self.model(inputs)
        # print(outputs.shape)
        return outputs.detach().cpu().numpy()


    def score(self, X, y, weights=None):
        if not isinstance(X, torch.Tensor):
            X = X.astype(np.float)
            X = torch.tensor(X)
            y = torch.tensor(y)
            if not weights is None:
                weights = torch.tensor(weights.astype(np.float)).to(self.device)

        inputs, labels = X.to(self.device), y.to(self.device)
        self.model = self.model.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        if weights is None:
            correct = (predicted == labels).sum().item()
            total = y.shape[0]
            acc = correct / total 
            f1 = self.f1score(X,y)
            return acc, f1
        else:
            correct = torch.tensor(predicted == labels).float()
            acc = torch.dot(correct, weights.float()) / weights.sum()
            return acc
        
    def f1score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = X.astype(np.float)
            X = torch.tensor(X)
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        self.model = self.model.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return f1_score(y.cpu().numpy(), predicted.cpu().numpy(), average='macro')

    def fit(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = X.astype(np.float)
            X = torch.tensor(X)
            y = torch.tensor(y)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        trainset = TensorDataset(X, y)
        trainloader = torch.utils.data.DataLoader(trainset,batch_size=self.batch_size, shuffle=True, num_workers=1)
        
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

        return None

    def fit_weight(self, X, y, weights):
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        if not isinstance(X, torch.Tensor):
            X = X.astype(np.float)
            X = torch.tensor(X)
            y = torch.tensor(y)
            weights = torch.tensor(weights).float()
            
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        trainset = TensorDataset(X, y, weights)
        trainloader = torch.utils.data.DataLoader(trainset,batch_size=self.batch_size, shuffle=True, num_workers=1)
        for epoch in tqdm(range(0,self.train_epochs+1)):
            running_loss = 0.0
            epoch_steps = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels, weights_batch = data
                inputs, labels, weights_batch = inputs.to(self.device), labels.to(self.device), weights_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss = torch.dot(loss.reshape(-1), weights_batch.reshape(-1))/loss.shape[0]
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_steps += 1
                
        return None


class chi_DRO():
    def __init__(self, input_dim=9, num_classes=2, size=0.1, reg=0.1, max_iter=100):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = MLP(input_dim, num_classes)
        self.criterion = RobustLoss(
            geometry='chi-square',
            size=size,
            reg=reg,
            max_iter=max_iter,
            is_regression=False, device=torch.device("cuda:6"))
        
    def update(self, config):
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.train_epochs = config["train_epochs"]
        self.device = config["device"]
        self.model = MLP(self.input_dim, self.num_classes, num_units=config["hidden_size"], dropout_ratio=config["dropout_ratio"]).to(self.device)
        self.criterion = RobustLoss(
            geometry='chi-square',
            size=config["size"],
            reg=config["reg"],
            max_iter=500,
            is_regression=False, device=config["device"])

    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = X.astype(np.float)
            X = torch.tensor(X)
        inputs = X.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.detach().cpu().numpy()

    def score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = X.astype(np.float)
            X = torch.tensor(X)
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        total = y.shape[0]
        acc = correct / total 
        f1 = self.f1score(X,y)
        return acc, f1

    def f1score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = X.astype(np.float)
            X = torch.tensor(X)
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return f1_score(y.cpu().numpy(), predicted.cpu().numpy(), average='macro')

    def fit(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = X.astype(np.float)
            X = torch.tensor(X)
            y = torch.tensor(y)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        trainset = TensorDataset(X, y)
        trainloader = torch.utils.data.DataLoader(trainset,batch_size=self.batch_size, shuffle=True)
        for epoch in range(0,self.train_epochs+1):
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
    
        return None

class cvar_DRO():
    def __init__(self, input_dim=9, num_classes=2, size=0.1, reg=0.1, max_iter=100):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = MLP(input_dim, num_classes)
        self.criterion = RobustLoss(
            geometry='cvar',
            size=size,
            reg=reg,
            max_iter=max_iter,
            is_regression=False, device=torch.device("cuda:6"))

    def update(self, config):
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.train_epochs = config["train_epochs"]
        self.device = config["device"]
        self.model = MLP(self.input_dim, self.num_classes, num_units=config["hidden_size"], dropout_ratio=config["dropout_ratio"]).to(self.device)
        self.criterion = RobustLoss(
            geometry='cvar',
            size=config["size"],
            reg=config["reg"],
            max_iter=config["max_iter"],
            is_regression=False, device=config["device"])

    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = X.astype(np.float)
            X = torch.tensor(X)
        inputs = X.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.detach().cpu().numpy()

    def score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = X.astype(np.float)
            X = torch.tensor(X)
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        total = y.shape[0]
        acc = correct / total 
        f1 = self.f1score(X,y)
        return acc, f1

    def f1score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = X.astype(np.float)
            X = torch.tensor(X)
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return f1_score(y.cpu().numpy(), predicted.cpu().numpy(), average='macro')

    def fit(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = X.astype(np.float)
            X = torch.tensor(X)
            y = torch.tensor(y)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        trainset = TensorDataset(X, y)
        trainloader = torch.utils.data.DataLoader(trainset,batch_size=self.batch_size, shuffle=True, num_workers=1)
        
        for epoch in range(0,self.train_epochs+1):
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
                
        return None

class cvar_DORO():
    def __init__(self, input_dim=9, num_classes=2):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = MLP(input_dim, num_classes)

    def update(self, config):
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.train_epochs = config["train_epochs"]
        self.device = config["device"]
        self.model = MLP(self.input_dim, self.num_classes, num_units=config["hidden_size"], dropout_ratio=config["dropout_ratio"]).to(self.device)
        self.eps = config["eps"]
        self.alpha = config["alpha"]


    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = X.astype(np.float)
            X = torch.tensor(X)
        inputs = X.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.detach().cpu().numpy()

    def score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = X.astype(np.float)
            X = torch.tensor(X)
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        total = y.shape[0]
        acc = correct / total 
        f1 = self.f1score(X,y)
        return acc, f1

    def f1score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = X.astype(np.float)
            X = torch.tensor(X)
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return f1_score(y.cpu().numpy(), predicted.cpu().numpy(), average='macro')

    def fit(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = X.astype(np.float)
            X = torch.tensor(X)
            y = torch.tensor(y)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        trainset = TensorDataset(X, y)
        trainloader = torch.utils.data.DataLoader(trainset,batch_size=self.batch_size, shuffle=True, num_workers=1)
        
        for epoch in range(0,self.train_epochs+1):
            running_loss = 0.0
            epoch_steps = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = cvar_doro_criterion(outputs, labels, self.eps, self.alpha)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_steps += 1
                
        return None

class chi_DORO():
    def __init__(self, input_dim=9, num_classes=2, eps=0.1, alpha=0.1,lr=1e-3):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = MLP(input_dim, num_classes)
        self.eps = eps 
        self.alpha = alpha 
        self.lr = lr
        self.batch_size=128
        self.train_epochs=100

    def update(self, config):
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.train_epochs = config["train_epochs"]
        self.device = config["device"]
        self.model = MLP(self.input_dim, self.num_classes, num_units=config["hidden_size"], dropout_ratio=config["dropout_ratio"]).to(self.device)
        self.eps = config["eps"]
        self.alpha = config["alpha"]


    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = X.astype(np.float)
            X = torch.tensor(X)
        inputs = X.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.detach().cpu().numpy()

    def score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = X.astype(np.float)
            X = torch.tensor(X)
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        total = y.shape[0]
        acc = correct / total 
        f1 = self.f1score(X,y)
        return acc, f1

    def f1score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = X.astype(np.float)
            X = torch.tensor(X)
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return f1_score(y.cpu().numpy(), predicted.cpu().numpy(), average='macro')

    def fit(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = X.astype(np.float)
            X = torch.tensor(X)
            y = torch.tensor(y)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        trainset = TensorDataset(X, y)
        trainloader = torch.utils.data.DataLoader(trainset,batch_size=self.batch_size, shuffle=True, num_workers=1)
        for epoch in range(0,self.train_epochs+1):
            running_loss = 0.0
            epoch_steps = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = chi_square_doro_criterion(outputs, labels, self.eps, self.alpha)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_steps += 1
                
        return None

