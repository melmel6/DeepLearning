import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module

class DenseNormalGamma(Module):
    def __init__(self, n_input, n_out_tasks=1):
        super(DenseNormalGamma, self).__init__()
        self.units = n_input
        self.n_out = 4 * n_out_tasks
        self.n_tasks = n_out_tasks
        self.dense = nn.Linear(self.units, self.n_out)

    def forward(self, x):
        x = self.dense(x)
        if len(x.shape) == 1:
            gamma, lognu, logalpha, logbeta = torch.split(x, self.n_tasks, dim=0)
        else:
            gamma, lognu, logalpha, logbeta = torch.split(x, self.n_tasks, dim=1)

        nu = F.softplus(lognu)
        alpha = F.softplus(logalpha) + 1.
        beta = F.softplus(logbeta)

        return torch.cat([gamma, nu, alpha, beta], axis=-1).to(x.device)

    



