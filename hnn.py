import torch
import numpy as np
from baseline_nn import BLNN


class HNN(torch.nn.Module):
    def __init__(self, d_in, baseline_model):
        super(HNN, self).__init__()
        self.baseline_model = baseline_model
        self.M = self.permutation_tensor(d_in)

    def forward(self, x):
        y = self.baseline_model(x)
        return y

    def time_derivative(self, x, t=None):
        F = self.forward(x)
        dF = torch.autograd.grad(F.sum(), x, create_graph=True)[0]
        vector_field = dF @ self.M.t()

        return vector_field

    def permutation_tensor(self, n):
        M = None
        M = torch.eye(n)
        M = torch.cat([M[n // 2:], -M[:n // 2]])

        return M
