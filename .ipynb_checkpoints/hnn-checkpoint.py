from baseline_nn import BLNN
import numpy as np
import torch

class HNN(BLNN):
    def __init__(self,  d_in, d_hidden, d_out, activation_fn):
        super(HNN, self).__init__(d_in, d_hidden, d_out, activation_fn)
        self.M = self.permutation_tensor(d_in)

    def forward(self, x):
        y = super().forward(x)
        
        return y

    def time_derivative(self, x):
        F = self.forward(x)
        dF = torch.autograd.grad(F.sum(), x, create_graph=True)[0]
        M = self.M.t()
        vector_field = dF @ M

        return vector_field

    def permutation_tensor(self, n):
        M = None
        M = torch.eye(n)
        M = torch.cat([M[n // 2:], -M[:n // 2]])

        return M

    def validate(self, args, data):
        '''
        Calculating losses on validation/testing dataset
        '''
        self.eval()
        loss_fn = torch.nn.MSELoss()
        if args.input_noise != '':
            # adding noise (optional)
            npnoise = np.array(args.input_noise, dtype="float")
            noise = torch.tensor(npnoise).to(device)

        for x, dxdt in data:
            dxdt_hat = self.time_derivative(x)
            # adding noise (optional)
            if args.input_noise != "":
                dxdt_hat += noise * torch.randn(*x.shape) 
            
            return loss_fn(dxdt_hat, dxdt)

