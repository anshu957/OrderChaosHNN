import torch
import numpy as numpy


class BLNN(torch.nn.Module):
    ''' Classic neural network implementation which serves as a baseline NN model '''

    def __init__(self, d_in, d_hidden, d_out, activation_fn):
        super(BLNN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.nonlinearity = []

        if activation_fn == 'Tanh':
            print('Using Tanh()...')
            nonlinear_fn = torch.nn.Tanh()
        elif activation_fn == 'ReLU':
            print('Using ReLU()...')
            nonlinear_fn = torch.nn.ReLU()

        self.layers.append(torch.nn.Linear(d_in, d_hidden[0]))
        self.nonlinearity.append(nonlinear_fn)

        for i in range(len(d_hidden) - 1):
            self.layers.append(torch.nn.Linear(d_hidden[i], d_hidden[i + 1]))
            self.nonlinearity.append(nonlinear_fn)

        self.last_layer = torch.nn.Linear(d_hidden[-1], d_out, bias=None)

        print(self.layers)

        for i in range(len(self.layers)):
            torch.nn.init.orthogonal_(self.layers[i].weight)

        torch.nn.init.orthogonal_(self.last_layer.weight)

    def forward(self, x):
        dict_layers = dict(zip(self.layers, self.nonlinearity))
        for layer, nonlinear_transform in dict_layers.items():
            out = nonlinear_transform(layer(x))
            x = out
        return self.last_layer(out)

    def time_derivative(self, x, t=None):
        return self.forward(x)

    def train(self, args, data, optim):
        
        
        # ----Training inputs-------
        x = torch.tensor(
            data['coords'], requires_grad=True, dtype=torch.float32)
        # ----Training labels-------
        dxdt = torch.Tensor(data['dcoords'])
        
        # ----Testing inputs-------
        test_x = torch.tensor(
            data['test_coords'], requires_grad=True, dtype=torch.float32)

        # ----Testing inputs-------
        test_dxdt = torch.Tensor(data['test_dcoords'])
            
        # number of batches
        no_batches = int(x.shape[0]/args.batch_size)
        
        L2_loss = torch.nn.MSE()
        

        stats = {'train_loss': [], 'test_loss': []}
        for epoch in tqdm(range(args.epochs)):
            
            for batch in range(no_batches):
                
                optim.zero_grad()
                ixs = torch.randperm(x.shape[0])[:args.batch_size]
                dxdt_hat = self.time_derivative(x[ixs])
                dxdt_hat += args.input_noise * torch.randn(
                    *x[ixs].shape)  # add noise, maybe
                loss = L2_loss(dxdt[ixs], dxdt_hat)
                loss.backward()
                grad = torch.cat([p.grad.flatten()
                                  for p in self.parameters()]).clone()
                optim.step()
                

            # run test data
            test_ixs = torch.randperm(test_x.shape[0])[:args.batch_size]
            test_dxdt_hat = self.time_derivative(test_x[test_ixs])
            test_dxdt_hat += args.input_noise * torch.randn(
                *test_x[test_ixs].shape)  # add noise, maybe
            test_loss = L2_loss(test_dxdt[test_ixs], test_dxdt_hat)

            # logging
            stats['train_loss'].append(loss.item())
            stats['test_loss'].append(test_loss.item())
            if args.verbose and step % args.print_every == 0:
                print(
                    "step {}, train_loss {:.4e}, test_loss {:.4e}, grad norm {:.4e}, grad std {:.4e}"
                    .format(step, loss.item(), test_loss.item(), grad @ grad,
                            grad.std()))

        return stats
