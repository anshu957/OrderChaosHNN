import torch
import numpy as np
from hnn import HNN
from baseline_nn import BLNN
import argparse
from data import DynamicalSystem

import os
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--hamiltonian',
                        default="(q1**2+q2**2+p1**2+p2**2)/(2)+ (q1**2*q2 - (q2**3)/(3))",
                        type=str, help='Hamiltonian of the system')

    parser.add_argument('--input_dim', default=4,
                        type=int, help='Input dimension')
    parser.add_argument('--hidden_dim', nargs="*", default=[200, 200],
                        type=int, help='hidden layers dimension')
    parser.add_argument('--learn_rate', default=1e-03,
                        type=float, help='learning rate')
    parser.add_argument('--batch_size', default=512,
                        type=int, help='batch size'),
    parser.add_argument('--input_noise', default=0.0,
                        type=float, help='noise strength added to the inputs')
    parser.add_argument('--epochs', default=2,
                        type=int, help='No. of training epochs')
    parser.add_argument('--integrator_scheme', default='RK45',
                        type=str, help='name of the integration scheme [RK4, RK45, Symplectic]')
    parser.add_argument('--activation_fn', default='Tanh', type=str,
                        help='which activation function to use [Tanh, ReLU]')
    parser.add_argument('--name', default='Henon_Heiles',
                        type=str, help='Name of the system')
    parser.add_argument('--model', default='baseline',
                        type=str, help='baseline or hamiltonian')
    parser.add_argument('--dsr', default=0.1, type=float,
                        help='data sampling rate')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='Verbose output or not')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str,
                        help='where to save the trained model')
    parser.set_defaults(feature=True)

    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    args.save_dir = THIS_DIR + '/' + 'TrainedNetworks'
    args.name = args.name + '_DSR_' + args.dsr

    state_symbols = ['q1', 'q2', 'p1', 'p2']
    tspan = [0, 1000]
    tpoints = int((1/args.dsr)*(tspan[1]))

    sys = DynamicalSystem(sys_hamiltonian=args.hamiltonian, tspan=tspan,
                          timesteps=tpoints, integrator=args.integrator_scheme, state_symbols=state_symbols, symplectic_order=4)

    data = sys.get_dataset(args.name, THIS_DIR)

    print('Hidden dimensions (excluding first and last layer) : {}'.format(
        args.hidden_dim))

    print('Training data size : {}'.format(data['coords'].shape))
    print('Testing data size : {}'.format(data['test_coords'].shape))

    if args.model == 'baseline':
        print('Training baseline model ...')
        out_dim = args.input_dim
        model = BLNN(args.input_dim, args.hidden_dim,
                     out_dim, args.activation_fn)
        optim = torch.optim.Adam(
            model.parameters(), args.learn_rate, weight_decay=1e-4)
        stats = model.train(args, data, optim)

    else:
        print('Training hamiltonian neural network ...')
        out_dim = 1
        nn_model = BLNN(args.input_dim, args.hidden_dim,
                        out_dim, args.activation_fn)
        model = HNN(args.input_dim, baseline_model=nn_model)
        optim = torch.optim.Adam(
            model.parameters(), args.learn_rate, weight_decay=1e-4)
        stats = model.train(args, data, optim)

    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None

    if args.model == 'baseline':
        label = 'baseline'
    else:
        label = 'hnn'

    path = '{}/{}_nlayers_{}-orbits-{}_integrator_{}_epochs_{}_BatchSize_{}.tar'.format(
        args.save_dir, args.name, len(args.hidden_dim), label, args.integrator_scheme, args.epochs, args.batch_size)
    torch.save(model.state_dict(), path)
