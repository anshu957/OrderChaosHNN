'''
File: data.py
File Created: Monday, 22 July 2019 13:36:30
Author: well-well-well
------------------------
Description: Calculates the trajectories of any dynamical
system for various initial conditions and combine all those
to make a data set which is split into (train, test) data
sets with some predefined ratio.
--------------------------
'''

# Inspired by Hamiltonian Neural Networks | 2019 ( Sam Greydanus, Misko Dzamba, Jason Yosinski)

import numpy as np
import math

from scipy import integrate

from IPython.core.debugger import set_trace
import sympy as sp
from sympy import sympify

from utils import to_pickle, from_pickle


class DynamicalSystem():
    '''
    Integrates a system (passed as in input string for the constructor) using user defined method (scipy's IVP, rk4[fixed step size], Custom (Python) symplectic integrators) for multiple random initial conditions.
    Returns (train, test) data.
    '''

    def __init__(self, sys_hamiltonian, split_ratio=0.3, timesteps=1000,
                 state_symbols=['q1', 'q2', 'p1', 'p2'],
                 no_ensembles=10, tspan=[0, 1000], verbose=False, integrator="RK45", symplectic_order=4):
        super(DynamicalSystem, self).__init__()
        self.split = split_ratio
        self.time_points = timesteps
        self.ensembles = no_ensembles
        self.tspan = tspan
        self.external_update_fn = None
        self.update_fn = self.update
        self.verbose = verbose
        self.integrator = integrator
        self.order = symplectic_order
        self.sys_fn = sympify(sys_hamiltonian, evaluate=False)
        self.state_symbols = state_symbols
        self.energyPoints = 50

        self.sys_dim = len(state_symbols)
        self.sys_eqns = []
        sympy_symbols = sp.symbols([tuple(state_symbols)])
        for i in range(self.sys_dim):
            fn = self.sys_fn.diff(self.state_symbols[i])
            self.sys_eqns.append(sp.lambdify(sympy_symbols, fn))

        self.sys_hamiltonian = sp.lambdify(sympy_symbols, self.sys_fn, 'numpy')

        # ----------- Getting symbolic expression of p2 = f(H,q1,q2,p1) and then lambdifying it --------------
        self.sym_energy = sp.Symbol('energy')
        # get the expression of p2 as f(H,q1,q2,p1)
        self.expr_p2 = sp.solve(
            self.sys_fn - self.sym_energy, sympy_symbols[0][-1])[1]
        # lambdify p2 for fast numpy evaluations

        self.expr_2_lam = sp.lambdify(
            (self.sym_energy,) + sympy_symbols[0][:-1], self.expr_p2, 'numpy')
        # ---------------------------------------------------------------------------------------------------

    def get_energy(self, state):
        x = self.sys_hamiltonian(state)
        return np.array(x, dtype=np.float)

    def update(self, t, state):
        '''
        Calculates the derivative of state --> equations of motion
        '''
        deriv_sympy = np.zeros_like(state)
        for i in range(self.sys_dim):
            deriv_sympy[i] = self.sys_eqns[i](state)

        deriv = np.array(deriv_sympy, dtype=np.float)
        # print(deriv.dtype)

        m1 = np.array([[0.0, 1.0], [-1.0, 0.0]])
        m2 = np.eye(int(self.sys_dim / 2))
        m3 = np.kron(m1, m2)
        m4 = np.matmul(m3, deriv)
        # m4 looks like -- [dH/dp1, dH/dp2, -dH/dq1, -dH/dq2]
        return m4.reshape(-1)

    def integrate_one_step(self, func, y0, t, dt, order=2):
        '''
        Advances one step of the symplectic integrator
        '''
        # set_trace()
        n_dim = len(y0)
        u, v = np.array(y0[:int(n_dim / 2)]), np.array(y0[int(n_dim / 2):])

        if order == 2:
            # uses verlat-leap frog method

            v_half = v + 0.5 * dt * func(t, v, u)['dv']
            u_next = u + dt * v_half
            v_next = v_half + 0.5 * dt * func(t, v_half, u_next)['dv']

        elif order == 4:
            # uses 4th order Forest-Ruth Algorithm
            theta = 1.35120719195966

            u_1 = u + theta * (0.5 * dt) * func(t, v, u)['du']
            v_1 = v + theta * dt * func(t, v, u_1)['dv']

            u_2 = u_1 + (1.0 - theta) * (0.5 * dt) * func(t, v_1, u_1)['du']
            v_2 = v_1 + (1.0 - 2 * theta) * dt * func(t, v_1, u_2)['dv']

            u_3 = u_2 + (1.0 - theta) * (0.5 * dt) * func(t, v_2, u_2)['du']
            v_next = v_2 + theta * dt * func(t, v_2, u_3)['dv']

            u_next = u_3 + theta * (0.5 * dt) * func(t, v_next, u_3)['du']

        return np.concatenate((u_next, v_next), axis=None)

    def integrate_symplectic(self, func, dt, t_span, y0, order=2):
        '''
        Symplectically integrate an IVP for time given by t_span
        '''
        print('order = ', order)
        tpoints = int((t_span[1] - t_span[0]) / dt)
        t_eval = np.linspace(t_span[0], t_span[1], tpoints)
        t_observed = []
        sol = []
        for i, t in enumerate(t_eval):
            if i % int(1 / dt) == 0:
                t_observed.append(t)
                sol.append(y0)
            y0 = self.integrate_one_step(func, y0, t, dt, order)

        return np.array(sol, dtype=float), t_observed

    def update_fn_symplectic(self, t, v, u):
        '''
        Helper function used by symplectic integrator
        '''

        if self.external_update_fn is None:
            dudv = self.update_fn(t, state=np.concatenate((u, v), axis=None))

        else:
            dudv = self.external_update_fn(
                t, y0=np.concatenate((u, v), axis=None))

        dv = dudv[int(self.sys_dim / 2):self.sys_dim]  # [-dH/dq1, -dH/dq2]
        du = dudv[0:int(self.sys_dim / 2)]  # [dH/dp1, dH/dp2]

        deriv = {'du': du, 'dv': dv}

        return deriv

    def rk4_integrate(self, fun, y0, t_span, dt, t_eval=1):
        '''
        Fixed step size RK4 integrator
        '''
        t1 = t_span[0]
        t2 = t_span[1]
        t_integrate = np.arange(t1, t2, dt, dtype=np.float)

        y = []
        t_observe = []
        j = 0
        for i, t in enumerate(t_integrate):
            if i % int(t_eval / dt) == 0:
                t_observe.append(t)
                y.append(y0)
                j = j + 1

            dt2 = dt / 2.0
            k1 = fun(t, y0)
            k2 = fun(t + dt2, y0 + dt2 * k1)
            k3 = fun(t + dt2, y0 + dt2 * k2)
            k4 = fun(t + dt, y0 + dt * k3)
            y0 = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return np.array(y, dtype=float), t_observe

    def get_orbit(self, state):
        '''
        Integrate the orbit based on intergrator asked and return it
        '''
        orbit_settings = {}

        if self.integrator == "RK45":

            if self.external_update_fn is not None:
                print('Using external  function...')
                self.update_fn = self.external_update_fn
            else:
                self.update_fn = self.update

        # print(self.update_fn)
        t_eval = np.linspace(
            self.tspan[0], self.tspan[1], self.time_points)

        path = integrate.solve_ivp(fun=self.update_fn,
                                   t_span=self.tspan,
                                   y0=state.flatten(),
                                   t_eval=t_eval, rtol=1e-12)

        orbit = path['y'].reshape(self.sys_dim, self.time_points)
        orbit_settings['t_eval'] = t_eval

        # fixed step size RK4
        if self.integrator == "RK4":
            path, t = self.rk4_integrate(
                self.update_fn, y0=state.flatten(), t_span=self.tspan, dt=0.01)

            orbit = path.T
            orbit_settings['t_eval'] = t

        if self.integrator == "Symplectic":

            path, t = self.integrate_symplectic(
                func=self.update_fn_symplectic, dt=0.01, t_span=self.tspan, y0=state.flatten(), order=self.order)

            orbit = path.T
            orbit_settings['t_eval'] = t

        return orbit, orbit_settings

    def random_config(self, energy):
        '''
            Train the network on different orbits having different energies
        '''
    #    result = False
    #    while not result:
    #        with np.errstate(invalid='raise'):
    #            try:
    #                q1, q2 = 0.3 * (1 - 2 * np.random.random(2))
    #                p1 = 0.2 * (1 - 2 * np.random.random())
    #                p2 = np.sqrt(2 * (energy - (q1**2 * q2 - q2**3 / 3.0))
    #                             - (q1**2 + q2**2 + p1**2))
    #                result = True
    #            except FloatingPointError:
    #                continue

        result = False
        while not result:
            with np.errstate(invalid='raise'):
                try:
                    q1, q2 = 0.3 * (1 - 2 * np.random.random(2))
                    p1 = 0.2 * (1 - 2 * np.random.random())

                    p2 = self.expr_2_lam(energy, q1, q2, p1)

                    result = True
                except FloatingPointError:
                    continue

        state = np.array([q1, q2, p1, p2])

        #cal_energy = self.get_energy(state)
        # print("sampled energy = {:.4e}, calculated energy = {:.4e}".format(
        #    energy, cal_energy))

        return state

    def sample_orbits(self):
        orbit_settings = {}
        if self.verbose:
            print("Making a data-set for Henon-Heiles system ...")

        # Energy range for training [0.02, 0.15]
        energies = np.linspace(0.02, 0.15, self.energyPoints)

        orbit_settings['energy_range'] = energies
        orbit_settings['ensembles'] = self.ensembles
        x, dx, e = [], [], []

        N = self.time_points * self.ensembles
        from tqdm import tqdm
        for energy in tqdm(energies):
            count = 0
            while count < N:
                state = self.random_config(energy)
                orbit, settings = self.get_orbit(state)
                batch = orbit.transpose(1, 0)

                for state in batch:
                    count += 1
                    dstate = self.update(None, state)

                    coords = state.T.flatten()
                    dcoords = dstate.T.flatten()
                    x.append(coords)
                    dx.append(dcoords)

                    shaped_state = state.copy()
                    e.append(self.get_energy(shaped_state))

        data = {
            'coords': np.stack(x)[:N*len(energies)],
            'dcoords': np.stack(dx)[:N*len(energies)],
            'energy': np.stack(e)[:N*len(energies)]
        }

        return data, settings

    def make_orbits_dataset(self):
        data, orbit_settings = self.sample_orbits()

        # spliting the data --> train + test
        split_idx = int(data['coords'].shape[0] * self.split)
        split_data = {}
        for k, v in data.items():
            split_data[k], split_data['test_' +
                                      k] = v[split_idx:], v[:split_idx]

        data = split_data
        data['meta'] = orbit_settings

        return data

    def get_dataset(self, experiment_name, save_dir):
        '''Returns the trajectory dataset. Also constructs
           the dataset if no saved version is available.'''

        path = '{}/{}-orbits-dataset_{}_EnsemblesPerEnergy_{}_OrbitLen_{}_Resolution_{}_energyPoints{}.pkl'.format(
            save_dir, experiment_name, self.integrator, self.ensembles, self.tspan[1], self.time_points, self.energyPoints)
        #path = "../Henon-Heiles-orbits-dataset_RK45_EnsemblesPerEnergy_20_OrbitLen_5000_Resolution_50000_energyPoints20.pkl"
        #path = "../Henon-Heiles-orbits-dataset_RK45_EnsemblesPerEnergy_20_OrbitLen_1000_Resolution_10000_energyPoints20.pkl"
        try:
            data = from_pickle(path)
            print("Successfully loaded data from {}".format(path))
        except:
            print(
                "Had a problem loading data from {}. Rebuilding dataset...".format(
                    path))
            data = self.make_orbits_dataset()
            to_pickle(data, path)

        return data
