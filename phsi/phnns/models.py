import torch
import torch.nn as nn
from scipy.special import binom
import numpy as np

from .dynamic_system_neural_network import DynamicSystemNN


class BaseNN(torch.nn.Module):
    def __init__(self, ninputs, noutputs, hidden_dim,
                 timedependent, statedependent):
        super().__init__()
        self.ninputs = ninputs
        self.noutputs = noutputs
        self.hidden_dim = hidden_dim
        self.timedependent = timedependent
        self.statedependent = statedependent
        input_dim = int(statedependent)*ninputs + int(timedependent)
        linear1 = nn.Linear(input_dim, hidden_dim)
        linear2 = nn.Linear(hidden_dim, hidden_dim)
        linear3 = nn.Linear(hidden_dim, noutputs)

        for lin in [linear1, linear2, linear3]:
            nn.init.orthogonal_(lin.weight)

        self.model = nn.Sequential(
            linear1,
            nn.Tanh(),
            linear2,
            nn.ReLU(),
            linear3,
        )

        if not statedependent:
            self.forward = self._forward_without_state
        elif not timedependent:
            self.forward = self._forward_without_time
        else:
            self.forward = self._forward_with_state_and_time

    def _forward_with_state_and_time(self, *args):
        return self.model(torch.cat(args, dim=-1))

    def _forward_without_time(self, *args):
        return self.model(args[0])

    def _forward_without_state(self, *args):
        return self.model(args[1])


class BaselineNN(BaseNN):
    def __init__(self, nstates, hidden_dim, timedependent,
                 statedependent):
        super().__init__(nstates, nstates, hidden_dim,
                         timedependent, statedependent)


class HamiltonianNN(BaseNN):
    def __init__(self, nstates, hidden_dim):
        super().__init__(nstates, 1, hidden_dim, False, True)


class ExternalForcesNN(BaseNN):
    def __init__(self, nstates, noutputs, hidden_dim, timedependent,
                 statedependent, external_forces_filter=None, ttype=torch.float32):
        super().__init__(nstates, noutputs, hidden_dim,
                         timedependent, statedependent)
        self.nstates = nstates
        self.noutputs = noutputs
        self.ttype = ttype

        self.external_forces_filter = self.format_external_forces_filter(external_forces_filter)

    def _forward_with_state_and_time(self, *args):
        return self.model(torch.cat(args, dim=-1))@self.external_forces_filter

    def _forward_without_time(self, *args):
        return self.model(args[0])@self.external_forces_filter

    def _forward_without_state(self, *args):
        return self.model(args[1])@self.external_forces_filter

    def format_external_forces_filter(self, external_forces_filter):
        if external_forces_filter is None:
            assert self.noutputs == self.nstates, (
                f'noutputs ({self.noutputs}) != nstates ({self.nstates}) is not allowed '
                + 'when external_forces_filter is not provided.')
            return torch.eye(self.noutputs, dtype=self.ttype)

        if not isinstance(external_forces_filter, torch.Tensor):
            external_forces_filter = torch.tensor(external_forces_filter > 0)
        external_forces_filter = external_forces_filter.int()

        if (len(external_forces_filter.shape) == 1) or (external_forces_filter.shape[-1] == 1):
            external_forces_filter = external_forces_filter.flatten()
            assert external_forces_filter.shape[-1] == self.nstates, (
                'external_forces_filter is a vector of '
                + f'length {external_forces_filter.shape[-1]} != nstates, but '
                + f'({self.nstates}). external_forces_filter must be a '
                + 'vector of length nstates or a matrix of shape (nstates x noutputs).')
            expanded = torch.zeros((self.nstates, external_forces_filter.sum()), dtype=self.ttype)
            c = 0
            for i, e in enumerate(external_forces_filter):
                if e > 0:
                    expanded[i, c] = 1
                    c += 1
            return expanded.T

        assert external_forces_filter.shape == (self.nstates, self.noutputs), (
            f'external_forces_filter.shape == {external_forces_filter.shape}, but '
            + 'external_forces_filter must be a vector of length nstates or '
            + 'a matrix of shape (naffected_states x noutputs).')
        return torch.tensor(external_forces_filter, dtype=self.ttype).T


class ExternalForcesSI(torch.nn.Module):
    def __init__(self, nstates, timedependent, statedependent, 
                 function_space, external_forces_filter=torch.tensor([[0. ,1.]]), degrees = 2):
        super().__init__()
        self.timedependent = timedependent
        self.statedependent = statedependent
        self.ninputs = int(statedependent)*nstates + int(timedependent)
        pol_terms = int(binom(self.ninputs + degrees, degrees)-1)
        trig_terms = 2*self.ninputs
        self.n_terms = function_space @ np.array([pol_terms, trig_terms])
        init_values = torch.ones(self.n_terms, requires_grad=True)*1
        self.layer = nn.Parameter(init_values)
        self.external_forces_filter = external_forces_filter

        self.pol_space = bool(function_space[0])
        self.trig_space = bool(function_space[1])
        if self.trig_space:
            init_values_omega = torch.ones(2*self.ninputs, requires_grad=True)*1.
            self.omegas = nn.Parameter(init_values_omega)
        self.deg = degrees

        if not statedependent:
            self.forward = self._forward_without_state
        elif not timedependent:
            self.forward = self._forward_without_time
        else:
            self.forward = self._forward_with_state_and_time

    def _forward_with_state_and_time(self, *args):
        features = self.features(torch.cat(args, dim=-1))
        return torch.sum((self.layer*features), dim=1).reshape(-1,1) @ self.external_forces_filter
    
    def _forward_without_time(self, *args):
        features = self.features(args[0])
        return torch.sum((self.layer*features), dim=1).reshape(-1,1) @ self.external_forces_filter
    
    def _forward_without_state(self, *args):
        features = self.features(args[1])
        return torch.sum((self.layer*features), dim=1).reshape(-1,1) @ self.external_forces_filter

    def features(self, x):
        x = x.clone().requires_grad_(True)
        n_data = x.shape[0]
        features = torch.zeros(n_data, 1)

        if self.pol_space:
            x_cat = torch.cat([ torch.transpose(x[:,0].unsqueeze(0),0,1) ]*(self.deg+1), dim=1).unsqueeze(dim=0)
            x_pow = torch.pow(x_cat, torch.arange(self.deg+1))
            for i in range(1,self.ninputs):
                i_cat = torch.cat([ torch.transpose(x[:,i].unsqueeze(0),0,1) ]*(self.deg+1), dim=1).unsqueeze(dim=0)
                i_pow = torch.pow(i_cat, torch.arange(self.deg+1))
                x_pow = torch.cat((x_pow, i_pow), dim=0)
            
            idx_list = []
            idx_init = [0]*self.ninputs

            def loop_rec(idx_array, n):
                nonlocal features, idx_list, x_pow
                for i in range(self.ninputs):
                    idx_copy = idx_array.copy()
                    idx_copy[i] += 1
                    if idx_copy not in idx_list:
                        idx_list.append(idx_copy)
                        idx_onehot = torch.zeros((self.ninputs, self.deg+1))
                        idx_onehot[torch.arange(self.ninputs), idx_copy] = 1
                        idx_bool = idx_onehot > 0
                        select_cols = x_pow.transpose(1,2)[idx_bool].transpose(0,1)
                        product = torch.prod(select_cols, dim=1).unsqueeze(1)
                        features = torch.cat((features, product), dim=1)
                    if n>0:
                        loop_rec(idx_copy, n-1)
            loop_rec(idx_init, self.deg - 1)

        if self.trig_space:
            cosx = torch.cos(self.omegas[0]*x); sinx = torch.sin(self.omegas[1]*x)
            features = torch.cat((features, cosx, sinx), dim=1)
        features = features[:,1:]
        return features




class R_NN(BaseNN):
    '''
    Three layer feed forward neural network estimating the
    parameters of the damping matrix. 

    parameters
    ----------
    nstates    : Number of states.
    hidden_dim : Size of hidden layers.
    diagonal   : If True, only damping coefficients on the diagonal
                  are estimated. If False, all nstates**2 entries in the
                  R matrix are estimated.
    '''
    def __init__(self, nstates, hidden_dim, diagonal=False):
        if diagonal:
            noutputs = nstates
            self.forward = self._forward_diag
        else:
            noutputs = nstates**2
            self.forward = self._forward
        super().__init__(nstates, noutputs, hidden_dim, False, True)
        
        self.nstates = nstates

    def _forward_diag(self, x):
        return torch.diag_embed(self.model(x)**2).reshape(x.shape[0], self.nstates, self.nstates)

    def _forward(self, x):
        return (self.model(x)**2).reshape(x.shape[0], self.nstates, self.nstates)


class R_estimator(torch.nn.Module):
    '''
    Creates an estimator of a diagonal damping matrix of shape (nstates, nstates),
    where only a chosen set of states are damped.

    parameters
    ----------
    state_is_damped:   Array/list of boolean values of length nstates. If
                        state_is_damped[i] is True, a learnable damping
                        parameter is created for state i. If not, the damping
                        of state i is set to zero.
    '''
    def __init__(self, state_is_damped, ttype=torch.float32):
        super().__init__()
        if not isinstance(state_is_damped, torch.Tensor):
            state_is_damped = torch.tensor(state_is_damped)
        self.state_is_damped = state_is_damped.bool()
        self.ttype = ttype
        nstates = self.state_is_damped.shape[0]
        self.rs = nn.Parameter(torch.zeros(torch.sum(self.state_is_damped), dtype=ttype))
        self.pick_rs = torch.zeros((nstates, torch.sum(self.state_is_damped)))
        c = 0
        for i in range(nstates):
            if self.state_is_damped[i]:
                self.pick_rs[i, c] = 1
                c += 1

    def forward(self, x):
        return torch.diag(torch.abs(self.pick_rs)@(self.rs))


def load_baseline_model(modelpath):

    metadict = torch.load(modelpath)

    nstates = metadict['nstates']
    init_sampler = metadict['init_sampler']
    controller = metadict['controller']
    ttype = metadict['ttype']
    hidden_dim = metadict['rhs_model']['hidden_dim']
    timedependent = metadict['rhs_model']['timedependent']
    statedependent = metadict['rhs_model']['statedependent']

    rhs_model = BaselineNN(nstates, hidden_dim, timedependent, statedependent)
    rhs_model.load_state_dict(metadict['rhs_model']['state_dict'])

    model = DynamicSystemNN(nstates, rhs_model=rhs_model, init_sampler=init_sampler,
                            controller=controller, ttype=ttype)

    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(metadict['traininginfo']['optimizer_state_dict'])

    return model, optimizer, metadict


def store_baseline_model(storepath, model, optimizer, **kwargs):

    metadict = {}

    metadict['nstates'] = model.nstates
    metadict['init_sampler'] = model._initial_condition_sampler
    metadict['controller'] = model.controller
    metadict['ttype'] = model.ttype

    metadict['rhs_model'] = {}
    metadict['rhs_model']['hidden_dim'] = model.rhs_model.hidden_dim
    metadict['rhs_model']['timedependent'] = model.rhs_model.timedependent
    metadict['rhs_model']['statedependent'] = model.rhs_model.statedependent
    metadict['rhs_model']['state_dict'] = model.rhs_model.state_dict()

    metadict['traininginfo'] = {}
    metadict['traininginfo']['optimizer_state_dict'] = optimizer.state_dict()
    for key, value in kwargs.items():
        metadict['traininginfo'][key] = value

    torch.save(metadict, storepath)
