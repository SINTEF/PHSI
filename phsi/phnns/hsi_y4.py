import torch
from torch import nn
import numpy as np
from scipy.special import binom

from ..utils.derivatives import time_derivative

from .dynamic_system_neural_network import DynamicSystemNN


class HSI_Y4(DynamicSystemNN):
    def __init__(self,
                nstates,
                structure_matrix,
                function_space,
                degrees = 2,
                **kwargs):
        super().__init__(nstates, **kwargs)
        self.deg = degrees
        self.pol_space = bool(function_space[0])
        self.trig_space = bool(function_space[1])
        self.S = torch.tensor(structure_matrix, dtype=self.ttype)
        self.S_T = self.S.T
        
        pol_terms = 2*(int(binom(nstates//2 + degrees, degrees)-1))
        trig_terms = nstates*2
        self.n_terms = function_space @ np.array([pol_terms, trig_terms]) #number of total terms
        init_values = torch.ones(self.n_terms, requires_grad=True)*0.2
        self.layer = nn.Parameter(init_values)
        self.rhs_model = self.x_dot
        self.full_si = False
        self.zero_vals = None
    
    def time_derivative(self, integrator, *args):
        return time_derivative(integrator, self.x_dot, *args)

    def x_dot(self, x, t):
        x = x.clone().requires_grad_(True)
        n_data = x.shape[0]
        features = torch.zeros(n_data, 1)

        x1 = x[:,:2]; x2 = x[:, 2:]
        nstates = self.nstates//2

        for xi in [x1, x2]:
            if self.pol_space:
                x_cat = torch.cat([ torch.transpose(xi[:,0].unsqueeze(0),0,1) ]*(self.deg+1), dim=1).unsqueeze(dim=0)
                x_pow = torch.pow(x_cat, torch.arange(self.deg+1))
                for i in range(1,nstates):
                    i_cat = torch.cat([ torch.transpose(xi[:,i].unsqueeze(0),0,1) ]*(self.deg+1), dim=1).unsqueeze(dim=0)
                    i_pow = torch.pow(i_cat, torch.arange(self.deg+1))
                    x_pow = torch.cat((x_pow, i_pow), dim=0)
                
                idx_list = []
                idx_init = [0]*nstates

                def loop_rec(idx_array, n):
                    nonlocal features, idx_list, x_pow
                    for i in range(nstates):
                        idx_copy = idx_array.copy()
                        idx_copy[i] += 1
                        if idx_copy not in idx_list:
                            idx_list.append(idx_copy)
                            idx_onehot = torch.zeros((nstates, self.deg+1))
                            idx_onehot[torch.arange(nstates),idx_copy] = 1
                            idx_bool = idx_onehot > 0
                            select_cols = x_pow.transpose(1,2)[idx_bool].transpose(0,1)
                            product = torch.prod(select_cols, dim=1).unsqueeze(1)
                            features = torch.cat((features, product), dim=1)
                        if n>0:
                            loop_rec(idx_copy, n-1)
                loop_rec(idx_init, self.deg - 1)

            if self.trig_space:
                cosx = torch.cos(xi); sinx = torch.sin(xi)
                features = torch.cat((features, cosx, sinx), dim=1)
        features = features[:,1:]

        H = (self.layer*features).sum(axis=1)
        dH = torch.autograd.grad(H.sum(), x, retain_graph=True, create_graph=True)[0]
        return dH@(self.S_T)
    
    def si_print(self):
        print(self.layer)
