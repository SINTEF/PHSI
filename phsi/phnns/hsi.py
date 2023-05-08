import torch
from torch import nn
import numpy as np
from scipy.special import binom
import re

from ..utils.derivatives import time_derivative

from .dynamic_system_neural_network import DynamicSystemNN


class HSI(DynamicSystemNN):
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
        
        pol_terms = int(binom(nstates + degrees, degrees)-1)
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

        if self.pol_space:
            x_cat = torch.cat([ torch.transpose(x[:,0].unsqueeze(0),0,1) ]*(self.deg+1), dim=1).unsqueeze(dim=0)
            x_pow = torch.pow(x_cat, torch.arange(self.deg+1))
            for i in range(1,self.nstates):
                i_cat = torch.cat([ torch.transpose(x[:,i].unsqueeze(0),0,1) ]*(self.deg+1), dim=1).unsqueeze(dim=0)
                i_pow = torch.pow(i_cat, torch.arange(self.deg+1))
                x_pow = torch.cat((x_pow, i_pow), dim=0)
            
            idx_list = []
            idx_init = [0]*self.nstates

            def loop_rec(idx_array, n):
                nonlocal features, idx_list, x_pow
                for i in range(self.nstates):
                    idx_copy = idx_array.copy()
                    idx_copy[i] += 1
                    if idx_copy not in idx_list:
                        idx_list.append(idx_copy)
                        idx_onehot = torch.zeros((self.nstates, self.deg+1))
                        idx_onehot[torch.arange(self.nstates),idx_copy] = 1
                        idx_bool = idx_onehot > 0
                        select_cols = x_pow.transpose(1,2)[idx_bool].transpose(0,1)
                        product = torch.prod(select_cols, dim=1).unsqueeze(1)
                        features = torch.cat((features, product), dim=1)
                    if n>0:
                        loop_rec(idx_copy, n-1)
            loop_rec(idx_init, self.deg - 1)

        if self.trig_space:
            cosx = torch.cos(x); sinx = torch.sin(x)
            features = torch.cat((features, cosx, sinx), dim=1)
        features = features[:,1:]


        H = (self.layer*features).sum(axis=1)
        dH = torch.autograd.grad(H.sum(), x, retain_graph=True, create_graph=True)[0]
        return dH@(self.S_T)
    
    def si_print(self):
        string = ""
        j = 0

        if self.pol_space:
            idx_list = []
            idx_init = [0]*self.nstates
            def loop_rec(idx_array, n):
                nonlocal idx_list, string, j
                for i in range(self.nstates):
                    idx_copy = idx_array.copy()
                    idx_copy[i] += 1
                    if idx_copy not in idx_list:
                        idx_list.append(idx_copy)
                        if float(self.layer[j]) != 0:
                            str_array = str(np.array(idx_copy))
                            string += f"x^{str_array}"
                            string += ":"
                            string += f"{float(self.layer[j]): .4f}"
                            string += ", "
                            last_line = string.split("\n")[-1]
                            line_len = len(re.findall(',', last_line))
                            if line_len == 5:
                                string += "\n"
                        j += 1
                    if n>0:
                        loop_rec(idx_copy, n-1)
            loop_rec(idx_init, self.deg - 1)
        
        if self.trig_space:
            for i in range(1,self.nstates+1):
                string += f"cos(x_{i}): "
                string += f"{float(self.layer[j+i]): .4f}, "
            for i in range(1,self.nstates+1):
                string += f"sin(x_{i}): "
                string += f"{float(self.layer[j+i+self.nstates]): .4f}, "
        
        print(string[:-2])
