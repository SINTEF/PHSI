import torch
from torch import nn
from scipy.special import binom
import numpy as np
import re

from .dynamic_system_neural_network import DynamicSystemNN
from ..utils.derivatives import time_derivative

class BaselineSI(DynamicSystemNN):

    def __init__(
        self,
        nstates,
        degrees,
        pol_space, trig_space,
        time_pol_space, time_trig_space,
        **kwargs):
        super().__init__(nstates, **kwargs)
        self.deg = degrees
        self.n_terms = int(binom(nstates + degrees, degrees))*pol_space
        self.n_terms += nstates*2*trig_space
        self.n_terms += degrees*time_pol_space
        self.n_terms += 2*time_trig_space
        init_values = torch.ones((self.n_terms, nstates), requires_grad=True)*0.2
        self.pol_space = pol_space; self.trig_space = trig_space
        self.time_pol_space = time_pol_space; self.time_trig_space = time_trig_space
        self.layer = nn.Parameter(init_values)
        self.rhs_model = self.x_dot
        self.full_si = False

        if time_trig_space:
            self.t_omega = nn.Parameter(torch.ones(2))

    def time_derivative(self, integrator, *args):
        return time_derivative(integrator, self.x_dot, *args)
        
    def x_dot(self, x, t):
        x = x.clone().requires_grad_(True)
        n_data = x.shape[0]
        features = torch.ones(n_data, 1)

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
        
        if self.time_pol_space:
            t_pow = torch.pow(t, torch.arange(1, self.deg+1))
            features = torch.cat((features, t_pow), dim=1)

        if self.time_trig_space:
            t_cos = torch.cos(self.t_omega[0]*t); t_sin = torch.sin(self.t_omega[1]*t)
            features = torch.cat((features, t_cos, t_sin), dim=1)

        dx = features @ self.layer

        return dx
    
    def si_print(self):
        string = ""
        j = 0

        for k in range(self.nstates):
            string += f"x{k+1}': "
            j=0
            if self.pol_space:
                idx_list = []
                idx_init = [0]*self.nstates
                if float(self.layer[0,k]) != 0:
                    string += f"x^{idx_init}:{self.layer[0,k]: .4f}, "
                def loop_rec(idx_array, n):
                    nonlocal idx_list, string, j
                    for i in range(self.nstates):
                        idx_copy = idx_array.copy()
                        idx_copy[i] += 1
                        if idx_copy not in idx_list:
                            idx_list.append(idx_copy)
                            if float(self.layer[j+1,k]) != 0:
                                str_array = str(np.array(idx_copy))
                                string += f"x^{str_array}:"
                                string += f"{float(self.layer[j+1,k]): .4f}"
                                string += ", "
                                last_line = string.split("\n")[-1]
                                line_len = len(re.findall(':', last_line))
                                if line_len == 5:
                                    string += "\n"
                            j += 1
                        if n>0:
                            loop_rec(idx_copy, n-1)
                loop_rec(idx_init, self.deg - 1)
            
            if self.trig_space:
                string += "\n"
                for i in range(1,self.nstates+1):
                    if self.layer[j+i,k] != 0:
                        string += f"cos(x_{i}): "
                        string += f"{float(self.layer[j+i,k]): .4f}, "
                for i in range(1,self.nstates+1):
                    if self.layer[j+i,k] != 0:
                        string += f"sin(x_{i}): "
                        string += f"{float(self.layer[j+i+self.nstates,k]): .4f}, "
            
            if self.time_pol_space:
                string += "\n"
                idx = int(binom(self.nstates + self.deg, self.deg))*self.pol_space \
                    + self.nstates*2*self.trig_space
                for i in range(self.deg):
                    if self.layer[i+idx,k] != 0:
                        string += f"t^{i+1}:"
                        string += f"{self.layer[i+idx,k]: .4f}, "
            if self.time_trig_space:
                string += "\n"
                idx = int(binom(self.nstates + self.deg, self.deg))*self.pol_space \
                    + self.nstates*2*self.trig_space + self.time_pol_space*self.deg
                if self.layer[idx, k] != 0:
                    string += f"cos(t): "
                    string += f"amp= {self.layer[idx, k]: .4f} "
                    string += f"freq= {self.t_omega[0]: .4f}, "
                if self.layer[idx+1, k] != 0:
                    string += f"sin(t): "
                    string += f"amp= {self.layer[idx+1, k]: .4f} "
                    string += f"freq= {self.t_omega[1]: .4f}, "
            string += "\n\n"

        print(string[:-2])