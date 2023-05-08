import torch
from torch import nn
import numpy as np
from scipy.special import binom
import re

from ..utils.derivatives import time_derivative

from .dynamic_system_neural_network import DynamicSystemNN


class PHSI(DynamicSystemNN):
    def __init__(self,
                nstates,
                structure_matrix,
                function_space,
                external_forces,
                dissipation_est,
                degrees = 2,
                full_si = False,
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
        self.external_forces = external_forces
        self.R = dissipation_est
        self.rhs_model = self.x_dot
        self.full_si = full_si
        self.zero_vals = None
    
    def time_derivative(self, integrator, *args):
        return time_derivative(integrator, self.x_dot, *args)

    def x_dot(self, x, t):
        x = x.clone().requires_grad_(True)   
        n_data = x.shape[0]
        features = torch.zeros(n_data, 1)
        R = self.R(x)

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
        if self.external_forces is not None:
            return dH@(self.S_T - R.T) + self.external_forces(x, t)
        
        
        return dH@(self.S_T - R.T)

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
                                string += f"\n"
                        j += 1
                    if n>0:
                        loop_rec(idx_copy, n-1)
            loop_rec(idx_init, self.deg - 1)
        
        if self.trig_space:
            for i in range(self.nstates):
                if float(self.layer[j+i] != 0):
                    string += f"cos(x_{i}): "
                    string += f"{float(self.layer[j+i]): .4f}, "
                    last_line = string.split("\n")[-1]
                    line_len = len(re.findall(',', last_line))
                    if line_len == 5:
                        string += f"\n"
            for i in range(self.nstates):
                if float(self.layer[j+i+self.nstates]) != 0:
                    string += f"sin(x_{i}): "
                    string += f"{float(self.layer[j+i+self.nstates]): .4f}, "
                    line_len = len(re.findall(',', last_line))
                    if line_len == 5:
                        string += f"\n"
        
        string += "\n\n"
        for i in range(self.R.rs.shape[0]):
            string += f"r{i+1}: {self.R.rs[i]: .4f}, "
        
        if self.full_si:
            string += "\nForces Parameters: \n"
            j=0
            if self.external_forces.pol_space:
                idx_list_p = []
                idx_init_p = [0]*self.external_forces.ninputs

                def loop_rec_p(idx_array, n):
                    nonlocal idx_list_p, j, string
                    for i in range(self.external_forces.ninputs):
                        idx_copy = idx_array.copy()
                        idx_copy[i] += 1
                        if idx_copy not in idx_list_p:
                            idx_list_p.append(idx_copy)
                            if float(self.external_forces.layer[j]) != 0:
                                str_array = str(np.array(idx_copy))
                                string += f"x^{str_array}"
                                string += ":"
                                string += f"{float(self.external_forces.layer[j]): .4f}, "
                                last_line = string.split("\n")[-1]
                                line_len = len(re.findall(',', last_line))
                                if line_len == 5:
                                    string += f"\n"
                            j += 1
                        if n>0:
                            loop_rec_p(idx_copy, n-1)
                loop_rec_p(idx_init_p, self.deg - 1)
            
            if self.external_forces.pol_space:
                for i in range(self.external_forces.ninputs):
                    if float(self.external_forces.layer[j+i]) != 0:
                        string += f"cos(x_{i}): "
                        string += f"amp={float(self.external_forces.layer[j+i]): .4f} "
                        string += f"freq={float(self.external_forces.omegas[i]): .4f}, "
                        last_line = string.split("\n")[-1]
                        line_len = len(re.findall(',', last_line))
                        if line_len == 5:
                            string += f"\n"
                for i in range(self.external_forces.ninputs):
                    if float(self.external_forces.layer[j+i+self.external_forces.ninputs]) != 0:
                        string += f"sin(x_{i}): "
                        string += f"amp={float(self.external_forces.layer[j+i+self.external_forces.ninputs]): .4f} "
                        string += f"freq={float(self.external_forces.omegas[self.external_forces.ninputs+i]): .4f}, "
                        last_line = string.split("\n")[-1]
                        line_len = len(re.findall(',', last_line))
                        if line_len == 5:
                            string += f"\n"

        print(string[:-2] + "\n")


def load_phsi_model(modelpath):

    metadict = torch.load(modelpath)

    model = metadict['model']

    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(metadict['traininginfo']['optimizer_state_dict'])

    return model, optimizer, metadict


def store_phsi_model(storepath, model, optimizer, **kwargs):
    metadict = {}
    metadict['model'] = model

    metadict['traininginfo'] = {}
    metadict['traininginfo']['optimizer_state_dict'] = optimizer.state_dict()
    for key, value in kwargs.items():
        metadict['traininginfo'][key] = value

    torch.save(metadict, storepath)