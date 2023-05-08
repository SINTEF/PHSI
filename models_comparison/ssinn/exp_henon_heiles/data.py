###############################################################################
# General Information
###############################################################################
# Sparse Symplectically Integrated Neural Networks (2020)
# Paper: https://arxiv.org/abs/2006.12972
# Daniel DiPietro, Shiying Xiong, Bo Zhu

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional
import numpy as np
import math
import random
import logging
import pickle
import argparse

import os
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from integrators import fourth_order, rk4
from function_spaces import gradient_wrapper, bivariate_poly
from ssinn import SSINN
from utils import get_logger

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--n', default=6250, type=int, help='number of points to generate')
    parser.add_argument('--dt', default=0.1, type=float, help='time step')
    parser.add_argument('--std', default=0.02, type=float, help='gaussian noise st dev')
    parser.set_defaults(feature=True)
    return parser.parse_args()

class Tp_hh(nn.Module):
    def forward(self, p):
        return p
class Vq_hh(nn.Module):
    def forward(self, q):
        return torch.tensor([[q[0][0] + 2*q[0][0]*q[0][1], q[0][1] + q[0][0]*q[0][0] - q[0][1]*q[0][1]]])

def Hamiltonian_hh(q_x, q_y, p_x, p_y):
    return 0.5*(q_x**2 + q_y**2 + p_x**2 + p_y**2) + (q_x**2)*(q_y) - (1/3)*(q_y**3)

def generate_data():
    # ~ Check if data exists already ~
    if (os.path.exists("hh_noise.pkl")):
        print('hh_noise.pkl found in local directory')
        return 0

    # ~ Fetch arguments ~
    args = get_args()
    device = torch.device('cpu')
    torch.set_default_tensor_type(torch.FloatTensor)

    # ~ Instantiate Model ~
    Tp = Tp_hh().to(device) # Kinetic
    Vq = Vq_hh().to(device) # Potential
    model = SSINN(Tp, Vq, fourth_order).to(device)
    dt = args.dt

    # ~ Generate Data ~
    points = []
    while (len(points) < args.n):
        # Generate random points within interior region
        q_x = random.uniform(-1, 1)
        q_y = random.uniform(-0.5, 1)
        p_x = random.uniform(-1, 1)
        p_y = random.uniform(-1, 1)

        # Check that these points correspond to chaotic Hamiltonians
        # If they don't, continue past.
        H = Hamiltonian_hh(q_x, q_y, p_x, p_y)
        if (H < 0.083333333332 or H > 0.16666666665):
            continue

        # Convert to tensors and predict future states; store state pairs as tuple
        t0 = torch.tensor([[0.]]).to(device)
        t1 = torch.tensor([[dt]]).to(device)
        p0 = torch.tensor([[p_x, p_y]])
        q0 = torch.tensor([[q_x, q_y]])
        p1, q1 = model(p0, q0, t0, t1)
        ele = (torch.cat((q0, p0), 1), torch.cat((q1, p1), 1))
        points.append(ele)
        print('{}/{}'.format(len(points),args.n), end='\r')

    # ~ Add Noise if Desired ~
    if (args.std != 0):
        for i in range(len(points)):
            noise1 = torch.empty(1,4).normal_(mean=0,std=args.std)
            noise2 = torch.empty(1,4).normal_(mean=0,std=args.std)
            points[i] = (points[i][0] + noise1, points[i][1] + noise2)

    # ~ Save Model ~
    with open("./hh_noise.pkl", 'wb') as handle:
        pickle.dump(points, handle)
    print('{} Henon-Heiles state-pairs (dt={}, noise stdev={}) saved in ./hh.pkl'.format(args.n, dt, args.std))

if __name__ == '__main__':
    generate_data()
