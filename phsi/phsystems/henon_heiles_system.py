import numpy as np
import torch

from .pseudo_Hamiltonian_system import PseudoHamiltonianSystem

class HenonHeilesSystem(PseudoHamiltonianSystem):

    def __init__(self, **kwargs):
        R = np.zeros((4,4))

        def ham(x):
            if x.ndim == 1:
                q_x = x[0]; q_y = x[1]; p_x = x[2]; p_y = x[3]
            else:
                q_x = x[:,0]; q_y = x[:,1]; p_x = x[:,2]; p_y = x[:,3]
            return 0.5*(q_x**2 + q_y**2 + p_x**2 + p_y**2) + (q_x**2)*(q_y) - (1/3)*(q_y**3)
        
        def ham_grad(x):
            if x.ndim == 1:
                q_x = x[0]; q_y = x[1]; p_x = x[2]; p_y = x[3]
            else:
                q_x = x[:,0]; q_y = x[:,1]; p_x = x[:,2]; p_y = x[:,3]
            return np.transpose(np.array([q_x + 2*q_x*q_y, q_y + q_x**2 - q_y**2, p_x, p_y]))
        
        super().__init__(nstates=4, hamiltonian=ham, grad_hamiltonian=ham_grad,
                         dissipation_matrix=R, **kwargs)
        
        self.structure_matrix = np.array([[0, 0, 1, 0],
                                          [0, 0, 0, 1],
                                          [-1, 0, 0, 0],
                                          [0, -1, 0, 0]])

def init_henon_heiles():
    f0 = 1.0
    omega = 2

    def F(x, t):
        return (f0*np.sin(omega*t)).reshape(x[..., 3:].shape)*np.array([0, 0, 0, 1])
    
    def zero(x, t):
        return np.array([0, 0, 0, 0])

    return HenonHeilesSystem(external_forces = zero, init_sampler = hh_init(0))

def hh_init(rand):
    def sampler(rng):
        mini=-1.; maxi=1.
        q_x_0 = rng.uniform(low=mini, high=maxi)
        q_y_0 = rng.uniform(low=mini, high=maxi)
        p_x_0 = rng.uniform(low=mini, high=maxi)
        p_y_0 = rng.uniform(low=mini, high=maxi)
        rand = np.array([q_x_0, q_y_0, p_x_0, p_y_0])
        return torch.tensor(rand)
    return sampler