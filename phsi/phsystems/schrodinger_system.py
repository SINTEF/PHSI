import numpy as np
import torch

from .pseudo_Hamiltonian_system import PseudoHamiltonianSystem

class SCHRSystem(PseudoHamiltonianSystem):
    def __init__(self, N=1, **kwargs):
        R = np.zeros((2*N,2*N))

        def ham(x):
            if N==1:
                if x.ndim == 1:
                    q = x[0]; p = x[1]
                else:
                    q = x[:,0]; p = x[:,1]
                return 1/4*(q**4 + 2*q**2*p**2 + p**4)
            elif N==2:
                if x.ndim == 1:
                    q_1 = x[0]; q_2 = x[1]; p_2 = x[2]; p_2 = x[3]
                else:
                    q_1 = x[:,0]; q_2 = x[:,1]; p_1 = x[:,2]; p_2 = x[:,3]
                return 1/4*(q_1**4 + 2*q_1**2*p_1**2 + p_1**4 + q_2**4 + 2*q_2**2*p_2**2 + p_2**4) \
                - (p_1**2*p_2**2 + q_1**2*q_2**2 - q_1**2*p_2**2 - p_1**2*q_2**2 + 4*p_1*p_2*q_1*q_2)

        def ham_grad(x):
            if N==1:
                if x.ndim == 1:
                    q = x[0]; p = x[1]
                else:
                    q = x[:,0]; p = x[:,1]
                return np.transpose(np.array([q**3 + q*p**2, p**3 + q**2*p]))
            elif N==2:
                if x.ndim == 1:
                    q_1 = x[0]; q_2 = x[1]; p_1 = x[2]; p_2 = x[3]
                else:
                    q_1 = x[:,0]; q_2 = x[:,1]; p_1 = x[:,2]; p_2 = x[:,3]
                return np.transpose(np.array([q_1**3 + q_1*p_1**2 - 2*q_1*q_2**2 + 2*q_1*p_2**2 - 4*p_1*p_2*q_2, \
                                              q_2**3 + q_2*p_2**2 - 2*q_1**2*q_2 + 2*q_2*p_1**2 - 4*p_1*p_2*q_1, \
                                              p_1**3 + q_1**2*p_1 - 2*p_1*p_2**2 + 2*p_1*q_2**2 - 4*p_2*q_1*q_2, \
                                              p_2**3 + q_2**2*p_2 - 2*p_1**2*p_2 + 2*q_1**2*p_2 - 4*p_1*q_1*q_2]))

        super().__init__(nstates=N*2, hamiltonian=ham, grad_hamiltonian=ham_grad,
                         dissipation_matrix=R, **kwargs)
        
        if N==1:
            self.structure_matrix = np.array([[0.0, 1.0],
                                          [-1.0, 0.0]])
        elif N==2:       
            self.structure_matrix = np.array([[0.0, 0.0, 1.0, 0.0],
                                            [0.0, 0.0, 0.0, 1.0],
                                            [-1.0, 0.0, 0.0, 0.0],
                                            [0.0, -1.0, 0.0, 0.0]])

def init_schr(N=1):
    def zero(x, t):
        return np.array([0]*2*N)

    return SCHRSystem(N=N, external_forces = zero, init_sampler = hh_init(N))

def hh_init(N):
    def sampler(rnd):
        q_1_init =rnd.uniform(low=-1, high=1)
        p_1_init =rnd.uniform(low=-1, high=1)
        q_2_init = rnd.uniform(low=-1, high=1)
        p_2_init = rnd.uniform(low=-1, high=1)
        rand = np.array([q_1_init, p_1_init, q_2_init, p_2_init])
        rand = rand[:(N*2)]
        rand = rand*0.2
        return torch.tensor(rand)
    return sampler