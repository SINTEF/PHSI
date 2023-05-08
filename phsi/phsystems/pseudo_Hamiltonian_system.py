import numpy as np
from scipy.integrate import solve_ivp
import torch

from ..utils.derivatives import time_derivative


class PseudoHamiltonianSystem():
    """
    Implements a pseudo-Hamiltonian system of the form
        dx/dt = (S(x) - R(x))*grad[H(x)] + F(x, t)
    where x is the system state, S is the interconection matrix,
    R is the dissipation-matrix, H is the Hamiltonian of the system, F is the external forces.

    parameters
    ----------
        nstates            : Number of system states.

        structure_matrix   : Corresponds to the S matrix. Must either be an (nstates, nstates) ndarray,
                             or callable taking an ndarray input of shape (nsamples, nstates) and
                             returning an ndarray of shape (nsamples, nstates, nstates). If none,
                             the system is assumed to be canonical, and the 
                             S matrix is set ot the skew-symmetric matrix [[0, I_n], [-I_n, 0]].

        dissipation_matrix : Corresponds to the R matrix. Must either be an (nstates, nstates) ndarray,
                             or callable taking an ndarray input of shape (nsamples, nstates) and
                             returning an ndarray of shape (nsamples, nstates, nstates).

        hamiltonian        : The Hamiltonian H of the system. Callable taking a torch tensor input
                             of shape (nsamples, nstates) and returning a torch tensor of shape (nsamples, 1).
                             If the gradient of the Hamiltonian is not provided, the gradient of this function
                             will be computed by torch and used instead. If this is not provided, the grad_hamiltonian
                             must be provided.

        grad_hamiltonian   : The gradient of the Hamiltonian H of the system. Callable taking an ndarray input
                             of shape (nsamples, nstates) and returning a torch tensor of shape (nsamples, nstates).
                             If this is not provided, the hamiltonian must be provided.

        external_forces      : The external forces affecting system. Callable taking two ndarrays as input,
                             x and t, of shape (nsamples, nstates), (nsamples, 1), respectively and returning
                             an ndarray of shape (nasamples, nstates).

        controller         : Additional external forces set by a controller. Callable taking an ndarray x
                             of shape (nstates,) and a scalar t as input and returning
                             an ndarray of shape (nstates,). Note that this function should not take batch inputs,
                             and that when calling PseudoHamiltonianSystem.sample_trajectory when a controller
                             is provided, the Runge-Kutta 4 method will be used for integration in favor of
                             Scipy's solve_ivp.

        init_sampler       : Function for sampling initial conditions. Callabale taking a numpy random generator
                             as input and returning an ndarray of shape (nstates,) with inital conditions for
                             the system. This sampler is used when calling PseudoHamiltonianSystem.sample_trajectory
                             if no initial condition is provided.
    """

    def __init__(self, nstates, structure_matrix=None,
                 dissipation_matrix=None, hamiltonian=None,
                 grad_hamiltonian=None, external_forces=None,
                 controller=None, init_sampler=None):

        self.nstates = nstates

        if structure_matrix is None:
            npos = nstates // 2
            structure_matrix = np.block([[np.zeros([npos, npos]), np.eye(npos)],
                                         [-np.eye(npos), np.zeros([npos, npos])]])

        if not callable(structure_matrix):
            self.structure_matrix = structure_matrix
            self.S = lambda x: structure_matrix
        else:
            self.structure_matrix = None
            self.S = structure_matrix

        if dissipation_matrix is None:
            dissipation_matrix = np.zeros((self.nstates, self.nstates))

        if not callable(dissipation_matrix):
            if len(dissipation_matrix.shape) == 1:
                dissipation_matrix = np.diag(dissipation_matrix)
            self.dissipation_matrix = dissipation_matrix
            self.R = lambda x: dissipation_matrix
        else:
            self.dissipation_matrix = None
            self.R = dissipation_matrix

        self.H = hamiltonian
        self.dH = grad_hamiltonian
        if grad_hamiltonian is None:
            self.dH = self._dH

        self.controller = controller

        self.external_forces = external_forces
        if external_forces is None:
            self.external_forces = zero_forces

        if init_sampler is not None:
            self._initial_condition_sampler = init_sampler

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def time_derivative(self, integrator, *args):
        return time_derivative(integrator, self.x_dot, *args)

    def x_dot(self, x, t, u=None):
        S = self.S(x)
        R = self.R(x)
        dH = self.dH(x)
        if (len(S.shape) == 3) or (len(R.shape) == 3):
            dynamics = np.matmul(S - R, np.atleast_3d(dH)).reshape(x.shape) + self.external_forces(x, t)
        else:
            dynamics = dH@(S.T - R.T) + self.external_forces(x, t)
        if u is not None:
            dynamics += u
        return dynamics

    def sample_trajectory(self, t, x0=None, noise_std=0, reference=None):
        if x0 is None:
            x0 = self._initial_condition_sampler(self.rng)

        if self.controller is None:
            x_dot = lambda t, x: self.x_dot(x.reshape(1, x.shape[-1]),
                                            np.array(t).reshape((1, 1)))
            out_ivp = solve_ivp(fun=x_dot, t_span=(t[0], t[-1]), y0=x0, t_eval=t, rtol=1e-10)
            x, t = out_ivp['y'].T, out_ivp['t'].T
            dxdt = self.x_dot(x, t)
            us = None
        else:  # Use RK4 integrator instead of solve_ivp when controller is provided
            self.controller.reset()
            if reference is not None:
                self.controller.set_reference(reference)
            x = np.zeros([t.shape[0], x0.shape[-1]])
            dxdt = np.zeros_like(x)
            us = np.zeros([t.shape[0] - 1, x0.shape[-1]])
            x[0, :] = x0
            for i, t_step in enumerate(t[:-1]):
                dt = t[i + 1] - t[i]
                us[i, :] = self.controller(x[i, :], t_step)
                dxdt[i, :] = self.time_derivative('rk4', x[i:i+1, :], x[i:i+1, :],
                                                  np.array([t_step]), np.array([t_step]), dt, u=us[i:i+1, :])
                x[i + 1, :] = x[i, :] + dt*dxdt[i, :]

        # Add noise:
        x += self.rng.normal(size=x.shape)*noise_std
        dxdt += self.rng.normal(size=dxdt.shape)*noise_std

        return x, dxdt, t, us

    def _dH(self, x):
        x = torch.tensor(x, requires_grad=True)
        return torch.autograd.grad(self.H(x).sum(), x, retain_graph=False, create_graph=False)[0].detach().numpy()

    def _initial_condition_sampler(self, rng=None):
        if rng is None:
            assert self.rng is not None
            rng = self.rng
        return rng.uniform(low=-1., high=1.0, size=self.nstates)


def zero_forces(x, t=None):
    return np.zeros_like(x)

def sample_trajectories(pH_system, ntrajectories, t_sample, noise_std=0, references=None):
    nsamples = t_sample.shape[0]
    x = np.zeros((ntrajectories, nsamples, pH_system.nstates))
    dxdt = np.zeros((ntrajectories, nsamples, pH_system.nstates))
    t = np.zeros((ntrajectories, nsamples))
    u = np.zeros((ntrajectories, nsamples - 1, pH_system.nstates))
    if references is None:
        references = [None] * ntrajectories

    for i in range(ntrajectories):
        x[i], dxdt[i], t[i], u[i] = pH_system.sample_trajectory(t_sample, noise_std=noise_std, reference=references[i])

    if pH_system.controller is None:
        u = None

    return x, dxdt, t, u
