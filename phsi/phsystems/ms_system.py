import numpy as np

from .pseudo_Hamiltonian_system import PseudoHamiltonianSystem


class MassSpringSystem(PseudoHamiltonianSystem):
    """
    Implements a general forced damped mass spring damper as a
    forces-Hamintonian formulation:
           .
        |  q |     |  0     1 |                 |     0    |
        |  . |  =  |          |*grad[H(q, p)] + |          |
        |  p |     | -1    -c |                 |f(q, p, t)|
    where q is the position, p the momentum and c the damping coefficient.

    parameters
    ----------
        mass            : Scalar mass
        spring_constant : Scalar spring coefficient
        damping         : Scalar damping coefficient. Corresponds to c.
        kwargs          : Keyword arguments that are passed to PseudoHamiltonianSystem constructor.
    """
    def __init__(self, mass=1.0, spring_constant=1.0, damping=0.0, **kwargs):
        R = np.array([[0, 0], [0, damping]])

        def ham(x):
            return np.dot(x**2, np.array([spring_constant / 2, 1/(2*mass)]))

        def ham_grad(x):
            return np.matmul(x, np.diag([spring_constant, 1/mass]))

        super().__init__(nstates=2, hamiltonian=ham, grad_hamiltonian=ham_grad,
                         dissipation_matrix=R, **kwargs)


def init_mssystem():
    f0 = 2
    omega = 1/2

    def F(x, t):
        return (f0*np.sin(omega*t)).reshape(x[..., 1:].shape)*np.array([0, 1])

    def zero(x, t):
        np.zeros(t.shape)
        return (np.zeros(t.shape)).reshape(x[..., 1:].shape)*np.array([0, 0])

    return MassSpringSystem(external_forces=zero, init_sampler=initial_condition_radial(1, 4.5))


def initial_condition_radial(r_min, r_max):
    def sampler(rng):
        a = rng.uniform(size=1)
        r = (r_max - r_min) * np.sqrt(a) + r_min
        theta = 2.*np.pi * rng.uniform(size=1)
        q = r * np.cos(theta)
        p = r * np.sin(theta)
        return np.array([q, p]).flatten()

    return sampler