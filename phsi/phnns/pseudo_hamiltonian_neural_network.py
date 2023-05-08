import torch

from .dynamic_system_neural_network import DynamicSystemNN
from .models import HamiltonianNN, R_NN, R_estimator, ExternalForcesNN


class PseudoHamiltonianNN(DynamicSystemNN):
    """
    Implements a pseudo-Hamiltonian neural network abiding to the pseudo-Hamiltonian formulation
        dx/dt = (S(x) - R(x))*grad[H(x)] + F(x, t)
    where x is the system state, S is the interconection matrix,
    R is the dissipation matrix, H is the Hamiltonian of the system and F is the external forces.
    
    It is possible to provide function estimators like neural networks to model R, H
    and F. All estimators must subclass torch.nn.Module, such that gradients can be recorded
    with pytorch.

    If R, H or F are known, they can be provided and used in favor of estimators. Note that
    R, H and F must be functions both taking as input and returning tensors, and that the
    gradients of H(x) must be availble through autograd unless the true gradient is provided.


    parameters
    ----------
        nstates            : Number of system states.

        structure_matrix   : Corresponds to the S matrix. Must either be an (nstates, nstates) tensor,
                             or callable taking a tensor input of shape (nsamples, nstates) and
                             returning an tensor of shape (nsamples, nstates, nstates).

        hamiltonian_true   : The known Hamiltonian H of the system. Callable taking a torch tensor input
                             of shape (nsamples, nstates) and returning a torch tensor of shape (nsamples, 1).
                             If the gradient of the Hamiltonian is not provided, the gradient of this function
                             will be computed by torch and used instead. If hamiltonian_true is provided,
                             hamiltonian_est will be ignored.

        grad_hamiltonian_true : The known gradient of the Hamiltonian. Callable taking a tensor input
                             of shape (nsamples, nstates) and returning a tensor of shape (nsamples, nstates).

        dissipation_true   : The known R matrix. Must either be a (nstates, nstates) tensor,
                             or callable taking a tensor input of shape (nsamples, nstates) and
                             returning a tesnsor of shape (nsamples, nstates, nstates). If dissipation_true
                             is provided, dissipation_est will be ignored.

        external_forces_true : The external forces affecting system. Callable taking two tensord as input,
                             x and t, of shape (nsamples, nstates), (nsamples, 1), respectively and returning
                             a tensor of shape (nasamples, nforcees). If external_forces_true
                             is provided, external_forces_est will be ignored.

        hamiltonian_est    : Estimator for the Hamiltonian. Takes a tensor of shape
                             (nsamples, nstates) as input, returning a tensor of shape
                             (nsamples, 1).

        dissipation_est    : Estimator for the R matrix. Takes a tensor of shape
                             (nsamples, nstates) as input, returning a tensor either of shape
                             (nstates, nstates) or of shape (nsamples, nstates, nstates).

        external_forces_est  : Estimator for the external forces. Takes a tensor of shape
                             (nsamples, nstates) as input, returning a tensor of shape
                             (nsamples, nstates).
    """

    def __init__(self,
                 nstates,
                 structure_matrix,
                 hamiltonian_true=None,
                 grad_hamiltonian_true=None,
                 dissipation_true=None,
                 external_forces_true=None,
                 hamiltonian_est=None,
                 dissipation_est=None,
                 external_forces_est=None,
                 **kwargs):
        super().__init__(nstates, **kwargs)
        self.S = None
        self.hamiltonian = None
        self.external_forces = None
        self.R = None
        self.hamiltonian_provided = False
        self.grad_hamiltonian_provided = False
        self.external_forces_provided = False
        self.dissipation_provided = False
        self.structure_matrix = structure_matrix
        self.hamiltonian_true = hamiltonian_true
        self.grad_hamiltonian_true = grad_hamiltonian_true
        self.dissipation_true = dissipation_true
        self.external_forces_true = external_forces_true

        if not callable(structure_matrix):
            self.S = self._structure_matrix
        else:
            self.S = structure_matrix

        if dissipation_true is not None:
            self.dissipation_provided = True
            if callable(dissipation_true):
                self.R = self._dissipation_true_callable
            else:
                self.R = self._dissipation_true_static
        else:
            self.R = dissipation_est

        if hamiltonian_true is not None:
            if grad_hamiltonian_true is None:
                self.hamiltonian = hamiltonian_true
                self.dH = self._dH_hamiltonian_true
            else:
                self.hamiltonian = self._hamiltonian_true
                self.dH = self._grad_hamiltonian_true
                self.grad_hamiltonian_provided = True
            self.hamiltonian_provided = True
        elif grad_hamiltonian_true is not None:
            self.dH = self._grad_hamiltonian_true
            self.grad_hamiltonian_provided = True
        else:
            self.hamiltonian = hamiltonian_est
            self.dH = self._dH_hamiltonian_est

        if external_forces_true is not None:
            self.external_forces = self._external_forces_true
            self.external_forces_provided = True
        else:
            self.external_forces = external_forces_est
        self.rhs_model = self._x_dot

    def _structure_matrix(self, x):
        return torch.tensor(self.structure_matrix, dtype=self.ttype)
    
    def _hamiltonian_true(self, x):
        return self.hamiltonian_true(x).detach()

    def _grad_hamiltonian_true(self, x):
        return self.grad_hamiltonian_true(x).detach()

    def _external_forces_true(self, x, t):
        return self.external_forces_true(x, t).detach()

    def _dissipation_true_callable(self, x):
        return self.dissipation_true(x).detach()

    def _dissipation_true_static(self, x):
        return torch.tensor(self.dissipation_true, dtype=self.ttype)

    def _dH_hamiltonian_est(self, x):
        x = x.detach().requires_grad_()
        return torch.autograd.grad(self.hamiltonian(x).sum(), x, retain_graph=self.training, create_graph=self.training)[0]

    def _dH_hamiltonian_true(self, x):
        x = x.detach().requires_grad_()
        return torch.autograd.grad(self.hamiltonian(x).sum(), x, retain_graph=False, create_graph=False)[0].detach()

    def _x_dot(self, x, t, u=None):
        S = self.S(x)
        R = self.R(x)
        dH = self.dH(x)
        if (len(S.shape) == 3) or (len(R.shape) == 3):
            dynamics = torch.matmul(S - R, torch.atleast_3d(dH)).reshape(x.shape) + self.external_forces(x, t)
        else:
            dynamics = dH@(S.T - R.T) + self.external_forces(x, t)
        if u is not None:
            dynamics += u
        return dynamics


def load_phnn_model(modelpath):

    metadict = torch.load(modelpath)

    nstates = metadict['nstates']
    structure_matrix = metadict['structure_matrix']
    hamiltonian_provided = metadict['hamiltonian_provided']
    grad_hamiltonian_provided = metadict['grad_hamiltonian_provided']
    external_forces_provided = metadict['external_forces_provided']
    dissipation_provided = metadict['dissipation_provided']
    init_sampler = metadict['init_sampler']
    controller = metadict['controller']
    ttype = metadict['ttype']

    if hamiltonian_provided:
        hamiltonian_true = metadict['hamiltonian']['true']
        hamiltonian_est = None
    else:
        hamiltonian_true = None
        hidden_dim = metadict['hamiltonian']['hidden_dim']
        hamiltonian_est = HamiltonianNN(nstates, hidden_dim)
        hamiltonian_est.load_state_dict(metadict['hamiltonian']['state_dict'].copy())

    if grad_hamiltonian_provided:
        grad_hamiltonian_true = metadict['grad_hamiltonian']['true']
    else:
        grad_hamiltonian_true = None

    if external_forces_provided:
        external_forces_true = metadict['external_forces']['true']
        external_forces_est = None
    else:
        external_forces_true = None
        noutputs = metadict['external_forces']['noutputs']
        hidden_dim = metadict['external_forces']['hidden_dim']
        timedependent = metadict['external_forces']['timedependent']
        statedependent = metadict['external_forces']['statedependent']
        external_forces_filter = metadict['external_forces']['external_forces_filter']
        ttype = metadict['external_forces']['ttype']
        external_forces_est = ExternalForcesNN(nstates, noutputs, hidden_dim, timedependent,
                                           statedependent, external_forces_filter, ttype)
        external_forces_est.load_state_dict(metadict['external_forces']['state_dict'].copy())

    if dissipation_provided:
        dissipation_true = metadict['dissipation']['true']
        dissipation_est = None
    else:
        dissipation_true = None
        dissipation_type = metadict['dissipation']['type'].lower()
        if 'r_estimator' in dissipation_type:
            state_is_damped = metadict['dissipation']['state_is_damped']
            ttype = metadict['dissipation']['ttype']
            dissipation_est = R_estimator(state_is_damped, ttype)
        else:
            hidden_dim = metadict['dissipation']['hidden_dim']
            diagonal = metadict['dissipation']['diagonal']
            dissipation_est = R_NN(nstates, hidden_dim, diagonal)
        dissipation_est.load_state_dict(metadict['dissipation']['state_dict'].copy())

    model = PseudoHamiltonianNN(nstates, structure_matrix,
                              hamiltonian_true=hamiltonian_true,
                              grad_hamiltonian_true=grad_hamiltonian_true,
                              dissipation_true=dissipation_true,
                              external_forces_true=external_forces_true,
                              hamiltonian_est=hamiltonian_est,
                              dissipation_est=dissipation_est,
                              external_forces_est=external_forces_est,
                              init_sampler=init_sampler,
                              controller=controller,
                              ttype=ttype)

    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(metadict['traininginfo']['optimizer_state_dict'])

    return model, optimizer, metadict


def store_phnn_model(storepath, model, optimizer, **kwargs):
    metadict = {}
    metadict['nstates'] = model.nstates
    metadict['structure_matrix'] = model.structure_matrix
    metadict['hamiltonian_provided'] = model.hamiltonian_provided
    metadict['grad_hamiltonian_provided'] = model.hamiltonian_provided
    metadict['external_forces_provided'] = model.external_forces_provided
    metadict['dissipation_provided'] = model.dissipation_provided
    metadict['init_sampler'] = model._initial_condition_sampler
    metadict['controller'] = model.controller
    metadict['ttype'] = model.ttype

    metadict['traininginfo'] = {}
    metadict['traininginfo']['optimizer_state_dict'] = optimizer.state_dict()
    for key, value in kwargs.items():
        metadict['traininginfo'][key] = value

    metadict['hamiltonian'] = {}
    metadict['grad_hamiltonian'] = {}
    metadict['external_forces'] = {}
    metadict['dissipation'] = {}

    if model.hamiltonian_provided:
        metadict['hamiltonian']['true'] = model.hamiltonian_true
        metadict['hamiltonian']['hidden_dim'] = None
        metadict['hamiltonian']['state_dict'] = None
    else:
        metadict['hamiltonian']['true'] = None
        metadict['hamiltonian']['hidden_dim'] = model.hamiltonian.hidden_dim
        metadict['hamiltonian']['state_dict'] = model.hamiltonian.state_dict()

    if model.grad_hamiltonian_provided:
        metadict['grad_hamiltonian']['true'] = model.grad_hamiltonian_true
    else:
        metadict['grad_hamiltonian']['true'] = None

    if model.external_forces_provided:
        metadict['external_forces']['true'] = model.external_forces_true
        metadict['external_forces']['noutputs'] = None
        metadict['external_forces']['hidden_dim'] = None
        metadict['external_forces']['timedependent'] = None
        metadict['external_forces']['statedependent'] = None
        metadict['external_forces']['external_forces_filter'] = None
        metadict['external_forces']['ttype'] = None
        metadict['external_forces']['state_dict'] = None
    else:
        metadict['external_forces']['true'] = None
        metadict['external_forces']['noutputs'] = model.external_forces.noutputs
        metadict['external_forces']['hidden_dim'] = model.external_forces.hidden_dim
        metadict['external_forces']['timedependent'] = model.external_forces.timedependent
        metadict['external_forces']['statedependent'] = model.external_forces.statedependent
        metadict['external_forces']['external_forces_filter'] = model.external_forces.external_forces_filter.T
        metadict['external_forces']['ttype'] = model.external_forces.ttype
        metadict['external_forces']['state_dict'] = model.external_forces.state_dict()

    if model.dissipation_provided:
        metadict['dissipation']['true'] = model.dissipation_true
        metadict['dissipation']['type'] = None
        metadict['dissipation']['state_is_damped'] = None
        metadict['dissipation']['ttype'] = None
        metadict['dissipation']['hidden_dim'] = None
        metadict['dissipation']['diagonal'] = None
        metadict['dissipation']['state_dict'] = None

    else:
        metadict['dissipation']['true'] = None
        metadict['dissipation']['type'] = str(type(model.R))
        if 'r_estimator' in metadict['dissipation']['type'].lower():
            metadict['dissipation']['state_is_damped'] = model.R.state_is_damped
            metadict['dissipation']['ttype'] = model.R.ttype
            metadict['dissipation']['hidden_dim'] = None
            metadict['dissipation']['diagonal'] = None
        else:
            metadict['dissipation']['state_is_damped'] = None
            metadict['dissipation']['ttype'] = None
            metadict['dissipation']['hidden_dim'] = model.R.hidden_dim
            metadict['dissipation']['diagonal'] = model.R.diagonal
        metadict['dissipation']['state_dict'] = model.R.state_dict()

    torch.save(metadict, storepath)
