import numpy as np
from qiskit import QuantumCircuit
from scipy.optimize import minimize

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from utils import set_random_params, sample_initial_qaoa_params

from Circuit import QAOACircuit

"""def optimise_params(qc: QuantumCircuit, gammas, betas, n: int, edges: np.ndarray, weights, params, x0: np.ndarray | None = None):

    p = len(gammas)
    if x0 is None:
        gamma_0, beta_0 = set_random_params(p)
        x0 = np.concatenate([gamma_0, beta_0])\

    def objective(x: np.ndarray) -> float:
        gamma_vals, beta_vals = x[:p], x[p:]
        qc_bound"""
class Gaussian_Process:
    def build_gp_prior(self):
        kernel = ConstantKernel(
            1.0,
            (1e-3, 1e3))
        kernel *= Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=1.5)
        kernel += WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e3))

        return GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10)

def function(point: tuple[np.ndarray[float], np.ndarray[float]], QAOA: QAOACircuit):
    # Step 2
    """This function receives a set of points, i.e. QAOA parameters, and evaluates the actual QAOA circuit on those parameters, returns the QAOA value found"""

    QAOA.bind_circuit_parameters(gamma_values=point[0], beta_values=point[1])
    results = QAOA.run_circuit()

def compute_bayesian_params(D: dict[]):
    # Step 4
    pass

def update_posterior(D: dict):
    # Step 6.1
    pass

def acquisition_function(D: dict, posterior, hyperparams):
    # Step 6.2/6.3
    pass

def bayesian_optimisation(QAOA: QAOACircuit, N_bayes: float, no_layers: int, points: list[tuple] = None):
    # points: list of parameters Θ = (𝛄, β) needed for the QAOA
    """This function optimises the QAOA parameters Θ = (𝛄, β) using Bayesian optimization."""
    if not isinstance(QAOA, QAOACircuit):
        raise TypeError("QAOA must be a class instance of QAOACircuit")

    if points is None: points = sample_initial_qaoa_params(1000, no_layers)

    gp = Gaussian_Process()
    prior = gp.build_gp_prior()

    training_set: dict = {}
    y: list = [] # keep separate list of observations for convenience
    for point in points:
        eval = function(point)
        training_set[point] = eval
        y.append(eval)

    sigma, l = compute_bayesian_params(training_set)
    n = 0
    posterior = None
    f_m = None # Best minimum of function
    while n <= N_bayes:
        maximising_point = acquisition_function(training_set, posterior, hyperparams = (sigma, l))
        eval = function(maximising_point)
        if eval < f_m or f_m is None:
            f_m = eval
        training_set[maximising_point] = eval
        sigma, l = compute_bayesian_params(training_set) # Find new hyperparams for the acquisition function given the updated training set
        n += 1

    return f_m
