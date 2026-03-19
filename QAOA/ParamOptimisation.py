import numpy as np
from qiskit import QuantumCircuit
from scipy.optimize import minimize
from scipy.stats import norm

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
        """
        This method builds the Gaussian-process prior used for Bayesian optimisation.

        :return: configured GaussianProcessRegressor with Constant*Matern+White kernel
        """
        kernel = ConstantKernel(
            1.0,
            (1e-3, 1e3))
        kernel *= Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=1.5)
        kernel += WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e3))

        return GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10)

def eval_QAOA_circuit(point: tuple[np.ndarray[float], np.ndarray[float]], QAOA: QAOACircuit) -> float:
    # Step 2
    """This function receives a set of points, i.e. QAOA parameters, and evaluates the actual QAOA circuit on those parameters, returns the QAOA value found"""

    QAOA.bind_circuit_parameters(gamma_values=point[0], beta_values=point[1])
    statevec = QAOA.run_circuit(return_statevector=True)

    product_states = {}
    for (i, j), w in zip(QAOA.edges, QAOA.weights):
        product_states[(i, j)] = QAOA.two_qubit_marginal(statevec, QAOA.n, i, j)
    energy = QAOA.qaoa_compute_energy(product_states=product_states, edges=QAOA.edges, weights=QAOA.weights)

    return float(np.real(energy))

def compute_bayesian_params(dataset: tuple, prior):
    # Step 4
    """
    This method fits the prior on observed data and extracts kernel hyperparameters.

    :param dataset: tuple (points, y) with sampled points and observed energies
    :param prior: Gaussian process regressor to fit on the dataset
    :return: tuple (sigma, l) with amplitude and length-scale hyperparameters (in the current implementation, also returning nothing would be fine)
    """
    points, y = dataset
    X = np.array([np.concatenate([np.asarray(point[0]), np.asarray(point[1])]) for point in points], dtype=float)
    prior.fit(X, np.array(y, dtype=float))
    sigma = float(np.sqrt(prior.kernel_.k1.k1.constant_value))
    l = float(prior.kernel_.k1.k2.length_scale)

    return sigma, l

def expected_improvement(prior, X_candidates: np.ndarray, f_min: float):
    """
    This method computes expected-improvement values for candidate points.

    :param prior: fitted Gaussian process regressor
    :param X_candidates: candidate points in flattened parameter space
    :param f_min: current best objective value (i.e. energy)
    :return: expected-improvement score for each candidate point
    """
    mean, std = prior.predict(X_candidates, return_std=True)
    covariance = np.maximum(std**2, 1e-12) # NOTE It seems that the more standard way outside of the scope of this paper is to use std and not std ** 2

    z = (f_min - mean) / covariance

    EI = norm.cdf(z)*(f_min - mean) + norm.pdf(z)*covariance

    return EI

def compute_acquisition_function(prior, f_min, candidate_points=None):
    # Step 6.2/6.3
    """
    This method selects the candidate with the maximum expected improvement.

    :param prior: fitted Gaussian process regressor
    :param f_min: current best objective value
    :param candidate_points: list of candidate QAOA parameter tuples (gamma, beta)
    :return: candidate point that maximises the acquisition function
    """
    assert candidate_points is not None
    X = np.array([np.concatenate([np.asarray(point[0]), np.asarray(point[1])]) for point in candidate_points], dtype=float)

    EI = expected_improvement(prior, X, f_min)
    best_index = int(np.argmax(EI))
    return candidate_points[best_index]

def optimise_cobyla(QAOA: QAOACircuit, no_layers: int, max_iter: int = 100):
    """
    NOTE: ALTERNATIVE OPTIMISATION FUNCTION; this one is standalone, in the sense that all the other methods in this file are only for the Bayesian optimisation, but this one is a separate method that can be used to optimise QAOA parameters using COBYLA instead of Bayesian optimisation. 
    :param no_layers: number of QAOA layers
    :param max_iter: maximum number of COBYLA iterations
    :return: scipy optimisation result object
    """
    assert isinstance(QAOA, QAOACircuit)

    gamma, beta = set_random_params(no_layers)
    x0 = np.concatenate([gamma, beta])

    def objective(theta: np.ndarray) -> float:
        gamma_vals = theta[:no_layers]
        beta_vals = theta[no_layers:]
        return -eval_QAOA_circuit((gamma_vals, beta_vals), QAOA)

    return minimize(objective, x0=x0, method="COBYLA", options={"maxiter": max_iter})
    #return minimize(objective, x0=x0, method="L-BFGS-B", options={"maxiter": max_iter})

def bayesian_optimisation(QAOA: QAOACircuit, N_bayes: float, no_layers: int, points: list[tuple] = None):
    # points: list of parameters Θ = (𝛄, β) needed for the QAOA
    """
    NOTE: MAIN OPTIMISATION FUNCTION; this one is the entry point for the Bayesian optimisation per the paper [provide citation]
    This function optimises the QAOA parameters Θ = (𝛄, β) using Bayesian optimization.

    :param QAOA: configured QAOA circuit instance
    :param N_bayes: number of Bayesian optimisation iterations
    :param no_layers: number of QAOA layers p
    :param points: optional initial list of QAOA parameter tuples (gamma, beta); if not provided, random initial points will be sampled
    :return: best energy value found during optimisation
    """
    if not isinstance(QAOA, QAOACircuit):
        raise TypeError("QAOA must be a class instance of QAOACircuit")

    if points is None: points = sample_initial_qaoa_params(1000, no_layers)

    gp = Gaussian_Process()
    prior = gp.build_gp_prior()

    #training_set: dict = {}
    y: list[float] = [] # keep separate list of observations for convenience
    for point in points:
        eval = eval_QAOA_circuit(point, QAOA)
        #training_set[point] = eval
        y.append(eval)

    sigma, l = compute_bayesian_params((points, y), prior)

    n = 0
    posterior = None
    f_m = min(y) # Best minimum of function
    print(f"Starting optimization with sigma={sigma}, l={l}, f_m={f_m}")
    while n <= N_bayes:
        # Step I: Posteerior update now happens at the end of the loop, where also the optimisation parameters sigma, l are updated
        assert len(points) == len(y)
        candidate_points = sample_initial_qaoa_params(len(points), no_layers)
        maximising_point = compute_acquisition_function(prior=prior, f_min=f_m, candidate_points=candidate_points)
        eval = eval_QAOA_circuit(maximising_point, QAOA=QAOA)
        if eval > f_m: #NOTE somehow the paper says to minimise, but we will now be maximising!
            f_m = eval
            print(f"New best energy found: {f_m}")
        #training_set[maximising_point] = eval
        y.append(eval)
        points.append(maximising_point)
        sigma, l = compute_bayesian_params((points, y), prior=prior) # Find new hyperparams for the acquisition function given the updated training set + here also the posterior distribution is updated
        n += 1
        print(f"Finished iteration {n} out of {N_bayes}. Current energy: {f_m}")

    return f_m
