import numpy as np
from numpy import pi
import random, itertools
from qiskit import QuantumCircuit
from scipy.optimize import minimize
from scipy.stats import norm
import time

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from utils import set_random_params, sample_initial_qaoa_params, get_benchmark_params

from Circuit import QAOACircuit

from qiskit_algorithms.optimizers import ADAM

"""def optimise_params(qc: QuantumCircuit, gammas, betas, n: int, edges: np.ndarray, weights, params, x0: np.ndarray | None = None):

    p = len(gammas)
    if x0 is None:
        gamma_0, beta_0 = set_random_params(p)
        x0 = np.concatenate([gamma_0, beta_0])\

    def objective(x: np.ndarray) -> float:
        gamma_vals, beta_vals = x[:p], x[p:]
        qc_bound"""

benchmark_params: dict = get_benchmark_params()
parameters = benchmark_params["parameter_vector"]

debug = benchmark_params["debug"]

Energies: list[float] = [] # Used to store energy values from each QAOA iteration; Used in eval_QAOA_circuit (right below)
Optimisation_time: list[float] = [] # Used to store time taken for each evaluation of the objective function during ADAM & COBYLA optimisation;

def eval_QAOA_circuit(point: tuple[np.ndarray, np.ndarray], QAOA: QAOACircuit) -> float:
    # Step 2
    """This function receives a set of points, i.e. QAOA parameters, and evaluates the actual QAOA circuit on those parameters, returns the QAOA value found"""

    QAOA.bind_circuit_parameters(gamma_values=point[0], beta_values=point[1])
    statevec = QAOA.run_circuit(return_statevector=True)
    E = QAOA.compute_energy_from_statevector(statevec)
    Energies.append(E)
    return E

class BayesianOptimiser:
    @staticmethod
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

    def expected_improvement(self, prior, X_candidates: np.ndarray, f_min: float):
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

    def compute_acquisition_function(self, prior, f_min, candidate_points=None):
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

        EI = self.expected_improvement(prior, X, f_min)
        best_index = int(np.argmax(EI))
        return candidate_points[best_index]

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

    def bayesian_optimisation(self, QAOA: QAOACircuit, N_bayes: float, no_layers: int, points: list[tuple] = None):
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

        prior = self.build_gp_prior()

        #training_set: dict = {}
        y: list[float] = [] # keep separate list of observations for convenience
        for point in points:
            eval = eval_QAOA_circuit(point, QAOA)
            #training_set[point] = eval
            y.append(eval)

        sigma, l = self.compute_bayesian_params((points, y), prior)

        n = 0
        posterior = None
        f_m = min(y) # Best minimum of function
        print(f"Starting optimization with sigma={sigma}, l={l}, f_m={f_m}")
        while n <= N_bayes:
            # Step I: Posteerior update now happens at the end of the loop, where also the optimisation parameters sigma, l are updated
            assert len(points) == len(y)
            candidate_points = sample_initial_qaoa_params(len(points), no_layers)
            maximising_point = self.compute_acquisition_function(prior=prior, f_min=f_m, candidate_points=candidate_points)
            eval = eval_QAOA_circuit(maximising_point, QAOA=QAOA)
            if eval > f_m: #NOTE somehow the paper says to minimise, but we will now be maximising!
                f_m = eval
                print(f"New best energy found: {f_m}")
            #training_set[maximising_point] = eval
            y.append(eval)
            points.append(maximising_point)
            sigma, l = self.compute_bayesian_params((points, y), prior=prior) # Find new hyperparams for the acquisition function given the updated training set + here also the posterior distribution is updated
            n += 1
            print(f"Finished iteration {n} out of {N_bayes}. Current energy: {f_m}")

        return f_m

def optimise_cobyla(
    QAOA: QAOACircuit,
    no_layers: int,
    max_iter: int = 1000,
    correlations: int = None,
    x0: np.ndarray | None = None,
):
    """
    NOTE: ALTERNATIVE OPTIMISATION FUNCTION; this one is standalone, in the sense that all the other methods in this file are only for the Bayesian optimisation, but this one is a separate method that can be used to optimise QAOA parameters using COBYLA instead of Bayesian optimisation.
    :param no_layers: number of QAOA layers
    :param max_iter: maximum number of COBYLA iterations
    :return: scipy optimisation result object
    """
    assert isinstance(QAOA, QAOACircuit)
    init_close_to_zero = get_benchmark_params()["init_QAOAparams_close_to_zero"]
    use_corr_init = benchmark_params["use_correlations_as_initial_params"]

    if x0 is None:
        gamma, beta = set_random_params(no_layers, init_close_to_zero=init_close_to_zero)
        print(f"Using random initial parameters with gamma {gamma} and beta {beta}")

        if use_corr_init and correlations is not None:
            corr_scalar = float(np.mean(list(correlations.values())))
            corr_scalar = float(np.clip(corr_scalar, -3.0, 3.0))
            gamma0 = np.pi * (3.0 - corr_scalar) / 6.0
            gamma = np.full(no_layers, gamma0, dtype=float)
            print(f"Using correlation-based initial parameters with scalar {corr_scalar} and gamma0 {gamma}")
        elif use_corr_init and correlations is None:
            raise ValueError("correlations must be provided if use_correlations_as_initial_params is True")

        x0 = np.concatenate([gamma, beta])
    else:
        x0 = np.asarray(x0, dtype=float)
        expected_dim = 2 * no_layers
        if x0.shape != (expected_dim,):
            raise ValueError(f"Expected x0 shape ({expected_dim},), got {x0.shape}")
        print(f"Using provided initial parameters x0 with shape {x0.shape}")

    def objective(theta: np.ndarray) -> float:
        start = time.time() # XXX Time
        gamma_vals = theta[:no_layers]
        beta_vals = theta[no_layers:]
        energy = -eval_QAOA_circuit((gamma_vals, beta_vals), QAOA)

        Optimisation_time.append(time.time() - start) # XXX Time

        return energy

    result = minimize(objective, x0=x0, method="COBYLA", options={"maxiter": max_iter})
    setattr(result, "initial_point", x0.copy())
    print(f"Energies observed during COBYLA optimization: {Energies}")
    print(f"Total optimisation time observed during COBYLA optimization: {sum(Optimisation_time):.6f} seconds")
    print(f"Median time per evaluation during COBYLA optimization: {np.median(Optimisation_time):.6f} seconds")
    return result
    # return minimize(objective, x0=x0, method="L-BFGS-B", options={"maxiter": max_iter})

def optimise_adam(
    QAOA: QAOACircuit,
    no_layers: int,
    steps: int = 300,
    learning_rate: float = 0.05,
    x0: np.ndarray | None = None,
):
    assert isinstance(QAOA, QAOACircuit)

    optimiser = ADAM(maxiter=steps, lr=learning_rate)
    print(f"ADAM optimizer configured with maxiter={steps} and learning_rate={learning_rate}")
    optimiser.set_max_evals_grouped(2 * no_layers)

    if x0 is None:
        init_close_to_zero = get_benchmark_params()["init_QAOAparams_close_to_zero"]
        gamma, beta = set_random_params(no_layers, init_close_to_zero=init_close_to_zero)
        x0 = np.concatenate([gamma, beta]).astype(float)
        print(f"Using random initial parameters with gamma {gamma} and beta {beta}")
    else:
        x0 = np.asarray(x0, dtype=float)
        expected_dim = 2 * no_layers
        if x0.shape != (expected_dim,):
            raise ValueError(f"Expected x0 shape ({expected_dim},), got {x0.shape}")
        print(f"Using provided initial parameters x0 with shape {x0.shape}")

    dimension = 2 * no_layers

    def objective_single(theta_single: np.ndarray) -> float:
        gamma_vals = theta_single[:no_layers]
        beta_vals = theta_single[no_layers:]
        value = eval_QAOA_circuit((gamma_vals, beta_vals), QAOA)
        if debug:
            print(f"Eval: {value}, negated: {-value}")
        return -value

    def objective(theta: np.ndarray) -> float:
        start = time.time() # XXX Time
        theta = np.asarray(theta, dtype=float)

        if theta.ndim != 1:
            raise ValueError(f"Expected 1D parameter array, got shape {theta.shape}")

        if theta.size == dimension:
            return objective_single(theta)

        if theta.size % dimension != 0:
            raise ValueError(
                f"Grouped ADAM evaluation received invalid size {theta.size} for dimension {dimension}"
            )

        values = [
            objective_single(theta[i:i + dimension])
            for i in range(0, theta.size, dimension)
        ]
        energy = np.array(values, dtype=float)
        Optimisation_time.append(time.time() - start) # XXX Time
        return energy

    result = optimiser.minimize(fun=objective, x0=x0)
    setattr(result, "initial_point", x0.copy())
    print(f"Energies observed during COBYLA optimization: {Energies}")
    print(f"Total optimisation time observed during ADAM optimization: {sum(Optimisation_time):.6f} seconds")
    print(f"Median time per evaluation during ADAM optimization: {np.median(Optimisation_time):.6f} seconds")
    return result


def grid_search_parameters(
    precision: float,
    p: int,
    shuffle: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if precision <= 0:
        raise ValueError("precision must be positive")
    if p <= 0:
        raise ValueError("p must be positive")

    gamma_values = np.arange(0, 2 * pi, precision, dtype=float)
    beta_values = np.arange(0, pi, precision, dtype=float)

    full_grid = [gamma_values] * p + [beta_values] * p

    parameters: list[tuple[np.ndarray, np.ndarray]] = []

    for params in itertools.product(*full_grid):
        gammas = np.array(params[:p], dtype=float)
        betas = np.array(params[p:], dtype=float)
        parameters.append((gammas, betas))

    if shuffle:
        random.shuffle(parameters) # Just for fun

    return parameters

def grid_search(QAOA: QAOACircuit, no_layers: int, precision: float):
    parameters_grid = grid_search_parameters(precision, no_layers)

    y: list[float] = []

    n = 1
    f_m = 0  # Best minimum of function
    print(f"Starting optimization with f_m={f_m}")
    for point in parameters_grid:
        # Step I: Posteerior update now happens at the end of the loop, where also the optimisation parameters sigma, l are updated
        #assert len(points) == len(y)
        eval = eval_QAOA_circuit(point, QAOA=QAOA)
        if eval > f_m:  # NOTE somehow the paper says to minimise, but we will now be maximising!
            f_m = eval
            print(f"New best energy found: {f_m} in iteration {n} out of {len(parameters_grid)}")
        # training_set[maximising_point] = eval
        y.append(eval)
        if n % 1000 == 0 or n == len(parameters_grid):
            print(f"Status Report: Finished iteration {n} out of {len(parameters_grid)}. Current energy: {f_m}. Timestamp: {time.strftime('%H:%M:%S', time.gmtime(time.time()))}")
        n += 1

    return f_m

if __name__ == "__main__":
    params = grid_search_parameters(0.5, 1, shuffle=False)
    print(len(params))
    print(params)
