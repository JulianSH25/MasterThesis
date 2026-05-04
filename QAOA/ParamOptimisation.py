import os

import numpy as np
from numpy import pi
import random, itertools
from qiskit import QuantumCircuit
from scipy.optimize import minimize
from scipy.stats import norm
import time
from joblib import Parallel, delayed

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
debug_allInfo = True
idx_counter = 1

Energies: list[float] = [] # Used to store energy values from each QAOA iteration; Used in eval_QAOA_circuit (right below)
Optimisation_time: list[float] = [] # Used to store time taken for each evaluation of the objective function during ADAM & COBYLA optimisation;

def eval_QAOA_circuit(point: list[np.ndarray], QAOA: QAOACircuit) -> float:
    # Step 2
    """This function receives a set of points, i.e. QAOA parameters, and evaluates the actual QAOA circuit on those parameters, returns the QAOA value found"""

    QAOA.bind_circuit_parameters(parameters=point)
    statevec = QAOA.run_circuit(return_statevector=True)
    E = QAOA.compute_energy_from_statevector(statevec)
    QAOA.debug_previous_result = E
    Energies.append(E)
    if debug and debug_allInfo:
        print(f"Debugging info for idx {idx_counter}:")
        print(f"Statevector: {statevec}")
        #idx_counter += 1g
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
        parameters = []
        for param_type_idx in range(QAOA.no_param_types):
            # Extract bounds for this specific parameter type
            if isinstance(QAOA.param_ranges, list):
                bounds = QAOA.param_ranges[param_type_idx] if param_type_idx < len(QAOA.param_ranges) else QAOA.param_ranges[-1]
            else:
                bounds = QAOA.param_ranges
            parameters.append(set_random_params(QAOA.p, bounds, init_close_to_zero=init_close_to_zero))
        print(f"Using random initial parameters with {QAOA.no_param_types} parameter types")

        if use_corr_init and correlations is not None:
            corr_scalar = float(np.mean(list(correlations.values())))
            corr_scalar = float(np.clip(corr_scalar, -3.0, 3.0))
            gamma0 = np.pi * (3.0 - corr_scalar) / 6.0
            gamma = np.full(no_layers, gamma0, dtype=float)
            print(f"Using correlation-based initial parameters with scalar {corr_scalar} and gamma0 {gamma}")
        elif use_corr_init and correlations is None:
            raise ValueError("correlations must be provided if use_correlations_as_initial_params is True")

        x0 = np.concatenate(parameters).astype(float)
    else:
        x0 = np.asarray(x0, dtype=float)
        expected_dim = QAOA.p * QAOA.no_param_types
        if x0.shape != (expected_dim,):
            raise ValueError(f"Expected x0 shape ({expected_dim},), got {x0.shape}")
        print(f"Using provided initial parameters x0 with shape {x0.shape}")

    def objective(x0: np.ndarray) -> float:
        start = time.time() # XXX Time
        gamma_vals = x0[:no_layers]
        beta_vals = x0[no_layers:]
        energy = -eval_QAOA_circuit(x0, QAOA)

        Optimisation_time.append(time.time() - start) # XXX Time

        return energy

    result = minimize(objective, x0=x0, method="COBYLA", options={"maxiter": max_iter})
    setattr(result, "initial_point", x0.copy())
    print(f"Energies observed during COBYLA optimization: {Energies}")
    print(f"Total optimisation time observed during COBYLA optimization: {sum(Optimisation_time):.6f} seconds")
    print(f"Median time per evaluation during COBYLA optimization: {np.median(Optimisation_time):.6f} seconds")
    return result
    # return minimize(objective, x0=x0, method="L-BFGS-B", options={"maxiter": max_iter})

######
# NOTE this is only a wrapper function for the adam optimiser. The actual optimisation happens below in '_adam_optimiser'
def optimise_adam(
    QAOA: QAOACircuit,
    no_layers: int,
    steps: int,
    learning_rate: float = 0.05,
    x0: np.ndarray | None = None,
):
    benchmark_params: dict = get_benchmark_params()
    if benchmark_params.get("optimiser_use_heuristic"):
        x0 = heuristic_optimiser(QAOA, no_layers, learning_rate=learning_rate)
    best_result_obj = None
    print("Running ADAM optimization with a single iteration (no debug loop)")
    best_result_obj = _adam_optimiser(QAOA, no_layers, steps=steps, learning_rate=learning_rate, x0=x0)
    return best_result_obj

# NOTE same as optimise_adam but iterates many times
def heuristic_optimiser(
        QAOA: QAOACircuit,
    no_layers: int,
    learning_rate: float = 0.05,
    ):
    benchmark_params: dict = get_benchmark_params()
    best_result_obj = None
    steps = benchmark_params.get("heuristic_optimiser_iterations")
    
    # Iterate over many runs of the ADAM optmiser with different randmly initialised parameters.
    worst_result, best_result_value = float("inf"), float("-inf")
    best_results_log: dict = {}
    points: dict = {}
    best_point = None
    for idx in range(benchmark_params.get("heuristic_optimiser_sampleSize")):
        print(f"Debug iteration {idx+1}/{benchmark_params.get('heuristic_optimiser_sampleSize', 1)}")
        result, point = _adam_optimiser(QAOA, no_layers, steps=steps, learning_rate=learning_rate, x0=None)
        # call adam optimiser many times with a small number of steps to test different random initial parameters (sampled in adam optimiser) and return the best point (i.e. best parameters) to optimise further with Adam but this time more steps
        points[idx] = point
        if -result.fun < worst_result:
            worst_result = -result.fun
        if -result.fun > best_result_value:
            best_result_value = -result.fun
            best_result_obj = result
            best_point = point
        best_results_log[idx] = -result.fun
    print(f"ADAM optimization heuristic: best_result={best_result_value}, worst_result={worst_result}")
    if benchmark_params.get("debug"):
        print(f"ADAM optimization heuristic: all results observed across iterations: {best_results_log}")
        print(f"ADAM optimization heuristic: best point found across iterations: {best_point}")
        print(f"Improvement over worst result: {best_result_value - worst_result}")
        print(f"Improvement over worst result (percentage): {(best_result_value - worst_result) / abs(worst_result) * 100:.5f}%")
    return best_point


def _adam_optimiser(
    QAOA: QAOACircuit,
    no_layers: int,
    steps: int,
    learning_rate: float = 0.05,
    x0: np.ndarray | None = None,
):
    assert isinstance(QAOA, QAOACircuit)
    optimiser = ADAM(maxiter=steps, lr=learning_rate)
    print(f"ADAM optimizer configured with maxiter={steps} and learning_rate={learning_rate}")
    optimiser.set_max_evals_grouped(QAOA.p * QAOA.no_param_types)

    if x0 is None:
        init_close_to_zero = get_benchmark_params()["init_QAOAparams_close_to_zero"]
        parameters = []
        for param_type_idx in range(QAOA.no_param_types):
            # Extract bounds for this specific parameter type
            if isinstance(QAOA.param_ranges, list):
                bounds = QAOA.param_ranges[param_type_idx] if param_type_idx < len(QAOA.param_ranges) else QAOA.param_ranges[-1]
            else:
                bounds = QAOA.param_ranges
            parameters.append(set_random_params(QAOA.p, bounds, init_close_to_zero=init_close_to_zero))
        print(f"Generated {QAOA.no_param_types} random initial parameter sets for ADAM optimization")
        x0 = np.concatenate(parameters).astype(float)
        print(f"Using random initial parameters with:")
        idx = 1
        for parameter_type in parameters:
            print(f"Param_type {idx}:  {parameter_type}")
            idx += 1
    else:
        x0 = np.asarray(x0, dtype=float)
        expected_dim = QAOA.p * QAOA.no_param_types
        if x0.shape != (expected_dim,):
            raise ValueError(f"Expected x0 shape ({expected_dim},), got {x0.shape}")
        print(f"Using provided initial parameters x0 with shape {x0.shape}")

    dimension = QAOA.p * QAOA.no_param_types

    def objective_single(x0: np.ndarray) -> float:
        # Split flat array into grouped parameters matching circuit structure
        x0_grouped = []
        offset = 0
        for _ in range(QAOA.no_param_types):
            x0_grouped.append(x0[offset:offset + QAOA.p])
            offset += QAOA.p
        value = eval_QAOA_circuit(x0_grouped, QAOA)
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

        batch_count = theta.size // dimension
        thread_env_values = []
        for env_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            try:
                thread_env_values.append(int(os.environ.get(env_name, "0") or 0))
            except ValueError:
                thread_env_values.append(0)

        n_jobs = int(os.environ.get("QAOA_JOBLIB_N_JOBS", "0") or 0)
        if n_jobs <= 0:
            if thread_env_values and max(thread_env_values) > 1:
                n_jobs = 1
            else:
                n_jobs = os.cpu_count() or 1
        n_jobs = max(1, min(n_jobs, batch_count))

        values = Parallel(n_jobs=n_jobs, prefer="threads", require="sharedmem")(
            delayed(objective_single)(theta[i:i + dimension])
            for i in range(0, theta.size, dimension)
        )
        #idx_counter += n_jobs
        energy = np.array(values, dtype=float)
        Optimisation_time.append(time.time() - start) # XXX Time
        return energy

    result = optimiser.minimize(fun=objective, x0=x0)
    setattr(result, "initial_point", x0.copy())
    print(f"Energies observed during ADAM optimization: {Energies}")
    print(f"Total optimisation time observed during ADAM optimization: {sum(Optimisation_time):.6f} seconds")
    print(f"Median time per evaluation during ADAM optimization: {np.median(Optimisation_time):.6f} seconds")
    return result, x0


def grid_search(QAOA: QAOACircuit, no_layers: int, precision: float, shuffle: bool = False):
    group_lengths = [len(parameter_group) for parameter_group in QAOA.qaoa_parameters]
    if not group_lengths:
        raise ValueError("QAOA.qaoa_parameters must be populated before grid search")

    if precision <= 0:
        raise ValueError("precision must be positive")
    if any(length <= 0 for length in group_lengths):
        raise ValueError("all group lengths must be positive")

    if no_layers is not None and any(length != no_layers for length in group_lengths):
        print(
            f"Grid search received no_layers={no_layers}, but QAOA.qaoa_parameters has group lengths {group_lengths}. "
            f"Using the circuit-defined lengths instead."
        )

    if isinstance(QAOA.param_ranges, list):
        if len(QAOA.param_ranges) == 0:
            raise ValueError("QAOA.param_ranges must not be empty when provided as a list")
        parameter_ranges = list(QAOA.param_ranges)
        if len(parameter_ranges) < len(group_lengths):
            parameter_ranges.extend([parameter_ranges[-1]] * (len(group_lengths) - len(parameter_ranges)))
        parameter_ranges = parameter_ranges[:len(group_lengths)]
    else:
        parameter_ranges = [QAOA.param_ranges] * len(group_lengths)

    value_spaces = [np.arange(lb, ub, precision, dtype=float) for lb, ub in parameter_ranges]
    parameter_axes = [value_spaces[group_index] for group_index, length in enumerate(group_lengths) for _ in range(length)]

    def parameter_stream():
        for params in itertools.product(*parameter_axes):
            offset = 0
            grouped_params = []
            for length in group_lengths:
                grouped_params.append(np.array(params[offset:offset + length], dtype=float))
                offset += length
            yield tuple(grouped_params)

    total_parameters = 1
    for bounds, length in zip(parameter_ranges, group_lengths):
        axis_size = len(np.arange(bounds[0], bounds[1], precision, dtype=float))
        total_parameters *= axis_size ** length

    y: list[float] = []

    n = 1
    f_m = 0  # Best minimum of function
    print(f"Starting optimization with f_m={f_m}")
    
    for point in parameter_stream():
        # Step I: Posteerior update now happens at the end of the loop, where also the optimisation parameters sigma, l are updated
        #assert len(points) == len(y)
        eval = eval_QAOA_circuit(point, QAOA=QAOA)
        if eval > f_m:  # NOTE somehow the paper says to minimise, but we will now be maximising!
            f_m = eval
            print(f"New best energy found: {f_m} in iteration {n} out of {total_parameters}")
        # training_set[maximising_point] = eval
        y.append(eval)
        if n % 1000 == 0 or n == total_parameters:
            print(f"Status Report: Finished iteration {n} out of {total_parameters}. Current energy: {f_m}. Timestamp: {time.strftime('%H:%M:%S', time.gmtime(time.time()))}")
        n += 1

    return f_m

if __name__ == "__main__":
    print("ParamOptimisation grid search helpers are intended to be used via Main.py.")
