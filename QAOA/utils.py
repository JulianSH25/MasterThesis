import numpy as np
import random

def set_random_params(p: int, seed: int | None = None):
    """Rather pointless method to generate random parameters for the QAOA circuit. Mainly used for initial testing"""
    rng = np.random.default_rng(seed)
    gamma_values = rng.uniform(0.0, 2*np.pi, size=p)
    beta_values = rng.uniform(0.0, np.pi, size=p)

    return gamma_values, beta_values

def sample_initial_qaoa_params(n: int, p: int) -> list[tuple[list[float], list[float]]]:
    return [
        (
            [random.uniform(0, 3.141592653589793) for _ in range(p)],
            [random.uniform(0, 3.141592653589793 / 2) for _ in range(p)],
        )
        for _ in range(n)
    ]