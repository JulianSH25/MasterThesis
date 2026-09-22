"""Create stable identifiers for benchmark configurations.

The launcher uses the resulting hash to identify equivalent benchmark rows
without relying on the full serialized configuration string.
"""

import json
import sys
import hashlib

"""Compute a unique run key based on the configuration and whitelist of parameters for benchmarking QAOA warm-starts; somewhat outdated, was mainly used for manual result mapping to check correctness of the benchmarking pipeline (easier lookup via key instead of full config)."""

def normalize(v):
    """Serialise one configuration value deterministically where possible.

    Args:
        v: Arbitrary configuration value.

    Returns:
        A stable string representation suitable for inclusion in a run key.
    """
    if v is None:
        return ""
    try:
        return json.dumps(v, sort_keys=True)
    except TypeError:
        return str(v).strip()

def compute_m(n, graph_type):
    """Return the deterministic edge count for a named graph family.

    Args:
        n: Number of graph vertices.
        graph_type: Supported generated family name.

    Returns:
        Its edge count, or ``"n/a"`` for non-deterministic data sources.

    Raises:
        ValueError: If ``graph_type`` is unsupported.
    """
    if graph_type == "line":
        return n - 1
    elif graph_type == "cycle":
        return n
    elif graph_type == "complete":
        return n * (n - 1) // 2
    elif graph_type == "random":
        return "n/a"
    elif graph_type == "HOG":
        return "n/a"
    else:
        raise ValueError("Unknown graph type: " + graph_type + "; expected 'line', 'cycle', 'complete', 'random', or 'HOG'.")

def main():
    """Read command-line benchmark data and print its SHA-1 run key.

    The positional arguments are the configuration file, whitelist file,
    number of vertices, QAOA depth, and number of optimiser iterations.
    """
    config_file = sys.argv[1]
    whitelist_file = sys.argv[2]
    n = int(sys.argv[3])
    p = int(sys.argv[4])
    iterations = int(sys.argv[5])

    with open(config_file) as f:
        config = json.load(f)

    with open(whitelist_file) as f:
        whitelist = json.load(f)["whitelist"]

    params = dict(config)
    params["n"] = n
    params["p"] = p
    params["precision/iterations"] = iterations

    if "m" in whitelist:
        params["m"] = compute_m(n, config.get("graph_generation_type"))

    key_string = "|".join(
        f"{k}={normalize(params.get(k))}" for k in whitelist
    )

    run_key = hashlib.sha1(key_string.encode()).hexdigest()
    print(run_key)

if __name__ == "__main__":
    main()
