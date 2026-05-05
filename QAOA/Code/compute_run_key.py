import json
import sys
import hashlib

def normalize(v):
    if v is None:
        return ""
    try:
        return json.dumps(v, sort_keys=True)
    except TypeError:
        return str(v).strip()

def compute_m(n, graph_type):
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