import random
from datetime import datetime
import csv
import json



def random_instance_generator(nodes: int, weights_static: bool, sparse: bool):
    edges = []
    weights = []
    avg_n_edges = random.random() if not sparse else random.uniform(0.8, 0.99) # random threshold for edge generation (directly correlating to the average number of edges generated among the existing vertices)
    for i in range(nodes):
        for j in range(nodes):
            if i != j and random.random() > avg_n_edges:
                if (i, j) in edges or (j, i) in edges:
                    continue
                edges.append((i, j))
                weights.append(random.uniform(1e-10, 1.0) if not weights_static else 1)

    return edges, weights, nodes

def get_edges_in_cut(cut, edges):
    edge_count = 0
    edges_in_cut = []
    print(f'Check the cut: {cut}')
    for i in range(len(cut)):
        for j in range(len(cut)):
            if i != j and cut[i] != cut[j] and (i, j) in edges:
                edge_count += 1
                edges_in_cut.append((i, j))

    return edge_count, edges_in_cut

def save_benchmark_csv(sol_sdp, sol_grb):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    date = datetime.now().strftime("%Y-%m-%d")
    #filename = f'Benchmarks/benchmark_{sol_sdp["n_vertices"]}_{sol_sdp["n_edges"]}_{now}.csv'
    filename = f'benchmarks_{date}.csv'
    merged = {**sol_sdp, **sol_grb, "current time": now}  # dict2 overwrites dict1 if keys overlap

    # If the CSV does not exist yet, write headers
    try:
        with open(filename, "x", newline="") as f:  # 'x' fails if file exists
            writer = csv.DictWriter(f, fieldnames=merged.keys())
            writer.writeheader()
            writer.writerow(merged)
    except FileExistsError:
        with open(filename, "a", newline="") as f:  # append
            writer = csv.DictWriter(f, fieldnames=merged.keys())
            writer.writerow(merged)