"""Tiny benchmark runner.

This is intentionally minimal: it just calls `Main.main_benchmark`.

Examples:
    python benchmark_pipeline.py
"""

from __future__ import annotations

import json
import time
import random

from Main import main_benchmark
from Gurobi_exact_solver import gurobi_maxcut
from SDP_solver import ABCParams
from Utilities import line_instance_generator, save_benchmark_csv, random_instance_generator
from gurobipy import GurobiError
from datetime import datetime
import uuid


MAX_RUNTIME_SECONDS = 600*6  # 2 hours
MAX_VERTICES = 15
MIN_RANDOM_VERTICES = 8
rounds = 1

# --- User configuration section -------------------------------------------------
# Adjust these values to explore different regimes without touching code below.
START_N_VERTICES = 3
PARAMS: ABCParams = {"a": 0, "b": 0, "c": 1}
SPARSE = True
# -------------------------------------------------------------------------------

now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
hash2 = f"{random.randrange(100):02d}"
name = f"roundings_{now}_{hash2}"

all_approx_ratios = []

def run_single_benchmark(n_vertices: int, params: ABCParams, sparse: bool) -> bool:

    if any(v not in (0, 1) for v in params.values()) or sum(params.values()) == 0:
        raise ValueError("--params must be three bits (0/1) and at least one must be 1")

    edges, weights, _ = random_instance_generator(
        nodes=n_vertices,
        weights_static=True,
        sparse=sparse,
    )

    uuid__ = uuid.uuid4().hex

    edge_count, edges_in_cut, cut, M_optimal, cut_variations = main_benchmark(
        n_vertices=n_vertices,
        params=params,
        instance=(edges, weights),
        sparse=sparse,
        benchm_filename=name,
        uuid__= uuid__,
    )
    print("Finished SDP + Rounding benchmark. Benchmarking Gurobi now...")
    try:
        obj, y_sol, z_sol, status = gurobi_maxcut(
            n_vertices,
            edges=edges,
            weights=weights,
            time_limit=1200,
            mip_gap=0.01,
            verbose=True,
        )
    except GurobiError as exc:
        print("Gurobi solver raised an error:")
        print(exc)
        return False

    if status == 2:  # Optimal
        print(f"Gurobi found optimal solution with objective {obj}")
        print(f"SDP rounded cut has {edge_count} edges in cut out of {len(edges)} total edges")
        sol_sdp = {
            "solver": "SDP+GW",
            "edges": json.dumps(edges),
            "params": json.dumps(params),
            "SDPGW_optimal_moment_matrix": json.dumps(M_optimal.tolist()),
            "SDPGW_numberOf_vertices": n_vertices,
            "SDPGW_numberOf_edges": len(edges),
            "SDPGW_params": json.dumps(params),
            "SDPGW_objective": edge_count,
            "SDPGW_edges_in_cut": json.dumps(edges_in_cut),
            "SDPGW_cut_assignment": json.dumps(list(map(int, cut))),
        }

        approx_ratio = (edge_count / obj) if obj not in (0, None) else None

        all_approx_ratios.append(approx_ratio)

        sol_grb = {
            "GRB_solver_exact": "Gurobi",
            "GRB_objective": obj,
            "GRB_status": status,
            "GRB_assignment": json.dumps(list(map(int, y_sol))),
            "Approximation_quality_percent": json.dumps(approx_ratio * 100) if approx_ratio is not None else None,
            "uuid": uuid__,
            "cut_variations": sorted(cut_variations),
            "quality_ratios": sorted({i / obj for i in cut_variations}),
            "minimum_ratio": min({i / obj for i in cut_variations})
        }

        save_benchmark_csv(sol_sdp, sol_grb, f"test_maxcut_1Rounds_{now}")
        return True
    else:
        print(f"Gurobi did not find optimal solution for size {n_vertices} vertices with {len(edges)} edges. Status code: {status}")
        return False


def adaptive_vertex_search(start_n: int, params: ABCParams, sparse: bool) -> tuple[list[tuple[int, bool]], int | None]:
    if MIN_RANDOM_VERTICES > MAX_VERTICES:
        raise ValueError("MIN_RANDOM_VERTICES cannot exceed MAX_VERTICES")

    start_n = max(1, int(start_n))
    start_n = min(MAX_VERTICES, start_n)
    deadline = time.perf_counter() + MAX_RUNTIME_SECONDS
    n_vertices = start_n
    step = max(1, start_n)
    best_success: int | None = None
    history: list[tuple[int, bool]] = []
    iteration = 0
    hit_cap = n_vertices >= MAX_VERTICES
    announced_random_mode = False

    while True:
        current_time = time.perf_counter()
        if current_time >= deadline:
            print("Reached time budget before starting a new iteration; stopping search.")
            break

        iteration += 1
        mode_label = "random-resample" if hit_cap else f"step={step}"
        print(f"\n=== Iteration {iteration}: testing n_vertices={n_vertices} ({mode_label}) ===")
        status = run_single_benchmark(n_vertices=n_vertices, params=params, sparse=sparse)
        history.append((n_vertices, status))

        if status:
            best_success = n_vertices if best_success is None else max(best_success, n_vertices)
            if hit_cap:
                n_vertices = random.randint(MIN_RANDOM_VERTICES, MAX_VERTICES)
            else:
                step = max(1, min(MAX_VERTICES, step * 2))
                next_n = n_vertices + step
                if next_n >= MAX_VERTICES:
                    n_vertices = MAX_VERTICES
                    hit_cap = True
                else:
                    n_vertices = next_n
        else:
            if hit_cap:
                n_vertices = random.randint(MIN_RANDOM_VERTICES, MAX_VERTICES)
            else:
                step = max(1, step // 2)
                n_vertices = max(1, n_vertices - step)
                if n_vertices >= MAX_VERTICES:
                    n_vertices = MAX_VERTICES
                    hit_cap = True

        if time.perf_counter() >= deadline:
            print("Time budget exhausted; ending search after this iteration.")
            break

        if hit_cap and not announced_random_mode:
            print(
                f"Reached MAX_VERTICES={MAX_VERTICES}. Switching to random sampling in the range "
                f"[{MIN_RANDOM_VERTICES}, {MAX_VERTICES}]."
            )
            announced_random_mode = True

        if step == 1 and not status and n_vertices == 1:
            print("Reached minimum vertex count with failure; stopping search.")
            break

    return history, best_success


if __name__ == "__main__":
    history, best_n = adaptive_vertex_search(
        start_n=START_N_VERTICES,
        params=PARAMS,
        sparse=SPARSE,
    )

    print("\n=== Search summary ===")
    for idx, (n_val, status) in enumerate(history, start=1):
        outcome = "success" if status else "failure"
        print(f"Iteration {idx}: n_vertices={n_val} -> {outcome}")

    if best_n is not None:
        print(f"Largest successful vertex count observed: {best_n}")
    else:
        print("No successful runs within the time budget.")

    print(f"Average approximation ratio: {sum(all_approx_ratios) / len(all_approx_ratios)}")
