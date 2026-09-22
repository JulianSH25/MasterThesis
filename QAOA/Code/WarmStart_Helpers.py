import time
import os
from typing import Any
from datetime import datetime
import numpy as np
import sys
from pathlib import Path
import json

# Local imports
from WarmStart import get_warm_start_state
from benchmark_utils import _current_graph_index_from_argv, _graph_index_already_failed

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    os.chdir(project_root)

warm_start_cache_stats: dict[str, Any] = {
    "cache_path": None,
    "cache_used": False,
    "compute_time_seconds": None,
    "load_time_seconds": 0.0,
    "save_time_seconds": 0.0,
    "effective_time_seconds": None,
}

def config_bool(params: dict, key: str, default: bool = False) -> bool:
    value = params.get(key)
    return default if value is None else bool(value)


def resolve_graph_generation_type(params: dict) -> str:
    graph_generation_type = params.get("graph_generation_type")
    assert isinstance(graph_generation_type, str), "graph_generation_type must be a string"
    graph_generation_type = graph_generation_type.lower()
    assert graph_generation_type in ("line", "cycle", "complete", "random", "hog")
    return graph_generation_type


def _normalise_edges_for_hash(edges: list[tuple[int, int]], weights: list[float]) -> list[tuple[int, int, float]]:
    normalised = []
    for (i, j), weight in zip(edges, weights):
        a, b = sorted((int(i), int(j)))
        normalised.append((a, b, float(weight)))
    return sorted(normalised)

def _graph_hash(edges: list[tuple[int, int]], weights: list[float]) -> str:
    payload = json.dumps(_normalise_edges_for_hash(edges, weights), sort_keys=True)
    import hashlib
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

def _expected_warm_start_metadata(
    configured_sdp_seed: int | None,
    benchmark_params: dict,
    edges: list[tuple[int, int]],
    weights: list[float],
    n_vertices: int,
) -> dict[str, Any]:
    warm_start_mode = str(benchmark_params.get("warm_start_mode") or "standard").lower()
    cache_warm_start_mode = "amplified" if warm_start_mode in {"amplified_king", "entangled_king"} else warm_start_mode
    sdp_solver_mode = str(benchmark_params.get("sdp_solver_mode") or "mosek").lower()
    lasserre_level = benchmark_params.get("lasserre_level")
    algorithm17_beta_mode = (
        str(benchmark_params.get("algorithm17_beta_mode") or "fixed").strip().lower()
        if lasserre_level == 2
        else "fixed"
    )
    return {
        "cache_format_version": 2,
        "graph_hash": _graph_hash(edges, weights),
        "n_vertices": int(n_vertices),
        "n_edges": int(len(edges)),
        "sdp_seed": configured_sdp_seed,
        "lasserre_level": lasserre_level,
        "initial_solver_level_M": benchmark_params.get("initial_solver_level_M"),
        "warm_start_mode": cache_warm_start_mode,
        "parameter_vector": list(benchmark_params.get("parameter_vector") or []),
        "sdp_solver_mode": sdp_solver_mode,
        "sdp_scs_eps": float(benchmark_params["sdp_scs_eps"]) if sdp_solver_mode == "scs" else None,
        "sdp_scs_max_iters": int(benchmark_params["sdp_scs_max_iters"]) if sdp_solver_mode == "scs" else None,
        "algorithm17_beta_mode": algorithm17_beta_mode,
    }

def _metadata_matches(found: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, expected_value in expected.items():
        found_value = found.get(key)

        if key == "warm_start_mode":
            found_mode = str(found_value or "standard").lower()
            expected_mode = str(expected_value or "standard").lower()
            if (
                found_mode != expected_mode
                and not (
                    found_mode == "amplified"
                    and expected_mode in {"standard", "entangled", "amplified_king", "entangled_king"}
                )
            ):
                return False
            continue

        if key == "algorithm17_beta_mode":
            found_mode = str(found_value or "fixed").strip().lower()
            expected_mode = str(expected_value or "fixed").strip().lower()
            if found_mode != expected_mode:
                return False
            continue

        if found_value != expected_value:
            return False
    return True

def _json_sanitise(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_sanitise(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitise(val) for val in value]
    if isinstance(value, np.ndarray):
        return _json_sanitise(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    return value

def _warm_start_mode_needs_moment_matrix(benchmark_params: dict) -> bool:
    warm_start_mode = str(benchmark_params.get("warm_start_mode") or "standard").lower()
    return (
        warm_start_mode in {"amplified", "entangled", "amplified_king", "entangled_king"}
        or config_bool(benchmark_params, "use_correlations_as_initial_params")
    )

def _save_warm_start_cache(
    cache_path: Path,
    metadata: dict[str, Any],
    initial_state: Any,
    product_states: Any,
    classical_cut: Any,
    moment_matrix: Any,
    warm_start_result: Any,
    compute_time_seconds: float,
    store_moment_matrix: bool,
) -> float:
    save_start = time.time()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    payload_metadata = dict(metadata)
    if isinstance(warm_start_result, dict) and warm_start_result.get("sdp_seed_used") is not None:
        payload_metadata["sdp_seed_used"] = int(warm_start_result["sdp_seed_used"])
    payload_metadata["warm_start_compute_time_seconds"] = float(compute_time_seconds)
    payload_metadata["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    has_moment_matrix = store_moment_matrix and moment_matrix is not None
    moment_matrix_array = np.asarray(moment_matrix) if has_moment_matrix else np.array([])

    if has_moment_matrix:
        print(
            "Warm-start cache: moment matrix before save "
            f"shape={moment_matrix_array.shape}, "
            f"dtype={moment_matrix_array.dtype}, "
            f"entries={moment_matrix_array.size}, "
            f"bytes={moment_matrix_array.nbytes} "
            f"({moment_matrix_array.nbytes / (1024 ** 2):.2f} MiB)",
            flush=True,
        )
    else:
        print("Warm-start cache: moment matrix not saved for this configuration.", flush=True)

    with open(tmp_path, "wb") as file:
        np.savez(
            file,
            metadata=json.dumps(payload_metadata),
            initial_state=np.asarray(initial_state, dtype=complex) if initial_state is not None else np.array([], dtype=complex),
            has_initial_state=np.array(initial_state is not None),
            product_states=np.asarray(product_states, dtype=complex) if product_states is not None else np.array([], dtype=complex),
            has_product_states=np.array(product_states is not None),
            classical_cut=np.asarray(classical_cut) if classical_cut is not None else np.array([]),
            has_classical_cut=np.array(classical_cut is not None),
            moment_matrix=moment_matrix_array,
            has_moment_matrix=np.array(has_moment_matrix),
            warm_start_result_json=json.dumps(_json_sanitise(warm_start_result or {})),
        )
    tmp_path.replace(cache_path)
    return time.time() - save_start

def _load_warm_start_cache(cache_path: Path, expected_metadata: dict[str, Any]):
    load_start = time.time()
    with np.load(cache_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if not _metadata_matches(metadata, expected_metadata):
            raise ValueError(
                f"Warm-start cache metadata mismatch for {cache_path}. "
                f"Expected {expected_metadata}, found {metadata}."
            )
        initial_state = data["initial_state"] if bool(data["has_initial_state"]) else None
        product_states = data["product_states"] if bool(data["has_product_states"]) else None
        classical_cut = data["classical_cut"] if bool(data["has_classical_cut"]) else None
        moment_matrix = data["moment_matrix"] if bool(data["has_moment_matrix"]) else None
        warm_start_result = json.loads(str(data["warm_start_result_json"]))
        compute_time_seconds = metadata.get("warm_start_compute_time_seconds")
    load_time_seconds = time.time() - load_start
    return (initial_state, product_states, classical_cut), moment_matrix, warm_start_result, compute_time_seconds, load_time_seconds


def get_or_create_cached_warm_start(
    configured_sdp_seed: int | None,
    benchmark_params: dict,
    edges: list[tuple[int, int]],
    weights: list[float],
    n_vertices: int,
):
    global warm_start_cache_stats

    cache_path_raw = os.environ.get("WARM_START_CACHE_PATH")
    current_graph_index = _current_graph_index_from_argv()
    if _graph_index_already_failed(cache_path_raw, current_graph_index):
        print(
            f"Warm-start for graph/index {current_graph_index} was already marked as failed. "
            "Exiting early before SDP construction/solve."
        )
        raise RuntimeError("WARM START GENERATION FAILED EARLIER FOR THIS INSTANCE - ABORTING")
    expected_metadata = _expected_warm_start_metadata(
        configured_sdp_seed,
        benchmark_params,
        edges,
        weights,
        n_vertices,
    )
    warm_start_cache_stats = {
        "cache_path": cache_path_raw,
        "cache_used": False,
        "compute_time_seconds": None,
        "load_time_seconds": 0.0,
        "save_time_seconds": 0.0,
        "effective_time_seconds": None,
    }

    if cache_path_raw:
        cache_path = Path(cache_path_raw)
        if not cache_path.exists():
            expected_mode = str(benchmark_params.get("warm_start_mode") or "standard").lower()
            if expected_mode in {"standard", "entangled", "amplified_king", "entangled_king"}:
                expected_suffix = f"_{expected_mode}.npz"
                if cache_path.name.endswith(expected_suffix):
                    amplified_cache_path = cache_path.with_name(
                        cache_path.name[:-len(expected_suffix)] + "_amplified.npz"
                    )
                    if amplified_cache_path.exists():
                        print(
                            "Warm-start cache: requested cache missing; "
                            f"using compatible amplified cache {amplified_cache_path}",
                            flush=True,
                        )
                        cache_path = amplified_cache_path
                        warm_start_cache_stats["cache_path"] = str(cache_path)

        if cache_path.exists():
            print(f"Warm-start cache: loading {cache_path}")
            warm_start_data, moment_matrix, warm_start_result, compute_time_seconds, load_time_seconds = _load_warm_start_cache(cache_path, expected_metadata)
            warm_start_cache_stats.update({
                "cache_used": True,
                "compute_time_seconds": compute_time_seconds,
                "load_time_seconds": load_time_seconds,
                "effective_time_seconds": compute_time_seconds,
            })
            print(f"Warm-start cache: loaded in {load_time_seconds:.4f} seconds; effective compute time={compute_time_seconds}")
            return warm_start_data, moment_matrix, warm_start_result

    compute_start = time.time()
    try:
        warm_start_data, moment_matrix, warm_start_result = get_warm_start_state((edges, weights), n_vertices)
    except Exception as exc:
        if cache_path_raw:
            failed_path = Path(cache_path_raw + ".failed")
            failed_path.parent.mkdir(parents=True, exist_ok=True)
            failed_payload = dict(expected_metadata)
            failed_payload.update({
                "failed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": repr(exc),
            })
            failed_path.write_text(json.dumps(failed_payload, indent=2), encoding="utf-8")
            print(f"Warm-start failure metadata written to: {failed_path}")
        print(f"Warm-start generation failed during Python execution: {exc!r}")
        raise

    compute_time_seconds = time.time() - compute_start
    warm_start_cache_stats.update({
        "cache_used": False,
        "compute_time_seconds": compute_time_seconds,
        "effective_time_seconds": compute_time_seconds,
    })

    if cache_path_raw:
        cache_path = Path(cache_path_raw)
        save_time_seconds = _save_warm_start_cache(
            cache_path=cache_path,
            metadata=expected_metadata,
            initial_state=warm_start_data[0],
            product_states=warm_start_data[1],
            classical_cut=warm_start_data[2],
            moment_matrix=moment_matrix,
            warm_start_result=warm_start_result,
            compute_time_seconds=compute_time_seconds,
            store_moment_matrix=_warm_start_mode_needs_moment_matrix(benchmark_params),
        )
        warm_start_cache_stats["save_time_seconds"] = save_time_seconds
        print(f"Warm-start cache: saved {cache_path} in {save_time_seconds:.4f} seconds")

    return warm_start_data, moment_matrix, warm_start_result
