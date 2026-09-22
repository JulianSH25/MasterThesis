"""Small helpers shared by QAOA benchmark workers.

These functions locate the per-run marker recording graph instances whose warm
start failed, so repeated workers can skip known failures cleanly.
"""

from pathlib import Path
import sys

"""Auxiliary functions for benchmarking QAOA warm-starts"""

def _failed_graph_index_file_from_cache_path(cache_path_raw: str | None) -> Path | None:
    """Derive the failure-marker path associated with a warm-start cache.

    Args:
        cache_path_raw: Configured cache path, if persistent caching is active.

    Returns:
        The run-local failure-marker path, or ``None`` without a cache path.
    """
    if not cache_path_raw:
        return None
    cache_path = Path(cache_path_raw)
    run_dir = cache_path.parent.parent
    run_tag = run_dir.name
    return run_dir / "failed_warm_starts" / f"failed_warm_start_graph_indices_{run_tag}.txt"


def _current_graph_index_from_argv() -> int | None:
    """Read the benchmark graph index from the worker command line.

    Returns:
        The integer graph index passed by the launcher, or ``None`` when the
        current invocation does not follow the benchmark-worker convention.
    """
    try:
        if len(sys.argv) >= 4:
            return int(sys.argv[3])
    except Exception:
        return None
    return None


def _graph_index_already_failed(cache_path_raw: str | None, graph_index: int | None) -> bool:
    """Check whether a graph instance was previously marked as a warm-start failure.

    Args:
        cache_path_raw: Configured persistent warm-start cache path.
        graph_index: Index of the graph considered by the current worker.

    Returns:
        ``True`` only when the run-local marker file lists ``graph_index``.
    """
    failed_file = _failed_graph_index_file_from_cache_path(cache_path_raw)
    if failed_file is None or graph_index is None or not failed_file.exists():
        return False

    try:
        failed_indices = {
            line.strip()
            for line in failed_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    except Exception as exc:
        print(f"Warning: could not read failed warm-start graph-index file {failed_file}: {exc!r}")
        return False

    return str(graph_index) in failed_indices
