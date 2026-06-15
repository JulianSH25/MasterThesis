def _failed_graph_index_file_from_cache_path(cache_path_raw: str | None) -> Path | None:
    if not cache_path_raw:
        return None
    cache_path = Path(cache_path_raw)
    run_dir = cache_path.parent.parent
    run_tag = run_dir.name
    return run_dir / "failed_warm_starts" / f"failed_warm_start_graph_indices_{run_tag}.txt"


def _current_graph_index_from_argv() -> int | None:
    try:
        if len(sys.argv) >= 4:
            return int(sys.argv[3])
    except Exception:
        return None
    return None


def _graph_index_already_failed(cache_path_raw: str | None, graph_index: int | None) -> bool:
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