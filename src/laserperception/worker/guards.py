"""An explicit external context supplements, and never replaces, owner authorization."""


def require_external_worker(external_worker: bool) -> None:
    """Reject local/default invocation before hardware imports, queries or child processes."""
    if external_worker is not True:
        raise ValueError(
            "GPU worker path requires explicit --external-worker context; "
            "owner scope still required"
        )
