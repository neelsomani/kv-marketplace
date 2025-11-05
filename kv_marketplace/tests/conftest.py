"""Pytest configuration and fixtures for kv-marketplace tests."""

import pytest
import torch


@pytest.fixture(autouse=True)
def reset_peer_access():
    """Reset peer access state between tests to avoid leakage."""
    # Only run if CUDA is available and we have 2+ GPUs
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        yield
        return
    
    try:
        from kv_marketplace.transport import p2p_cuda
    except Exception:
        yield
        return
    
    # Best-effort disable both directions; ignore "not enabled" errors
    try:
        # Try to disable peer access in both directions
        for a, b in [(0, 1), (1, 0)]:
            try:
                p2p_cuda.disable_peer_access(a, b)
            except Exception:
                pass
    except Exception:
        pass
    
    yield
    
    # Cleanup after test (same as before)
    try:
        for a, b in [(0, 1), (1, 0)]:
            try:
                p2p_cuda.disable_peer_access(a, b)
            except Exception:
                pass
    except Exception:
        pass

