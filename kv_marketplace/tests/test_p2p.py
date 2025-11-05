"""Tests for CUDA peer-to-peer transport."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ensure_peer_access():
    """Test peer access setup."""
    if torch.cuda.device_count() < 2:
        pytest.skip("Need at least 2 GPUs")
    
    from kv_marketplace.transport import PeerCopy
    
    # Same device
    assert PeerCopy.ensure_peer_access(0, 0) is True
    
    # Different devices (may or may not support peer access)
    result = PeerCopy.ensure_peer_access(0, 1)
    assert isinstance(result, bool)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_peer_copy_checksum():
    """Test peer copy with checksum validation."""
    if torch.cuda.device_count() < 2:
        pytest.skip("Need at least 2 GPUs")
    
    import hashlib
    size = 1024 * 1024  # 1MB
    
    # Allocate first, before any import that might enable peer access
    src = torch.randn(size, device='cuda:0', dtype=torch.float16)
    dst = torch.zeros(size, device='cuda:1', dtype=torch.float16)
    
    # Now import
    try:
        from kv_marketplace.transport import p2p_cuda
        from kv_marketplace.transport import _get_stream_ptr
    except ImportError:
        pytest.skip("CUDA extension not built")
    
    # If peer access was already on, ensure() should treat it as success
    if not p2p_cuda.ensure_peer_access(0, 1):
        pytest.skip("Peer access not available between GPU 0 and 1")
    
    # Checksum on src
    src_hash = hashlib.md5(src.cpu().numpy().tobytes()).hexdigest()
    
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    
    with torch.cuda.device(1):
        stream = torch.cuda.Stream(device=1)
        with torch.cuda.stream(stream):
            p2p_cuda.memcpy_peer_async(
                dst.data_ptr(), 1,
                src.data_ptr(), 0,
                size * 2,  # bytes (float16 = 2 bytes)
                _get_stream_ptr(stream),
            )
        stream.synchronize()
    
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    
    dst_hash = hashlib.md5(dst.cpu().numpy().tobytes()).hexdigest()
    assert src_hash == dst_hash

