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
    
    try:
        from kv_marketplace.transport import p2p_cuda
    except ImportError:
        pytest.skip("CUDA extension not built")
    
    # Create dummy buffers on GPU 0
    size = 1024 * 1024  # 1MB
    src = torch.randn(size, device='cuda:0', dtype=torch.float16)
    
    # Compute source checksum
    src_cpu = src.cpu().numpy()
    src_hash = hashlib.md5(src_cpu.tobytes()).hexdigest()
    
    # Create destination buffer on GPU 1
    dst = torch.zeros(size, device='cuda:1', dtype=torch.float16)
    
    # Ensure peer access
    if not p2p_cuda.ensure_peer_access(0, 1):
        pytest.skip("Peer access not available between GPU 0 and 1")
    
    # Copy via P2P
    from kv_marketplace.transport import _get_stream_ptr
    
    stream = torch.cuda.Stream(device=1)
    stream_ptr = _get_stream_ptr(stream)
    
    with torch.cuda.stream(stream):
        p2p_cuda.memcpy_peer_async(
            dst.data_ptr(), 1,
            src.data_ptr(), 0,
            size * 2,  # bytes (float16 = 2 bytes)
            stream_ptr
        )
    
    stream.synchronize()
    
    # Verify checksum
    dst_cpu = dst.cpu().numpy()
    dst_hash = hashlib.md5(dst_cpu.tobytes()).hexdigest()
    
    assert src_hash == dst_hash, "Checksums don't match after peer copy"

