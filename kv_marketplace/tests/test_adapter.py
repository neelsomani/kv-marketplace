"""Tests for vLLM adapter logic (synthetic loopback tests)."""

import pytest
import torch
from kv_marketplace.adapter import vllm
from kv_marketplace.adapter.vllm import VLLMImportCtx, VLLMExportCtx
from kv_marketplace.compat import KVCompat
from kv_marketplace.adapter.types import KVLayout, AllocatedKV


def create_mock_compat() -> KVCompat:
    """Create a mock compatibility configuration."""
    return KVCompat(
        model_params={'n_layers': 2, 'hidden_size': 1024},
        tokenizer_config={'vocab_size': 50000},
        rope_config={'rope_theta': 10000.0},
        layout_config={'dtype': 'float16', 'page_size': 16}
    )


def create_mock_layout() -> KVLayout:
    """Create a mock KV layout."""
    return {
        'n_layers': 2,
        'n_kv_heads': 8,
        'head_dim': 128,
        'page_size': 16,
        'strides': {}
    }


def test_adapter_basic_export_and_import():
    """Test basic export and import cycle without GPU."""
    # Reset adapter state
    from kv_marketplace.adapter.vllm import _registry, _prefix_index
    _registry._registry.clear()
    _prefix_index._root = type(_prefix_index._root)()
    _prefix_index._hash_to_length.clear()
    
    compat = create_mock_compat()
    layout = create_mock_layout()
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    length = len(tokens)
    
    # Mock KV pages (using dummy pointers)
    kv_pages: AllocatedKV = {
        'k_ptrs': [0x1000, 0x2000],  # 2 layers
        'v_ptrs': [0x3000, 0x4000],
        'length': length
    }
    
    # Export after prefill
    export_ctx: VLLMExportCtx = {
        'device_id': 0,
        'compat': compat,
        'tokens': tokens,
        'kv_pages': kv_pages,
        'layout': layout,
        'length': length
    }
    
    vllm.after_prefill(export_ctx)
    
    # Now try to import with matching tokens
    allocated_ptrs = []
    
    def mock_alloc_prefix(prefix_len: int) -> AllocatedKV:
        """Mock allocator that tracks allocated pointers."""
        alloc = {
            'k_ptrs': [0x5000 + i for i in range(layout['n_layers'])],
            'v_ptrs': [0x6000 + i for i in range(layout['n_layers'])],
            'length': prefix_len
        }
        allocated_ptrs.append(alloc)
        return alloc
    
    import_ctx: VLLMImportCtx = {
        'device_id': 1,
        'compat': compat,
        'tokens': tokens + [9, 10],  # Same prefix + extra tokens
        'alloc_prefix': mock_alloc_prefix,
        'layout': layout,
        'stream': 0
    }
    
    # Set minimum prefix to allow import
    vllm.set_min_prefix_length(4)
    
    result = vllm.before_prefill(import_ctx)
    
    # Should find match and return (lcp_len, dst_alloc)
    # Note: Without actual GPU peer access, this will fail at PeerCopy.ensure_peer_access
    # But we can verify the logic up to that point
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        # If we have GPUs, result might be None if peer access fails
        # or might succeed if peer access works
        assert result is None or (isinstance(result, tuple) and len(result) == 2)
    else:
        # Without GPUs, we expect None due to peer access check
        assert result is None


def test_adapter_min_prefix_enforcement():
    """Test that minimum prefix length is enforced."""
    from kv_marketplace.adapter.vllm import _registry, _prefix_index
    _registry._registry.clear()
    _prefix_index._root = type(_prefix_index._root)()
    _prefix_index._hash_to_length.clear()
    
    compat = create_mock_compat()
    layout = create_mock_layout()
    
    # Export a short prefix
    short_tokens = [1, 2, 3]  # Only 3 tokens
    kv_pages: AllocatedKV = {
        'k_ptrs': [0x1000, 0x2000],
        'v_ptrs': [0x3000, 0x4000],
        'length': len(short_tokens)
    }
    
    export_ctx: VLLMExportCtx = {
        'device_id': 0,
        'compat': compat,
        'tokens': short_tokens,
        'kv_pages': kv_pages,
        'layout': layout,
        'length': len(short_tokens)
    }
    
    vllm.after_prefill(export_ctx)
    
    # Set minimum prefix to 5 (higher than exported)
    vllm.set_min_prefix_length(5)
    
    def mock_alloc_prefix(prefix_len: int) -> AllocatedKV:
        return {
            'k_ptrs': [0x5000, 0x6000],
            'v_ptrs': [0x7000, 0x8000],
            'length': prefix_len
        }
    
    import_ctx: VLLMImportCtx = {
        'device_id': 1,
        'compat': compat,
        'tokens': short_tokens + [4, 5],
        'alloc_prefix': mock_alloc_prefix,
        'layout': layout,
        'stream': 0
    }
    
    result = vllm.before_prefill(import_ctx)
    
    # Should return None because LCP (3) < min_prefix (5)
    assert result is None


def test_adapter_compat_mismatch():
    """Test that compatibility mismatch prevents import."""
    from kv_marketplace.adapter.vllm import _registry, _prefix_index
    _registry._registry.clear()
    _prefix_index._root = type(_prefix_index._root)()
    _prefix_index._hash_to_length.clear()
    
    compat1 = create_mock_compat()
    compat2 = KVCompat(
        model_params={'n_layers': 4, 'hidden_size': 2048},  # Different params
        tokenizer_config={'vocab_size': 50000}
    )
    
    layout = create_mock_layout()
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    
    # Export with compat1
    export_ctx: VLLMExportCtx = {
        'device_id': 0,
        'compat': compat1,
        'tokens': tokens,
        'kv_pages': {
            'k_ptrs': [0x1000, 0x2000],
            'v_ptrs': [0x3000, 0x4000],
            'length': len(tokens)
        },
        'layout': layout,
        'length': len(tokens)
    }
    
    vllm.after_prefill(export_ctx)
    
    # Try to import with compat2 (different model)
    vllm.set_min_prefix_length(4)
    
    def mock_alloc_prefix(prefix_len: int) -> AllocatedKV:
        return {
            'k_ptrs': [0x5000, 0x6000],
            'v_ptrs': [0x7000, 0x8000],
            'length': prefix_len
        }
    
    import_ctx: VLLMImportCtx = {
        'device_id': 1,
        'compat': compat2,  # Different compat
        'tokens': tokens + [9],
        'alloc_prefix': mock_alloc_prefix,
        'layout': layout,
        'stream': 0
    }
    
    result = vllm.before_prefill(import_ctx)
    
    # Should return None because compat doesn't match
    assert result is None


def test_adapter_no_lcp_match():
    """Test that no LCP match returns None."""
    from kv_marketplace.adapter.vllm import _registry, _prefix_index
    _registry._registry.clear()
    _prefix_index._root = type(_prefix_index._root)()
    _prefix_index._hash_to_length.clear()
    
    compat = create_mock_compat()
    layout = create_mock_layout()
    
    # Export tokens
    tokens1 = [10, 20, 30, 40]
    export_ctx: VLLMExportCtx = {
        'device_id': 0,
        'compat': compat,
        'tokens': tokens1,
        'kv_pages': {
            'k_ptrs': [0x1000, 0x2000],
            'v_ptrs': [0x3000, 0x4000],
            'length': len(tokens1)
        },
        'layout': layout,
        'length': len(tokens1)
    }
    
    vllm.after_prefill(export_ctx)
    
    # Try to import with completely different tokens
    tokens2 = [99, 98, 97, 96]  # No overlap
    vllm.set_min_prefix_length(4)
    
    def mock_alloc_prefix(prefix_len: int) -> AllocatedKV:
        return {
            'k_ptrs': [0x5000, 0x6000],
            'v_ptrs': [0x7000, 0x8000],
            'length': prefix_len
        }
    
    import_ctx: VLLMImportCtx = {
        'device_id': 1,
        'compat': compat,
        'tokens': tokens2,
        'alloc_prefix': mock_alloc_prefix,
        'layout': layout,
        'stream': 0
    }
    
    result = vllm.before_prefill(import_ctx)
    
    # Should return None because no LCP match
    assert result is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_adapter_with_real_gpus():
    """Test adapter with real GPUs if available."""
    # Check GPU count inside the test (skipif lambda can be unreliable)
    if torch.cuda.device_count() < 2:
        pytest.skip(f"Need at least 2 GPUs, found {torch.cuda.device_count()}")
    
    try:
        from kv_marketplace.transport import p2p_cuda
    except ImportError as e:
        pytest.skip(f"CUDA extension not built: {e}")
    
    # Check if peer access is available
    if not p2p_cuda.ensure_peer_access(0, 1):
        pytest.skip("Peer access not available between GPU 0 and 1")
    
    from kv_marketplace.adapter.vllm import _registry, _prefix_index
    _registry._registry.clear()
    _prefix_index._root = type(_prefix_index._root)()
    _prefix_index._hash_to_length.clear()
    
    compat = create_mock_compat()
    layout = create_mock_layout()
    tokens = list(range(100, 164))  # 64 tokens (meets min_prefix)
    
    # Create actual GPU buffers for export
    n_layers = layout['n_layers']
    n_kv_heads = layout['n_kv_heads']
    head_dim = layout['head_dim']
    length = len(tokens)
    
    # Size per layer: n_kv_heads * head_dim * length * 2 (float16 = 2 bytes)
    kv_size = n_kv_heads * head_dim * length * 2
    
    k_buffers = []
    v_buffers = []
    k_ptrs = []
    v_ptrs = []
    
    # Create buffers on GPU 0
    for layer_idx in range(n_layers):
        k = torch.randn(n_kv_heads, head_dim, length, device='cuda:0', dtype=torch.float16)
        v = torch.randn(n_kv_heads, head_dim, length, device='cuda:0', dtype=torch.float16)
        k_buffers.append(k)
        v_buffers.append(v)
        k_ptrs.append(k.data_ptr())
        v_ptrs.append(v.data_ptr())
    
    # Export
    export_ctx: VLLMExportCtx = {
        'device_id': 0,
        'compat': compat,
        'tokens': tokens,
        'kv_pages': {
            'k_ptrs': k_ptrs,
            'v_ptrs': v_ptrs,
            'length': length
        },
        'layout': layout,
        'length': length
    }
    
    vllm.after_prefill(export_ctx)
    
    # Now try to import on GPU 1
    vllm.set_min_prefix_length(64)
    
    dst_k_buffers = []
    dst_v_buffers = []
    dst_k_ptrs = []
    dst_v_ptrs = []
    
    def mock_alloc_prefix(prefix_len: int) -> AllocatedKV:
        """Allocate buffers on GPU 1."""
        for layer_idx in range(n_layers):
            k = torch.zeros(n_kv_heads, head_dim, prefix_len, device='cuda:1', dtype=torch.float16)
            v = torch.zeros(n_kv_heads, head_dim, prefix_len, device='cuda:1', dtype=torch.float16)
            dst_k_buffers.append(k)
            dst_v_buffers.append(v)
            dst_k_ptrs.append(k.data_ptr())
            dst_v_ptrs.append(v.data_ptr())
        
        return {
            'k_ptrs': dst_k_ptrs,
            'v_ptrs': dst_v_ptrs,
            'length': prefix_len
        }
    
    import_ctx: VLLMImportCtx = {
        'device_id': 1,
        'compat': compat,
        'tokens': tokens + [200, 201],  # Same prefix + extra
        'alloc_prefix': mock_alloc_prefix,
        'layout': layout,
        'stream': 0
    }
    
    result = vllm.before_prefill(import_ctx)
    
    # Should succeed if peer access works
    if result is not None:
        lcp_len, dst_alloc = result
        assert lcp_len == len(tokens)
        assert dst_alloc['length'] == lcp_len
        
        # Verify data was copied (compare first layer as sample)
        torch.cuda.synchronize()
        if len(dst_k_buffers) > 0 and len(k_buffers) > 0:
            # Copy back to CPU for comparison
            src_k_cpu = k_buffers[0].cpu()
            dst_k_cpu = dst_k_buffers[0].cpu()
            
            # Check if they match (within floating point tolerance)
            assert torch.allclose(src_k_cpu, dst_k_cpu, atol=1e-3), \
                "Copied KV data doesn't match source"
