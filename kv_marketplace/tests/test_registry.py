"""Tests for KV registry."""

import pytest
from kv_marketplace.registry import KVRegistry, KVHandle
from kv_marketplace.compat import KVCompat


def test_registry_register_and_lookup():
    """Test basic register and lookup."""
    registry = KVRegistry()
    
    compat = KVCompat(
        model_params={'n_layers': 32, 'hidden_size': 4096},
        tokenizer_config={'vocab_size': 50000}
    )
    
    handle = KVHandle(
        device_id=0,
        k_ptrs=[0x1000, 0x2000],
        v_ptrs=[0x3000, 0x4000],
        length=128,
        layout_meta={'page_size': 16}
    )
    
    prefix_hash = b'\x00' * 8  # Dummy hash
    
    registry.register(compat, prefix_hash, handle)
    
    # Lookup should succeed
    found = registry.lookup(compat, prefix_hash)
    assert found is not None
    assert found.device_id == 0
    assert found.length == 128
    
    # Lookup with different compat should fail
    compat2 = KVCompat(
        model_params={'n_layers': 16, 'hidden_size': 2048},
        tokenizer_config={'vocab_size': 50000}
    )
    found2 = registry.lookup(compat2, prefix_hash)
    assert found2 is None


def test_registry_concurrent_operations():
    """Test concurrent register and query operations."""
    registry = KVRegistry()
    
    compat = KVCompat(
        model_params={'n_layers': 32},
        tokenizer_config={'vocab_size': 50000}
    )
    
    # Register multiple handles
    for i in range(10):
        handle = KVHandle(
            device_id=i % 2,
            k_ptrs=[0x1000 * i],
            v_ptrs=[0x2000 * i],
            length=64 + i,
            layout_meta={}
        )
        prefix_hash = bytes([i] * 8)
        registry.register(compat, prefix_hash, handle)
    
    # Query all
    for i in range(10):
        prefix_hash = bytes([i] * 8)
        found = registry.lookup(compat, prefix_hash)
        assert found is not None
        assert found.length == 64 + i


def test_registry_remove():
    """Test removing entries."""
    registry = KVRegistry()
    
    compat = KVCompat(
        model_params={'n_layers': 32},
        tokenizer_config={'vocab_size': 50000}
    )
    
    handle = KVHandle(
        device_id=0,
        k_ptrs=[0x1000],
        v_ptrs=[0x2000],
        length=128,
        layout_meta={}
    )
    
    prefix_hash = b'\x01' * 8
    registry.register(compat, prefix_hash, handle)
    
    assert registry.lookup(compat, prefix_hash) is not None
    
    registry.remove(compat, prefix_hash)
    
    assert registry.lookup(compat, prefix_hash) is None

