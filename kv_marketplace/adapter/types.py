"""Thin typed views over vLLM KV layout."""

from typing import TypedDict, List


class KVLayout(TypedDict):
    """KV cache layout metadata."""
    n_layers: int
    n_kv_heads: int
    head_dim: int
    page_size: int
    strides: dict  # per-layer or global, as needed


class AllocatedKV(TypedDict):
    """Allocated KV cache structure."""
    k_ptrs: List[int]   # per-layer device addresses or page IDs
    v_ptrs: List[int]
    length: int         # tokens materialized
