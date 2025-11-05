"""Thin typed views over vLLM KV layout."""

from typing import TypedDict, List, Tuple, Optional, Any


class KVLayout(TypedDict):
    """KV cache layout metadata."""
    n_layers: int
    n_kv_heads: int
    head_dim: int
    page_size: int
    strides: dict  # per-layer or global, as needed


class AllocatedKVRequired(TypedDict):
    """Required fields for AllocatedKV."""
    k_ptrs: List[int]   # per-layer device addresses or page IDs
    v_ptrs: List[int]
    length: int         # tokens materialized


class AllocatedKV(AllocatedKVRequired, total=False):
    """Allocated KV cache structure."""
    # Optional: page ranges for coalescing optimization
    # List where each element corresponds to a layer. Each element can be:
    # - Tuple[int, int]: single (start_page, end_page) range for that layer
    # - List[Tuple[int, int]]: multiple (start_page, end_page) ranges to coalesce
    # If not provided, assumes contiguous pages from 0
    page_ranges: List[Any]  # Optional, for coalescing; Any to support Union[Tuple, List[Tuple]]
