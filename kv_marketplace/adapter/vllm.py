"""vLLM adapter implementing the callback contract."""

from typing import Callable, Optional, Tuple
from .types import KVLayout, AllocatedKV
from ..compat import KVCompat


class VLLMImportCtx:
    """Context for importing KV cache before prefill."""
    device_id: int
    compat: KVCompat
    tokens: list
    alloc_prefix: Callable[[int], AllocatedKV]
    layout: KVLayout
    stream: int  # cudaStream_t as int


class VLLMExportCtx:
    """Context for exporting KV cache after prefill."""
    device_id: int
    compat: KVCompat
    tokens: list
    kv_pages: AllocatedKV   # [0:prompt_len]
    layout: KVLayout
    length: int


def before_prefill(ctx: VLLMImportCtx) -> Optional[Tuple[int, AllocatedKV]]:
    """Hook called before prefill to attempt KV cache import.
    
    Args:
        ctx: Import context with request information
        
    Returns:
        Tuple of (lcp_len, dst_alloc) if import succeeds, None otherwise
    """
    # TODO: Implement in milestone 4
    # - Build KVCompat from ctx.compat
    # - Query KVRegistry for a match and LCP
    # - If lcp_len >= min_prefix:
    #   - Allocate destination KV
    #   - Copy via PeerCopy
    #   - Return (lcp_len, dst_alloc)
    return None


def after_prefill(ctx: VLLMExportCtx) -> None:
    """Hook called after prefill to export KV cache.
    
    Args:
        ctx: Export context with KV pages
    """
    # TODO: Implement in milestone 4
    # - Build KVHandle from ctx.kv_pages
    # - Register in KVRegistry
    pass

