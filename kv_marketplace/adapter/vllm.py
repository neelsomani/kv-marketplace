"""vLLM adapter implementing the callback contract."""

import logging
from typing import Callable, Optional, Tuple, List, TypedDict
from .types import KVLayout, AllocatedKV
from ..compat import KVCompat
from ..registry import KVRegistry, KVHandle
from ..prefix_index import PrefixIndex
from ..transport.p2p import PeerCopy

logger = logging.getLogger(__name__)

# Global instances for registry and prefix index
_registry = KVRegistry()
_prefix_index = PrefixIndex()

# Default minimum prefix length (can be overridden by vLLM flags)
_MIN_PREFIX_LENGTH = 64


def set_min_prefix_length(min_length: int):
    """Set the minimum prefix length for import.
    
    Args:
        min_length: Minimum number of tokens required for import
    """
    global _MIN_PREFIX_LENGTH
    _MIN_PREFIX_LENGTH = min_length


def get_min_prefix_length() -> int:
    """Get the current minimum prefix length."""
    return _MIN_PREFIX_LENGTH


class VLLMImportCtx(TypedDict):
    """Context for importing KV cache before prefill."""
    device_id: int
    compat: KVCompat
    tokens: List[int]
    alloc_prefix: Callable[[int], AllocatedKV]
    layout: KVLayout
    stream: int  # cudaStream_t as int


class VLLMExportCtx(TypedDict):
    """Context for exporting KV cache after prefill."""
    device_id: int
    compat: KVCompat
    tokens: List[int]
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
    try:
        # Get compatibility from context
        compat = ctx['compat']
        tokens = ctx['tokens']
        device_id = ctx['device_id']
        layout = ctx['layout']
        min_prefix = get_min_prefix_length()
        
        # Convert compat dict to KVCompat if needed
        if not isinstance(compat, KVCompat):
            if isinstance(compat, dict):
                compat = KVCompat(
                    model_params=compat.get("model_params", {}),
                    tokenizer_config=compat.get("tokenizer_config", {}),
                    rope_config=compat.get("rope_config", {}),
                    layout_config=compat.get("kv_layout", {})
                )
            else:
                logger.error(f"Invalid compat type: {type(compat)}")
                return None
        
        # Find longest common prefix using prefix index
        lcp_result = _prefix_index.find_lcp(tokens)
        if lcp_result is None:
            logger.debug(f"No LCP found for tokens (length={len(tokens)})")
            return None
        
        lcp_len, prefix_hash = lcp_result
        
        # Check if LCP meets minimum length requirement
        if lcp_len < min_prefix:
            logger.debug(f"LCP length {lcp_len} < min_prefix {min_prefix}, skipping import")
            return None
        
        # Lookup handle in registry
        handle = _registry.lookup(compat, prefix_hash)
        if handle is None:
            logger.debug(f"No handle found for compat={compat.checksum.hex()[:8]}..., prefix_hash={prefix_hash.hex()[:8]}...")
            return None
        
        logger.info(f"Found KV cache match: lcp_len={lcp_len}, src_dev={handle.device_id}, dst_dev={device_id}")
        
        # Allocate destination KV cache
        dst_alloc = ctx['alloc_prefix'](lcp_len)
        
        # Ensure peer access is enabled
        if not PeerCopy.ensure_peer_access(handle.device_id, device_id):
            logger.warning(f"Peer access not available between GPU {handle.device_id} and GPU {device_id}, skipping import")
            return None
        
        # Copy KV tensors from source to destination
        # Pass the stream directly - PeerCopy.copy_kv accepts int (CUDA stream pointer)
        PeerCopy.copy_kv(
            dst_dev=device_id,
            dst_k_ptrs=dst_alloc['k_ptrs'],
            dst_v_ptrs=dst_alloc['v_ptrs'],
            src_dev=handle.device_id,
            src_k_ptrs=handle.k_ptrs,
            src_v_ptrs=handle.v_ptrs,
            n_layers=layout['n_layers'],
            length=lcp_len,
            meta={
                'n_kv_heads': layout['n_kv_heads'],
                'head_dim': layout['head_dim'],
                'page_size': layout['page_size'],
                **layout.get('strides', {})
            },
            stream=ctx['stream']  # Pass int stream pointer directly
        )
        
        logger.info(f"Successfully imported {lcp_len} tokens from GPU {handle.device_id} to GPU {device_id}")
        return (lcp_len, dst_alloc)
        
    except Exception as e:
        logger.error(f"Error in before_prefill: {e}", exc_info=True)
        # Return None on error to fall back to standard execution
        return None


def after_prefill(ctx: VLLMExportCtx) -> None:
    """Hook called after prefill to export KV cache.
    
    Args:
        ctx: Export context with KV pages
    """
    try:
        compat = ctx['compat']
        tokens = ctx['tokens']
        length = ctx['length']
        device_id = ctx['device_id']
        kv_pages = ctx['kv_pages']
        layout = ctx['layout']
        
        # Convert compat dict to KVCompat if needed
        if not isinstance(compat, KVCompat):
            if isinstance(compat, dict):
                compat = KVCompat(
                    model_params=compat.get("model_params", {}),
                    tokenizer_config=compat.get("tokenizer_config", {}),
                    rope_config=compat.get("rope_config", {}),
                    layout_config=compat.get("kv_layout", {})
                )
            else:
                logger.error(f"Invalid compat type: {type(compat)}")
                return
        
        # Only export the prefix up to length
        prefix_tokens = tokens[:length]
        
        # Compute prefix hash and insert into prefix index
        prefix_hash = _prefix_index.insert(prefix_tokens)
        
        # Build KVHandle from context
        handle = KVHandle(
            device_id=device_id,
            k_ptrs=kv_pages['k_ptrs'],
            v_ptrs=kv_pages['v_ptrs'],
            length=length,
            layout_meta={
                'n_layers': layout['n_layers'],
                'n_kv_heads': layout['n_kv_heads'],
                'head_dim': layout['head_dim'],
                'page_size': layout['page_size'],
                **layout.get('strides', {})
            }
        )
        
        # Register in registry
        _registry.register(compat, prefix_hash, handle)
        
        logger.info(f"Exported KV cache: length={length}, device={device_id}, prefix_hash={prefix_hash.hex()[:8]}...")
        
    except Exception as e:
        logger.error(f"Error in after_prefill: {e}", exc_info=True)
        # Don't raise - export failure shouldn't break inference

