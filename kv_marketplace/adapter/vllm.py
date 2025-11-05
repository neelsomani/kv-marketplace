"""vLLM adapter implementing the callback contract."""

import logging
import json
import os
import tempfile
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

# Statistics tracking
_import_hits = 0
_import_misses = 0
_import_lcp_lengths = []

# File-based stats sharing for multiprocessing
_stats_file = None
_stats_lock_file = None


def _get_stats_file_path():
    """Get the path to the stats file."""
    global _stats_file
    if _stats_file is None:
        # Use a temp file in /tmp for cross-process access
        stats_dir = os.path.join(tempfile.gettempdir(), 'kv_marketplace_stats')
        os.makedirs(stats_dir, exist_ok=True)
        _stats_file = os.path.join(stats_dir, 'stats.json')
    return _stats_file


def _get_lock_file_path():
    """Get the path to the lock file."""
    global _stats_lock_file
    if _stats_lock_file is None:
        stats_dir = os.path.join(tempfile.gettempdir(), 'kv_marketplace_stats')
        os.makedirs(stats_dir, exist_ok=True)
        _stats_lock_file = os.path.join(stats_dir, 'stats.lock')
    return _stats_lock_file


def _write_stats_to_file():
    """Write current stats to file for cross-process access.
    
    Each process writes its local stats. The file structure is:
    {
        "processes": {
            "pid1": { "import_hits": ..., "import_misses": ..., ... },
            "pid2": { ... },
            ...
        }
    }
    """
    try:
        import os as os_module
        pid = os_module.getpid()
        
        stats_file = _get_stats_file_path()
        
        # Read existing stats
        existing_stats = _read_stats_from_file_raw()
        
        # Update this process's stats
        if 'processes' not in existing_stats:
            existing_stats['processes'] = {}
        
        existing_stats['processes'][str(pid)] = {
            'import_hits': _import_hits,
            'import_misses': _import_misses,
            'import_lcp_lengths': _import_lcp_lengths.copy(),
            'registry_size': len(_registry._registry),
            'prefix_index_size': len(_prefix_index._hash_to_length),
        }
        
        # Atomic write: write to temp then rename
        temp_file = stats_file + '.tmp'
        with open(temp_file, 'w') as f:
            json.dump(existing_stats, f)
        os.replace(temp_file, stats_file)
    except Exception as e:
        logger.warning(f"Failed to write stats to file: {e}")


def _read_stats_from_file_raw():
    """Read raw stats file (returns dict with per-process stats)."""
    try:
        stats_file = _get_stats_file_path()
        if not os.path.exists(stats_file):
            return {'processes': {}}
        
        with open(stats_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read stats from file: {e}")
        return {'processes': {}}


def _read_stats_from_file():
    """Read stats from file and aggregate across all processes.
    
    Returns aggregated stats summing counts from all worker processes.
    """
    try:
        raw_stats = _read_stats_from_file_raw()
        processes = raw_stats.get('processes', {})
        
        if not processes:
            return {
                'import_hits': 0,
                'import_misses': 0,
                'import_lcp_lengths': [],
                'registry_size': 0,
                'prefix_index_size': 0,
            }
        
        # Aggregate stats from all processes
        total_hits = 0
        total_misses = 0
        all_lcp_lengths = []
        max_registry_size = 0
        max_prefix_index_size = 0
        
        for pid, proc_stats in processes.items():
            total_hits += proc_stats.get('import_hits', 0)
            total_misses += proc_stats.get('import_misses', 0)
            all_lcp_lengths.extend(proc_stats.get('import_lcp_lengths', []))
            max_registry_size = max(max_registry_size, proc_stats.get('registry_size', 0))
            max_prefix_index_size = max(max_prefix_index_size, proc_stats.get('prefix_index_size', 0))
        
        return {
            'import_hits': total_hits,
            'import_misses': total_misses,
            'import_lcp_lengths': all_lcp_lengths,
            'registry_size': max_registry_size,
            'prefix_index_size': max_prefix_index_size,
        }
    except Exception as e:
        logger.warning(f"Failed to aggregate stats from file: {e}")
        return {
            'import_hits': 0,
            'import_misses': 0,
            'import_lcp_lengths': [],
            'registry_size': 0,
            'prefix_index_size': 0,
        }


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


def get_stats() -> dict:
    """Get statistics about imports and exports.
    
    Reads from both in-process stats and file-based stats (for cross-process access).
    In multiprocessing scenarios, file-based stats aggregate across all worker processes.
    
    Returns:
        Dictionary with statistics:
        - import_hits: Number of successful imports (aggregated across processes)
        - import_misses: Number of failed import attempts (aggregated)
        - import_lcp_lengths: List of LCP lengths for successful imports (combined)
        - registry_size: Maximum registry size across processes
        - prefix_index_size: Maximum prefix index size across processes
    """
    global _import_hits, _import_misses, _import_lcp_lengths
    
    # Get in-process stats
    local_stats = {
        'import_hits': _import_hits,
        'import_misses': _import_misses,
        'import_lcp_lengths': _import_lcp_lengths.copy(),
        'registry_size': len(_registry._registry),
        'prefix_index_size': len(_prefix_index._hash_to_length),
    }
    
    # Read aggregated stats from file (includes all worker processes)
    file_stats = _read_stats_from_file()
    
    # File stats are aggregated across all worker processes (summed for hits/misses)
    # Use file stats as primary source since hooks run in worker processes
    # Also merge with local stats to handle cases where hooks might run in main process
    merged_stats = {
        # File stats are aggregated (summed), so use them directly
        # They will be 0 if no worker processes have written yet
        'import_hits': file_stats['import_hits'],
        'import_misses': file_stats['import_misses'],
        # Combine LCP lengths from both sources (deduplicate)
        'import_lcp_lengths': list(set(local_stats['import_lcp_lengths'] + file_stats.get('import_lcp_lengths', []))),
        # For sizes, use max (one process might have more entries than others)
        'registry_size': max(local_stats['registry_size'], file_stats['registry_size']),
        'prefix_index_size': max(local_stats['prefix_index_size'], file_stats['prefix_index_size']),
    }
    
    return merged_stats


def reset_stats():
    """Reset statistics counters."""
    global _import_hits, _import_misses, _import_lcp_lengths
    _import_hits = 0
    _import_misses = 0
    _import_lcp_lengths = []
    # Also reset file-based stats
    try:
        stats_file = _get_stats_file_path()
        if os.path.exists(stats_file):
            os.remove(stats_file)
        # Write empty stats
        _write_stats_to_file()
    except Exception as e:
        logger.warning(f"Failed to reset file stats: {e}")


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
        prefix_index_size = _prefix_index.size()
        logger.info(f"kv-marketplace before_prefill: Looking for LCP in prefix_index (size={prefix_index_size}, tokens_len={len(tokens)}, first_10_tokens={tokens[:10]})")
        lcp_result = _prefix_index.find_lcp(tokens)
        if lcp_result is None:
            logger.info(f"kv-marketplace before_prefill: No LCP found for tokens (length={len(tokens)}, prefix_index_size={prefix_index_size})")
            global _import_misses
            _import_misses += 1
            _write_stats_to_file()
            return None
        
        lcp_len, prefix_hash = lcp_result
        logger.info(f"kv-marketplace before_prefill: Found LCP (length={lcp_len}, prefix_hash={prefix_hash.hex()[:8]}..., min_prefix={min_prefix})")
        
        # Check if LCP meets minimum length requirement
        if lcp_len < min_prefix:
            logger.info(f"kv-marketplace before_prefill: LCP length {lcp_len} < min_prefix {min_prefix}, skipping import")
            _import_misses += 1
            _write_stats_to_file()
            return None
        
        # Lookup handle in registry
        logger.info(f"kv-marketplace before_prefill: Looking up handle in registry (size={len(_registry._registry)}, compat_hash={compat.checksum.hex()[:8]}..., prefix_hash={prefix_hash.hex()[:8]}...)")
        handle = _registry.lookup(compat, prefix_hash)
        if handle is None:
            logger.info(f"kv-marketplace before_prefill: No handle found in registry for compat/prefix_hash")
            _import_misses += 1
            _write_stats_to_file()
            return None
        
        # Check if handle has valid GPU pointers (not just block IDs)
        # GPU pointers are typically large integers (memory addresses)
        # Block IDs are small integers (0 to num_blocks)
        if handle.k_ptrs and isinstance(handle.k_ptrs[0], int):
            # Rough heuristic: GPU pointers are usually > 0x1000000 (16MB)
            # Block IDs are typically < 10000
            if handle.k_ptrs[0] < 0x1000000:
                logger.warning(
                    f"kv-marketplace before_prefill: Handle contains likely block_ids (first={handle.k_ptrs[0]}), "
                    f"not GPU pointers. Pointer extraction may have failed. Attempting import anyway..."
                )
                # Continue anyway - might still work if it's actually a valid small pointer
        
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
        
        # Track successful import
        global _import_hits, _import_lcp_lengths
        _import_hits += 1
        _import_lcp_lengths.append(lcp_len)
        # Write to file for cross-process access
        _write_stats_to_file()
        
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
        
        # Check if we have valid pointers
        k_ptrs = kv_pages.get('k_ptrs', [])
        v_ptrs = kv_pages.get('v_ptrs', [])
        
        logger.debug(
            f"kv-marketplace after_prefill: length={length}, device_id={device_id}, "
            f"k_ptrs={len(k_ptrs)}, v_ptrs={len(v_ptrs)}, "
            f"k_ptrs_sample={k_ptrs[:5] if k_ptrs else []}, "
            f"v_ptrs_sample={v_ptrs[:5] if v_ptrs else []}"
        )
        
        # If pointers are empty, we can't export (get_prefill_pages may have failed)
        if not k_ptrs or not v_ptrs:
            logger.warning(
                f"KV cache pointers are empty for export: length={length}, "
                f"k_ptrs={len(k_ptrs)}, v_ptrs={len(v_ptrs)}. "
                f"This may indicate get_prefill_pages needs implementation or blocks aren't available yet."
            )
            return
        
        # Compute prefix hash and insert into prefix index
        prefix_hash = _prefix_index.insert(prefix_tokens)
        
        # Build KVHandle from context
        handle = KVHandle(
            device_id=device_id,
            k_ptrs=k_ptrs,
            v_ptrs=v_ptrs,
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
        
        # Update file stats (registry size changed)
        _write_stats_to_file()
        
        logger.info(f"Exported KV cache: length={length}, device={device_id}, prefix_hash={prefix_hash.hex()[:8]}..., "
                   f"num_blocks={len(k_ptrs)}")
        
    except Exception as e:
        logger.error(f"Error in after_prefill: {e}", exc_info=True)
        # Don't raise - export failure shouldn't break inference

