"""vLLM adapter implementing the callback contract."""

import logging
import json
import os
import tempfile
from typing import Callable, Optional, Tuple, List, TypedDict
import torch
from .types import KVLayout, AllocatedKV
from ..compat import KVCompat
from ..registry import KVRegistry, KVHandle
from ..registry_backend import FileBasedRegistryBackend
from ..prefix_index import PrefixIndex, TrieNode
from ..prefix_index_backend import FileBasedPrefixIndex
from ..transport.p2p import PeerCopy
from ..transport import p2p_cuda

logger = logging.getLogger(__name__)
# Set kv-marketplace logger to INFO to avoid debug overhead in hot paths
# (vLLM may set root logger to DEBUG, but we want to avoid debug logging overhead)
logger.setLevel(logging.INFO)

# Detect if we should use file-based backend for multi-process sharing
# Use environment variable to enable, or auto-detect based on data parallelism
_USE_FILE_BACKEND = os.environ.get('KV_MARKETPLACE_FILE_BACKEND', '').lower() in ('1', 'true', 'yes')

# Global instances for registry and prefix index
# Use file-based backend if enabled for multi-process sharing on same machine
if _USE_FILE_BACKEND:
    logger.info("kv-marketplace: Using file-based registry and prefix index backends for multi-process sharing")
    _registry = KVRegistry(backend=FileBasedRegistryBackend())
    _prefix_index = FileBasedPrefixIndex()  # File-based prefix index for cross-process persistence
else:
    _registry = KVRegistry()  # Default: in-process backend
    _prefix_index = PrefixIndex()  # Default: in-process prefix index

# Default minimum prefix length (can be overridden by vLLM flags)
_MIN_PREFIX_LENGTH = 64

# Statistics tracking
_import_hits = 0
_import_misses = 0
_import_lcp_lengths = []
_local_hits = 0  # Cache hits on same GPU (skipped by marketplace)
_cross_hits = 0  # Cross-GPU cache hits (marketplace imports)

# File-based stats sharing for multiprocessing
_stats_file = None
_stats_lock_file = None


def get_stable_device_id(local_device_id: int) -> Optional[str]:
    """Get a stable, globally unique device identifier.
    
    Uses PCI bus ID (preferred) or UUID as fallback. This ensures correct
    device identification even when CUDA_VISIBLE_DEVICES makes each process
    see its GPU as device 0.
    
    Args:
        local_device_id: Local device ordinal (e.g., from torch.cuda.current_device())
        
    Returns:
        Stable device identifier string (PCI bus ID or UUID), or None if unavailable.
        Returns None (not device name) to avoid collisions when multiple GPUs share the same name.
    """
    try:
        props = torch.cuda.get_device_properties(local_device_id)
        # Prefer PCI bus ID (most stable)
        if getattr(props, 'pci_bus_id', None):
            return props.pci_bus_id
        # PyTorch >=2.1 exposes props.uuid (bytes-like or str depending on build)
        uuid_attr = getattr(props, 'uuid', None)
        if uuid_attr:
            return str(uuid_attr)
        # Fallback via our CUDA ext (NVML/CUDA runtime)
        if hasattr(p2p_cuda, 'get_device_uuid'):
            uuid = p2p_cuda.get_device_uuid(local_device_id)
            if uuid:
                return uuid
    except Exception as e:
        logger.warning(f"Could not get stable device ID for device {local_device_id}: {e}")
    
    # No stable id → return None (force "not same GPU" classification)
    return None


def resolve_stable_id_to_local_ordinal(stable_device_id: str) -> Optional[int]:
    """Map a stable device identifier back to local device ordinal.
    
    Args:
        stable_device_id: Stable device identifier (PCI bus ID or UUID)
        
    Returns:
        Local device ordinal if found, None otherwise.
        Note: Returns None if the device is not visible in this process
        (e.g., when CUDA_VISIBLE_DEVICES hides it).
    """
    try:
        num_devices = torch.cuda.device_count()
        for local_idx in range(num_devices):
            props = torch.cuda.get_device_properties(local_idx)
            
            # Check PCI bus ID first
            if getattr(props, 'pci_bus_id', None) == stable_device_id:
                return local_idx
            
            # Check UUID via PyTorch props
            uuid_attr = getattr(props, 'uuid', None)
            if uuid_attr and str(uuid_attr) == stable_device_id:
                return local_idx
            
            # Check UUID via our CUDA extension
            if hasattr(p2p_cuda, 'get_device_uuid'):
                uuid = p2p_cuda.get_device_uuid(local_idx)
                if uuid == stable_device_id:
                    return local_idx
    except Exception as e:
        logger.warning(f"Could not resolve stable device ID {stable_device_id} to local ordinal: {e}")
    
    return None


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
            'local_hits': _local_hits,
            'cross_hits': _cross_hits,
            'registry_size': _registry.size(),
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
                'local_hits': 0,
                'cross_hits': 0,
                'import_lcp_lengths': [],
                'registry_size': 0,
                'prefix_index_size': 0,
            }
        
        # Aggregate stats from all processes
        total_hits = 0
        total_misses = 0
        total_local_hits = 0
        total_cross_hits = 0
        all_lcp_lengths = []
        max_registry_size = 0
        max_prefix_index_size = 0
        
        for pid, proc_stats in processes.items():
            total_hits += proc_stats.get('import_hits', 0)
            total_misses += proc_stats.get('import_misses', 0)
            total_local_hits += proc_stats.get('local_hits', 0)
            total_cross_hits += proc_stats.get('cross_hits', 0)
            all_lcp_lengths.extend(proc_stats.get('import_lcp_lengths', []))
            max_registry_size = max(max_registry_size, proc_stats.get('registry_size', 0))
            max_prefix_index_size = max(max_prefix_index_size, proc_stats.get('prefix_index_size', 0))
        
        return {
            'import_hits': total_hits,
            'import_misses': total_misses,
            'local_hits': total_local_hits,
            'cross_hits': total_cross_hits,
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
            'local_hits': 0,
            'cross_hits': 0,
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
        - local_hits: Number of same-GPU cache hits (skipped by marketplace)
        - cross_hits: Number of cross-GPU cache hits (marketplace imports)
        - import_lcp_lengths: List of LCP lengths for successful imports (combined)
        - registry_size: Maximum registry size across processes
        - prefix_index_size: Maximum prefix index size across processes
    """
    global _import_hits, _import_misses, _import_lcp_lengths, _local_hits, _cross_hits
    
    # Get in-process stats
    # For registry_size, use the backend's size() method (works for both in-process and file-based)
    # For prefix_index_size, use the size() method if available, otherwise fall back to _hash_to_length
    registry_size = _registry.size() if hasattr(_registry, 'size') else len(_registry._registry)
    prefix_index_size = _prefix_index.size() if hasattr(_prefix_index, 'size') else len(_prefix_index._hash_to_length)
    
    local_stats = {
        'import_hits': _import_hits,
        'import_misses': _import_misses,
        'local_hits': _local_hits,
        'cross_hits': _cross_hits,
        'import_lcp_lengths': _import_lcp_lengths.copy(),
        'registry_size': registry_size,
        'prefix_index_size': prefix_index_size,
    }
    
    # Read aggregated stats from file (includes all worker processes)
    file_stats = _read_stats_from_file()
    
    # File stats are aggregated across all worker processes (summed for hits/misses)
    # Use file stats as primary source since hooks run in worker processes
    # Also merge with local stats to handle cases where hooks might run in main process
    # For registry_size and prefix_index_size, prefer the current actual size over stale file stats
    # (file stats might be stale from previous runs, but current size is authoritative)
    merged_stats = {
        # File stats are aggregated (summed), so use them directly
        # They will be 0 if no worker processes have written yet
        'import_hits': file_stats['import_hits'],
        'import_misses': file_stats['import_misses'],
        'local_hits': file_stats.get('local_hits', 0),
        'cross_hits': file_stats.get('cross_hits', 0),
        # Combine LCP lengths from both sources (deduplicate)
        'import_lcp_lengths': list(set(local_stats['import_lcp_lengths'] + file_stats.get('import_lcp_lengths', []))),
        # For sizes, always use the current actual size from local_stats (authoritative)
        # File stats might be stale from previous runs, but current size reflects the actual state
        'registry_size': local_stats['registry_size'],
        'prefix_index_size': local_stats['prefix_index_size'],
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


def clear_registry():
    """Clear the registry and prefix index to ensure an empty cache.
    
    This clears all stored KV cache handles and prefix sequences.
    For file-based backends, this clears the underlying files.
    For in-process backends, this clears the internal data structures.
    Also clears the stats file to remove stale statistics.
    """
    global _registry, _prefix_index
    
    # Clear registry
    try:
        if isinstance(_registry._backend, FileBasedRegistryBackend):
            # For file-based backend, clear the registry file (create empty file if it doesn't exist)
            lock_fd = _registry._backend._acquire_lock()
            try:
                _registry._backend._write_registry({})
            finally:
                _registry._backend._release_lock(lock_fd)
        else:
            # For in-process backend, clear the internal dict
            _registry._backend._registry.clear()
    except Exception as e:
        logger.warning(f"Failed to clear registry: {e}")
    
    # Clear prefix index
    try:
        if isinstance(_prefix_index, FileBasedPrefixIndex):
            # For file-based prefix index, clear the file (create empty file if it doesn't exist)
            prefix_index_file = _prefix_index._prefix_index_file
            lock_fd = _prefix_index._acquire_lock()
            try:
                _prefix_index._write_sequences([])
            finally:
                _prefix_index._release_lock(lock_fd)
            # Also clear the in-memory trie
            _prefix_index._root = TrieNode()
            _prefix_index._hash_to_length.clear()
        else:
            # For in-process prefix index, clear the internal structures
            _prefix_index._root = TrieNode()
            _prefix_index._hash_to_length.clear()
    except Exception as e:
        logger.warning(f"Failed to clear prefix index: {e}")
    
    # Clear stats file to remove stale statistics
    try:
        stats_file = _get_stats_file_path()
        if os.path.exists(stats_file):
            os.remove(stats_file)
        # Write empty stats to ensure file exists
        _write_stats_to_file()
    except Exception as e:
        logger.warning(f"Failed to clear stats file: {e}")


def get_registry_keys() -> List[Tuple[bytes, bytes]]:
    """Get all keys in the registry.
    
    Returns:
        List of (compat_checksum, prefix_hash) tuples
    """
    global _registry
    try:
        return _registry.list_keys()
    except Exception as e:
        logger.warning(f"Failed to get registry keys: {e}")
        return []


class VLLMImportCtx(TypedDict, total=False):
    """Context for importing KV cache before prefill."""
    device_id: int
    compat: KVCompat
    tokens: List[int]
    alloc_prefix: Callable[[int], AllocatedKV]
    layout: KVLayout
    stream: int  # cudaStream_t as int
    allow_pcie: bool  # Optional: allow import even without peer access (via PCIe)


class VLLMExportCtx(TypedDict):
    """Context for exporting KV cache after prefill."""
    device_id: int
    compat: KVCompat
    tokens: List[int]
    kv_pages: AllocatedKV   # [0:prompt_len]
    layout: KVLayout
    length: int


def _validate_meta(meta: dict):
    """Validate that meta dictionary has all required fields with valid values."""
    for k in ("n_layers", "n_kv_heads", "head_dim", "page_size"):
        v = meta.get(k)
        if not isinstance(v, int) or v <= 0:
            raise RuntimeError(f"[kv-mkt] invalid meta[{k}]={v}")


def _valid_ptr_list(ptrs, n_layers):
    """Validate pointer list: must be list, correct length, integers (not bools), and nonzero."""
    # ints and nonzero
    if not isinstance(ptrs, list) or len(ptrs) != n_layers:
        return False
    # prevent bools sneaking in (bool is subclass of int)
    return all(isinstance(p, int) and not isinstance(p, bool) and p != 0 for p in ptrs)


def before_prefill(ctx: VLLMImportCtx) -> Optional[Tuple[int, AllocatedKV]]:
    """Hook called before prefill to attempt KV cache import.
    
    Args:
        ctx: Import context with request information
        
    Returns:
        Tuple of (lcp_len, dst_alloc) if import succeeds, None otherwise
    """
    global _import_misses, _import_hits, _local_hits, _cross_hits
    
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
        # logger.debug(f"kv-marketplace before_prefill: Looking for LCP in prefix_index (size={prefix_index_size}, tokens_len={len(tokens)}, first_10_tokens={tokens[:10]})")
        lcp_result = _prefix_index.find_lcp(tokens)
        if lcp_result is None:
            logger.debug(f"kv-marketplace before_prefill: No LCP found for tokens (length={len(tokens)}, prefix_index_size={prefix_index_size})")
            _import_misses += 1
            _write_stats_to_file()
            return None
        
        lcp_len, prefix_hash = lcp_result
        # logger.debug(f"kv-marketplace before_prefill: Found LCP (length={lcp_len}, prefix_hash={prefix_hash.hex()[:8]}..., min_prefix={min_prefix})")
        
        # Check if LCP meets minimum length requirement
        if lcp_len < min_prefix:
            logger.debug(f"kv-marketplace before_prefill: LCP length {lcp_len} < min_prefix {min_prefix}, skipping import")
            _import_misses += 1
            _write_stats_to_file()
            return None
        
        # Lookup handle in registry
        # logger.debug(f"kv-marketplace before_prefill: Looking up handle in registry (size={_registry.size()}, compat_hash={compat.checksum.hex()[:8]}..., prefix_hash={prefix_hash.hex()[:8]}...)")
        handle = _registry.lookup(compat, prefix_hash)
        if handle is None:
            logger.debug(f"kv-marketplace before_prefill: No handle found in registry for compat/prefix_hash")
            _import_misses += 1
            _write_stats_to_file()
            return None
        
        # Use handle's layout_meta (authoritative) for the copy, with fallback to layout
        meta = handle.layout_meta or {
            'n_layers': layout['n_layers'],
            'n_kv_heads': layout['n_kv_heads'] or layout.get('n_heads', 0),
            'head_dim': layout['head_dim'],
            'page_size': layout['page_size'],
            **layout.get('strides', {})
        }
        
        # Validate meta before using it
        _validate_meta(meta)
        
        # Allocate destination KV cache
        dst_alloc = ctx['alloc_prefix'](lcp_len)
        
        # Require k_ptrs and v_ptrs to exist - fail loudly if missing
        if 'k_ptrs' not in dst_alloc:
            logger.error(
                f"kv-marketplace: alloc_prefix returned dict missing 'k_ptrs' key. "
                f"Returned keys: {list(dst_alloc.keys())}, type: {type(dst_alloc)}"
            )
            _import_misses += 1
            _write_stats_to_file()
            return None
        
        if 'v_ptrs' not in dst_alloc:
            logger.error(
                f"kv-marketplace: alloc_prefix returned dict missing 'v_ptrs' key. "
                f"Returned keys: {list(dst_alloc.keys())}, type: {type(dst_alloc)}"
            )
            _import_misses += 1
            _write_stats_to_file()
            return None
        
        # Harden pointer validation before copying (length + nonzero)
        n_layers = meta['n_layers']
        k_ptrs = dst_alloc['k_ptrs']
        v_ptrs = dst_alloc['v_ptrs']
        
        if not _valid_ptr_list(k_ptrs, n_layers) or not _valid_ptr_list(v_ptrs, n_layers):
            logger.error(
                f"kv-marketplace: invalid dst ptrs "
                f"(k_len={len(k_ptrs)}, v_len={len(v_ptrs)}, "
                f"expected n_layers={n_layers}, "
                f"k_ptrs={k_ptrs[:5] if len(k_ptrs) > 0 else []}, "
                f"v_ptrs={v_ptrs[:5] if len(v_ptrs) > 0 else []}); treating as miss"
            )
            _import_misses += 1
            _write_stats_to_file()
            return None
        
        # Require k_ptrs and v_ptrs to exist on handle - fail loudly if missing
        if not hasattr(handle, 'k_ptrs'):
            logger.error(
                f"kv-marketplace: handle missing 'k_ptrs' attribute. "
                f"Handle type: {type(handle)}, handle attributes: {dir(handle)}"
            )
            try:
                _registry.delete(compat, prefix_hash)
            except Exception:
                pass
            _import_misses += 1
            _write_stats_to_file()
            return None
        
        if not hasattr(handle, 'v_ptrs'):
            logger.error(
                f"kv-marketplace: handle missing 'v_ptrs' attribute. "
                f"Handle type: {type(handle)}, handle attributes: {dir(handle)}"
            )
            try:
                _registry.delete(compat, prefix_hash)
            except Exception:
                pass
            _import_misses += 1
            _write_stats_to_file()
            return None
        
        handle_k_ptrs = handle.k_ptrs
        handle_v_ptrs = handle.v_ptrs
        
        if not _valid_ptr_list(handle_k_ptrs, n_layers) or not _valid_ptr_list(handle_v_ptrs, n_layers):
            logger.error(
                f"kv-marketplace: invalid src ptrs in handle "
                f"(k_len={len(handle_k_ptrs)}, v_len={len(handle_v_ptrs)}, "
                f"expected n_layers={n_layers}, "
                f"k_ptrs={handle_k_ptrs[:5] if len(handle_k_ptrs) > 0 else []}, "
                f"v_ptrs={handle_v_ptrs[:5] if len(handle_v_ptrs) > 0 else []}); dropping handle"
            )
            try:
                _registry.delete(compat, prefix_hash)
            except Exception:
                pass
            _import_misses += 1
            _write_stats_to_file()
            return None
        
        # Log the lengths you're about to use so you can see the mismatch instantly
        logger.info(
            "kv-marketplace: ptr lens: dst(k=%d,v=%d) src(k=%d,v=%d) n_layers=%d",
            len(k_ptrs), len(v_ptrs),
            len(handle_k_ptrs), len(handle_v_ptrs), n_layers
        )
        
        # Get stable device IDs for comparison
        handle_global_id = getattr(handle, 'device_global_id', None) or getattr(handle, 'device_uuid', None)
        current_global_id = get_stable_device_id(device_id)
        
        logger.info(f"kv-marketplace: Registry lookup succeeded! handle.device_id={handle.device_id}, handle.device_global_id={handle_global_id}, current_device_id={device_id}, current_global_id={current_global_id}, lcp_len={lcp_len}")
        
        # Check if cache hit is on the same GPU using stable global ID
        # This is reliable across processes even when CUDA_VISIBLE_DEVICES makes both see device 0
        # Only compare when both IDs are available (not None)
        is_same_gpu = (handle_global_id is not None and
                      current_global_id is not None and
                      handle_global_id == current_global_id)
        
        if is_same_gpu:
            # Same GPU - let vLLM's native cache management handle local reuse
            # This avoids unnecessary memory allocation and copying overhead
            logger.info(f"kv-marketplace: Cache hit on same GPU {device_id} (global_id={current_global_id}, lcp_len={lcp_len}), skipping marketplace import (let vLLM handle local reuse)")
            _local_hits += 1
            _write_stats_to_file()
            return None
        
        # Cross-GPU cache hit - map source stable ID to local ordinal for P2P operations
        src_dev_local = None
        if handle_global_id:
            # Map source stable ID to local ordinal
            src_dev_local = resolve_stable_id_to_local_ordinal(handle_global_id)
            if src_dev_local is None:
                logger.warning(f"Could not map source global ID {handle_global_id} to local ordinal, trying UUID fallback")
                # Try UUID-based mapping as fallback
                if hasattr(p2p_cuda, 'uuid_to_local_ordinal') and handle_global_id:
                    src_dev_local = p2p_cuda.uuid_to_local_ordinal(handle_global_id)
                    if src_dev_local < 0:
                        src_dev_local = None
        
        if src_dev_local is None:
            # Source device is not visible in this process (e.g., CUDA_VISIBLE_DEVICES hides it).
            # Peer-to-peer copy requires both devices to be visible in the same process.
            logger.warning(
                f"Could not resolve source device {handle_global_id} to local ordinal. "
                f"This usually means the source GPU is not visible in this process "
                f"(e.g., CUDA_VISIBLE_DEVICES is set). Peer-to-peer copy requires both "
                f"GPUs to be visible. Cannot proceed with cross-GPU import."
            )
            _import_misses += 1
            _write_stats_to_file()
            return None
        
        # Cross-GPU cache hit - proceed with marketplace import
        logger.info(f"kv-marketplace: Cross-GPU cache hit found! src_dev={src_dev_local} (global_id={handle_global_id}), dst_dev={device_id} (global_id={current_global_id}), lcp_len={lcp_len}, proceeding with import")
        
        # Check peer access requirement using mapped local ordinals
        # Note: PeerCopy.ensure_peer_access has C++-level caching, so no need to cache here
        allow_pcie = ctx.get('allow_pcie', False)
        has_peer_access = PeerCopy.ensure_peer_access(src_dev_local, device_id)
        
        if not has_peer_access:
            if not allow_pcie:
                logger.warning(
                    f"Peer access not available between GPU {src_dev_local} and GPU {device_id}, "
                    f"and --kv-allow-pcie not set. Skipping import."
                )
                return None
            else:
                logger.warning(
                    f"Peer access not available between GPU {src_dev_local} and GPU {device_id}, "
                    f"but --kv-allow-pcie is set. Attempting import anyway (may fail or use slower path)."
                )
        
        # Extract page ranges from dst_alloc if available (for coalescing optimization)
        dst_page_ranges = dst_alloc.get('page_ranges')
        # Handle page ranges from handle if available
        src_page_ranges = None
        if hasattr(handle, 'layout_meta') and handle.layout_meta:
            src_page_ranges = handle.layout_meta.get('page_ranges')
        
        # Get dtype size from cache config or compat
        dtype_size = 2  # fp16 default
        try:
            # Try to get from compat's layout_config
            cache_dtype = ''
            if isinstance(compat, KVCompat):
                cache_dtype = str(compat.layout_config.get('dtype', '')).lower()
            elif isinstance(compat, dict):
                cache_dtype = str(compat.get('kv_layout', {}).get('dtype', '')).lower()
            
            if "float8" in cache_dtype or "fp8" in cache_dtype:
                dtype_size = 1
            elif "float16" in cache_dtype or "fp16" in cache_dtype or "half" in cache_dtype:
                dtype_size = 2
            elif "bfloat16" in cache_dtype or "bf16" in cache_dtype:
                dtype_size = 2
            elif "float32" in cache_dtype or "fp32" in cache_dtype:
                dtype_size = 4
        except Exception:
            pass
        
        # Log bytes to copy and estimate expected bandwidth using meta (authoritative)
        bytes_per_tok = meta['n_layers'] * meta['n_kv_heads'] * meta['head_dim'] * 2 * dtype_size  # K and V
        nbytes = bytes_per_tok * lcp_len
        logger.info(f"kv-marketplace: will copy ~{nbytes/1e6:.1f} MB for LCP={lcp_len}")
        
        # Copy KV tensors from source to destination with coalescing
        # Use mapped local ordinals (src_dev_local, device_id) for correct P2P routing
        # Pass the stream directly - PeerCopy.copy_kv accepts int (CUDA stream pointer)
        PeerCopy.copy_kv(
            dst_dev=device_id,
            dst_k_ptrs=k_ptrs,
            dst_v_ptrs=v_ptrs,
            src_dev=src_dev_local,  # Use mapped local ordinal, not handle.device_id
            src_k_ptrs=handle_k_ptrs,
            src_v_ptrs=handle_v_ptrs,
            n_layers=meta['n_layers'],
            length=lcp_len,
            meta={
                'n_kv_heads': meta['n_kv_heads'],
                'head_dim': meta['head_dim'],
                'page_size': meta['page_size'],
                **meta.get('strides', {})
            },
            stream=ctx['stream'],  # Pass int stream pointer directly
            dst_page_ranges=dst_page_ranges,
            src_page_ranges=src_page_ranges
        )
        
        # Synchronize copy completion before suffix prefill (MVP requirement)
        # This ensures the imported KV cache is fully materialized before vLLM continues
        PeerCopy.synchronize_stream(ctx['stream'])
        
        logger.info(f"kv-marketplace: Successfully imported {lcp_len} tokens from GPU {src_dev_local} to GPU {device_id} (coalesced segments, synchronized)")
        
        # Track successful import
        global _import_hits, _import_lcp_lengths, _cross_hits
        _import_hits += 1
        _cross_hits += 1
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
        
        # Require k_ptrs and v_ptrs to exist - fail loudly if missing
        if 'k_ptrs' not in kv_pages:
            logger.error(
                f"kv-marketplace: kv_pages dict missing 'k_ptrs' key. "
                f"Returned keys: {list(kv_pages.keys())}, type: {type(kv_pages)}, "
                f"length={length}, device_id={device_id}"
            )
            return
        
        if 'v_ptrs' not in kv_pages:
            logger.error(
                f"kv-marketplace: kv_pages dict missing 'v_ptrs' key. "
                f"Returned keys: {list(kv_pages.keys())}, type: {type(kv_pages)}, "
                f"length={length}, device_id={device_id}"
            )
            return
        
        k_ptrs = kv_pages['k_ptrs']
        v_ptrs = kv_pages['v_ptrs']
        
        """
        logger.debug(
            f"kv-marketplace after_prefill: length={length}, device_id={device_id}, "
            f"k_ptrs={len(k_ptrs)}, v_ptrs={len(v_ptrs)}, "
            f"k_ptrs_sample={k_ptrs[:5] if k_ptrs else []}, "
            f"v_ptrs_sample={v_ptrs[:5] if v_ptrs else []}"
        )
        """
        
        # If pointers are empty, we can't export (get_prefill_pages may have failed)
        if not k_ptrs or not v_ptrs:
            logger.error(
                f"kv-marketplace: KV cache pointers are empty for export: length={length}, "
                f"k_ptrs={len(k_ptrs)}, v_ptrs={len(v_ptrs)}, "
                f"k_ptrs={k_ptrs[:5] if len(k_ptrs) > 0 else []}, "
                f"v_ptrs={v_ptrs[:5] if len(v_ptrs) > 0 else []}. "
                f"This may indicate get_prefill_pages needs implementation or blocks aren't available yet."
            )
            return
        
        # Compute prefix hash and insert into prefix index
        prefix_hash = _prefix_index.insert(prefix_tokens)
        
        # Get stable globally unique device identifier (PCI bus ID or UUID)
        # This ensures correct device identification even when CUDA_VISIBLE_DEVICES
        # makes each process see its GPU as device 0
        device_global_id = get_stable_device_id(device_id)
        
        # Also get UUID for backward compatibility
        device_uuid = None
        if hasattr(p2p_cuda, 'get_device_uuid'):
            device_uuid = p2p_cuda.get_device_uuid(device_id)
        
        logger.debug(f"Exporting KV cache: device_id={device_id}, device_global_id={device_global_id}, device_uuid={device_uuid}")
        
        # Build KVHandle from context
        page_ranges = kv_pages.get('page_ranges')
        layout_meta = {
            'n_layers': layout['n_layers'],
            'n_kv_heads': layout['n_kv_heads'],
            'head_dim': layout['head_dim'],
            'page_size': layout['page_size'],
            **layout.get('strides', {})
        }
        if page_ranges:
            layout_meta['page_ranges'] = page_ranges
        
        handle = KVHandle(
            device_id=device_id,  # Local ordinal (for backward compatibility)
            k_ptrs=k_ptrs,
            v_ptrs=v_ptrs,
            length=length,
            layout_meta=layout_meta,
            device_uuid=device_uuid,  # UUID (for backward compatibility)
            device_global_id=device_global_id  # Stable global ID (PCI bus ID or UUID, preferred)
        )
        
        # Register in registry
        _registry.register(compat, prefix_hash, handle)
        
        # Update file stats (registry size changed)
        _write_stats_to_file()
        
        logger.debug(f"Exported KV cache: length={length}, device={device_id}, prefix_hash={prefix_hash.hex()[:8]}..., "
                   f"num_blocks={len(k_ptrs)}")
        
    except Exception as e:
        logger.error(f"Error in after_prefill: {e}", exc_info=True)
        # Don't raise - export failure shouldn't break inference
