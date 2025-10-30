"""KVRegistry: in-process map (compat, prefix hash) → KVHandle."""

from typing import Dict, Optional, Tuple
from .compat import KVCompat


class KVHandle:
    """Handle to exported KV cache with metadata for transfer."""
    
    def __init__(self, device_id: int, k_ptrs: list, v_ptrs: list, 
                 length: int, layout_meta: dict):
        """Initialize a KV handle.
        
        Args:
            device_id: GPU device ID where KV cache resides
            k_ptrs: Per-layer device pointers/addresses for K tensors
            v_ptrs: Per-layer device pointers/addresses for V tensors
            length: Number of tokens materialized
            layout_meta: Layout metadata (strides, page_size, etc.)
        """
        self.device_id = device_id
        self.k_ptrs = k_ptrs
        self.v_ptrs = v_ptrs
        self.length = length
        self.layout_meta = layout_meta


class KVRegistry:
    """Registry mapping (compat, prefix_hash) to KVHandle.
    
    Stores exported KV caches for reuse by matching requests.
    """
    
    def __init__(self):
        """Initialize an empty registry."""
        self._registry: Dict[Tuple[KVCompat, bytes], KVHandle] = {}
    
    def register(self, compat: KVCompat, prefix_hash: bytes, handle: KVHandle):
        """Register a KV handle for a given compatibility and prefix.
        
        Args:
            compat: Compatibility configuration
            prefix_hash: Hash of the token prefix
            handle: KV handle to register
        """
        key = (compat, prefix_hash)
        self._registry[key] = handle
    
    def lookup(self, compat: KVCompat, prefix_hash: bytes) -> Optional[KVHandle]:
        """Look up a KV handle by compatibility and prefix hash.
        
        Args:
            compat: Compatibility configuration to match
            prefix_hash: Hash of the token prefix to match
            
        Returns:
            KVHandle if found, None otherwise
        """
        key = (compat, prefix_hash)
        return self._registry.get(key)
    
    def remove(self, compat: KVCompat, prefix_hash: bytes):
        """Remove a registered KV handle.
        
        Args:
            compat: Compatibility configuration
            prefix_hash: Hash of the token prefix
        """
        key = (compat, prefix_hash)
        self._registry.pop(key, None)

