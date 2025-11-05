"""KVRegistry: map (compat, prefix hash) → KVHandle with pluggable backends."""

from typing import Optional
from .compat import KVCompat
from .registry_backend import RegistryBackend, InProcessRegistryBackend


class KVHandle:
    """Handle to exported KV cache with metadata for transfer."""
    
    def __init__(self, device_id: int, k_ptrs: list, v_ptrs: list, 
                 length: int, layout_meta: dict, device_uuid: Optional[str] = None):
        """Initialize a KV handle.
        
        Args:
            device_id: GPU device ID where KV cache resides (local ordinal, for backward compatibility)
            k_ptrs: Per-layer device pointers/addresses for K tensors
            v_ptrs: Per-layer device pointers/addresses for V tensors
            length: Number of tokens materialized
            layout_meta: Layout metadata (strides, page_size, etc.)
            device_uuid: Globally unique device UUID (CUDA UUID). If None, device_id will be used (legacy mode).
        """
        self.device_id = device_id  # Local ordinal (for backward compatibility)
        self.device_uuid = device_uuid  # Globally unique UUID (preferred)
        self.k_ptrs = k_ptrs
        self.v_ptrs = v_ptrs
        self.length = length
        self.layout_meta = layout_meta


class KVRegistry:
    """Registry mapping (compat, prefix_hash) to KVHandle.
    
    Stores exported KV caches for reuse by matching requests.
    Uses a pluggable backend for storage (in-process, file-based, or distributed).
    
    Example:
        # Default: in-process backend
        registry = KVRegistry()
        
        # File-based for multi-process on same machine
        from kv_marketplace.registry_backend import FileBasedRegistryBackend
        backend = FileBasedRegistryBackend()
        registry = KVRegistry(backend=backend)
    """
    
    def __init__(self, backend: Optional[RegistryBackend] = None):
        """Initialize registry with a backend.
        
        Args:
            backend: Registry backend to use. If None, uses InProcessRegistryBackend.
        """
        if backend is None:
            backend = InProcessRegistryBackend()
        self._backend = backend
    
    def register(self, compat: KVCompat, prefix_hash: bytes, handle: KVHandle):
        """Register a KV handle for a given compatibility and prefix.
        
        Args:
            compat: Compatibility configuration
            prefix_hash: Hash of the token prefix
            handle: KV handle to register
        """
        self._backend.register(compat, prefix_hash, handle)
    
    def lookup(self, compat: KVCompat, prefix_hash: bytes) -> Optional[KVHandle]:
        """Look up a KV handle by compatibility and prefix hash.
        
        Args:
            compat: Compatibility configuration to match
            prefix_hash: Hash of the token prefix to match
            
        Returns:
            KVHandle if found, None otherwise
        """
        return self._backend.lookup(compat, prefix_hash)
    
    def remove(self, compat: KVCompat, prefix_hash: bytes):
        """Remove a registered KV handle.
        
        Args:
            compat: Compatibility configuration
            prefix_hash: Hash of the token prefix
        """
        self._backend.remove(compat, prefix_hash)
    
    def size(self) -> int:
        """Get the number of registered handles.
        
        Returns:
            Number of registered handles
        """
        return self._backend.size()
    
    @property
    def _registry(self):
        """Backward compatibility: access internal dict for in-process backend.
        
        Note: This property is deprecated. Use size() method instead.
        """
        if isinstance(self._backend, InProcessRegistryBackend):
            return self._backend._registry
        # For other backends, return a dummy dict to avoid breaking existing code
        # that accesses _registry directly
        return {}

