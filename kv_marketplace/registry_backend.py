"""Registry backend abstraction for KV cache storage.

Supports multiple backends:
- InProcessRegistryBackend: Single-process, in-memory storage
- FileBasedRegistryBackend: Multi-process on same machine via file system
- Future: DistributedRegistryBackend for cluster-wide sharing
"""

import os
import json
import tempfile
import fcntl
from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple, TYPE_CHECKING

# Import at runtime to avoid circular dependency
# These will be imported when needed in methods


class RegistryBackend(ABC):
    """Abstract base class for registry storage backends."""
    
    @abstractmethod
    def register(self, compat: 'KVCompat', prefix_hash: bytes, handle: 'KVHandle') -> None:
        """Register a KV handle for a given compatibility and prefix.
        
        Args:
            compat: Compatibility configuration
            prefix_hash: Hash of the token prefix
            handle: KV handle to register
        """
        pass
    
    @abstractmethod
    def lookup(self, compat: 'KVCompat', prefix_hash: bytes) -> Optional['KVHandle']:
        """Look up a KV handle by compatibility and prefix hash.
        
        Args:
            compat: Compatibility configuration to match
            prefix_hash: Hash of the token prefix to match
            
        Returns:
            KVHandle if found, None otherwise
        """
        pass
    
    @abstractmethod
    def remove(self, compat: 'KVCompat', prefix_hash: bytes) -> None:
        """Remove a registered KV handle.
        
        Args:
            compat: Compatibility configuration
            prefix_hash: Hash of the token prefix
        """
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get the number of registered handles.
        
        Returns:
            Number of registered handles
        """
        pass


class InProcessRegistryBackend(RegistryBackend):
    """In-process, in-memory registry backend.
    
    Suitable for single-process use cases or when registry is already
    shared via other means (e.g., tensor parallelism).
    """
    
    def __init__(self):
        """Initialize an empty in-process registry."""
        self._registry: Dict[Tuple[bytes, bytes], KVHandle] = {}
    
    def register(self, compat: 'KVCompat', prefix_hash: bytes, handle: 'KVHandle') -> None:
        """Register a KV handle."""
        key = (compat.checksum, prefix_hash)
        self._registry[key] = handle
    
    def lookup(self, compat: 'KVCompat', prefix_hash: bytes) -> Optional['KVHandle']:
        """Look up a KV handle."""
        key = (compat.checksum, prefix_hash)
        return self._registry.get(key)
    
    def remove(self, compat: 'KVCompat', prefix_hash: bytes) -> None:
        """Remove a registered KV handle."""
        key = (compat.checksum, prefix_hash)
        self._registry.pop(key, None)
    
    def size(self) -> int:
        """Get the number of registered handles."""
        return len(self._registry)


class FileBasedRegistryBackend(RegistryBackend):
    """File-based registry backend for multi-process sharing on same machine.
    
    Uses a JSON file with file locking for synchronization across processes.
    This is a stopgap solution for multi-GPU setups on the same machine.
    For production clusters, use a distributed registry service.
    """
    
    def __init__(self, registry_file: Optional[str] = None):
        """Initialize file-based registry.
        
        Args:
            registry_file: Path to registry file. If None, uses default temp location.
        """
        if registry_file is None:
            registry_dir = os.path.join(tempfile.gettempdir(), 'kv_marketplace_registry')
            os.makedirs(registry_dir, exist_ok=True)
            registry_file = os.path.join(registry_dir, 'registry.json')
        
        self._registry_file = registry_file
        self._lock_file = registry_file + '.lock'
    
    def _acquire_lock(self):
        """Acquire file lock for exclusive access."""
        # Ensure lock file exists
        os.makedirs(os.path.dirname(self._lock_file), exist_ok=True)
        lock_fd = os.open(self._lock_file, os.O_CREAT | os.O_RDWR)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        return lock_fd
    
    def _release_lock(self, lock_fd):
        """Release file lock."""
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    
    def _read_registry(self) -> Dict:
        """Read registry from file."""
        if not os.path.exists(self._registry_file):
            return {}
        
        try:
            with open(self._registry_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    
    def _write_registry(self, registry: Dict) -> None:
        """Write registry to file atomically."""
        temp_file = self._registry_file + '.tmp'
        with open(temp_file, 'w') as f:
            json.dump(registry, f)
        os.replace(temp_file, self._registry_file)
    
    def _serialize_handle(self, handle: 'KVHandle') -> Dict:
        """Serialize KVHandle to JSON-serializable dict."""
        return {
            'device_id': handle.device_id,
            'k_ptrs': handle.k_ptrs,
            'v_ptrs': handle.v_ptrs,
            'length': handle.length,
            'layout_meta': handle.layout_meta,
        }
    
    def _deserialize_handle(self, data: Dict) -> 'KVHandle':
        """Deserialize dict to KVHandle."""
        # Import here to avoid circular dependency
        from .registry import KVHandle
        return KVHandle(
            device_id=data['device_id'],
            k_ptrs=data['k_ptrs'],
            v_ptrs=data['v_ptrs'],
            length=data['length'],
            layout_meta=data['layout_meta'],
        )
    
    def _make_key(self, compat: 'KVCompat', prefix_hash: bytes) -> str:
        """Convert (compat, prefix_hash) to a JSON-serializable string key."""
        return f"{compat.checksum.hex()}:{prefix_hash.hex()}"
    
    def register(self, compat: 'KVCompat', prefix_hash: bytes, handle: 'KVHandle') -> None:
        """Register a KV handle."""
        lock_fd = self._acquire_lock()
        try:
            registry = self._read_registry()
            key = self._make_key(compat, prefix_hash)
            registry[key] = self._serialize_handle(handle)
            self._write_registry(registry)
        finally:
            self._release_lock(lock_fd)
    
    def lookup(self, compat: 'KVCompat', prefix_hash: bytes) -> Optional['KVHandle']:
        """Look up a KV handle."""
        lock_fd = self._acquire_lock()
        try:
            registry = self._read_registry()
            key = self._make_key(compat, prefix_hash)
            handle_data = registry.get(key)
            if handle_data is None:
                return None
            return self._deserialize_handle(handle_data)
        finally:
            self._release_lock(lock_fd)
    
    def remove(self, compat: 'KVCompat', prefix_hash: bytes) -> None:
        """Remove a registered KV handle."""
        lock_fd = self._acquire_lock()
        try:
            registry = self._read_registry()
            key = self._make_key(compat, prefix_hash)
            registry.pop(key, None)
            self._write_registry(registry)
        finally:
            self._release_lock(lock_fd)
    
    def size(self) -> int:
        """Get the number of registered handles."""
        lock_fd = self._acquire_lock()
        try:
            registry = self._read_registry()
            return len(registry)
        finally:
            self._release_lock(lock_fd)
