"""File-based backend for PrefixIndex to enable cross-process sharing."""

import os
import json
import tempfile
import fcntl
from typing import List, Optional, Set, Tuple
from .prefix_index import PrefixIndex
from .shared_memory_store import SharedMemoryJSONStore


class FileBasedPrefixIndex(PrefixIndex):
    """PrefixIndex with file-based persistence for multi-process sharing.
    
    Stores token sequences to a file and rebuilds the trie on initialization.
    This allows the prefix index to persist across process boundaries.
    """
    
    def __init__(self, prefix_index_file: Optional[str] = None):
        """Initialize file-based prefix index.
        
        Args:
            prefix_index_file: Path to prefix index file. If None, uses default temp location.
        """
        # Initialize parent PrefixIndex
        super().__init__()
        
        if prefix_index_file is None:
            prefix_index_dir = os.path.join(tempfile.gettempdir(), 'kv_marketplace_registry')
            os.makedirs(prefix_index_dir, exist_ok=True)
            prefix_index_file = os.path.join(prefix_index_dir, 'prefix_index.json')
        
        self._prefix_index_file = prefix_index_file
        self._lock_file = prefix_index_file + '.lock'
        
        # Load existing sequences from file and rebuild trie
        self._load_sequences()
    
    def _acquire_lock(self):
        """Acquire file lock for exclusive access."""
        os.makedirs(os.path.dirname(self._lock_file), exist_ok=True)
        lock_fd = os.open(self._lock_file, os.O_CREAT | os.O_RDWR)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        return lock_fd
    
    def _release_lock(self, lock_fd):
        """Release file lock."""
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    
    def _read_sequences(self) -> List[List[int]]:
        """Read token sequences from file."""
        if not os.path.exists(self._prefix_index_file):
            return []
        
        try:
            with open(self._prefix_index_file, 'r') as f:
                data = json.load(f)
                # Convert hex strings back to token sequences
                sequences = data.get('sequences', [])
                return sequences
        except (json.JSONDecodeError, IOError, KeyError):
            return []
    
    def _write_sequences(self, sequences: List[List[int]]) -> None:
        """Write token sequences to file atomically."""
        temp_file = self._prefix_index_file + '.tmp'
        data = {
            'sequences': sequences
        }
        with open(temp_file, 'w') as f:
            json.dump(data, f)
        os.replace(temp_file, self._prefix_index_file)
    
    def _load_sequences(self) -> None:
        """Load sequences from file and rebuild trie."""
        sequences = self._read_sequences()
        # Rebuild trie by inserting all stored sequences
        for tokens in sequences:
            # Use parent's insert method to rebuild trie
            super().insert(tokens)
    
    def insert(self, tokens: List[int], prefix_hash: Optional[bytes] = None) -> bytes:
        """Insert a token sequence into the trie and persist to file.
        
        Args:
            tokens: Token IDs to insert
            prefix_hash: Optional pre-computed hash, otherwise computed
            
        Returns:
            The hash of the token sequence
        """
        # Insert into trie (parent method)
        result_hash = super().insert(tokens, prefix_hash)
        
        # Persist to file (with locking)
        lock_fd = self._acquire_lock()
        try:
            sequences = self._read_sequences()
            # Only add if not already present (avoid duplicates)
            if tokens not in sequences:
                sequences.append(tokens)
                self._write_sequences(sequences)
        finally:
            self._release_lock(lock_fd)
        
        return result_hash


class SharedMemoryPrefixIndex(PrefixIndex):
    """PrefixIndex that stores token sequences inside shared memory."""

    def __init__(
        self,
        shm_name: Optional[str] = None,
        size_bytes: Optional[int] = None,
    ) -> None:
        super().__init__()
        name = shm_name or 'kv_marketplace_prefix_index'
        size = size_bytes or (32 * 1024 * 1024)
        self._store = SharedMemoryJSONStore(
            name=name,
            size_bytes=size,
            default_factory=lambda: {'sequences': []},
        )
        self._sequence_cache: Set[Tuple[int, ...]] = set()
        self._load_sequences()

    @property
    def storage_backend(self) -> str:
        return getattr(self._store, 'backend', 'unknown')

    def _load_sequences(self) -> None:
        data = self._store.load(shared=True)
        sequences = data.get('sequences', []) if isinstance(data, dict) else []
        for tokens in sequences:
            if not isinstance(tokens, list):
                continue
            seq_tuple = tuple(tokens)
            if seq_tuple in self._sequence_cache:
                continue
            self._sequence_cache.add(seq_tuple)
            super().insert(list(tokens))

    def insert(self, tokens: List[int], prefix_hash: Optional[bytes] = None) -> bytes:
        result_hash = super().insert(tokens, prefix_hash)
        tokens_copy = list(tokens)
        seq_tuple = tuple(tokens_copy)
        if seq_tuple in self._sequence_cache:
            return result_hash

        inserted = {'added': False}

        def _mutator(payload):
            if not isinstance(payload, dict):
                payload = {'sequences': []}
            sequences = payload.setdefault('sequences', [])
            if tokens_copy not in sequences:
                sequences.append(tokens_copy)
                inserted['added'] = True
            return payload

        self._store.update(_mutator)
        if inserted['added']:
            self._sequence_cache.add(seq_tuple)
        return result_hash
