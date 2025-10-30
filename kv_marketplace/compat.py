"""KVCompat: gathers model, tokenizer, rope, layout and produces a 128-bit checksum."""

from typing import Dict, Any
import hashlib


class KVCompat:
    """Compatibility checker that produces a checksum for model configuration.
    
    Ensures that KV caches are only reused across compatible model configurations,
    tokenizers, positional encodings, and memory layouts.
    """
    
    def __init__(self, model_params: Dict[str, Any], tokenizer_config: Dict[str, Any],
                 rope_config: Dict[str, Any] = None, layout_config: Dict[str, Any] = None):
        """Initialize compatibility checker with model configuration.
        
        Args:
            model_params: Model parameters (n_layers, n_heads, hidden_size, etc.)
            tokenizer_config: Tokenizer configuration (vocab_size, etc.)
            rope_config: RoPE configuration if applicable
            layout_config: Memory layout configuration (page_size, dtype, etc.)
        """
        self.model_params = model_params
        self.tokenizer_config = tokenizer_config
        self.rope_config = rope_config or {}
        self.layout_config = layout_config or {}
        self._checksum = self._compute_checksum()
    
    def _serialize_value(self, value: Any) -> bytes:
        """Recursively serialize a value to bytes in a deterministic way.
        
        Handles nested dictionaries, lists, and other types.
        """
        if isinstance(value, dict):
            # Sort by key and recursively serialize values
            items = sorted(value.items())
            serialized = b'{'
            for k, v in items:
                serialized += self._serialize_value(k) + b':' + self._serialize_value(v) + b','
            serialized += b'}'
            return serialized
        elif isinstance(value, (list, tuple)):
            # Serialize each element
            serialized = b'[' if isinstance(value, list) else b'('
            for item in value:
                serialized += self._serialize_value(item) + b','
            serialized += b']' if isinstance(value, list) else b')'
            return serialized
        elif isinstance(value, (str, bytes)):
            return repr(value).encode() if isinstance(value, str) else value
        else:
            # For primitive types (int, float, bool, None)
            return repr(value).encode()
    
    def _compute_checksum(self) -> bytes:
        """Compute a 128-bit (16-byte) checksum from all configuration."""
        h = hashlib.sha256()
        
        # Serialize all config in a deterministic order with proper nesting handling
        h.update(self._serialize_value(self.model_params))
        h.update(self._serialize_value(self.tokenizer_config))
        h.update(self._serialize_value(self.rope_config))
        h.update(self._serialize_value(self.layout_config))
        
        # Return first 16 bytes (128 bits)
        return h.digest()[:16]
    
    @property
    def checksum(self) -> bytes:
        """Get the 128-bit compatibility checksum."""
        return self._checksum
    
    def __eq__(self, other: 'KVCompat') -> bool:
        """Check if two configurations are compatible."""
        if not isinstance(other, KVCompat):
            return False
        return self.checksum == other.checksum
    
    def __hash__(self) -> int:
        """Hash based on checksum."""
        return int.from_bytes(self.checksum, byteorder='big')

