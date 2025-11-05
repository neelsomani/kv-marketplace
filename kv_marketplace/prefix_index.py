"""PrefixIndex: LCP (Longest Common Prefix) over token IDs using a trie with rolling hashes."""

from typing import List, Optional, Tuple
import xxhash


class TrieNode:
    """Node in the prefix trie."""
    
    def __init__(self):
        self.children: dict[int, 'TrieNode'] = {}  # token_id -> child node
        self.prefix_hash: Optional[bytes] = None  # hash of sequence ending at this node
        self.length: int = 0  # length of prefix ending at this node


class PrefixIndex:
    """Efficient prefix index using a trie structure with rolling hashes for LCP lookup.
    
    Uses a trie for efficient longest common prefix matching, combined with
    xxh3_64 rolling hashes for fast verification.
    """
    
    def __init__(self):
        """Initialize an empty prefix index."""
        self._root = TrieNode()
        self._hash_to_length: dict[bytes, int] = {}  # cache for hash -> prefix length
    
    def size(self) -> int:
        """Get the number of unique prefixes stored in the index."""
        return len(self._hash_to_length)
    
    def insert(self, tokens: List[int], prefix_hash: Optional[bytes] = None) -> bytes:
        """Insert a token sequence into the trie.
        
        Args:
            tokens: Token IDs to insert
            prefix_hash: Optional pre-computed hash, otherwise computed
            
        Returns:
            The hash of the token sequence
        """
        if prefix_hash is None:
            prefix_hash = self._hash_sequence(tokens)
        
        # Insert into trie, marking all prefix nodes
        node = self._root
        depth = 0
        for token in tokens:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
            depth += 1
            
            # Mark this prefix node with its hash
            # This allows finding partial prefix matches
            prefix = tokens[:depth]
            prefix_hash_at_depth = self._hash_sequence(prefix)
            node.prefix_hash = prefix_hash_at_depth
            node.length = depth
            self._hash_to_length[prefix_hash_at_depth] = depth
        
        return prefix_hash
    
    def find_lcp(self, query_tokens: List[int]) -> Optional[Tuple[int, bytes]]:
        """Find the longest common prefix match using trie traversal.
        
        Args:
            query_tokens: Token sequence to find LCP for
            
        Returns:
            Tuple of (lcp_length, prefix_hash) if found, None otherwise
        """
        if not query_tokens:
            return None
        
        node = self._root
        best_lcp = 0
        best_hash = None
        
        # Traverse trie following query_tokens
        for i, token in enumerate(query_tokens):
            if token not in node.children:
                break
            
            node = node.children[token]
            
            # If this node marks the end of a stored prefix, update best match
            if node.prefix_hash is not None:
                # Verify the hash matches (rolling hash verification)
                prefix = query_tokens[:i + 1]
                expected_hash = self._hash_sequence(prefix)
                if expected_hash == node.prefix_hash:
                    best_lcp = i + 1
                    best_hash = node.prefix_hash
        
        if best_lcp > 0 and best_hash is not None:
            return (best_lcp, best_hash)
        return None
    
    def _hash_sequence(self, tokens: List[int]) -> bytes:
        """Compute xxh3_64 rolling hash of token sequence.
        
        Args:
            tokens: Token IDs to hash
            
        Returns:
            8-byte hash
        """
        # Use xxhash for fast rolling hash
        h = xxhash.xxh3_64()
        for token in tokens:
            h.update(token.to_bytes(8, byteorder='big'))
        return h.digest()
    
    def verify(self, prefix_hash: bytes, tokens: List[int]) -> bool:
        """Verify that a hash matches a token sequence.
        
        Args:
            prefix_hash: Hash to verify
            tokens: Token sequence to check
            
        Returns:
            True if hash matches the sequence
        """
        expected_hash = self._hash_sequence(tokens)
        return expected_hash == prefix_hash
