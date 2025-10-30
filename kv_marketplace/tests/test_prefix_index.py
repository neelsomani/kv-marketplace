"""Tests for prefix index with LCP lookup."""

import pytest
from kv_marketplace.prefix_index import PrefixIndex


def test_prefix_index_insert_and_lookup():
    """Test basic insert and LCP lookup."""
    index = PrefixIndex()
    
    tokens1 = [1, 2, 3, 4, 5]
    tokens2 = [1, 2, 3, 6, 7]
    tokens3 = [10, 11, 12]
    
    hash1 = index.insert(tokens1)
    hash2 = index.insert(tokens2)
    hash3 = index.insert(tokens3)
    
    # Query with matching prefix
    result = index.find_lcp([1, 2, 3, 4, 5, 8, 9])
    assert result is not None
    lcp_len, prefix_hash = result
    assert lcp_len == 5
    assert prefix_hash == hash1
    
    # Query with shorter matching prefix
    result = index.find_lcp([1, 2, 3, 99])
    assert result is not None
    lcp_len, prefix_hash = result
    assert lcp_len == 3
    
    # Query with no match
    result = index.find_lcp([99, 100])
    assert result is None


def test_prefix_index_lcp_ground_truth():
    """Test LCP against O(N) ground truth for correctness."""
    import random
    
    index = PrefixIndex()
    
    # Generate random token sequences
    sequences = []
    for _ in range(10):
        seq = [random.randint(0, 1000) for _ in range(random.randint(5, 20))]
        sequences.append(seq)
        index.insert(seq)
    
    # Test queries
    for query_seq in sequences[:5]:
        # Full match
        result = index.find_lcp(query_seq)
        assert result is not None
        assert result[0] == len(query_seq)
        
        # Prefix match
        if len(query_seq) > 1:
            prefix = query_seq[:len(query_seq)//2]
            result = index.find_lcp(prefix)
            assert result is not None
            assert result[0] == len(prefix)


def test_prefix_index_verify():
    """Test hash verification."""
    index = PrefixIndex()
    
    tokens = [1, 2, 3, 4, 5]
    prefix_hash = index.insert(tokens)
    
    assert index.verify(prefix_hash, tokens)
    assert not index.verify(prefix_hash, [1, 2, 3, 4, 6])

