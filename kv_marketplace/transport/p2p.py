"""Python wrapper over CUDA P2P extension for peer-to-peer memory copy."""

import torch
from typing import List, Dict, Any, Optional
from . import p2p_cuda


def _get_stream_ptr(stream: torch.cuda.Stream) -> int:
    """Get the CUDA stream pointer from a PyTorch stream object.
    
    Args:
        stream: PyTorch CUDA stream object
        
    Returns:
        Integer pointer to the CUDA stream
    """
    # Try multiple methods to get the stream pointer
    if hasattr(stream, 'cuda_stream'):
        return stream.cuda_stream
    elif hasattr(stream, '_cudastream'):
        # Internal PyTorch attribute (may change across versions)
        return stream._cudastream
    else:
        # Fallback: get current stream for the device
        # This works when stream is the current active stream
        device = stream.device
        current_stream = torch.cuda.current_stream(device)
        if hasattr(current_stream, 'cuda_stream'):
            return current_stream.cuda_stream
        # Last resort: return 0 for default stream
        # This will use the default CUDA stream
        return 0


class PeerCopy:
    """Wrapper for CUDA peer-to-peer memory copy operations."""
    
    @staticmethod
    def ensure_peer_access(src_dev: int, dst_dev: int) -> bool:
        """Ensure peer access is enabled between two devices.
        
        Args:
            src_dev: Source GPU device ID
            dst_dev: Destination GPU device ID
            
        Returns:
            True if peer access is available/enabled, False otherwise
        """
        if src_dev == dst_dev:
            return True  # Same device, no peer access needed
        return p2p_cuda.ensure_peer_access(src_dev, dst_dev)
    
    @staticmethod
    def copy_kv(dst_dev: int, dst_k_ptrs: List[int], dst_v_ptrs: List[int],
                src_dev: int, src_k_ptrs: List[int], src_v_ptrs: List[int],
                n_layers: int, length: int, meta: Dict[str, Any],
                stream: Optional[torch.cuda.Stream] = None):
        """Copy KV tensors from source to destination GPU.
        
        Args:
            dst_dev: Destination GPU device ID
            dst_k_ptrs: Destination K tensor pointers (per-layer)
            dst_v_ptrs: Destination V tensor pointers (per-layer)
            src_dev: Source GPU device ID
            src_k_ptrs: Source K tensor pointers (per-layer)
            src_v_ptrs: Source V tensor pointers (per-layer)
            n_layers: Number of layers
            length: Number of tokens to copy
            meta: Layout metadata
            stream: CUDA stream for async operations (optional)
        """
        if stream is None:
            stream = torch.cuda.current_stream()
        
        # Get CUDA stream pointer
        stream_ptr = _get_stream_ptr(stream)
        
        for layer_idx in range(n_layers):
            # Calculate size per layer based on meta
            # This is a placeholder - actual size calculation depends on layout
            n_kv_heads = meta.get('n_kv_heads', 1)
            head_dim = meta.get('head_dim', 128)
            page_size = meta.get('page_size', 16)
            
            # Size in bytes per token for K and V
            kv_size_per_token = n_kv_heads * head_dim * 2 * 2  # 2 for K+V, 2 for float16
            
            # Total size for this layer
            total_size = length * kv_size_per_token
            
            # Copy K
            p2p_cuda.memcpy_peer_async(
                dst_k_ptrs[layer_idx], dst_dev,
                src_k_ptrs[layer_idx], src_dev,
                total_size // 2,  # K size
                stream_ptr
            )
            
            # Copy V
            p2p_cuda.memcpy_peer_async(
                dst_v_ptrs[layer_idx], dst_dev,
                src_v_ptrs[layer_idx], src_dev,
                total_size // 2,  # V size
                stream_ptr
            )

