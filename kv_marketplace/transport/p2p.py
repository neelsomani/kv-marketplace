"""Python wrapper over CUDA P2P extension for peer-to-peer memory copy."""

import torch
from typing import List, Dict, Any, Optional, Union, Tuple
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
    def disable_peer_access(src_dev: int, dst_dev: int) -> bool:
        """Disable peer access between two devices.
        
        Args:
            src_dev: Source GPU device ID
            dst_dev: Destination GPU device ID
            
        Returns:
            True if peer access was disabled or already disabled, False on error
        """
        if src_dev == dst_dev:
            return True  # Same device, no peer access to disable
        return p2p_cuda.disable_peer_access(src_dev, dst_dev)
    
    @staticmethod
    def _coalesce_page_ranges(page_range: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]], 
                              length: int, page_size: int) -> List[Tuple[int, int]]:
        """Coalesce page ranges into contiguous segments for a single layer.
        
        Args:
            page_range: Optional page range for a layer. Can be:
                       - None: assume contiguous from page 0
                       - Tuple[int, int]: single (start_page, end_page) range
                       - List[Tuple[int, int]]: multiple (start_page, end_page) ranges to coalesce
            length: Number of tokens
            page_size: Tokens per page
            
        Returns:
            List of (start_page, end_page) tuples representing contiguous segments
        """
        if page_range is None:
            # No page range info, assume contiguous from page 0
            num_pages = (length + page_size - 1) // page_size
            return [(0, num_pages)] if num_pages > 0 else []
        
        # Normalize to list of tuples
        if isinstance(page_range, tuple):
            # Single range, return as-is
            return [page_range]
        
        if not page_range:
            # Empty list, assume contiguous
            num_pages = (length + page_size - 1) // page_size
            return [(0, num_pages)] if num_pages > 0 else []
        
        # Multiple ranges: sort and coalesce overlapping or adjacent ranges
        sorted_ranges = sorted(page_range, key=lambda x: x[0])
        coalesced = []
        
        for start, end in sorted_ranges:
            if not coalesced:
                coalesced.append((start, end))
            else:
                last_start, last_end = coalesced[-1]
                # If current range overlaps or is adjacent to last range
                if start <= last_end + 1:
                    # Merge ranges
                    coalesced[-1] = (last_start, max(last_end, end))
                else:
                    # Add new range
                    coalesced.append((start, end))
        
        return coalesced
    
    @staticmethod
    def copy_kv(dst_dev: int, dst_k_ptrs: List[int], dst_v_ptrs: List[int],
                src_dev: int, src_k_ptrs: List[int], src_v_ptrs: List[int],
                n_layers: int, length: int, meta: Dict[str, Any],
                stream: Optional[Union[torch.cuda.Stream, int]] = None,
                dst_page_ranges: Optional[List[Tuple[int, int]]] = None,
                src_page_ranges: Optional[List[Tuple[int, int]]] = None):
        """Copy KV tensors from source to destination GPU with coalesced segments.
        
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
            stream: CUDA stream for async operations. Can be:
                   - torch.cuda.Stream object
                   - int (raw CUDA stream pointer)
                   - None (uses current stream)
            dst_page_ranges: Optional list of (start_page, end_page) tuples per layer for destination
            src_page_ranges: Optional list of (start_page, end_page) tuples per layer for source
        """
        # Get CUDA stream pointer
        if stream is None:
            stream = torch.cuda.current_stream()
            stream_ptr = _get_stream_ptr(stream)
        elif isinstance(stream, int):
            # Already a raw pointer
            stream_ptr = stream
        else:
            # torch.cuda.Stream object
            stream_ptr = _get_stream_ptr(stream)
        
        # Layout metadata
        n_kv_heads = meta.get('n_kv_heads', 1)
        head_dim = meta.get('head_dim', 128)
        page_size = meta.get('page_size', 16)
        
        # Size in bytes per token for K and V
        kv_size_per_token = n_kv_heads * head_dim * 2 * 2  # 2 for K+V, 2 for float16
        bytes_per_page = page_size * kv_size_per_token
        
        # Set device context once before all copies (optimization: avoid 48 device switches)
        # All copies go to the same destination device, so we only need to set it once
        torch.cuda.set_device(dst_dev)
        
        for layer_idx in range(n_layers):
            # Coalesce page ranges if provided, otherwise use contiguous assumption
            if dst_page_ranges and layer_idx < len(dst_page_ranges):
                dst_ranges = PeerCopy._coalesce_page_ranges(
                    dst_page_ranges[layer_idx], length, page_size
                )
            else:
                dst_ranges = PeerCopy._coalesce_page_ranges(None, length, page_size)
            
            if src_page_ranges and layer_idx < len(src_page_ranges):
                src_ranges = PeerCopy._coalesce_page_ranges(
                    src_page_ranges[layer_idx], length, page_size
                )
            else:
                src_ranges = PeerCopy._coalesce_page_ranges(None, length, page_size)
            
            # Use the ranges to determine segments to copy
            # For now, assume pointers are to the start of the layer's KV cache
            # and we need to calculate offsets based on page ranges
            # If ranges are provided, copy each coalesced segment
            # Otherwise, fall back to single contiguous copy
            
            if dst_ranges and src_ranges and len(dst_ranges) == len(src_ranges):
                # Copy coalesced segments
                for (dst_start, dst_end), (src_start, src_end) in zip(dst_ranges, src_ranges):
                    num_pages = dst_end - dst_start
                    segment_bytes = num_pages * bytes_per_page
                    
                    # Calculate offsets (assuming pointers point to start of layer KV cache)
                    dst_k_offset = dst_start * bytes_per_page // 2  # K size
                    dst_v_offset = dst_start * bytes_per_page // 2  # V size
                    src_k_offset = src_start * bytes_per_page // 2
                    src_v_offset = src_start * bytes_per_page // 2
                    
                    # Copy K segment
                    p2p_cuda.memcpy_peer_async(
                        dst_k_ptrs[layer_idx] + dst_k_offset, dst_dev,
                        src_k_ptrs[layer_idx] + src_k_offset, src_dev,
                        segment_bytes // 2,  # K size
                        stream_ptr
                    )
                    
                    # Copy V segment
                    p2p_cuda.memcpy_peer_async(
                        dst_v_ptrs[layer_idx] + dst_v_offset, dst_dev,
                        src_v_ptrs[layer_idx] + src_v_offset, src_dev,
                        segment_bytes // 2,  # V size
                        stream_ptr
                    )
            else:
                # Fallback: single contiguous copy (original behavior)
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
    
    @staticmethod
    def synchronize_stream(stream: Optional[Union[torch.cuda.Stream, int]] = None):
        """Synchronize CUDA stream to ensure copy operations complete.
        
        Args:
            stream: CUDA stream to synchronize. Can be:
                   - torch.cuda.Stream object
                   - int (raw CUDA stream pointer)
                   - None (synchronizes current stream)
        """
        if stream is None:
            torch.cuda.synchronize()
        elif isinstance(stream, int):
            # Use precise stream synchronization via C++ extension
            # This only blocks until this specific stream completes,
            # allowing other streams to continue running (much faster)
            p2p_cuda.synchronize_stream(stream)
        else:
            # torch.cuda.Stream object
            stream.synchronize()

