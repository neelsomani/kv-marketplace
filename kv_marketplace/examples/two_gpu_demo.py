#!/usr/bin/env python3
"""Demo script: transport test without vLLM.

Copies dummy buffers GPU0→GPU1 and validates checksums.
"""

import torch
import hashlib
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from kv_marketplace.transport import p2p_cuda
except ImportError as e:
    print(f"Error importing CUDA extension: {e}")
    print("Make sure to build the extension first: pip install -e .")
    sys.exit(1)


def compute_checksum(tensor):
    """Compute MD5 checksum of tensor data."""
    cpu_data = tensor.cpu().numpy()
    return hashlib.md5(cpu_data.tobytes()).hexdigest()


def main():
    if not torch.cuda.is_available():
        print("CUDA not available!")
        sys.exit(1)
    
    if torch.cuda.device_count() < 2:
        print("Need at least 2 GPUs for this demo!")
        sys.exit(1)
    
    print(f"Found {torch.cuda.device_count()} GPUs")
    
    # Create source buffer on GPU 0
    size = 10 * 1024 * 1024  # 10MB
    print(f"\nCreating {size // (1024*1024)}MB buffer on GPU 0...")
    src = torch.randn(size, device='cuda:0', dtype=torch.float16)
    src_hash = compute_checksum(src)
    print(f"Source checksum: {src_hash}")
    
    # Create destination buffer on GPU 1
    print("Creating destination buffer on GPU 1...")
    dst = torch.zeros(size, device='cuda:1', dtype=torch.float16)
    
    # Ensure peer access
    print("\nEnabling peer access between GPU 0 and GPU 1...")
    if not p2p_cuda.ensure_peer_access(0, 1):
        print("ERROR: Failed to enable peer access!")
        print("Make sure GPUs support peer-to-peer access (usually requires NVLink or PCIe gen3+)")
        sys.exit(1)
    print("Peer access enabled successfully!")
    
    # Perform copy
    print("\nPerforming peer-to-peer copy...")
    from kv_marketplace.transport import _get_stream_ptr
    
    stream = torch.cuda.Stream(device=1)
    stream_ptr = _get_stream_ptr(stream)
    
    with torch.cuda.stream(stream):
        p2p_cuda.memcpy_peer_async(
            dst.data_ptr(), 1,  # dst
            src.data_ptr(), 0,  # src
            size * 2,  # bytes (float16 = 2 bytes per element)
            stream_ptr
        )
    
    stream.synchronize()
    print("Copy completed!")
    
    # Verify checksum
    print("\nVerifying checksum...")
    dst_hash = compute_checksum(dst)
    print(f"Destination checksum: {dst_hash}")
    
    if src_hash == dst_hash:
        print("\n✓ SUCCESS: Checksums match! Peer copy validated.")
        return 0
    else:
        print("\n✗ FAILURE: Checksums don't match!")
        return 1


if __name__ == '__main__':
    sys.exit(main())

