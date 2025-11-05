"""Transport module for CUDA peer-to-peer memory transfer."""

# Import p2p_cuda module lazily to avoid enabling peer access at import time
from .p2p import PeerCopy, _get_stream_ptr

__all__ = ['PeerCopy', '_get_stream_ptr']

# Note: We do NOT import p2p_cuda at module level to avoid enabling
# peer access automatically when the module is imported

