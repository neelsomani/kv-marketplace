#include <torch/extension.h>
#include <cuda_runtime.h>

// Ensure peer access between two devices
bool ensure_peer_access(int src_dev, int dst_dev) {
    if (src_dev == dst_dev) {
        return true;
    }
    
    int can_access = 0;
    cudaError_t err = cudaDeviceCanAccessPeer(&can_access, dst_dev, src_dev);
    if (err != cudaSuccess || !can_access) {
        return false;
    }
    
    // Set device contexts
    cudaSetDevice(dst_dev);
    err = cudaDeviceEnablePeerAccess(src_dev, 0);
    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
        return false;
    }
    
    return true;
}

// Asynchronous peer-to-peer memory copy
void memcpy_peer_async(uintptr_t dst, int dst_dev, uintptr_t src, int src_dev,
                       size_t bytes, uintptr_t stream_ptr) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    void* dst_ptr = reinterpret_cast<void*>(dst);
    void* src_ptr = reinterpret_cast<void*>(src);
    
    cudaMemcpyPeerAsync(dst_ptr, dst_dev, src_ptr, src_dev, bytes, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ensure_peer_access", &ensure_peer_access,
          "Ensure peer access between two CUDA devices");
    m.def("memcpy_peer_async", &memcpy_peer_async,
          "Asynchronous peer-to-peer memory copy");
}
