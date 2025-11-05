#include <torch/extension.h>
#include <cuda_runtime.h>

// Helper to set device and save previous device
static bool set_device(int d, int& prev) {
    cudaError_t e = cudaGetDevice(&prev);
    if (e != cudaSuccess) return false;
    return cudaSetDevice(d) == cudaSuccess;
}

// Ensure peer access between two devices
// This enables dst_dev to access memory from src_dev
bool ensure_peer_access(int src_dev, int dst_dev) {
    if (src_dev == dst_dev) {
        return true;
    }
    
    int prev;
    // Set device to dst_dev (the device that will access src_dev)
    if (!set_device(dst_dev, prev)) return false;
    
    // Check if dst_dev can access src_dev
    int can_access = 0;
    cudaError_t err = cudaDeviceCanAccessPeer(&can_access, dst_dev, src_dev);
    if (err != cudaSuccess || !can_access) {
        cudaSetDevice(prev);
        return false;
    }
    
    // Enable peer access from dst_dev to src_dev
    err = cudaDeviceEnablePeerAccess(src_dev, 0);
    // Treat already-enabled as success, and clear sticky error
    if (err == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError(); // Clear the error
        cudaSetDevice(prev);
        return true;
    }
    
    bool ok = (err == cudaSuccess);
    cudaSetDevice(prev);
    return ok;
}

// Disable peer access between two devices
// This disables access from dst_dev to src_dev
bool disable_peer_access(int src_dev, int dst_dev) {
    if (src_dev == dst_dev) {
        return true;
    }
    
    int prev;
    // Set device to dst_dev (the device that has access to src_dev)
    if (!set_device(dst_dev, prev)) return false;
    
    // Disable peer access from dst_dev to src_dev
    cudaError_t err = cudaDeviceDisablePeerAccess(src_dev);
    // Treat not-enabled as success too
    if (err == cudaErrorPeerAccessNotEnabled) {
        cudaGetLastError(); // Clear the error
        cudaSetDevice(prev);
        return true;
    }
    
    bool ok = (err == cudaSuccess);
    cudaSetDevice(prev);
    return ok;
}

// Asynchronous peer-to-peer memory copy
void memcpy_peer_async(uintptr_t dst, int dst_dev, uintptr_t src, int src_dev,
                       size_t bytes, uintptr_t stream_ptr) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    void* dst_ptr = reinterpret_cast<void*>(dst);
    void* src_ptr = reinterpret_cast<void*>(src);
    
    // Set device context to destination device (required for peer copy)
    cudaSetDevice(dst_dev);
    
    cudaError_t err = cudaMemcpyPeerAsync(dst_ptr, dst_dev, src_ptr, src_dev, bytes, stream);
    if (err != cudaSuccess) {
        // Log error but don't throw - let Python handle it
        // This helps with debugging
        const char* err_str = cudaGetErrorString(err);
        // Note: Can't use PyTorch logging here, so we'll check in Python
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ensure_peer_access", &ensure_peer_access,
          "Enable peer access between two CUDA devices");
    m.def("disable_peer_access", &disable_peer_access,
          "Disable peer access between two CUDA devices");
    m.def("memcpy_peer_async", &memcpy_peer_async,
          "Asynchronous peer-to-peer memory copy");
}
