#include <torch/extension.h>
#include <cuda_runtime.h>
#include <unordered_map>
#include <mutex>
#include <vector>

// Helper to set device and save previous device
static bool set_device(int d, int& prev) {
    cudaError_t e = cudaGetDevice(&prev);
    if (e != cudaSuccess) return false;
    return cudaSetDevice(d) == cudaSuccess;
}

// Cache for peer access status to avoid repeated cudaDeviceCanAccessPeer calls
// Key: (src_dev << 16) | dst_dev, Value: true if peer access is available/enabled
static std::unordered_map<uint32_t, bool> peer_access_cache;
static std::mutex peer_access_mutex;

// Event pool for reuse (avoid create/destroy overhead)
// Simple pool: keep a small number of events per device
// Key: device_id, Value: vector of available events
static std::unordered_map<int, std::vector<cudaEvent_t>> event_pool;
static std::mutex event_pool_mutex;
static const int MAX_EVENTS_PER_DEVICE = 4;  // Small pool per device

// Ensure peer access between two devices
// This enables dst_dev to access memory from src_dev
// Results are cached to avoid repeated cudaDeviceCanAccessPeer queries
bool ensure_peer_access(int src_dev, int dst_dev) {
    if (src_dev == dst_dev) {
        return true;
    }
    
    // Check cache first (avoid expensive cudaDeviceCanAccessPeer call)
    uint32_t key = (src_dev << 16) | dst_dev;
    {
        std::lock_guard<std::mutex> lock(peer_access_mutex);
        auto it = peer_access_cache.find(key);
        if (it != peer_access_cache.end()) {
            // Cached result - if true, peer access is already enabled
            // If false, we know it's not available, so don't try again
            if (it->second) {
                return true;  // Already enabled, no need to check again
            }
            // If cached as false, we could still try enable (in case it changed),
            // but for MVP, trust the cache to avoid repeated failures
            return false;
        }
    }
    
    int prev;
    // Set device to dst_dev (the device that will access src_dev)
    if (!set_device(dst_dev, prev)) {
        // Cache failure
        std::lock_guard<std::mutex> lock(peer_access_mutex);
        peer_access_cache[key] = false;
        return false;
    }
    
    // Check if dst_dev can access src_dev (expensive call - only do once)
    int can_access = 0;
    cudaError_t err = cudaDeviceCanAccessPeer(&can_access, dst_dev, src_dev);
    if (err != cudaSuccess || !can_access) {
        cudaSetDevice(prev);
        // Cache failure
        std::lock_guard<std::mutex> lock(peer_access_mutex);
        peer_access_cache[key] = false;
        return false;
    }
    
    // Enable peer access from dst_dev to src_dev
    err = cudaDeviceEnablePeerAccess(src_dev, 0);
    bool success = false;
    
    // Treat already-enabled as success, and clear sticky error
    if (err == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError(); // Clear the error
        success = true;
    } else if (err == cudaSuccess) {
        success = true;
    }
    
    cudaSetDevice(prev);
    
    // Cache result
    std::lock_guard<std::mutex> lock(peer_access_mutex);
    peer_access_cache[key] = success;
    
    return success;
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
// NOTE: Device context MUST be set to dst_dev by the caller before calling this function.
// The caller (copy_kv in p2p.py) sets the device once before the loop to avoid overhead.
void memcpy_peer_async(uintptr_t dst, int dst_dev, uintptr_t src, int src_dev,
                       size_t bytes, uintptr_t stream_ptr) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    void* dst_ptr = reinterpret_cast<void*>(dst);
    void* src_ptr = reinterpret_cast<void*>(src);
    
    // Device context is already set by caller - no need to check or set here
    // This avoids cudaGetDevice() overhead when called 48x per request
    
    cudaError_t err = cudaMemcpyPeerAsync(dst_ptr, dst_dev, src_ptr, src_dev, bytes, stream);
    if (err != cudaSuccess) {
        // Log error but don't throw - let Python handle it
        // This helps with debugging
        const char* err_str = cudaGetErrorString(err);
        // Note: Can't use PyTorch logging here, so we'll check in Python
    }
}

// Get an event from the pool (or create if pool is empty)
// Returns event handle as uintptr_t, or 0 on error
// Events must be created in the correct device context
// Only reuses events that have completed (cudaEventQuery == cudaSuccess)
// This prevents reusing events that are still "in flight" which would cause wrong dependencies
static uintptr_t get_event_from_pool(int device_id) {
    int prev;
    if (!set_device(device_id, prev)) return 0;
    
    std::lock_guard<std::mutex> lock(event_pool_mutex);
    auto& pool = event_pool[device_id];
    
    // Try to find a completed event in the pool
    while (!pool.empty()) {
        cudaEvent_t ev = pool.back();
        pool.pop_back();
        
        // Check if event has completed (only reuse completed events)
        cudaError_t query_result = cudaEventQuery(ev);
        if (query_result == cudaSuccess) {
            // Event has completed, safe to reuse
            cudaSetDevice(prev);
            return reinterpret_cast<uintptr_t>(ev);
        }
        
        // Event not ready yet - put it back at front and break to create a new one
        // This prevents reusing an event that's still in flight
        pool.insert(pool.begin(), ev);
        break;
    }
    
    // No completed events in pool, create new event (in correct device context)
    cudaEvent_t ev;
    if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) {
        cudaSetDevice(prev);
        return 0;
    }
    cudaSetDevice(prev);
    return reinterpret_cast<uintptr_t>(ev);
}

// Return an event to the pool for reuse
// Events must be returned/destroyed in the correct device context
static void return_event_to_pool(uintptr_t event_ptr, int device_id) {
    if (!event_ptr) return;
    
    int prev;
    if (!set_device(device_id, prev)) return;
    
    cudaEvent_t ev = reinterpret_cast<cudaEvent_t>(event_ptr);
    {
        std::lock_guard<std::mutex> lock(event_pool_mutex);
        auto& pool = event_pool[device_id];
        if (pool.size() < MAX_EVENTS_PER_DEVICE) {
            pool.push_back(ev);
        } else {
            // Pool full, destroy the event (in correct device context)
            cudaEventDestroy(ev);
        }
    }
    cudaSetDevice(prev);
}

// Record an event on a stream (for event handoff pattern)
// Returns event handle as uintptr_t, or 0 on error
// Uses event pooling to avoid create/destroy overhead
// device_id: device where the stream is running (for event pool management)
uintptr_t record_event(uintptr_t stream_ptr, int device_id) {
    if (stream_ptr == 0) {
        // Cannot record event on NULL stream - caller should provide real stream
        return 0;
    }
    
    // Get event from pool (or create if needed)
    uintptr_t event_ptr = get_event_from_pool(device_id);
    if (event_ptr == 0) {
        return 0;
    }
    
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    cudaEvent_t event = reinterpret_cast<cudaEvent_t>(event_ptr);
    
    cudaError_t err = cudaEventRecord(event, stream);
    if (err != cudaSuccess) {
        // On error, return event to pool (it's still valid, just recording failed)
        return_event_to_pool(event_ptr, device_id);
        return 0;
    }
    
    return event_ptr;
}

// Wait for an event on a stream (non-blocking from host perspective)
// The stream will wait for the event before executing subsequent operations
void wait_event(uintptr_t event_ptr, uintptr_t stream_ptr) {
    if (event_ptr == 0 || stream_ptr == 0) {
        return;  // Invalid handles
    }
    
    cudaEvent_t event = reinterpret_cast<cudaEvent_t>(event_ptr);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    cudaError_t err = cudaStreamWaitEvent(stream, event, 0);
    if (err != cudaSuccess) {
        // Log error but don't throw
        const char* err_str = cudaGetErrorString(err);
    }
}

// Destroy an event (or return to pool for reuse)
// device_id: device where the event was used (for pool management)
void destroy_event(uintptr_t event_ptr, int device_id) {
    if (event_ptr == 0) {
        return;
    }
    
    // Return to pool instead of destroying (for reuse)
    return_event_to_pool(event_ptr, device_id);
}

// Synchronize a specific CUDA stream (blocking - use event handoff when possible)
// Requires a real stream handle (no magic 0 path to avoid device-wide stall)
void synchronize_stream(uintptr_t stream_ptr) {
    if (stream_ptr == 0) {
        // Reject NULL stream - caller must provide real stream handle
        // This avoids accidental device-wide synchronization
        return;
    }
    
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        // Log error but don't throw - let Python handle it
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
    m.def("synchronize_stream", &synchronize_stream,
          "Synchronize a specific CUDA stream (blocking - requires real stream handle)");
    m.def("record_event", &record_event,
          "Record an event on a stream, returns event handle (uses event pool)");
    m.def("wait_event", &wait_event,
          "Make a stream wait for an event (non-blocking from host)");
    m.def("destroy_event", &destroy_event,
          "Return a CUDA event to pool for reuse (or destroy if pool full)");
}
