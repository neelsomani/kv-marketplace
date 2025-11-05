#!/usr/bin/env python3
"""Demo script: vLLM dual-GPU demo with shared prefix.

Uses vllm-kvm-dev fork with KV marketplace enabled.
Demonstrates import hits, LCP lengths, and latency improvements.

NOTE: vLLM uses multiprocessing with worker processes. Stats are automatically
aggregated across all processes using a file-based system, so the printed
statistics are accurate. The stats file is located at:
/tmp/kv_marketplace_stats/stats.json

For multi-GPU setups (data parallelism), the registry is automatically shared
via file-based backend when KV_MARKETPLACE_FILE_BACKEND=1 is set.
"""

import sys
import os
import time
import argparse
import logging
from typing import List, Dict, Tuple, Any
from collections import defaultdict
from multiprocessing import Process, Queue

# CRITICAL: Set multiprocessing start method to 'spawn' before any CUDA initialization
# This must be done at module level, before any imports that might initialize CUDA
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, or can't be set (e.g., in a child process)
    pass

# Enable logging to see hook activity
logging.basicConfig(level=logging.INFO)

# LLM import will be done lazily to avoid import-time issues
LLM = None


def maybe_get_adapter_funcs():
    """Lazily import adapter functions only when kv-marketplace is enabled.
    
    Returns:
        Tuple of (get_stats_fn, reset_stats_fn) or (None, None) if import fails
    """
    try:
        from kv_marketplace.adapter.vllm import get_stats as _get_stats, reset_stats as _reset_stats
        return _get_stats, _reset_stats
    except Exception:
        return None, None


def create_prefixes_from_system_prompt(system_prompt: str, num_prefixes: int = 10) -> List[str]:
    """Split system prompt into N different prefixes by splitting on '. '.
    
    Args:
        system_prompt: The system prompt to split
        num_prefixes: Number of prefixes to create
        
    Returns:
        List of prefixes (each is a cumulative prefix up to that sentence)
    """
    sentences = system_prompt.split('. ')
    
    # If we have fewer sentences than requested prefixes, use what we have
    if len(sentences) < num_prefixes:
        num_prefixes = len(sentences)
    
    prefixes = []
    current_prefix = ""
    for i in range(num_prefixes):
        if i == 0:
            current_prefix = sentences[i]
        else:
            current_prefix = current_prefix + '. ' + sentences[i]
        prefixes.append(current_prefix)
    
    return prefixes


def create_shared_prefix_prompts(system_prompt: str, user_prompts: List[str]) -> List[str]:
    """Create prompts with shared system prefix."""
    return [f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:" for prompt in user_prompts]


def measure_latency_concurrent(llm, prompts: List[str], sampling_params: Any, 
                               batch_name: str = "batch") -> Tuple[List[float], List[str], List[int], float]:
    """Measure generation latency for prompts using batched generation.
    
    Uses a single batched call to llm.generate() to avoid thread-safety issues.
    vLLM handles batching internally and can distribute requests across GPUs.
    
    Returns:
        Tuple of (latencies in seconds, generated texts, device_ids, total_wall_time)
    """
    import torch
    
    # Measure total wall-clock time for batched execution
    batch_start = time.time()
    
    # Single batched call - vLLM is thread-safe for batched requests
    # This avoids the socket protocol corruption issue from concurrent calls
    try:
        results = llm.generate(prompts, sampling_params)
    except Exception as e:
        print(f"  Error in batched generation: {e}")
        # Return empty results on error
        return ([0.0] * len(prompts), [""] * len(prompts), [-1] * len(prompts), 0.0)
    
    batch_end = time.time()
    total_wall_time = batch_end - batch_start
    
    # Extract outputs and compute per-request latencies
    # Note: For batched calls, we can't measure individual latencies,
    # so we approximate by dividing total time equally
    latencies = []
    outputs = []
    device_ids = []
    
    # Try to detect which GPU was used (approximate)
    device_id = torch.cuda.current_device() if torch.cuda.is_available() else -1
    
    for idx, result in enumerate(results):
        output = ""
        if result and hasattr(result, 'outputs') and len(result.outputs) > 0:
            output = result.outputs[0].text
        
        # Approximate per-request latency as total_time / num_requests
        # This is not exact but gives a reasonable estimate
        latency = total_wall_time / len(prompts) if prompts else 0.0
        
        outputs.append(output)
        latencies.append(latency)
        device_ids.append(device_id)
        
        print(f"  {batch_name}: Request {idx+1} completed (latency: {latency:.4f}s)")
    
    return latencies, outputs, device_ids, total_wall_time


def measure_latency(llm, prompts: List[str], sampling_params: Any) -> Tuple[List[float], List[str]]:
    """Measure generation latency for prompts (sequential, for backward compatibility)."""
    latencies, outputs, _, _ = measure_latency_concurrent(llm, prompts, sampling_params, "sequential")
    return latencies, outputs


def child_run(device_mask: str, model: str, shared_kwargs: Dict, prompts: List[str], 
              sampling_params_dict: Dict, result_queue: Queue, phase_name: str):
    """Child process function that runs one phase of the benchmark.
    
    Sets CUDA_VISIBLE_DEVICES, initializes LLM, runs batched generation,
    and returns results via Queue.
    
    Args:
        device_mask: CUDA_VISIBLE_DEVICES value (e.g., "0" or "1")
        model: Model name or path
        shared_kwargs: Shared LLM initialization kwargs
        prompts: List of prompts to process
        sampling_params_dict: Sampling parameters as dict (to avoid serialization issues)
        result_queue: Queue to put results in
        phase_name: Name of phase for logging
    """
    import os
    import sys
    import logging
    import time
    
    # Set device mask BEFORE any CUDA/PyTorch initialization
    os.environ['CUDA_VISIBLE_DEVICES'] = device_mask
    
    # Force vLLM's internal workers to use spawn instead of fork
    # This must be set before any torch/vllm imports
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_USE_RAY", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    
    # Disable all logging for timing runs (reduce overhead)
    # Also silence Ray/uvloop noise
    os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    os.environ.setdefault("RAY_DISABLE_IMPORT_WARNING", "1")
    logging.disable(logging.CRITICAL)
    
    # Check if kv-marketplace is enabled
    kvm_enabled = bool(shared_kwargs.get("kv_marketplace", False))
    
    # Only enable file backend if kv-marketplace is enabled
    if kvm_enabled:
        os.environ.setdefault('KV_MARKETPLACE_FILE_BACKEND', '1')
    else:
        os.environ.pop('KV_MARKETPLACE_FILE_BACKEND', None)
    
    llm = None
    get_stats = None
    
    try:
        # Import vllm - this will use the editable install from vllm/ folder (which is correct)
        import vllm
        from vllm.entrypoints.llm import LLM
        from vllm.sampling_params import SamplingParams
        
        # Recreate SamplingParams in child process to avoid multiprocessing serialization issues
        # This ensures the object is created fresh in this process context
        sampling_params = SamplingParams(**sampling_params_dict)

        if kvm_enabled:
            # Only import kv_marketplace adapter if it's enabled
            from kv_marketplace.adapter.vllm import get_stats
        else:
            # Skip adapter import entirely if kv-marketplace is disabled
            get_stats = None
        
        # Local version of measure_latency_concurrent for child process.
        # Uses batched generation to avoid thread-safety issues.
        def measure_latency_concurrent_local(llm, prompts, sampling_params, batch_name):
            """Local version of measure_latency_concurrent for child process.
            
            Uses a single batched call to avoid socket protocol corruption from
            concurrent calls to llm.generate() from multiple threads.
            """
            latencies = []
            outputs = []
            batch_start = time.time()
            
            # Single batched call - vLLM is thread-safe for batched requests
            # This avoids the socket protocol corruption issue from concurrent calls
            try:
                results = llm.generate(prompts, sampling_params)
            except Exception as e:
                print(f"  Error in batched generation: {e}")
                # Return empty results on error
                return ([0.0] * len(prompts), [""] * len(prompts), [], 0.0)
            
            batch_end = time.time()
            total_wall_time = batch_end - batch_start
            
            # Extract outputs and compute per-request latencies
            # Note: For batched calls, we can't measure individual latencies,
            # so we approximate by dividing total time equally
            for idx, result in enumerate(results):
                output = ""
                if result and hasattr(result, 'outputs') and len(result.outputs) > 0:
                    output = result.outputs[0].text
                
                # Approximate per-request latency as total_time / num_requests
                # This is not exact but gives a reasonable estimate
                latency = total_wall_time / len(prompts) if prompts else 0.0
                
                outputs.append(output)
                latencies.append(latency)
            
            return latencies, outputs, [], total_wall_time
        
        # NOTE: Do NOT reset_stats() here - it's already called in parent before spawning.
        # Resetting here could accidentally wipe Phase-1 exports before Phase-2 reads them.
        
        # Initialize LLM (will now see only the specified GPU)
        llm = LLM(model=model, **shared_kwargs)
        
        # Run batched generation
        lats, outs, _, wall = measure_latency_concurrent_local(llm, prompts, sampling_params, phase_name)
        
        # Get adapter stats - default to zeros, overwrite only if enabled and callable succeeds
        stats = zero_stats()
        if kvm_enabled and get_stats is not None:
            try:
                s = get_stats()
                stats.update({
                    "import_hits": s.get("import_hits", 0),
                    "import_misses": s.get("import_misses", 0),
                    "local_hits": s.get("local_hits", 0),
                    "cross_hits": s.get("cross_hits", 0),
                    "lcp_lengths": s.get("import_lcp_lengths", []),
                    "registry_size": s.get("registry_size", 0),
                    "prefix_index_size": s.get("prefix_index_size", 0),
                })
            except Exception:
                pass  # keep zeros
        
        # Put results in queue
        result_queue.put({
            "latencies": lats,
            "outputs": outs,
            "wall": wall,
            "stats": stats
        })
    except Exception as e:
        import traceback
        # Put error in queue with traceback for debugging
        result_queue.put({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "latencies": [],
            "outputs": [],
            "wall": 0.0,
            "stats": zero_stats()
        })
    finally:
        # CRITICAL: Always shut down the LLM engine to prevent hanging
        try:
            if llm is not None:
                llm.shutdown()
        except Exception:
            pass
        
        # Belt & suspenders: sync CUDA before exit
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass


def zero_stats() -> Dict:
    """Return a zero-initialized stats dictionary with all required keys."""
    return {
        "import_hits": 0,
        "import_misses": 0,
        "local_hits": 0,
        "cross_hits": 0,
        "lcp_lengths": [],
        "registry_size": 0,
        "prefix_index_size": 0,
    }


def norm_stats(s: Dict) -> Dict:
    """Normalize a stats dictionary, ensuring all required keys exist with defaults."""
    return {
        "import_hits": s.get("import_hits", 0),
        "import_misses": s.get("import_misses", 0),
        "local_hits": s.get("local_hits", 0),
        "cross_hits": s.get("cross_hits", 0),
        "lcp_lengths": s.get("lcp_lengths", []),
        "registry_size": s.get("registry_size", 0),
        "prefix_index_size": s.get("prefix_index_size", 0),
    }


def get_registry_stats(get_stats_fn=None) -> Dict:
    """Get statistics from the KV registry and prefix index.
    
    Args:
        get_stats_fn: Optional function to get stats (lazily imported when kv-marketplace is enabled)
    
    Returns:
        Dictionary with stats or empty defaults if not available
    """
    if get_stats_fn:
        try:
            return get_stats_fn()
        except Exception as e:
            print(f"Warning: Could not get stats: {e}")
    
    # Fallback
    return {
        'registry_size': 0,
        'prefix_index_size': 0,
        'import_hits': 0,
        'import_misses': 0,
        'lcp_lengths': [],
    }


def print_stats_table(results: List[Dict], title: str):
    """Print a formatted statistics table."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    
    if not results:
        print("No results to display.")
        return
    
    # Calculate averages
    avg_latency = sum(r['avg_latency'] for r in results) / len(results)
    avg_throughput = sum(r.get('throughput', 0) for r in results) / len(results) if any('throughput' in r for r in results) else 0
    
    print(f"\nAverage Latency: {avg_latency:.4f}s")
    if avg_throughput > 0:
        print(f"Average Throughput: {avg_throughput:.2f} req/s")
    
    print(f"\n{'Run':<6} {'Latency (s)':<15} {'Throughput':<15} {'Import Hits':<15} {'Avg LCP Len':<15}")
    print("-" * 80)
    
    for i, r in enumerate(results, 1):
        throughput = r.get('throughput', 0)
        import_hits = r.get('import_hits', 0)
        lcp_lengths = r.get('import_lcp_lengths', [])
        avg_lcp = sum(lcp_lengths) / len(lcp_lengths) if lcp_lengths else 0
        print(f"{i:<6} {r['avg_latency']:<15.4f} {throughput:<15.2f} {import_hits:<15} {avg_lcp:<15.1f}")


def run_benchmark(
    model: str,
    system_prompt: str,
    user_prompts: List[str],
    num_runs: int = 3,
    kv_marketplace: bool = True,
    kv_min_prefix: int = 64,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    max_model_len: int = 1024,
    enable_prefix_caching: bool = False,
    **llm_kwargs
) -> Dict:
    """Run benchmark with or without kv-marketplace.
    
    Args:
        model: Model name or path
        system_prompt: Shared system prompt prefix
        user_prompts: List of user prompts (will be prefixed with system_prompt)
        num_runs: Number of benchmark runs
        kv_marketplace: Enable kv-marketplace
        kv_min_prefix: Minimum prefix length for import
        gpu_memory_utilization: GPU memory utilization
        tensor_parallel_size: Tensor parallelism (1 = single GPU)
        max_model_len: Maximum model length
        **llm_kwargs: Additional LLM arguments
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*80}")
    print(f"  Benchmark: {'WITH kv-marketplace' if kv_marketplace else 'WITHOUT kv-marketplace'}")
    print(f"{'='*80}")
    print(f"Model: {model}")
    print(f"System prompt length: {len(system_prompt)} chars")
    print(f"Number of user prompts: {len(user_prompts)}")
    print(f"Number of runs: {num_runs}")
    print(f"KV Marketplace: {kv_marketplace}")
    if kv_marketplace:
        print(f"KV Min Prefix: {kv_min_prefix}")
    
    # Create 10 different prefixes from system prompt by splitting on '. '
    print(f"\nCreating {10} different prefixes from system prompt...")
    prefixes = create_prefixes_from_system_prompt(system_prompt, num_prefixes=10)
    print(f"Created {len(prefixes)} prefixes (lengths: {[len(p) for p in prefixes]})")
    
    # Create prompts using these prefixes (one user prompt per prefix)
    # Use first N user prompts, or repeat if needed
    num_prefixes = len(prefixes)
    user_prompts_to_use = (user_prompts * ((num_prefixes // len(user_prompts)) + 1))[:num_prefixes]
    
    # Phase 1 prompts: each prefix + user prompt
    phase1_prompts = [f"{prefix}\n\nUser: {user_prompt}\n\nAssistant:" 
                     for prefix, user_prompt in zip(prefixes, user_prompts_to_use)]
    
    # Phase 2 prompts: scrambled prefixes (use random permutation)
    import random
    scrambled_prefixes = prefixes.copy()
    random.seed(42)  # For reproducibility
    random.shuffle(scrambled_prefixes)
    phase2_prompts = [f"{prefix}\n\nUser: {user_prompt}\n\nAssistant:" 
                     for prefix, user_prompt in zip(scrambled_prefixes, user_prompts_to_use)]
    
    print(f"\nPhase 1: {len(phase1_prompts)} prompts (warm-up, batched)")
    print(f"Phase 2: {len(phase2_prompts)} prompts (test with scrambled prefixes, batched)")
    
    # Only enable file-based backend if kv-marketplace is enabled and we have multiple GPUs
    # This allows sharing registry across processes on the same machine
    # IMPORTANT: Both processes must see the same file backend path (default: /tmp/kv_marketplace_stats)
    # Do NOT delete the registry between phases - only clear stats, not the registry itself
    import torch
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and kv_marketplace:
        os.environ['KV_MARKETPLACE_FILE_BACKEND'] = '1'
        print(f"Detected {torch.cuda.device_count()} GPUs, file-based registry backend enabled for cross-process sharing")
    else:
        os.environ.pop('KV_MARKETPLACE_FILE_BACKEND', None)
    
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("ERROR: Need at least 2 GPUs for cross-GPU benchmark")
        return None
    
    # WARNING: Device identity issue across processes
    # Each child process sets CUDA_VISIBLE_DEVICES to a single GPU, so both see their GPU as device 0 locally.
    # If the adapter stores device_id as torch.cuda.current_device() (local ordinal), both will store device_id=0,
    # causing confusion in peer-copy logic. The adapter should store globally unique device IDs (UUID or PCI bus ID)
    # and translate to local ordinals at import time. This requires adapter changes - see kv_marketplace/adapter/vllm.py
    
    # Lazy import adapter functions only if kv-marketplace is enabled
    get_stats_fn, reset_stats_fn = None, None
    if kv_marketplace:
        get_stats_fn, reset_stats_fn = maybe_get_adapter_funcs()
        if reset_stats_fn:
            try:
                reset_stats_fn()
            except Exception:
                pass
    
    # Shared kwargs for both child processes
    shared_kwargs = {
        'kv_marketplace': kv_marketplace,
        'kv_min_prefix': kv_min_prefix,
        **({'kv_cache_memory_bytes': 36413060300} if kv_marketplace else {'gpu_memory_utilization': gpu_memory_utilization}),
        'tensor_parallel_size': 1,  # Force TP=1 for data parallelism
        'max_model_len': max_model_len,
        'enable_prefix_caching': enable_prefix_caching,  # Disable by default for clean benchmark
        'distributed_executor_backend': 'mp',  # Use multiprocessing instead of Ray
        # Note: worker_multiprocess_method is controlled via VLLM_WORKER_MULTIPROC_METHOD env var
        # (set in child_run before any torch/vllm imports)
        **llm_kwargs
    }
    
    # Convert SamplingParams to dict for multiprocessing (avoids serialization issues with vLLM)
    # This prevents msgspec validation errors when the object is pickled/unpickled across processes
    # We'll recreate it fresh in the child process to ensure proper vLLM serialization
    sampling_params_dict = {
        'temperature': 0.7,
        'top_p': .9,
        'max_tokens': 256,
    }
    
    print("\nUsing separate processes for cross-GPU testing (one process per GPU)")
    
    # Run benchmark with two-phase approach (using separate processes)
    all_latencies = []
    all_outputs = []
    run_stats = []
    
    for run_idx in range(num_runs):
        print(f"\n{'='*80}")
        print(f"Run {run_idx + 1}/{num_runs}")
        print(f"{'='*80}")
        
        # Get stats before run (from parent process, if available)
        stats_before = get_registry_stats(get_stats_fn)
        if run_idx == 0 and kv_marketplace:
            print(f"Stats before run: registry_size={stats_before['registry_size']}, "
                  f"prefix_index_size={stats_before['prefix_index_size']}")
        
        # Create queues for child processes
        q1 = Queue()
        q2 = Queue()
        
        # Phase 1: Process on GPU 0
        print(f"\n--- Phase 1: Warm-up ({len(phase1_prompts)} requests on GPU 0) ---")
        p1 = Process(target=child_run, args=("0", model, shared_kwargs, phase1_prompts, 
                                             sampling_params_dict, q1, "Phase1"))
        p1.start()
        
        # Wait for child result first; if it wedges, we can terminate it
        try:
            r1 = q1.get(timeout=300)  # 5 minute timeout (adjust as needed)
        except Exception as e:
            if p1.is_alive():
                p1.terminate()
                p1.join(10)
            raise RuntimeError(f"Phase 1 child timed out before returning a result: {e}")
        
        # Close queue to prevent leaked semaphores
        q1.close()
        q1.join_thread()
        
        # Now safe to join (should be quick if child exited)
        p1.join(10)
        if p1.is_alive():
            p1.terminate()
            p1.join(10)
        
        if "error" in r1:
            print(f"ERROR in Phase 1: {r1['error']}")
            if "traceback" in r1:
                print(f"Traceback:\n{r1['traceback']}")
            return None
        
        phase1_latencies = r1["latencies"]
        phase1_outputs = r1["outputs"]
        phase1_total_time = r1["wall"]
        phase1_stats = norm_stats(r1["stats"])
        
        print(f"Phase 1 total time: {phase1_total_time:.4f}s")
        print(f"Phase 1 average latency: {sum(phase1_latencies) / len(phase1_latencies):.4f}s per request")
        if kv_marketplace:
            print(f"Phase 1 stats: registry_size={phase1_stats['registry_size']}, "
                  f"local_hits={phase1_stats['local_hits']}, "
                  f"cross_hits={phase1_stats['cross_hits']}")
        
        # Barrier
        time.sleep(0.5)
        
        # Phase 2: Process on GPU 1 (forces cross-GPU)
        print(f"\n--- Phase 2: Test ({len(phase2_prompts)} requests on GPU 1, scrambled prefixes) ---")
        p2 = Process(target=child_run, args=("1", model, shared_kwargs, phase2_prompts, 
                                             sampling_params_dict, q2, "Phase2"))
        p2.start()
        
        # Wait for child result first; if it wedges, we can terminate it
        try:
            r2 = q2.get(timeout=300)  # 5 minute timeout (adjust as needed)
        except Exception as e:
            if p2.is_alive():
                p2.terminate()
                p2.join(10)
            raise RuntimeError(f"Phase 2 child timed out before returning a result: {e}")
        
        # Close queue to prevent leaked semaphores
        q2.close()
        q2.join_thread()
        
        # Now safe to join (should be quick if child exited)
        p2.join(10)
        if p2.is_alive():
            p2.terminate()
            p2.join(10)
        
        if "error" in r2:
            print(f"ERROR in Phase 2: {r2['error']}")
            if "traceback" in r2:
                print(f"Traceback:\n{r2['traceback']}")
            return None
        
        phase2_latencies = r2["latencies"]
        phase2_outputs = r2["outputs"]
        phase2_total_time = r2["wall"]
        phase2_stats = norm_stats(r2["stats"])
        
        print(f"Phase 2 total time: {phase2_total_time:.4f}s")
        print(f"Phase 2 average latency: {sum(phase2_latencies) / len(phase2_latencies):.4f}s per request")
        if kv_marketplace:
            print(f"Phase 2 stats: registry_size={phase2_stats['registry_size']}, "
                  f"local_hits={phase2_stats['local_hits']}, "
                  f"cross_hits={phase2_stats['cross_hits']}")
        
        # Combine results
        latencies = phase1_latencies + phase2_latencies
        outputs = phase1_outputs + phase2_outputs
        all_latencies.extend(latencies)
        all_outputs.extend(outputs)
        
        # Calculate run statistics
        avg_latency = sum(latencies) / len(latencies)
        total_time = phase1_total_time + phase2_total_time  # Sum of wall-clock times
        throughput = len(latencies) / total_time if total_time > 0 else 0
        
        # Combine stats from both phases
        import_hits = phase1_stats['import_hits'] + phase2_stats['import_hits']
        import_misses = phase1_stats['import_misses'] + phase2_stats['import_misses']
        local_hits = phase1_stats['local_hits'] + phase2_stats['local_hits']
        cross_hits = phase1_stats['cross_hits'] + phase2_stats['cross_hits']
        lcp_lengths = phase1_stats['lcp_lengths'] + phase2_stats['lcp_lengths']
        avg_lcp = sum(lcp_lengths) / len(lcp_lengths) if lcp_lengths else 0
        
        if kv_marketplace:
            # Use combined stats from both phases
            total_registry_size = max(phase1_stats['registry_size'], phase2_stats['registry_size'])
            total_prefix_index_size = max(phase1_stats['prefix_index_size'], phase2_stats['prefix_index_size'])
            print(f"  Combined stats: registry_size={total_registry_size}, "
                  f"prefix_index_size={total_prefix_index_size}, "
                  f"total_import_hits={import_hits}, "
                  f"total_import_misses={import_misses}, "
                  f"local_hits={local_hits}, "
                  f"cross_hits={cross_hits}")
            print(f"  Run stats: local_hits={local_hits}, cross_hits={cross_hits} (cross_hits should be > 0 for benefit)")
        
        # Calculate total_registry_size and total_prefix_index_size for run_stat
        total_registry_size = max(phase1_stats['registry_size'], phase2_stats['registry_size']) if kv_marketplace else 0
        total_prefix_index_size = max(phase1_stats['prefix_index_size'], phase2_stats['prefix_index_size']) if kv_marketplace else 0
        
        run_stat = {
            'run': run_idx + 1,
            'avg_latency': avg_latency,
            'total_latency': total_time,
            'throughput': throughput,
            'registry_size': total_registry_size,
            'prefix_index_size': total_prefix_index_size,
            'import_hits': import_hits,
            'import_misses': import_misses,
            'local_hits': local_hits,
            'cross_hits': cross_hits,
            'import_lcp_lengths': lcp_lengths,
        }
        run_stats.append(run_stat)
        
        print(f"  Average latency: {avg_latency:.4f}s")
        print(f"  Total time: {total_time:.4f}s")
        print(f"  Throughput: {throughput:.2f} req/s")
        if kv_marketplace:
            print(f"  Registry size: {total_registry_size}")
            print(f"  Prefix index size: {total_prefix_index_size}")
        if kv_marketplace:
            print(f"  Import hits: {import_hits} (local={local_hits}, cross={cross_hits})")
            print(f"  Import misses: {import_misses}")
            if avg_lcp > 0:
                print(f"  Average LCP length: {avg_lcp:.1f} tokens")
            if cross_hits == 0:
                print(f"  WARNING: No cross-GPU hits! Marketplace benefit requires cross_hits > 0")
    
    # Calculate overall statistics
    # For batched runs, we need to sum the wall-clock times from each run, not individual latencies
    overall_avg_latency = sum(all_latencies) / len(all_latencies)
    # Sum up the batched wall-clock times from each run
    overall_total_time = sum(run_stat['total_latency'] for run_stat in run_stats)
    overall_throughput = len(phase1_prompts + phase2_prompts) * num_runs / overall_total_time if overall_total_time > 0 else 0
    
    final_stats = get_registry_stats(get_stats_fn)
    
    # Aggregate import stats across all runs
    total_import_hits = sum(r.get('import_hits', 0) for r in run_stats)
    total_import_misses = sum(r.get('import_misses', 0) for r in run_stats)
    all_lcp_lengths = []
    for r in run_stats:
        all_lcp_lengths.extend(r.get('import_lcp_lengths', []))
    avg_lcp_overall = sum(all_lcp_lengths) / len(all_lcp_lengths) if all_lcp_lengths else 0
    
    result = {
        'kv_marketplace': kv_marketplace,
        'num_runs': num_runs,
        'num_prompts': len(phase1_prompts) + len(phase2_prompts),
        'avg_latency': overall_avg_latency,
        'total_latency': overall_total_time,
        'throughput': overall_throughput,
        'registry_size': final_stats['registry_size'],
        'prefix_index_size': final_stats['prefix_index_size'],
        'import_hits': total_import_hits,
        'import_misses': total_import_misses,
        'local_hits': final_stats.get('local_hits', 0),
        'cross_hits': final_stats.get('cross_hits', 0),
        'import_lcp_lengths': all_lcp_lengths,
        'avg_lcp_length': avg_lcp_overall,
        'run_stats': run_stats,
        'all_latencies': all_latencies,
        'all_outputs': all_outputs,
    }
    
    return result


def save_results_to_file(results: Dict, output_file: str):
    """Save benchmark results to a JSON file."""
    import json
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {output_file}")
    except Exception as e:
        print(f"\n✗ Failed to save results to {output_file}: {e}")


def create_comparison_chart(results_with: Dict, results_without: Dict):
    """Create a simple text-based comparison chart."""
    print(f"\n{'='*80}")
    print("  COMPARISON CHART")
    print(f"{'='*80}\n")
    
    if not results_with or not results_without:
        print("Cannot create comparison: missing results")
        return
    
    metrics = [
        ('Average Latency', 'avg_latency', 's', lambda x: x),
        ('Total Latency', 'total_latency', 's', lambda x: x),
        ('Throughput', 'throughput', 'req/s', lambda x: x),
        ('Cross-GPU Hits', 'cross_hits', '', lambda x: int(x)),
        ('Local Hits', 'local_hits', '', lambda x: int(x)),
        ('Avg LCP Length', 'avg_lcp_length', 'tokens', lambda x: x),
        ('Registry Size', 'registry_size', '', lambda x: int(x)),
    ]
    
    print(f"{'Metric':<20} {'Without kv-mkt':<20} {'With kv-mkt':<20} {'Improvement':<20}")
    print("-" * 80)
    
    for name, key, unit, formatter in metrics:
        without_val = results_without.get(key, 0)
        with_val = results_with.get(key, 0)
        
        if without_val > 0:
            if key in ['avg_latency', 'total_latency']:
                # For latency, lower is better
                improvement = ((without_val - with_val) / without_val) * 100
                improvement_str = f"{improvement:+.1f}%"
            elif key == 'throughput':
                # For throughput, higher is better
                improvement = ((with_val - without_val) / without_val) * 100
                improvement_str = f"{improvement:+.1f}%"
            else:
                improvement_str = "N/A"
        else:
            improvement_str = "N/A"
        
        without_str = f"{formatter(without_val)}{unit}" if without_val > 0 else "N/A"
        with_str = f"{formatter(with_val)}{unit}" if with_val > 0 else "N/A"
        
        print(f"{name:<20} {without_str:<20} {with_str:<20} {improvement_str:<20}")
    
    print("\n" + "="*80)


def main():
    # Set multiprocessing start method to "spawn" for reliable process isolation
    # This ensures CUDA_VISIBLE_DEVICES changes in children take effect before CUDA init
    # Fork can inherit CUDA/Ray state, which causes issues
    from multiprocessing import set_start_method
    try:
        set_start_method("spawn")
    except RuntimeError:
        # Already set, ignore
        pass
    
    parser = argparse.ArgumentParser(
        description="vLLM dual-GPU demo with kv-marketplace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python vllm_dual_gpu_demo.py --model gpt2

  # Run with custom system prompt
  python vllm_dual_gpu_demo.py --model gpt2 --system-prompt "You are a helpful assistant."

  # Run comparison (with and without kv-marketplace)
  python vllm_dual_gpu_demo.py --model gpt2 --compare
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Model name or path')
    parser.add_argument('--system-prompt', type=str,
                       default="You are a helpful AI assistant. Answer the user's questions concisely and accurately.",
                       help='Shared system prompt (default: helpful assistant prompt)')
    parser.add_argument('--user-prompts', type=str, nargs='+',
                       default=[
                           "What is the capital of France?",
                           "Explain quantum computing in simple terms.",
                           "Write a short poem about AI.",
                           "What are the benefits of renewable energy?",
                           "How does photosynthesis work?",
                       ],
                       help='User prompts (will be prefixed with system prompt)')
    parser.add_argument('--num-runs', type=int, default=3,
                       help='Number of benchmark runs (default: 3)')
    parser.add_argument('--kv-min-prefix', type=int, default=64,
                       help='Minimum prefix length for import (default: 64)')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                       help='GPU memory utilization (default: 0.9)')
    parser.add_argument('--tensor-parallel-size', type=int, default=1,
                       help='Tensor parallelism size (default: 1)')
    parser.add_argument('--max-model-len', type=int, default=1024,
                       help='Maximum model length (default: 1024)')
    parser.add_argument('--enable-prefix-caching', action='store_true', default=False,
                       help='Enable vLLM prefix caching (default: False, disabled for clean benchmark)')
    parser.add_argument('--compare', action='store_true',
                       help='Run comparison with and without kv-marketplace')
    parser.add_argument('--no-kv-marketplace', action='store_true',
                       help='Disable kv-marketplace (default: enabled)')
    parser.add_argument('--save-results', type=str, default=None,
                       help='Save benchmark results to JSON file (e.g., results.json)')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    try:
        import torch
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available!")
            sys.exit(1)
        
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPU(s)")
        
        if args.tensor_parallel_size > 1 and num_gpus < args.tensor_parallel_size:
            print(f"ERROR: Need at least {args.tensor_parallel_size} GPUs for tensor_parallel_size={args.tensor_parallel_size}")
            sys.exit(1)
    
    except ImportError:
        print("WARNING: PyTorch not available, cannot check GPU count")
    
    # Run benchmarks
    results_with = None
    results_without = None
    
    if args.compare:
        # Run without kv-marketplace first
        print("\n" + "="*80)
        print("  RUNNING WITHOUT kv-marketplace")
        print("="*80)
        results_without = run_benchmark(
            model=args.model,
            system_prompt=args.system_prompt,
            user_prompts=args.user_prompts,
            num_runs=args.num_runs,
            kv_marketplace=False,
            kv_min_prefix=args.kv_min_prefix,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
        )
        
        # Run with kv-marketplace
        print("\n" + "="*80)
        print("  RUNNING WITH kv-marketplace")
        print("="*80)
        results_with = run_benchmark(
            model=args.model,
            system_prompt=args.system_prompt,
            user_prompts=args.user_prompts,
            num_runs=args.num_runs,
            kv_marketplace=True,
            kv_min_prefix=args.kv_min_prefix,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
        )
        
        # Create comparison chart
        if results_with and results_without:
            create_comparison_chart(results_with, results_without)
        
        # Save results if requested
        if args.save_results:
            combined_results = {
                'without_kv_marketplace': results_without,
                'with_kv_marketplace': results_with,
                'comparison': {
                    'latency_improvement_pct': ((results_without.get('avg_latency', 0) - results_with.get('avg_latency', 0)) / results_without.get('avg_latency', 1)) * 100 if results_without.get('avg_latency', 0) > 0 else 0,
                    'throughput_improvement_pct': ((results_with.get('throughput', 0) - results_without.get('throughput', 0)) / results_without.get('throughput', 1)) * 100 if results_without.get('throughput', 0) > 0 else 0,
                }
            }
            save_results_to_file(combined_results, args.save_results)
    
    else:
        # Run single benchmark
        kv_marketplace = not args.no_kv_marketplace
        results = run_benchmark(
            model=args.model,
            system_prompt=args.system_prompt,
            user_prompts=args.user_prompts,
            num_runs=args.num_runs,
            kv_marketplace=kv_marketplace,
            kv_min_prefix=args.kv_min_prefix,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
        )
        
        if results:
            print_stats_table([results], "Benchmark Results")
            
            # Save results if requested
            if args.save_results:
                save_results_to_file(results, args.save_results)
    
    print("\n✓ Demo completed!")


if __name__ == '__main__':
    main()
