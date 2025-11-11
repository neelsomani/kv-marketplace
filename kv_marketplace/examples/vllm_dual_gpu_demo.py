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
import random
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
from multiprocessing import Process, Queue, Event

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
        Tuple of (get_stats_fn, reset_stats_fn, clear_registry_fn, get_registry_keys_fn) or (None, None, None, None) if import fails
    """
    try:
        from kv_marketplace.adapter.vllm import get_stats as _get_stats, reset_stats as _reset_stats, clear_registry as _clear_registry, get_registry_keys as _get_registry_keys
        return _get_stats, _reset_stats, _clear_registry, _get_registry_keys
    except Exception:
        return None, None, None, None


def create_prefixes_from_system_prompt(system_prompt: str, num_prefixes: int = 10) -> List[str]:
    """Split system prompt into N different prefixes by splitting on '. '.
    
    Creates prefixes with the same sentences but in scrambled order.
    
    Args:
        system_prompt: The system prompt to split
        num_prefixes: Number of prefixes to create
        
    Returns:
        List of prefixes (each is a cumulative prefix with sentences in different order)
    """
    sentences = system_prompt.split('. ')
    
    # If we have fewer sentences than requested prefixes, use what we have
    if len(sentences) < num_prefixes:
        num_prefixes = len(sentences)
    
    prefixes = []
    for _ in range(num_prefixes):
        # Shuffle sentences for each prefix to create different orderings
        shuffled_sentences = sentences.copy()
        random.shuffle(shuffled_sentences)
        
        # Create cumulative prefix from shuffled order
        current_prefix = ""
        for i, sentence in enumerate(shuffled_sentences):
            if i == 0:
                current_prefix = sentence
            else:
                current_prefix = current_prefix + '. ' + sentence
        prefixes.append(current_prefix)
    
    return prefixes


def create_shared_prefix_prompts(system_prompt: str, user_prompts: List[str]) -> List[str]:
    """Create prompts with shared system prefix."""
    return [f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:" for prompt in user_prompts]


def build_shared_kwargs(
    kv_marketplace: bool,
    kv_min_prefix: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    max_model_len: int,
    dtype: str,
    tokenizer_mode: str,
    enable_speculative: bool,
    speculative_config: Optional[Dict],
    llm_kwargs: Optional[Dict] = None,
) -> Dict:
    """Construct shared LLM kwargs used by child processes."""
    llm_kwargs = dict(llm_kwargs or {})
    resolved_spec_config = None
    if enable_speculative:
        resolved_spec_config = speculative_config or {
            "method": "ngram",
            "num_speculative_tokens": 6,
            "prompt_lookup_min": 3,
            "prompt_lookup_max": 8,
        }

    shared_kwargs = {
        'kv_marketplace': kv_marketplace,
        'kv_min_prefix': kv_min_prefix,
        'gpu_memory_utilization': gpu_memory_utilization,
        'tensor_parallel_size': 1,  # Force TP=1 for data parallelism between processes
        'max_model_len': max_model_len,
        'enable_prefix_caching': True,
        'distributed_executor_backend': 'mp',
        **llm_kwargs,
    }
    shared_kwargs.setdefault('dtype', dtype)
    shared_kwargs.setdefault('tokenizer_mode', tokenizer_mode)
    shared_kwargs.pop('device', None)

    if enable_speculative and resolved_spec_config:
        shared_kwargs.setdefault('speculative_config', resolved_spec_config)
    else:
        shared_kwargs.pop('speculative_config', None)

    return shared_kwargs


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
              sampling_params_dict: Dict, result_queue: Queue, phase_name: str,
              prefetch_only: bool = False, rehome_after_import: bool = False,
              prefetch_prompts: Optional[List[str]] = None,
              prefetch_sampling_params_dict: Optional[Dict] = None,
              hold_gpu_event: Optional[Event] = None,
              dump_kv_dir: Optional[str] = None,
              dump_kv_label: Optional[str] = None,
              capture_logits: bool = False,
              capture_label: Optional[str] = None,
              disable_import: bool = False):
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
    
    # Do not force torch current device here; vLLM will pick device 0 of the visible list.
    
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
    
    if dump_kv_dir:
        dump_target_dir = os.path.abspath(os.path.expanduser(dump_kv_dir))
        os.environ['KV_MARKETPLACE_DUMP_DIR'] = dump_target_dir
        if dump_kv_label:
            os.environ['KV_MARKETPLACE_DUMP_LABEL'] = dump_kv_label
        else:
            os.environ.pop('KV_MARKETPLACE_DUMP_LABEL', None)
        try:
            os.makedirs(dump_target_dir, exist_ok=True)
        except Exception:
            pass
    else:
        os.environ.pop('KV_MARKETPLACE_DUMP_DIR', None)
        os.environ.pop('KV_MARKETPLACE_DUMP_LABEL', None)

    if disable_import:
        os.environ['KV_MARKETPLACE_DISABLE_IMPORT'] = '1'
    else:
        os.environ.pop('KV_MARKETPLACE_DISABLE_IMPORT', None)
    
    # Check if kv-marketplace is enabled
    kvm_enabled = bool(shared_kwargs.get("kv_marketplace", False))
    
    # Only enable file backend if kv-marketplace is enabled
    if kvm_enabled:
        os.environ.setdefault('KV_MARKETPLACE_SHM_BACKEND', '1')
        if rehome_after_import:
            os.environ['KV_MARKETPLACE_REHOME_AFTER_IMPORT'] = '1'
        else:
            os.environ.pop('KV_MARKETPLACE_REHOME_AFTER_IMPORT', None)
    else:
        os.environ.pop('KV_MARKETPLACE_FILE_BACKEND', None)
        os.environ.pop('KV_MARKETPLACE_SHM_BACKEND', None)
        os.environ.pop('KV_MARKETPLACE_REHOME_AFTER_IMPORT', None)
    
    llm = None
    get_stats = None
    
    try:
        # Import vllm - this will use the editable install from vllm/ folder (which is correct)
        import torch

        # NVML cannot initialize in some containerized environments, which makes
        # vLLM fall back to UnspecifiedPlatform (device string == ""). Manually fix
        # the platform before EngineArgs inspects it so DeviceConfig always sees
        # a concrete device type.
        try:
            import vllm.platforms as vllm_platforms
            from vllm.platforms.cuda import CudaPlatform
            from vllm.platforms.cpu import CpuPlatform
            vllm_platforms.current_platform = (
                CudaPlatform() if torch.cuda.is_available() else CpuPlatform()
            )
        except Exception:
            pass

        import vllm
        from vllm.entrypoints.llm import LLM
        from vllm.sampling_params import SamplingParams
        
        # Recreate SamplingParams in child process to avoid multiprocessing serialization issues
        # This ensures the object is created fresh in this process context
        sampling_params_kwargs = dict(sampling_params_dict)
        if capture_logits:
            sampling_params_kwargs['logprobs'] = -1
        sampling_params = SamplingParams(**sampling_params_kwargs)

        reset_stats_fn = None
        if kvm_enabled:
            # Only import kv_marketplace adapter if it's enabled
            from kv_marketplace.adapter.vllm import (
                get_stats,
                reset_stats as adapter_reset_stats,
                set_min_prefix_length,
            )
            reset_stats_fn = adapter_reset_stats
            kv_prefix_len = int(shared_kwargs.get('kv_min_prefix', 64))
            try:
                set_min_prefix_length(kv_prefix_len)
            except Exception as exc:
                print(f"Warning: failed to set kv_min_prefix to {kv_prefix_len}: {exc}")
        else:
            # Skip adapter import entirely if kv-marketplace is disabled
            get_stats = None
        
        # Local version of measure_latency_concurrent for child process.
        # Uses batched generation to avoid thread-safety issues.
        def measure_latency_concurrent_local(llm, prompts, sampling_params, batch_name, capture_raw=False):
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
                return ([0.0] * len(prompts), [""] * len(prompts), [], 0.0, None)
            
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
            raw_payload = results if capture_raw else None
            
            return latencies, outputs, [], total_wall_time, raw_payload
        
        # NOTE: Do NOT reset_stats() here - it's already called in parent before spawning.
        # Resetting here could accidentally wipe Phase-1 exports before Phase-2 reads them.
        
        # Initialize LLM (will now see only the specified GPU)
        llm = LLM(model=model, **shared_kwargs)
        
        # Print prefix caching configuration to verify it's enabled
        print(f"[{phase_name}] cfg.enable_prefix_caching: {llm.llm_engine.cache_config.enable_prefix_caching}")
    
        # Sanity logs: print device IDs to verify correct GPU selection
        if torch.cuda.is_available():
            local_device_id = torch.cuda.current_device()
            print(f"  [{phase_name}] Local device ordinal: {local_device_id}")
            # Try to get stable device ID if adapter is available
            if kvm_enabled:
                try:
                    from kv_marketplace.adapter.vllm import get_stable_device_id
                    stable_id = get_stable_device_id(local_device_id)
                    if stable_id:
                        print(f"  [{phase_name}] Stable device ID: {stable_id}")
                    else:
                        print(f"  [{phase_name}] Stable device ID: None (unavailable)")
                except Exception:
                    pass  # Skip if adapter not available
        
        prefetch_info = None
        if prefetch_prompts and prefetch_sampling_params_dict:
            pref_sampling_params = SamplingParams(**prefetch_sampling_params_dict)
            pref_lats, pref_outs, _, pref_wall, _ = measure_latency_concurrent_local(
                llm, prefetch_prompts, pref_sampling_params, f"{phase_name}-prefetch"
            )
            pref_stats = zero_stats()
            if kvm_enabled and get_stats is not None:
                try:
                    pref_stats = get_stats()
                except Exception:
                    pref_stats = zero_stats()
            prefetch_info = {
                "latencies": pref_lats,
                "outputs": pref_outs,
                "wall": pref_wall,
                "stats": pref_stats,
            }

            if prefetch_only:
                result_queue.put({
                    "latencies": pref_lats,
                    "outputs": pref_outs,
                    "wall": pref_wall,
                    "stats": norm_stats(pref_stats),
                    "prefetch_only": True,
                    "prefetch": prefetch_info,
                })
                return

            if kvm_enabled and reset_stats_fn is not None:
                try:
                    reset_stats_fn()
                except Exception:
                    pass

        # Run batched generation for measured prompts
        lats, outs, _, wall, raw_outputs = measure_latency_concurrent_local(
            llm, prompts, sampling_params, phase_name, capture_raw=capture_logits
        )
        
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
        result = {
            "latencies": lats,
            "outputs": outs,
            "wall": wall,
            "stats": stats,
            "prefetch_only": prefetch_only,
        }
        captured_logits_payload = None
        if capture_logits and raw_outputs:
            captured_logits_payload = []
            for req in raw_outputs:
                if not req or not getattr(req, 'outputs', None):
                    continue
                completion = req.outputs[0]
                sample_logprobs = completion.logprobs or []
                if not sample_logprobs:
                    continue
                position_logprobs = sample_logprobs[0]
                if not position_logprobs:
                    continue
                token_ids = sorted(position_logprobs.keys())
                logprob_values = [position_logprobs[token_id].logprob for token_id in token_ids]
                captured_logits_payload.append({
                    'token_ids': token_ids,
                    'logprobs': logprob_values,
                })

        if prefetch_info is not None:
            result["prefetch"] = prefetch_info
        if captured_logits_payload is not None:
            result["captured_logits"] = captured_logits_payload
        result_queue.put(result)
        if hold_gpu_event is not None:
            try:
                hold_gpu_event.wait()
            except KeyboardInterrupt:
                pass
    except Exception as e:
        import traceback
        # Put error in queue with traceback for debugging
        result_queue.put({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "latencies": [],
            "outputs": [],
            "wall": 0.0,
            "stats": zero_stats(),
            "captured_logits": None,
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
    max_model_len: int = 4096,
    prefetch_phase2: bool = True,
    print_outputs: bool = False,
    dtype: str = "float16",
    speculative_config: Optional[Dict] = None,
    enable_speculative: bool = True,
    dump_kv_dir: Optional[str] = None,
    tokenizer_mode: str = "mistral",
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
        prefetch_phase2: Prefetch Phase 2 prefixes onto destination GPU before measurement
        print_outputs: Print generated text for each phase/run
        enable_speculative: Enable speculative decoding (ngram draft) in vLLM
        dump_kv_dir: Optional directory to dump exported KV pages for offline diffing
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
    if not enable_speculative:
        print("Speculative decoding: DISABLED")
    if dump_kv_dir:
        dump_kv_dir = os.path.abspath(os.path.expanduser(dump_kv_dir))
        os.makedirs(dump_kv_dir, exist_ok=True)
        print(f"KV cache dumps enabled: {dump_kv_dir}")
    
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
    
    # Phase 2 prompts: same as Phase 1 (to test cross-GPU imports with matching hashes)
    phase2_prompts = phase1_prompts.copy()
    
    print(f"\nPhase 1: {len(phase1_prompts)} prompts (warm-up, batched)")
    print(f"Phase 2: {len(phase2_prompts)} prompts (test with same prompts on different GPU, batched)")
    
    # Only enable file-based backend if kv-marketplace is enabled and we have multiple GPUs
    # This allows sharing registry across processes on the same machine
    # IMPORTANT: Both processes must see the same file backend path (default: /tmp/kv_marketplace_stats)
    # Do NOT delete the registry between phases - only clear stats, not the registry itself
    import torch
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and kv_marketplace:
        # Use shared-memory backend for cross-process sharing; file backend adds heavy IO.
        os.environ.pop('KV_MARKETPLACE_FILE_BACKEND', None)
        os.environ.setdefault('KV_MARKETPLACE_SHM_BACKEND', '1')
        print(f"Detected {torch.cuda.device_count()} GPUs, shared-memory registry backend enabled for cross-process sharing")
    else:
        os.environ.pop('KV_MARKETPLACE_FILE_BACKEND', None)

    # Prefer FlashInfer samplers whenever nvcc/flashinfer wheels are available.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "1")
    os.environ.setdefault("VLLM_USE_FLASHINFER", "1")
    
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("ERROR: Need at least 2 GPUs for cross-GPU benchmark")
        return None
    
    # WARNING: Device identity issue across processes
    # Each child process sets CUDA_VISIBLE_DEVICES to a single GPU, so both see their GPU as device 0 locally.
    # If the adapter stores device_id as torch.cuda.current_device() (local ordinal), both will store device_id=0,
    # causing confusion in peer-copy logic. The adapter should store globally unique device IDs (UUID or PCI bus ID)
    # and translate to local ordinals at import time. This requires adapter changes - see kv_marketplace/adapter/vllm.py
    
    # Lazy import adapter functions only if kv-marketplace is enabled
    get_stats_fn, reset_stats_fn, clear_registry_fn, get_registry_keys_fn = None, None, None, None
    shutdown_backends_fn = None
    if kv_marketplace:
        get_stats_fn, reset_stats_fn, clear_registry_fn, get_registry_keys_fn = maybe_get_adapter_funcs()
        if reset_stats_fn:
            try:
                reset_stats_fn()
            except Exception:
                pass
        # Clear registry before first run to ensure empty cache
        if clear_registry_fn:
            try:
                clear_registry_fn()
                print("Cleared registry and prefix index to ensure empty cache")
            except Exception as e:
                print(f"Warning: Failed to clear registry: {e}")
        try:
            from kv_marketplace.adapter.vllm import shutdown_backends as _shutdown_backends
            shutdown_backends_fn = _shutdown_backends
        except Exception:
            shutdown_backends_fn = None
        os.environ.setdefault('KV_MARKETPLACE_HANDLE_CACHE_SIZE', 'inf')
    
    # Shared kwargs for both child processes
    shared_kwargs = build_shared_kwargs(
        kv_marketplace=kv_marketplace,
        kv_min_prefix=kv_min_prefix,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        dtype=dtype,
        tokenizer_mode=tokenizer_mode,
        enable_speculative=enable_speculative,
        speculative_config=speculative_config,
        llm_kwargs=llm_kwargs,
    )
    
    # Convert SamplingParams to dict for multiprocessing (avoids serialization issues with vLLM)
    # This prevents msgspec validation errors when the object is pickled/unpickled across processes
    # We'll recreate it fresh in the child process to ensure proper vLLM serialization
    phase1_sampling_params_dict = {
        'temperature': 0.0,
        'top_p': 1.0,
        'max_tokens': 256,
    }
    phase2_sampling_params_dict = dict(phase1_sampling_params_dict)
    
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
        
        phase1_hold_event = Event() if kv_marketplace else None
        phase1_dump_label = f"run{run_idx + 1}_phase1" if dump_kv_dir else None
        phase2_dump_label = f"run{run_idx + 1}_phase2" if dump_kv_dir else None
        child_kwargs_phase1 = {}
        if phase1_hold_event is not None:
            child_kwargs_phase1['hold_gpu_event'] = phase1_hold_event
        if dump_kv_dir:
            child_kwargs_phase1['dump_kv_dir'] = dump_kv_dir
            child_kwargs_phase1['dump_kv_label'] = phase1_dump_label
        
        # Phase 1: Process on GPU 0
        print(f"\n--- Phase 1: Warm-up ({len(phase1_prompts)} requests on GPU 0) ---")
        p1 = Process(
            target=child_run,
            args=("0", model, shared_kwargs, phase1_prompts, phase1_sampling_params_dict, q1, "Phase1"),
            kwargs=child_kwargs_phase1,
        )
        p1.start()
        
        phase1_cleanup_done = False
        def cleanup_phase1():
            nonlocal phase1_cleanup_done
            if phase1_cleanup_done:
                return
            if phase1_hold_event is not None:
                phase1_hold_event.set()
            try:
                p1.join(10)
            except Exception:
                pass
            if p1.is_alive():
                p1.terminate()
                p1.join(10)
            phase1_cleanup_done = True
        
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
        
        if "error" in r1:
            print(f"ERROR in Phase 1: {r1['error']}")
            if "traceback" in r1:
                print(f"Traceback:\n{r1['traceback']}")
            cleanup_phase1()
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
        
        try:
            # Reset stats so measured Phase 2 excludes Phase 1 and prefetch activity
            if kv_marketplace and reset_stats_fn:
                try:
                    reset_stats_fn()
                    print("Reset stats before measured Phase 2 (registry preserved)")
                except Exception as e:
                    print(f"Warning: Failed to reset stats before Phase 2: {e}")
        
            # Barrier
            time.sleep(0.5)
        
            prefetch_kwargs = {}
            if kv_marketplace and prefetch_phase2:
                print(f"\n--- Prefetching Phase 2 prefixes onto GPU 1 (will reuse same worker) ---")
                prefetch_sampling_params = dict(phase2_sampling_params_dict)
                prefetch_sampling_params['max_tokens'] = max(1, prefetch_sampling_params.get('max_tokens', 1))
                prefetch_sampling_params['temperature'] = 0.0
                prefetch_sampling_params['top_p'] = 1.0
                prefetch_kwargs = {
                    'prefetch_prompts': phase2_prompts,
                    'prefetch_sampling_params_dict': prefetch_sampling_params,
                    'rehome_after_import': True,
                }
            child_kwargs_phase2 = dict(prefetch_kwargs)
            if dump_kv_dir:
                child_kwargs_phase2['dump_kv_dir'] = dump_kv_dir
                child_kwargs_phase2['dump_kv_label'] = phase2_dump_label
            # Phase 2: Process on GPU 1 (forces cross-GPU)
            # Make both GPUs visible so Phase 2 can import from GPU 0
            print(f"\n--- Phase 2: Test ({len(phase2_prompts)} requests on GPU 1) ---")
            # Make GPU1 the first visible device so vLLM picks it, but keep GPU0 visible for P2P
            p2 = Process(
                target=child_run,
                args=("1,0", model, shared_kwargs, phase2_prompts, phase2_sampling_params_dict, q2, "Phase2"),
                kwargs=child_kwargs_phase2,
            )
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
            if kv_marketplace and prefetch_phase2:
                pref_info = r2.get("prefetch")
                if pref_info:
                    print(f"Prefetch completed in {pref_info.get('wall', 0.0):.4f}s")
        
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
                      f"prefix_index_size={phase2_stats['prefix_index_size']}")
        
            if print_outputs:
                print(f"\nRun {run_idx + 1} outputs:")
                for idx, output in enumerate(phase1_outputs, 1):
                    text = output.strip()
                    print(f"  [Phase 1 #{idx}] {text if text else '(empty output)'}")
                for idx, output in enumerate(phase2_outputs, 1):
                    text = output.strip()
                    print(f"  [Phase 2 #{idx}] {text if text else '(empty output)'}")

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
        
            # Calculate total_registry_size and total_prefix_index_size for run_stat
            total_registry_size = max(phase1_stats['registry_size'], phase2_stats['registry_size']) if kv_marketplace else 0
            total_prefix_index_size = max(phase1_stats['prefix_index_size'], phase2_stats['prefix_index_size']) if kv_marketplace else 0
        
            # Calculate Phase 1 metrics
            phase1_avg_latency = sum(phase1_latencies) / len(phase1_latencies) if phase1_latencies else 0
            phase1_throughput = len(phase1_latencies) / phase1_total_time if phase1_total_time > 0 else 0
        
            # Calculate Phase 2 metrics
            phase2_avg_latency = sum(phase2_latencies) / len(phase2_latencies) if phase2_latencies else 0
            phase2_throughput = len(phase2_latencies) / phase2_total_time if phase2_total_time > 0 else 0
        
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
                # Phase 1 metrics
                'phase1_avg_latency': phase1_avg_latency,
                'phase1_total_time': phase1_total_time,
                'phase1_throughput': phase1_throughput,
                'phase1_local_hits': phase1_stats.get('local_hits', 0),
                'phase1_cross_hits': phase1_stats.get('cross_hits', 0),
                'phase1_import_hits': phase1_stats.get('import_hits', 0),
                'phase1_import_misses': phase1_stats.get('import_misses', 0),
                # Phase 2 metrics
                'phase2_avg_latency': phase2_avg_latency,
                'phase2_total_time': phase2_total_time,
                'phase2_throughput': phase2_throughput,
                'phase2_local_hits': phase2_stats.get('local_hits', 0),
                'phase2_cross_hits': phase2_stats.get('cross_hits', 0),
                'phase2_import_hits': phase2_stats.get('import_hits', 0),
                'phase2_import_misses': phase2_stats.get('import_misses', 0),
                'num_requests': len(latencies),
            }
            run_stats.append(run_stat)
        
            print(f"  Average latency: {avg_latency:.4f}s")
            print(f"  Total time: {total_time:.4f}s")
            print(f"  Throughput: {throughput:.2f} req/s")
            if kv_marketplace:
                print(f"  Registry size: {total_registry_size}")
                print(f"  Prefix index size: {total_prefix_index_size}")
                if avg_lcp > 0:
                    print(f"  Average LCP length: {avg_lcp:.1f} tokens")
                if cross_hits == 0 and not prefetch_phase2:
                    print(f"  WARNING: No cross-GPU hits! Marketplace benefit requires cross_hits > 0")
        finally:
            cleanup_phase1()
    
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

    if kv_marketplace and shutdown_backends_fn:
        try:
            shutdown_backends_fn()
        except Exception as exc:
            print(f"Warning: Failed to shut down kv-marketplace backends cleanly: {exc}")
    
    return result


def run_logits_divergence_check(
    model: str,
    system_prompt: str,
    user_prompts: List[str],
    kv_min_prefix: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    max_model_len: int,
    seed: int,
    dtype: str = "float16",
    tokenizer_mode: str = "mistral",
    llm_kwargs: Optional[Dict] = None,
):
    """Run four deterministic passes and compare logits across phases."""
    import torch

    if not user_prompts:
        raise ValueError("Need at least one user prompt for logits divergence check")

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Logits divergence check requires at least 2 CUDA devices")

    prompt = create_shared_prefix_prompts(system_prompt, [user_prompts[0]])[0]
    prompts = [prompt]
    sampling_params_dict = {
        'temperature': 0.0,
        'top_p': 1.0,
        'top_k': 1,
        'max_tokens': 1,
        'min_tokens': 1,
        'seed': seed,
    }

    print("\n" + "=" * 80)
    print("  LOGITS DIVERGENCE CHECK")
    print("=" * 80)
    print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print(f"Seed: {seed}")
    print("Decoding params: temperature=0.0, top_k=1, top_p=1.0, max_tokens=1")

    os.environ.pop('KV_MARKETPLACE_FILE_BACKEND', None)
    os.environ.setdefault('KV_MARKETPLACE_SHM_BACKEND', '1')
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "1")
    os.environ.setdefault("VLLM_USE_FLASHINFER", "1")

    diag_kv_min_prefix = kv_min_prefix if kv_min_prefix <= 1 else 1
    if diag_kv_min_prefix != kv_min_prefix:
        print(f"Forcing kv_min_prefix={diag_kv_min_prefix} for logits divergence check (was {kv_min_prefix})")

    diagnostic_llm_kwargs = dict(llm_kwargs or {})
    diagnostic_llm_kwargs['max_logprobs'] = -1

    shared_kwargs_no_kv = build_shared_kwargs(
        kv_marketplace=False,
        kv_min_prefix=diag_kv_min_prefix,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        dtype=dtype,
        tokenizer_mode=tokenizer_mode,
        enable_speculative=False,
        speculative_config=None,
        llm_kwargs=diagnostic_llm_kwargs,
    )
    shared_kwargs_kv = build_shared_kwargs(
        kv_marketplace=True,
        kv_min_prefix=diag_kv_min_prefix,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        dtype=dtype,
        tokenizer_mode=tokenizer_mode,
        enable_speculative=False,
        speculative_config=None,
        llm_kwargs=diagnostic_llm_kwargs,
    )

    get_stats_fn, reset_stats_fn, clear_registry_fn, _ = maybe_get_adapter_funcs()
    if clear_registry_fn:
        try:
            clear_registry_fn()
            print("Cleared registry and prefix index before logits check")
        except Exception as exc:
            print(f"Warning: Failed to clear registry before logits check: {exc}")
    if reset_stats_fn:
        try:
            reset_stats_fn()
        except Exception:
            pass

    run_plan = [
        {
            'key': 'A',
            'desc': 'Phase1 baseline (kv disabled)',
            'device_mask': '0',
            'shared_kwargs': shared_kwargs_no_kv,
            'disable_import': False,
            'phase': 'Phase1',
        },
        {
            'key': 'B',
            'desc': 'Phase1 control (kv enabled, import disabled)',
            'device_mask': '0',
            'shared_kwargs': shared_kwargs_kv,
            'disable_import': True,
            'phase': 'Phase1',
        },
        {
            'key': 'C',
            'desc': 'Phase2 baseline (import disabled)',
            'device_mask': '1,0',
            'shared_kwargs': shared_kwargs_kv,
            'disable_import': True,
            'phase': 'Phase2',
        },
        {
            'key': 'D',
            'desc': 'Phase2 reuse enabled',
            'device_mask': '1,0',
            'shared_kwargs': shared_kwargs_kv,
            'disable_import': False,
            'phase': 'Phase2',
        },
    ]

    def _execute(run_cfg: Dict) -> Dict:
        q = Queue()
        process = Process(
            target=child_run,
            args=(
                run_cfg['device_mask'],
                model,
                dict(run_cfg['shared_kwargs']),
                prompts,
                sampling_params_dict,
                q,
                run_cfg['phase'],
            ),
            kwargs={
                'capture_logits': True,
                'capture_label': run_cfg['key'],
                'disable_import': run_cfg['disable_import'],
            },
        )
        process.start()
        try:
            result = q.get(timeout=180)
        finally:
            q.close()
            q.join_thread()
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        if 'error' in result:
            raise RuntimeError(f"Run {run_cfg['key']} failed: {result['error']}\n{result.get('traceback', '')}")
        return result

    results: Dict[str, Dict] = {}
    for cfg in run_plan:
        print(f"\n--- {cfg['key']}: {cfg['desc']} ---")
        if cfg['key'] == 'B' and clear_registry_fn:
            try:
                clear_registry_fn()
                if reset_stats_fn:
                    reset_stats_fn()
                print("Reset registry before control export run (B)")
            except Exception as exc:
                print(f"Warning: Failed to reset registry before run B: {exc}")
        elif cfg['key'] in ('C', 'D') and reset_stats_fn:
            try:
                reset_stats_fn()
            except Exception:
                pass
        results[cfg['key']] = _execute(cfg)
        stats_snapshot = norm_stats(results[cfg['key']].get('stats', {}))
        print(
            f"Run {cfg['key']} stats: registry_size={stats_snapshot['registry_size']}, "
            f"prefix_index_size={stats_snapshot['prefix_index_size']}, "
            f"import_hits={stats_snapshot['import_hits']}, cross_hits={stats_snapshot['cross_hits']}"
        )

    def _tensor_from_result(key: str) -> Tuple[torch.Tensor, torch.Tensor]:
        payload = results[key].get('captured_logits')
        if not payload:
            raise RuntimeError(f"Run {key} did not return logits payload")
        record = payload[-1]
        token_ids = record.get('token_ids')
        logprob_values = record.get('logprobs')
        if token_ids is None or logprob_values is None:
            raise RuntimeError(f"Run {key} returned malformed logits payload")
        token_tensor = torch.tensor(token_ids, dtype=torch.int64)
        logprob_tensor = torch.tensor(logprob_values, dtype=torch.float32)
        return token_tensor, logprob_tensor

    logits = {key: _tensor_from_result(key) for key in results}

    def _compare(lhs_key: str, rhs_key: str) -> Tuple[bool, float]:
        lhs_tokens, lhs_vals = logits[lhs_key]
        rhs_tokens, rhs_vals = logits[rhs_key]
        if lhs_tokens.shape != rhs_tokens.shape or not torch.equal(lhs_tokens, rhs_tokens):
            raise AssertionError(
                f"Token ID mismatch between runs {lhs_key} and {rhs_key}; cannot compare logits"
            )
        diff = (lhs_vals - rhs_vals).abs().max().item()
        atol = 1e-6
        return bool(torch.allclose(lhs_vals, rhs_vals, atol=atol, rtol=atol)), diff

    eq_ab, diff_ab = _compare('A', 'B')
    eq_ac, diff_ac = _compare('A', 'C')
    eq_ad, diff_ad = _compare('A', 'D')

    print("\nComparison (max |Δ|):")
    print(f"  A vs B: {diff_ab:.3e} {'OK' if eq_ab else 'MISMATCH'}")
    print(f"  A vs C: {diff_ac:.3e} {'OK' if eq_ac else 'MISMATCH'}")
    print(f"  A vs D: {diff_ad:.3e} {'OK' if eq_ad else 'MISMATCH'}")

    if not eq_ab:
        raise AssertionError(
            f"Control mismatch: Phase1 baseline (A) != Phase1 control (B). max |Δ|={diff_ab:.3e}"
        )

    if eq_ac and not eq_ad:
        raise AssertionError(
            "Import path divergence detected: Phase2 without reuse matches baseline, "
            f"but enabling reuse changes logits (max |Δ|={diff_ad:.3e})."
        )

    if not eq_ac:
        raise AssertionError(
            "Phase2 baseline diverges before reuse: C != A. "
            f"Investigate Phase2 setup (max |Δ|={diff_ac:.3e})."
        )

    if not eq_ad:
        raise AssertionError(
            "Unexpected mismatch after reuse: Phase2 import did not match baseline."
        )

    print("\n✓ Logits are identical across all four runs (no divergence detected)")
    return {
        'results': results,
        'logits': logits,
    }


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
    
    # Calculate Phase 1 and Phase 2 averages from run_stats
    run_stats_with = results_with.get('run_stats', [])
    run_stats_without = results_without.get('run_stats', [])
    
    # Calculate Phase 1 combined metrics
    phase1_metrics = [
        ('Phase 1 Avg Latency', 'phase1_avg_latency', 's', lambda x: x),
        ('Phase 1 Total Time', 'phase1_total_time', 's', lambda x: x),
        ('Phase 1 Throughput', 'phase1_throughput', 'req/s', lambda x: x),
    ]
    
    # Calculate Phase 2 combined metrics
    phase2_metrics = [
        ('Phase 2 Avg Latency', 'phase2_avg_latency', 's', lambda x: x),
        ('Phase 2 Total Time', 'phase2_total_time', 's', lambda x: x),
        ('Phase 2 Throughput', 'phase2_throughput', 'req/s', lambda x: x),
    ]
    
    def calc_avg_from_runs(run_stats, key):
        """Calculate average value from run_stats."""
        if not run_stats:
            return 0
        values = [r.get(key, 0) for r in run_stats if r.get(key, 0) > 0]
        return sum(values) / len(values) if values else 0
    
    print(f"{'Metric':<25} {'Without kv-mkt':<20} {'With kv-mkt':<20} {'Improvement':<20}")
    print("-" * 85)
    
    # Print Phase 1 combined metrics
    print("\n  PHASE 1 (Avg across runs):")
    for name, key, unit, formatter in phase1_metrics:
        without_val = calc_avg_from_runs(run_stats_without, key)
        with_val = calc_avg_from_runs(run_stats_with, key)
        
        if without_val > 0:
            if 'latency' in key.lower() or 'time' in key.lower():
                # For latency, lower is better
                improvement = ((without_val - with_val) / without_val) * 100
                improvement_str = f"{improvement:+.1f}%"
            elif 'throughput' in key.lower():
                # For throughput, higher is better
                improvement = ((with_val - without_val) / without_val) * 100
                improvement_str = f"{improvement:+.1f}%"
            else:
                improvement_str = "N/A"
        else:
            improvement_str = "N/A"
        
        without_str = f"{formatter(without_val)}{unit}" if without_val > 0 else "N/A"
        with_str = f"{formatter(with_val)}{unit}" if with_val > 0 else "N/A"
        
        print(f"  {name:<23} {without_str:<20} {with_str:<20} {improvement_str:<20}")
    
    # Print Phase 2 combined metrics
    print("\n  PHASE 2 (Avg across runs):")
    for name, key, unit, formatter in phase2_metrics:
        without_val = calc_avg_from_runs(run_stats_without, key)
        with_val = calc_avg_from_runs(run_stats_with, key)
        
        if without_val > 0:
            if 'latency' in key.lower() or 'time' in key.lower():
                # For latency, lower is better
                improvement = ((without_val - with_val) / without_val) * 100
                improvement_str = f"{improvement:+.1f}%"
            elif 'throughput' in key.lower():
                # For throughput, higher is better
                improvement = ((with_val - without_val) / without_val) * 100
                improvement_str = f"{improvement:+.1f}%"
            else:
                improvement_str = "N/A"
        else:
            improvement_str = "N/A"
        
        without_str = f"{formatter(without_val)}{unit}" if without_val > 0 else "N/A"
        with_str = f"{formatter(with_val)}{unit}" if with_val > 0 else "N/A"
        
        print(f"  {name:<23} {without_str:<20} {with_str:<20} {improvement_str:<20}")
    
    # Phase 1 metrics per run
    print(f"\n{'='*80}")
    print("  PHASE 1 METRICS (per run)")
    print(f"{'='*80}\n")
    
    if run_stats_with and run_stats_without:
        phase1_metrics = [
            ('Avg Latency', 'phase1_avg_latency', 's', lambda x: x),
            ('Total Time', 'phase1_total_time', 's', lambda x: x),
            ('Throughput', 'phase1_throughput', 'req/s', lambda x: x),
        ]
        
        # Find max number of runs
        max_runs = max(len(run_stats_with), len(run_stats_without))
        
        print(f"{'Run':<6} {'Metric':<20} {'Without kv-mkt':<20} {'With kv-mkt':<20} {'Improvement':<20}")
        print("-" * 80)
        
        for run_idx in range(max_runs):
            run_with = run_stats_with[run_idx] if run_idx < len(run_stats_with) else {}
            run_without = run_stats_without[run_idx] if run_idx < len(run_stats_without) else {}
            
            for name, key, unit, formatter in phase1_metrics:
                without_val = run_without.get(key, 0)
                with_val = run_with.get(key, 0)
                
                if without_val > 0:
                    if key in ['phase1_avg_latency', 'phase1_total_time']:
                        # For latency, lower is better
                        improvement = ((without_val - with_val) / without_val) * 100
                        improvement_str = f"{improvement:+.1f}%"
                    elif key == 'phase1_throughput':
                        # For throughput, higher is better
                        improvement = ((with_val - without_val) / without_val) * 100
                        improvement_str = f"{improvement:+.1f}%"
                    else:
                        improvement_str = "N/A"
                else:
                    improvement_str = "N/A"
                
                without_str = f"{formatter(without_val)}{unit}" if without_val > 0 else "N/A"
                with_str = f"{formatter(with_val)}{unit}" if with_val > 0 else "N/A"
                
                run_label = f"Run {run_idx + 1}" if name == phase1_metrics[0][0] else ""
                print(f"{run_label:<6} {name:<20} {without_str:<20} {with_str:<20} {improvement_str:<20}")
            
            if run_idx < max_runs - 1:
                print("-" * 80)
    
    # Phase 2 metrics per run
    print(f"\n{'='*80}")
    print("  PHASE 2 METRICS (per run)")
    print(f"{'='*80}\n")
    
    if run_stats_with and run_stats_without:
        phase2_metrics = [
            ('Avg Latency', 'phase2_avg_latency', 's', lambda x: x),
            ('Total Time', 'phase2_total_time', 's', lambda x: x),
            ('Throughput', 'phase2_throughput', 'req/s', lambda x: x),
        ]
        
        # Find max number of runs
        max_runs = max(len(run_stats_with), len(run_stats_without))
        
        print(f"{'Run':<6} {'Metric':<20} {'Without kv-mkt':<20} {'With kv-mkt':<20} {'Improvement':<20}")
        print("-" * 80)
        
        for run_idx in range(max_runs):
            run_with = run_stats_with[run_idx] if run_idx < len(run_stats_with) else {}
            run_without = run_stats_without[run_idx] if run_idx < len(run_stats_without) else {}
            
            for name, key, unit, formatter in phase2_metrics:
                without_val = run_without.get(key, 0)
                with_val = run_with.get(key, 0)
                
                if without_val > 0:
                    if key in ['phase2_avg_latency', 'phase2_total_time']:
                        # For latency, lower is better
                        improvement = ((without_val - with_val) / without_val) * 100
                        improvement_str = f"{improvement:+.1f}%"
                    elif key == 'phase2_throughput':
                        # For throughput, higher is better
                        improvement = ((with_val - without_val) / without_val) * 100
                        improvement_str = f"{improvement:+.1f}%"
                    else:
                        improvement_str = "N/A"
                else:
                    improvement_str = "N/A"
                
                without_str = f"{formatter(without_val)}{unit}" if without_val > 0 else "N/A"
                with_str = f"{formatter(with_val)}{unit}" if with_val > 0 else "N/A"
                
                run_label = f"Run {run_idx + 1}" if name == phase2_metrics[0][0] else ""
                print(f"{run_label:<6} {name:<20} {without_str:<20} {with_str:<20} {improvement_str:<20}")
            
            if run_idx < max_runs - 1:
                print("-" * 80)
    
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

  # Run only kv-marketplace version (skip baseline when using --compare)
  python vllm_dual_gpu_demo.py --model gpt2 --compare --kv-marketplace-only
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Model name or path')
    parser.add_argument('--system-prompt', type=str,
                       default="You are a helpful AI assistant. Answer the user's questions concisely and accurately.",
                       help='Shared system prompt (default: helpful assistant prompt)')
    parser.add_argument('--system-prompt-file', type=str, default=None,
                       help='Path to a file containing the system prompt (overrides --system-prompt if set)')
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
    parser.add_argument('--max-model-len', type=int, default=4096,
                       help='Maximum model length (default: 4096)')
    parser.add_argument('--compare', action='store_true',
                       help='Run comparison with and without kv-marketplace')
    parser.add_argument('--no-kv-marketplace', action='store_true',
                       help='Disable kv-marketplace (default: enabled)')
    parser.add_argument('--save-results', type=str, default=None,
                       help='Save benchmark results to JSON file (e.g., results.json)')
    parser.add_argument('--kv-marketplace-only', action='store_true',
                       help='Run only kv-marketplace version (skip baseline when using --compare)')
    parser.add_argument('--print-registry-keys', action='store_true', default=False,
                       help='Print registry keys at the end (default: False)')
    parser.add_argument('--no-prefetch-phase2', action='store_true',
                       help='Disable Phase 2 prefix prefetch onto destination GPU (default: enabled)')
    parser.add_argument('--enable-logging', action='store_true',
                       help='Enable kv-marketplace INFO logging (default: disabled)')
    parser.add_argument('--print-outputs', action='store_true',
                       help='Print all generated outputs for each run (default: hidden)')
    parser.add_argument('--disable-speculative', action='store_true',
                       help='Disable speculative decoding when constructing the LLM (default: enabled)')
    parser.add_argument('--dump-kv-dir', type=str, default=None,
                       help='Directory to dump exported KV caches for offline comparison (default: disabled)')
    parser.add_argument('--logits-divergence-check', action='store_true',
                       help='Run deterministic Phase1/Phase2 logits comparison and exit')
    parser.add_argument('--logits-check-seed', type=int, default=1234,
                       help='Seed to use when running --logits-divergence-check (default: 1234)')
    
    args = parser.parse_args()

    if args.compare and args.no_kv_marketplace:
        print("ERROR: --compare always runs both baseline and kv-marketplace passes. "
              "Remove --no-kv-marketplace or drop --compare.")
        sys.exit(2)
    if not args.compare and args.kv_marketplace_only:
        print("ERROR: --kv-marketplace-only only applies when --compare is set.")
        sys.exit(2)
    if args.logits_divergence_check and args.compare:
        print("ERROR: --logits-divergence-check cannot be combined with --compare")
        sys.exit(2)

    if args.enable_logging:
        logging.getLogger("kv_marketplace").setLevel(logging.INFO)
    
    if args.system_prompt_file:
        prompt_path = os.path.expanduser(args.system_prompt_file)
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                args.system_prompt = f.read().strip()
            print(f"Loaded system prompt from {prompt_path} ({len(args.system_prompt)} chars)")
        except Exception as e:
            print(f"ERROR: Failed to load system prompt file '{prompt_path}': {e}")
            sys.exit(1)
    
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

    if args.logits_divergence_check:
        run_logits_divergence_check(
            model=args.model,
            system_prompt=args.system_prompt,
            user_prompts=args.user_prompts,
            kv_min_prefix=args.kv_min_prefix,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            seed=args.logits_check_seed,
        )
        return

    # Run benchmarks
    results_with = None
    results_without = None
    
    prefetch_phase2 = not args.no_prefetch_phase2

    if args.compare:
        # Run without kv-marketplace first (skip if --kv-marketplace-only is set)
        if not args.kv_marketplace_only:
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
                prefetch_phase2=prefetch_phase2,
                print_outputs=args.print_outputs,
                enable_speculative=not args.disable_speculative,
                dump_kv_dir=args.dump_kv_dir,
            )
        else:
            results_without = None
        
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
            prefetch_phase2=prefetch_phase2,
            print_outputs=args.print_outputs,
            enable_speculative=not args.disable_speculative,
            dump_kv_dir=args.dump_kv_dir,
        )
        
        # Create comparison chart (only if both results exist)
        if results_with and results_without:
            create_comparison_chart(results_with, results_without)
        elif args.kv_marketplace_only:
            print("\nSkipped baseline run (--kv-marketplace-only flag set)")
        
        # Save results if requested
        if args.save_results:
            if results_without:
                combined_results = {
                    'without_kv_marketplace': results_without,
                    'with_kv_marketplace': results_with,
                    'comparison': {
                        'latency_improvement_pct': ((results_without.get('avg_latency', 0) - results_with.get('avg_latency', 0)) / results_without.get('avg_latency', 1)) * 100 if results_without.get('avg_latency', 0) > 0 else 0,
                        'throughput_improvement_pct': ((results_with.get('throughput', 0) - results_without.get('throughput', 0)) / results_without.get('throughput', 1)) * 100 if results_without.get('throughput', 0) > 0 else 0,
                    }
                }
            else:
                combined_results = {
                    'with_kv_marketplace': results_with,
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
            prefetch_phase2=prefetch_phase2,
            print_outputs=args.print_outputs,
            enable_speculative=not args.disable_speculative,
            dump_kv_dir=args.dump_kv_dir,
        )
        
        if results:
            print_stats_table([results], "Benchmark Results")
            
            # Save results if requested
            if args.save_results:
                save_results_to_file(results, args.save_results)
    
    # Print registry keys at the end if requested and kv-marketplace is enabled
    if args.print_registry_keys:
        kv_marketplace_final = not args.no_kv_marketplace if not args.compare else True
        if kv_marketplace_final:
            get_stats_fn_final, _, _, get_registry_keys_fn_final = maybe_get_adapter_funcs()
            if get_registry_keys_fn_final:
                try:
                    registry_keys = get_registry_keys_fn_final()
                    print(f"\n{'='*80}")
                    print(f"  Registry Keys ({len(registry_keys)} total)")
                    print(f"{'='*80}")
                    if registry_keys:
                        for i, (compat_checksum, prefix_hash) in enumerate(registry_keys, 1):
                            print(f"  {i}. compat_checksum={compat_checksum.hex()[:16]}... prefix_hash={prefix_hash.hex()[:16]}...")
                    else:
                        print("  (Registry is empty)")
                except Exception as e:
                    print(f"\nWarning: Could not retrieve registry keys: {e}")
    
    print("\n✓ Demo completed!")


if __name__ == '__main__':
    main()
