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
from typing import List, Dict, Tuple
from collections import defaultdict

# Enable logging to see hook activity
logging.basicConfig(level=logging.INFO)

# IMPORTANT: Import vLLM BEFORE adding parent directory to sys.path
# Otherwise Python will import from the local vllm/ directory instead of installed package
try:
    from vllm.sampling_params import SamplingParams
    # Pre-initialize vllm module to ensure __getattr__ has cached PoolingRequestOutput
    import vllm
    _ = vllm.PoolingRequestOutput  # Trigger __getattr__ to cache it
except ImportError as e:
    print("ERROR: Could not import from vLLM.")
    print(f"Import error: {e}")
    print("Make sure you're using the vllm-kvm-dev fork.")
    print("Install it with: cd vllm && pip install -e .")
    sys.exit(1)

# Add parent directory to path (after vLLM imports to avoid conflicts)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Check for multi-GPU setup BEFORE importing adapter
# This allows the adapter to use file-based backend if needed
# The adapter initializes the registry at module import time, so we need to set
# the env var before importing
try:
    import torch
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        os.environ['KV_MARKETPLACE_FILE_BACKEND'] = '1'
        print(f"Detected {torch.cuda.device_count()} GPUs, enabling file-based registry backend for cross-process sharing")
except ImportError:
    pass

# LLM import will be done lazily to avoid import-time issues
LLM = None

try:
    from kv_marketplace.adapter.vllm import get_stats, reset_stats
    STATS_AVAILABLE = True
except ImportError:
    print("WARNING: Could not import kv-marketplace adapter. Stats may not be available.")
    STATS_AVAILABLE = False


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


def measure_latency_concurrent(llm, prompts: List[str], sampling_params: SamplingParams, 
                               batch_name: str = "batch") -> Tuple[List[float], List[str], List[int]]:
    """Measure generation latency for prompts concurrently.
    
    Uses ThreadPoolExecutor to send requests concurrently so they can be
    distributed across GPUs and see each other's caches.
    
    Returns:
        Tuple of (latencies in seconds, generated texts, device_ids)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import torch
    
    def generate_single(prompt: str, prompt_idx: int) -> Tuple[int, float, str]:
        """Generate for a single prompt and return device_id, latency, output."""
        # Try to detect which GPU this request goes to by checking current device
        # Note: This is approximate - vLLM may use different devices internally
        device_before = torch.cuda.current_device() if torch.cuda.is_available() else -1
        
        start_time = time.time()
        result = llm.generate([prompt], sampling_params)
        end_time = time.time()
        
        device_after = torch.cuda.current_device() if torch.cuda.is_available() else -1
        
        latency = end_time - start_time
        
        output = ""
        if result and len(result) > 0 and len(result[0].outputs) > 0:
            output = result[0].outputs[0].text
        
        # Use device_after as best guess (vLLM may have set it during processing)
        device_id = device_after if device_after >= 0 else device_before
        
        print(f"  {batch_name}: Request {prompt_idx+1} completed on GPU {device_id} (latency: {latency:.4f}s)")
        
        return (device_id, latency, output)
    
    latencies = []
    outputs = []
    device_ids = []
    
    # Use ThreadPoolExecutor to send requests concurrently
    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = {executor.submit(generate_single, prompt, idx): idx 
                  for idx, prompt in enumerate(prompts)}
        
        results = {}
        for future in as_completed(futures):
            prompt_idx = futures[future]
            try:
                device_id, latency, output = future.result()
                results[prompt_idx] = (device_id, latency, output)
            except Exception as e:
                print(f"  Error generating prompt {prompt_idx}: {e}")
                results[prompt_idx] = (-1, 0.0, "")
        
        # Reconstruct in order
        for idx in range(len(prompts)):
            device_id, latency, output = results[idx]
            device_ids.append(device_id)
            latencies.append(latency)
            outputs.append(output)
    
    return latencies, outputs, device_ids


def measure_latency(llm, prompts: List[str], sampling_params: SamplingParams) -> Tuple[List[float], List[str]]:
    """Measure generation latency for prompts (sequential, for backward compatibility)."""
    latencies, outputs, _ = measure_latency_concurrent(llm, prompts, sampling_params, "sequential")
    return latencies, outputs


def get_registry_stats() -> Dict:
    """Get statistics from the KV registry and prefix index."""
    if STATS_AVAILABLE:
        try:
            return get_stats()
        except Exception as e:
            print(f"Warning: Could not get stats: {e}")
    
    # Fallback
    return {
        'registry_size': 0,
        'prefix_index_size': 0,
        'import_hits': 0,
        'import_misses': 0,
        'import_lcp_lengths': [],
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
    
    print(f"\nPhase 1: {len(phase1_prompts)} prompts (warm-up, concurrent)")
    print(f"Phase 2: {len(phase2_prompts)} prompts (test with scrambled prefixes, concurrent)")
    
    # Note: File-based backend is enabled at module import time if multiple GPUs detected
    # This allows sharing registry across processes on the same machine
    
    # Reset stats if kv-marketplace is enabled
    if kv_marketplace and STATS_AVAILABLE:
        try:
            reset_stats()
        except Exception:
            pass
    
    # Lazy import LLM to avoid circular import issues
    global LLM
    if LLM is None:
        try:
            from vllm.entrypoints.llm import LLM
        except ImportError as e:
            print("ERROR: Could not import LLM from vLLM.")
            print(f"Import error: {e}")
            print("Make sure you're using the vllm-kvm-dev fork.")
            print("Install it with: cd vllm && pip install -e .")
            return None
    
    # Initialize LLM
    print(f"\nInitializing LLM...")
    llm_kwargs.update({
        'kv_marketplace': kv_marketplace,
        'kv_min_prefix': kv_min_prefix,
        'gpu_memory_utilization': gpu_memory_utilization,
        'tensor_parallel_size': tensor_parallel_size,
        'max_model_len': max_model_len,
    })
    
    try:
        llm = LLM(model=model, **llm_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to initialize LLM: {e}")
        return None
    
    print("LLM initialized successfully!")
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=256,
    )
    
    # Run benchmark with two-phase approach
    all_latencies = []
    all_outputs = []
    run_stats = []
    
    for run_idx in range(num_runs):
        print(f"\n{'='*80}")
        print(f"Run {run_idx + 1}/{num_runs}")
        print(f"{'='*80}")
        
        # Get stats before run
        stats_before = get_registry_stats()
        if run_idx == 0 and kv_marketplace:
            print(f"Stats before run: registry_size={stats_before['registry_size']}, "
                  f"prefix_index_size={stats_before['prefix_index_size']}")
        
        # Phase 1: Warm-up (concurrent)
        print(f"\n--- Phase 1: Warm-up (concurrent, {len(phase1_prompts)} requests) ---")
        phase1_start = time.time()
        phase1_latencies, phase1_outputs, phase1_device_ids = measure_latency_concurrent(
            llm, phase1_prompts, sampling_params, batch_name="Phase1"
        )
        phase1_end = time.time()
        phase1_total_time = phase1_end - phase1_start
        
        # Log device distribution for Phase 1
        device_counts = {}
        for device_id in phase1_device_ids:
            device_counts[device_id] = device_counts.get(device_id, 0) + 1
        print(f"Phase 1 device distribution: {device_counts}")
        print(f"Phase 1 total time: {phase1_total_time:.4f}s (concurrent)")
        print(f"Phase 1 average latency: {sum(phase1_latencies) / len(phase1_latencies):.4f}s per request")
        
        # Get stats after Phase 1
        stats_after_phase1 = get_registry_stats()
        if kv_marketplace:
            print(f"Stats after Phase 1: registry_size={stats_after_phase1['registry_size']}, "
                  f"prefix_index_size={stats_after_phase1['prefix_index_size']}")
        
        # Small delay to ensure Phase 1 exports complete
        time.sleep(0.5)
        
        # Phase 2: Test with scrambled prefixes (concurrent)
        print(f"\n--- Phase 2: Test (concurrent, {len(phase2_prompts)} requests with scrambled prefixes) ---")
        phase2_start = time.time()
        phase2_latencies, phase2_outputs, phase2_device_ids = measure_latency_concurrent(
            llm, phase2_prompts, sampling_params, batch_name="Phase2"
        )
        phase2_end = time.time()
        phase2_total_time = phase2_end - phase2_start
        
        # Log device distribution for Phase 2
        device_counts = {}
        for device_id in phase2_device_ids:
            device_counts[device_id] = device_counts.get(device_id, 0) + 1
        print(f"Phase 2 device distribution: {device_counts}")
        print(f"Phase 2 total time: {phase2_total_time:.4f}s (concurrent)")
        print(f"Phase 2 average latency: {sum(phase2_latencies) / len(phase2_latencies):.4f}s per request")
        
        # Combine results
        latencies = phase1_latencies + phase2_latencies
        outputs = phase1_outputs + phase2_outputs
        all_latencies.extend(latencies)
        all_outputs.extend(outputs)
        
        # Get stats after run
        stats_after = get_registry_stats()
        
        # Calculate run statistics
        avg_latency = sum(latencies) / len(latencies)
        total_time = phase1_total_time + phase2_total_time  # Use concurrent total time, not sum of latencies
        throughput = len(latencies) / total_time if total_time > 0 else 0
        
        # Calculate import statistics
        import_hits = stats_after['import_hits'] - stats_before['import_hits']
        import_misses = stats_after['import_misses'] - stats_before['import_misses']
        lcp_lengths = stats_after.get('import_lcp_lengths', [])
        # Get only the new LCP lengths from this run (subtract before from after)
        before_lcp_count = len(stats_before.get('import_lcp_lengths', []))
        after_lcp_count = len(lcp_lengths)
        new_lcp_lengths = lcp_lengths[before_lcp_count:] if after_lcp_count > before_lcp_count else []
        avg_lcp = sum(new_lcp_lengths) / len(new_lcp_lengths) if new_lcp_lengths else 0
        
        if kv_marketplace:
            print(f"  Stats after run: registry_size={stats_after['registry_size']}, "
                  f"prefix_index_size={stats_after['prefix_index_size']}, "
                  f"total_import_hits={stats_after['import_hits']}, "
                  f"total_import_misses={stats_after['import_misses']}")
        
        run_stat = {
            'run': run_idx + 1,
            'avg_latency': avg_latency,
            'total_latency': total_time,
            'throughput': throughput,
            'registry_size': stats_after['registry_size'],
            'prefix_index_size': stats_after['prefix_index_size'],
            'import_hits': import_hits,
            'import_misses': import_misses,
            'import_lcp_lengths': new_lcp_lengths,
        }
        run_stats.append(run_stat)
        
        print(f"  Average latency: {avg_latency:.4f}s")
        print(f"  Total time: {total_time:.4f}s")
        print(f"  Throughput: {throughput:.2f} req/s")
        print(f"  Registry size: {stats_after['registry_size']}")
        print(f"  Prefix index size: {stats_after['prefix_index_size']}")
        if kv_marketplace:
            print(f"  Import hits: {import_hits}")
            print(f"  Import misses: {import_misses}")
            if avg_lcp > 0:
                print(f"  Average LCP length: {avg_lcp:.1f} tokens")
    
    # Calculate overall statistics
    overall_avg_latency = sum(all_latencies) / len(all_latencies)
    overall_total_time = sum(all_latencies)
    overall_throughput = len(prompts) * num_runs / overall_total_time if overall_total_time > 0 else 0
    
    final_stats = get_registry_stats()
    
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
        'num_prompts': len(prompts),
        'avg_latency': overall_avg_latency,
        'total_latency': overall_total_time,
        'throughput': overall_throughput,
        'registry_size': final_stats['registry_size'],
        'prefix_index_size': final_stats['prefix_index_size'],
        'import_hits': total_import_hits,
        'import_misses': total_import_misses,
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
        ('Import Hits', 'import_hits', '', lambda x: int(x)),
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
