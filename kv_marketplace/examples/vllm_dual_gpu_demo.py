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


def create_shared_prefix_prompts(system_prompt: str, user_prompts: List[str]) -> List[str]:
    """Create prompts with shared system prefix."""
    return [f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:" for prompt in user_prompts]


def measure_latency(llm, prompts: List[str], sampling_params: SamplingParams) -> Tuple[List[float], List[str]]:
    """Measure generation latency for prompts.
    
    Returns:
        Tuple of (latencies in seconds, generated texts)
    """
    latencies = []
    outputs = []
    
    for prompt in prompts:
        start_time = time.time()
        result = llm.generate([prompt], sampling_params)
        end_time = time.time()
        
        latency = end_time - start_time
        latencies.append(latency)
        
        if result and len(result) > 0 and len(result[0].outputs) > 0:
            outputs.append(result[0].outputs[0].text)
        else:
            outputs.append("")
    
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
    
    # Create prompts with shared prefix
    prompts = create_shared_prefix_prompts(system_prompt, user_prompts)
    
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
    
    # Run benchmark
    all_latencies = []
    all_outputs = []
    run_stats = []
    
    for run_idx in range(num_runs):
        print(f"\nRun {run_idx + 1}/{num_runs}...")
        
        # Get stats before run
        stats_before = get_registry_stats()
        if run_idx == 0 and kv_marketplace:
            print(f"  Stats before run: registry_size={stats_before['registry_size']}, "
                  f"prefix_index_size={stats_before['prefix_index_size']}")
        
        # Measure latency
        latencies, outputs = measure_latency(llm, prompts, sampling_params)
        all_latencies.extend(latencies)
        all_outputs.extend(outputs)
        
        # Get stats after run
        stats_after = get_registry_stats()
        
        # Calculate run statistics
        avg_latency = sum(latencies) / len(latencies)
        total_time = sum(latencies)
        throughput = len(prompts) / total_time if total_time > 0 else 0
        
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
