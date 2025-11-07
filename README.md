# Cross-GPU KV Cache Marketplace

## Abstract

We propose a distributed inference runtime that enables cross-GPU reuse of transformer attention states. In autoregressive language models, each token produces per-layer key and value (KV) tensors that are cached to accelerate subsequent decoding. Today, every process or GPU recomputes these KV tensors independently, even when multiple requests share identical or overlapping prefixes, a major source of redundant computation and GPU memory waste.

The Cross-GPU KV Cache Marketplace treats these attention states as first-class, shareable artifacts. Each process exports completed prefix caches, indexed by a hash of the input token sequence and model version, into a distributed registry. Other processes with matching prefixes can import those caches directly via GPU-to-GPU Remote Direct Memory Access (RDMA) or NVLink peer copies, bypassing host memory and avoiding recomputation. The marketplace manages registration, eviction, and lifetime tracking of exported tensors, ensuring correctness and low-latency transfer.

This design effectively transforms transformer inference into a cooperative caching network (a "memcached for attention"). In workloads such as chat serving, retrieval-augmented generation, or multi-tenant inference with common system prompts, it can eliminate repeated prefix computation, improving throughput and reducing GPU utilization. Beyond efficiency, the framework opens new research avenues in distributed memory consistency for neural inference, prefix deduplication, and cache-aware load balancing across heterogeneous GPUs.

## Minimum Viable Prototype

### Initial Release

This initial release will focus on node-local reuse of transformer KV tensors within vLLM.

* A development branch of vLLM with two integration hooks to:
  * import KV state before prefill when a matching prefix exists,
  * export KV state after prefill for later reuse.
* A CUDA peer-to-peer and CUDA IPC transport component enabling direct movement of KV tensors across GPUs on the same machine.
* A prefix registry supporting exact-match prefix reuse (LCP planned).
* Configuration compatibility enforcement (model parameters, tokenizer, positional encoding, memory layout, dtype).
* Automatic fallback to standard execution when no applicable prefix is found.
* Tests confirming that reuse preserves next-token outputs within expected floating-point tolerance.
* Example scripts demonstrating how to enable reuse and observe its effects.

### Out of Scope for MVP

The MVP will not include:

* Cross-host KV transfer or a distributed registry.
* Any changes to global scheduling or routing within vLLM.
* Prefix eviction, scoring, or lifetime management policies.
* Tensor-parallel or pipeline-parallel sharded KV import.
* Compression or quantization of KV tensors during transfer.
* Integration with speculative decoding mechanisms.
* Longest Common Prefix (LCP) matching to enable partial reuse even when prompts differ slightly.

Future iterations may expand into these areas after the node-local integration path is complete and validated.

### vLLM fork required for cross-GPU KV import

To try cross-GPU KV import with vLLM, you must use the companion fork and branch:

- Repo: `neelsomani/vllm`
- Branch: `vllm-kvm-dev`

This fork adds two integration hooks (`before_prefill`, `after_prefill`) and a minimal plugin shim used by this project.

## Installation

### Prerequisites

* Debian 12
* Python 3.11 (uv)
* PyTorch 2.0+ with CUDA support (CUDA 12.8)
* CUDA toolkit (for building the CUDA extension)
* A CUDA-capable GPU (or multiple GPUs for peer-to-peer tests)
* For peer-to-peer transport: GPUs must support CUDA peer-to-peer access (typically requires NVLink or PCIe Gen3+)
* On GCP: At least a2-highgpu-2g with 2x A100 GPUs (40GB each)

### Setup

Install the package and dependencies:
```bash
# Verify this gives 12.8
nvidia-smi

# Verify this works as well for 12.8
sudo ln -sfn /usr/local/cuda-12.8 /usr/local/cuda
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
nvcc --version

# Install Python matching CUDA 12.8
pip install -U pip wheel setuptools
pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch torchvision torchaudio

# Verify CUDA is working
python - <<'PY'
import torch, numpy
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("numpy:", numpy.__version__)
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPUs: {torch.cuda.device_count()}')
PY

# vLLM dependencies
pip install -U pip wheel setuptools ninja cmake packaging
pip install "ray[cgraph]>=2.48.0" numba==0.61.2 \
  openai-harmony>=0.0.3 anthropic==0.71.0 \
  pybase64 cbor2 setproctitle fastapi pydantic<3 uvicorn httpx safetensors

# Clone and install forked vLLM for demo - might require resolving dependency conflicts
export TORCH_CUDA_ARCH_LIST="8.0"  # A100
export VLLM_DISABLE_FP8=1
export VLLM_USE_XFORMERS=0
export MAX_JOBS=8
git clone https://github.com/neelsomani/vllm.git
cd vllm
git fetch origin vllm-kvm-dev
git checkout vllm-kvm-dev
pip install -r requirements/build.txt
pip install --no-deps --no-build-isolation -e .
pip install -r requirements/common.txt

# Alternatively - dependency fixes I needed
unset VLLM_USE_PRECOMPILED
unset VLLM_PRECOMPILED_WHEEL_LOCATION
cd vllm
python use_existing_torch.py
export MAX_JOBS=8
pip install -r requirements/build.txt
pip install --no-build-isolation -e .
pip install -r requirements/common.txt

# Clone repo and install
cd ../
git clone git@github.com:neelsomani/kv-marketplace.git
cd kv-marketplace
pip3 install -e .

# Verify the installation
python -c "from kv_marketplace.transport import PeerCopy; print('Installation successful')"
```

### Building the CUDA Extension

The CUDA extension is automatically built during installation. If you need to rebuild it manually:

```bash
pip install -e . --force-reinstall --no-deps
```

Or use the setup script directly:
```bash
cd kv_marketplace/transport
python setup.py build_ext --inplace
```

## Running Tests

The test suite includes unit tests for core components and integration tests for CUDA peer-to-peer transport.

### Running All Tests

```bash
pytest kv_marketplace/tests/
```

### Running Specific Test Suites

Core component tests (no CUDA required):
```bash
pytest kv_marketplace/tests/test_prefix_index.py
pytest kv_marketplace/tests/test_registry.py
```

CUDA peer-to-peer tests (requires 2+ GPUs):
```bash
pytest kv_marketplace/tests/test_p2p.py
```

These tests will automatically skip if:
* CUDA is not available
* Less than 2 GPUs are detected
* GPUs don't support peer-to-peer access

## Testing vLLM Integration

To verify that the patched vLLM fork correctly loads and calls the kv-marketplace hooks:

### Quick Smoke Test

Run vLLM with the `--kv-marketplace` flag enabled on a small model:

```bash
# Test that vLLM runs with kv-marketplace enabled (hooks should be called)
python -m vllm.entrypoints.llm \
    --model distilbert/distilgpt2 \
    --kv-marketplace \
    --kv-min-prefix 64 \
    --gpu-memory-utilization 0.1 \
    --enforce-eager \
    --max-model-len 128

# Optional: Allow imports even without peer access (uses slower PCIe path)
# --kv-allow-pcie
```

Then in another terminal, send a test request:

```bash
python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='distilbert/distilgpt2', kv_marketplace=True, gpu_memory_utilization=0.1, enforce_eager=True)
outputs = llm.generate(['Hello'], sampling_params=SamplingParams())
print(outputs[0].outputs[0].text)
"
```

Expected behavior:
* vLLM should start without errors
* The request should complete successfully
* With logging enabled, you may see messages like "kv-marketplace: Exported prefix..." (hooks are being called)

### Verify Plugin Loading

Test that the kv-marketplace adapter can be loaded:

```bash
python -c "from vllm.kv_marketplace_shim import load_plugin; plugin = load_plugin(); print('Plugin loaded:', plugin is not None); print('Has before_prefill:', hasattr(plugin, 'before_prefill') if plugin else False)"
```

Should output: `Plugin loaded: True` and `Has before_prefill: True`

### Check Logs

Run with logging to see hook activity:

```bash
VLLM_LOGGING_LEVEL=INFO python -m vllm.entrypoints.llm \
    --model distilbert/distilgpt2 \
    --kv-marketplace \
    --gpu-memory-utilization 0.1 \
    --enforce-eager
```

Look for log messages containing "kv-marketplace" indicating the hooks are being called.

## End-to-End Examples

### Two-GPU Demo

The `two_gpu_demo.py` script demonstrates peer-to-peer memory transfer between two GPUs without vLLM:

```bash
python kv_marketplace/examples/two_gpu_demo.py
```

This script:
* Creates random data buffers on GPU 0
* Transfers them to GPU 1 via CUDA peer-to-peer copy
* Validates the transfer using checksums

Requirements:
* 2+ GPUs
* CUDA extension built
* GPUs must support peer-to-peer access

Expected output:
```
Found 2 GPUs
Creating 10MB buffer on GPU 0...
Source checksum: <hash>
Creating destination buffer on GPU 1...
Enabling peer access between GPU 0 and GPU 1...
Peer access enabled successfully!
Performing peer-to-peer copy...
Copy completed!
Verifying checksum...
Destination checksum: <hash>
✓ SUCCESS: Checksums match! Peer copy validated.
```

### vLLM Integration Demo

The `vllm_dual_gpu_demo.py` script demonstrates the full integration with vLLM (requires the `vllm-kvm-dev` fork).

Run with default settings (kv-marketplace enabled):

```bash
python kv_marketplace/examples/vllm_dual_gpu_demo.py --model gpt2
```

Run a side-by-side comparison with and without kv-marketplace:

```bash
python kv_marketplace/examples/vllm_dual_gpu_demo.py --model gpt2 --kv-min-prefix 16 --compare
```

Use custom system prompts and user prompts to see the benefit for longer prefixes:

```bash
python kv_marketplace/examples/vllm_dual_gpu_demo.py \
    --model gpt2-medium \
    --system-prompt "You are an exceptionally knowledgeable and experienced AI assistant with deep expertise spanning multiple domains including advanced science, cutting-edge technology, comprehensive history, world literature, complex mathematics, intricate philosophy, detailed psychology, modern economics, contemporary politics, environmental science, medical knowledge, engineering principles, artistic theory, cultural studies, and current global events. Your responses are meticulously researched, factually accurate, thoroughly comprehensive, and exceptionally well-structured. You excel at breaking down highly complex topics into understandable components while maintaining intellectual rigor. You provide relevant examples, cite important facts, and offer multiple perspectives when appropriate. You adapt your communication style dynamically to match the user's level of expertise while always maintaining clarity, precision, and educational value. When discussing technical or academic topics, you seamlessly provide both high-level conceptual overviews and detailed technical explanations as the context requires. You always strive to be maximally helpful, completely harmless, and rigorously honest in every interaction. Your knowledge base is extensive and you are able to synthesize information from multiple disciplines to provide insightful and nuanced responses. You value evidence-based reasoning, logical consistency, and intellectual humility. When you encounter uncertainty or limitations in your knowledge, you acknowledge them transparently. Your goal is to empower users with accurate information, foster deeper understanding, and facilitate meaningful learning experiences through thoughtful and well-crafted responses." \
    --user-prompts \
        "What is the capital of France?" \
        "Explain quantum computing." \
        "Write a poem about AI." \
        "What are renewable energy benefits?" \
        "How does photosynthesis work?" \
    --num-runs 5 \
    --kv-min-prefix 64 \
    --compare
```

The demo will display:

1. Benchmark runs with per-run statistics:
   * Average latency per request
   * Total time
   * Throughput (requests/second)
   * Registry size (number of cached prefixes)
   * Prefix index size
   * Import hits and misses (when kv-marketplace is enabled)
   * Average LCP length for successful imports

2. Comparison chart (when using `--compare`):
   * Side-by-side metrics with and without kv-marketplace
   * Improvement percentages for latency and throughput
   * Import statistics

Example output snippet:
```
[...]
  Stats after run: registry_size=5, prefix_index_size=75, total_import_hits=10, total_import_misses=5
  Average latency: 0.8995s
  Total time: 4.4977s
  Throughput: 1.11 req/s
  Registry size: 5
  Prefix index size: 75
  Import hits: 5
  Import misses: 0

================================================================================
  COMPARISON CHART
================================================================================

Metric               Without kv-mkt       With kv-mkt          Improvement         
--------------------------------------------------------------------------------
Average Latency      0.8560004552205404s  0.8420368671417237s  +1.6%               
Total Latency        12.840006828308105s  12.630553007125854s  +1.6%               
Throughput           1.168223677804423req/s 1.1875964569039346req/s +1.7%               
Import Hits          N/A                  10                   N/A                 
Avg LCP Length       N/A                  32.0tokens           N/A                 
Registry Size        5                    5                    N/A                 

================================================================================

✓ Demo completed!
```
