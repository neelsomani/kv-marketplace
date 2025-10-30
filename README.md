# Cross-GPU KV Cache Marketplace

## Abstract

We propose a distributed inference runtime that enables cross-GPU reuse of transformer attention states. In autoregressive language models, each token produces per-layer key and value (KV) tensors that are cached to accelerate subsequent decoding. Today, every process or GPU recomputes these KV tensors independently, even when multiple requests share identical or overlapping prefixes, a major source of redundant computation and GPU memory waste.

The Cross-GPU KV Cache Marketplace treats these attention states as first-class, shareable artifacts. Each process exports completed prefix caches, indexed by a hash of the input token sequence and model version, into a distributed registry. Other processes with matching prefixes can import those caches directly via GPU-to-GPU Remote Direct Memory Access (RDMA) or NVLink peer copies, bypassing host memory and avoiding recomputation. The marketplace manages registration, eviction, and lifetime tracking of exported tensors, ensuring correctness and low-latency transfer.

This design effectively transforms transformer inference into a cooperative caching network (a "memcached for attention"). In workloads such as chat serving, retrieval-augmented generation, or multi-tenant inference with common system prompts, it can eliminate repeated prefix computation, improving throughput and reducing GPU utilization by up to several times. Beyond efficiency, the framework opens new research avenues in distributed memory consistency for neural inference, prefix deduplication, and cache-aware load balancing across heterogeneous GPUs.

## Minimum Viable Prototype

This initial release will focus on node-local reuse of transformer KV tensors within vLLM. The planned MVP includes:

* A development branch of vLLM with two integration hooks to:
  * import KV state before prefill when a matching prefix exists,
  * export KV state after prefill for later reuse.
* A CUDA peer-to-peer and CUDA IPC transport component enabling direct movement of KV tensors across GPUs on the same machine.
* A prefix registry supporting longest-common-prefix lookup using token IDs.
* Configuration compatibility enforcement (model parameters, tokenizer, positional encoding, memory layout, dtype).
* Automatic fallback to standard execution when no applicable prefix is found.
* Tests confirming that reuse preserves next-token outputs within expected floating-point tolerance.
* Example scripts demonstrating how to enable reuse and observe its effects.

## Out of Scope for MVP

The MVP will not include:

* Cross-host KV transfer or a distributed registry.
* Any changes to global scheduling or routing within vLLM.
* Prefix eviction, scoring, or lifetime management policies.
* Tensor-parallel or pipeline-parallel sharded KV import.
* Compression or quantization of KV tensors during transfer.
* Integration with speculative decoding mechanisms.

Future iterations may expand into these areas after the node-local integration path is complete and validated.

## Installation

### Prerequisites

* Python 3.8 or higher
* PyTorch 2.0+ with CUDA support
* CUDA toolkit (for building the CUDA extension)
* A CUDA-capable GPU (or multiple GPUs for peer-to-peer tests)
* For peer-to-peer transport: GPUs must support CUDA peer-to-peer access (typically requires NVLink or PCIe Gen3+)
* On GCP: At least a2-highgpu-2g with 2x A100 GPUs (40GB each) (Deep Learning VM M129 image)

### Setup

1. Install the package and dependencies:
```bash
# Verify this works, or otherwise install the correct Nvidia drivers for your machine
nvidia-smi

# Verify this works as well, or otherwise install CUDA toolchain
nvcc --version

# Install Torch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install xxhash pytest

# Verify CUDA is working
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"

# Clone repo and install
git clone git@github.com:neelsomani/kv-marketplace.git
cd kv-marketplace
pip3 install -e .
```

This will:
* Install the package dependencies (PyTorch, xxhash, etc.)
* Build the CUDA extension (`p2p_cuda`) for peer-to-peer memory transfer

2. Verify the installation:
```bash
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

## Examples

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

The `vllm_dual_gpu_demo.py` script demonstrates the full integration with vLLM (requires the `vllm-kvm-dev` fork):

```bash
python kv_marketplace/examples/vllm_dual_gpu_demo.py
```
