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
