# 🥣 Soup Kernel Challenge — Decentrathon 6.0
> **High-Performance LLM Fine-Tuning Optimization for NVIDIA Tesla T4 (`sm_75`)**  
> **Team:** LunarTeam  
> **Batch Optimization, Harness & Testing Engineer:** Darkhan (Дархан)

---

## 🚀 Overview & Key Results

This project optimizes the QLoRA fine-tuning pipeline for **Qwen2.5-1.5B-Instruct** on NVIDIA Tesla T4 GPU (16GB GDDR6, Turing `sm_75` architecture). 

By designing and replacing critical memory and compute bottlenecks with **custom Triton kernels** and **architectural batch optimizations**, we achieved dramatic VRAM savings and speedups while maintaining mathematical convergence and 100% strict compliance with the hackathon harness rules.

### 📊 Benchmark Summary (Kaggle Tesla T4)

| Metric | Baseline (HuggingFace + PEFT + BnB) | Our Solution (LunarTeam) | Improvement |
| :--- | :---: | :---: | :---: |
| **GPU Peak Memory (`nvidia-smi`)** | `14,345 MiB` | **`4,231 MiB`** | **3.39x memory reduction (-70.5% VRAM)** |
| **PyTorch Max Allocated** | `12,184 MiB` | **`3,855 MiB`** | **-8,329 MiB saved** |
| **Training Step Time** | `0.934 s / step` | **`0.565 s / step`** | **1.65x loop speedup** |
| **Total Wall Time** | `91.80 s` | **`65.64 s`** | **+26.16 s faster** |
| **Model Quality (`eval_loss`)** | `2.04964` | **`2.04952`** | **-0.006% (strictly preserved)** |
| **Harness Rules Compliance** | — | **10 / 10 PASS** | **20 / 20 Gate Points** |
| **Own Triton Kernels Share** | — | **12.1%** | (Requirement: $\ge 5.0\%$) |
| **TestSuite Verification** | 79 passed | **79 passed (0.78s)** | 100% Green |

---

## 🧠 Architectural Highlights & Engineering Innovations

### 1. 2D Block-Tiled Group-128 INT8 SwiGLU Activation Compression
* **The Problem:** In Qwen2.5-1.5B (`intermediate_size = 8960`), caching full `gate` and `up` FP16 activations across 28 layers consumes $\approx 1.96\text{ GiB}$ of VRAM.
* **Our Solution:** A custom `@triton.jit` kernel that computes exact forward `silu(gate) * up` in FP32 registers while simultaneously quantizing saved activations into symmetric Group-128 `int8` with per-group dynamic scales.
* **Impact:** Cut SwiGLU activation memory by **48.4%** ($\approx 950\text{ MiB}$ saved) and halved memory read traffic during backward pass.

### 2. Norm-Only FP32 Selective K-Bit Preparation
* **The Problem:** Standard `peft.prepare_model_for_kbit_training` upcasts *all* non-quantized weights to FP32, including the tied `embed_tokens`/`lm_head` ($151936 \times 1536$). This turns a $445\text{ MiB}$ FP16 tensor into an $890\text{ MiB}$ FP32 monster and demotes the entire 28-layer residual stream to FP32.
* **Our Solution:** Selectively upcast *only* normalization layer weights (`RMSNorm`) to FP32, keeping embeddings, LM head, and linear biases in native FP16.
* **Impact:** Saved **$1.23\text{ GiB}$** of static/dynamic VRAM and kept residual adds on 65 TFLOPS FP16 Tensor Cores instead of 8.1 TFLOPS FP32 SIMT cores.

### 3. Fused Chunked Cross-Entropy + Streaming Metrics
* **The Problem:** Unrolling full $151,936$ vocabulary logits for a sequence of 512 tokens creates a massive $311\text{ MiB}$ tensor per layer, blowing up peak memory.
* **Our Solution:** Fused chunked cross-entropy with online softmax in FP32 registers, computing Cross-Entropy loss, Shannon entropy, and argmax token accuracy in a single pass directly into an in-place gradient buffer (`CHUNK_SIZE = 128`).
* **Impact:** Reduced peak logit memory from $4.5\text{ GiB}$ down to $< 100\text{ MiB}$ with zero host-device synchronization stalls.

### 4. Fused Q+K RoPE (Rotary Position Embedding)
* **The Problem:** Upstream PyTorch RoPE invokes multiple slice, `rotate_half`, multiplication, and addition kernels for both Query ($12$ heads) and Key ($2$ heads).
* **Our Solution:** A single 2D grid Triton kernel that fuses forward and backward RoPE passes, outputting contiguous `(B, H, S, D)` layouts ready for Flash / Cutlass SDPA without intermediate allocations.

### 5. Fast NF4 Vectorized Double-Dequantization
* **The Problem:** `bitsandbytes` dequantization runs multiple separate CUDA kernels for nested scales and 4-bit unpacking.
* **Our Solution:** Byte-packed Triton kernel loading 2 weights per byte, fusing nested double-quantization scaling in SRAM registers, and writing directly into reusable per-device transient workspaces.

---

## 🛠️ Repository Structure

```text
├── solution/
│   ├── dynamic_padding.py     # Pre-H2D padding truncation & 8-token tensor core alignment
│   ├── fused_cross_entropy.py # Fused chunked cross-entropy, streaming entropy & accuracy
│   ├── fused_rmsnorm.py       # Fused RMSNorm forward & backward Triton kernels
│   ├── fused_rope.py          # Fused Q+K rotary position embedding kernel
│   ├── kbit_mp.py             # Selective FP16 residual stream & zero-clone LoRA forward
│   ├── nf4_dequant.py         # Vectorized byte-packed NF4 dequantization
│   ├── swiglu.py              # Group-128 INT8 activation-compressed SwiGLU
│   ├── test_kernels.py        # Correctness & numerical gradient unit tests
│   └── __init__.py            # Clean idempotent `apply()` hook
├── harness/                   # Hackathon evaluation & rule compliance harness
│   ├── check_rules.py         # Official 10-check validation suite
│   ├── run_task.py            # End-to-end benchmark runner
│   └── score.py               # Official scoring formula
├── tests/                     # 79 standard harness integration tests
└── README.md                  # Project overview & technical documentation
```

---

## 🚦 How to Run & Verify

### 1. Requirements
* Linux / Windows with NVIDIA GPU (tested on Tesla T4, Turing `sm_75`)
* Python 3.10+
* PyTorch 2.1+, Triton 3.0+, Transformers, PEFT, Bitsandbytes, TRL

### 2. Run Official Benchmark
```bash
python harness/run_task.py
```
This automatically runs:
1. Warm-up pass.
2. Standard baseline training & evaluation.
3. Our patched solution training & evaluation.
4. Output comparison and writes `runs/result.json`.

### 3. Verify All Rules (Gate 20/20)
```bash
python harness/check_rules.py
```
Verifies no banned libraries, all 9 custom Triton kernels active, kernel execution time share $\ge 5\%$, training curve fidelity, and executes all 79 test suites.

---

## 👨‍💻 Team LunarTeam
* **Darkhan (Дархан)** — Batch Optimization, Mixed Precision & Harness Testing Engineer
* **Bekbolat (Бекболат)** — Core GPU Kernels (NF4 Dequant, Fused Cross-Entropy)
* **Nurdaulet (Нұрдәулет)** — Activation-Compressed SwiGLU, RMSNorm & RoPE
* **Togzhan (Тоғжан)** — Research, Benchmarks & Validation

*Developed for Decentrathon 6.0 Hackathon.*
