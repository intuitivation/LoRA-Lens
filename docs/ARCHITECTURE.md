# Technical Architecture

## Compression Pipeline

```
Input: LoRA.safetensors (e.g., 144 MB)
│
├─ Stage 1: Format Detection
│  ├─ Identify: SD 1.5 / SDXL / FLUX
│  ├─ Detect precision: Float16 / BFloat16 / Float32
│  └─ Map weight keys: lora_up/down (SD/SDXL) or lora_A/B (FLUX)
│
├─ Stage 2: SVD Analysis & Rank Optimization
│  ├─ Singular Value Decomposition on each layer pair
│  ├─ Calculate effective rank via variance threshold
│  ├─ Identify and prune unused dimensions
│  ├─ Enforce rank ceiling (never exceeds original)
│  └─ Reduction: 30-90% depending on format
│
├─ Stage 3: Quantization (Pro/Studio only)
│  ├─ Symmetric 8-bit quantization (BFloat16/Float16 → Int8)
│  ├─ Per-tensor scale factor calculation
│  ├─ Store: quantized weights + scale factors
│  └─ Additional reduction: ~50%
│
└─ Output: Optimized LoRA (e.g., 9 MB)
   ├─ Compatible with ComfyUI, A1111, Forge, etc.
   └─ Quality retention: 99%+
```

## Dual-Stage Compression Detail

### Stage 1: SVD-Based Rank Optimization

LoRA weights are inherently low-rank by design (that's the "Low-Rank" in LoRA). However, many trained LoRAs use higher ranks than necessary — a rank-128 LoRA may only need rank-16 to capture 99%+ of the learned information.

LoRA Lens performs SVD on each layer's weight matrices:

1. **Decompose** the weight matrix into singular values
2. **Analyze** the cumulative variance explained by each singular value
3. **Determine** the optimal rank where 99%+ variance is retained
4. **Reconstruct** the weight matrices at the reduced rank

This is particularly effective for SD 1.5 and SDXL LoRAs (80-90% reduction) because many community-trained LoRAs use unnecessarily high ranks. FLUX LoRAs tend to use their rank more efficiently, yielding 30-40% reduction from rank optimization alone.

### Stage 2: 8-Bit Symmetric Quantization

After rank optimization, LoRA Lens can further compress by quantizing the remaining weights:

```
For each tensor:
  1. Calculate scale = max(abs(tensor)) / 127
  2. Quantize: int8_tensor = round(tensor / scale)
  3. Store: int8_tensor (1 byte/weight) + scale (4 bytes/tensor)

Reconstruction:
  float_tensor = int8_tensor * scale
```

This halves the storage of the already-compressed weights. The per-tensor scale factor preserves the relative magnitudes within each layer.

**Quality metrics computed during quantization:**
- **SNR (Signal-to-Noise Ratio):** Typically 46-52 dB (higher is better)
- **MSE (Mean Squared Error):** Typically 0.0008-0.0015 (lower is better)
- **MAE (Mean Absolute Error):** Reported for transparency

### Ultra Compress Mode

Combines both stages in sequence: rank optimization followed by quantization. This is the recommended mode for maximum compression with quality retention.

## Smart Precision Handling

LoRA Lens detects and preserves the native precision of each LoRA:

| Source Format | Detection | Optimization |
|---------------|-----------|-------------|
| Float16 | SD 1.5, SDXL LoRAs | Standard SVD + optional Int8 |
| BFloat16 | FLUX LoRAs | BFloat16-aware SVD + optional Int8 |
| Float32 | Rare, older LoRAs | Downcast + SVD + optional Int8 |

Key behaviors:
- Never increases file size (a common bug in naive implementations)
- Enforces rank ceiling (pruned rank ≤ original rank)
- Detects both `lora_up/lora_down` (SD/SDXL) and `lora_A/lora_B` (FLUX) naming conventions

## Processing Performance

| Task | GPU (RTX 3090) | CPU (Ryzen 9 5950X) |
|------|:--------------:|:-------------------:|
| Analysis | 2-5 sec | 5-10 sec |
| Rank Optimization | 10-15 sec | 20-30 sec |
| 8-bit Quantization | 5-8 sec | 10-15 sec |
| Ultra Compress | 15-25 sec | 30-45 sec |

LoRA Lens runs entirely locally — no data leaves your machine.
