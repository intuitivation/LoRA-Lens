# Benchmarks

All benchmarks performed with LoRA Lens v1.6 on RTX 3090 (GPU) and Ryzen 9 5950X (CPU).

## Compression Ratios by Format

| Format | Original Size | After Rank Opt | After Quant (Pro) | Total Reduction |
|--------|:------------:|:--------------:|:-----------------:|:---------------:|
| **SD 1.5** | 144 MB | 18 MB (87.5%) | 9 MB | **93.8%** |
| **SDXL** | 144 MB | 16 MB (88.9%) | 8 MB | **94.4%** |
| **FLUX** | 487 MB | 175 MB (64.1%) | 87 MB | **82.1%** |

## Quality Retention

| Metric | SD 1.5 | SDXL | FLUX |
|--------|:------:|:----:|:----:|
| **Variance Retained** | 99.2% | 99.4% | 99.1% |
| **MSE** | 0.0012 | 0.0008 | 0.0015 |
| **SNR** | 48.2 dB | 51.7 dB | 46.8 dB |
| **Visual Quality** | Identical | Identical | Identical |

## 10-LoRA Test Database

Download these LoRAs and run them through LoRA Lens yourself to verify results:

| LoRA | Original | Optimized | Savings |
|------|:--------:|:---------:|:-------:|
| flux_koda_style | 342.0 MB | 1.4 MB | 99.6% |
| flux_anime_style | 42.8 MB | 1.9 MB | 95.6% |
| dmd2_sdxl_4step | 750.9 MB | 154.3 MB | 79.4% |
| hypersd_sdxl_2step | 750.9 MB | 186.0 MB | 75.2% |
| hypersd_sdxl_1step | 750.9 MB | 193.9 MB | 74.2% |
| hypersd_sdxl_8step | 750.9 MB | 239.6 MB | 68.1% |
| hypersd_sdxl_4step | 750.9 MB | 245.4 MB | 67.3% |
| lcm_lora_sd15 | 128.4 MB | 71.0 MB | 44.7% |
| hypersd_sd15_4step | 256.7 MB | 142.8 MB | 44.4% |
| hypersd_sd15_8step | 256.7 MB | 143.1 MB | 44.3% |
| **TOTAL** | **4,780.9 MB** | **1,379.3 MB** | **71.2%** |

**As .loradb collection:** 727.9 MB (47% additional compression via differential storage)

**Extraction quality:** All 10 LoRAs extract with EXCELLENT quality (max diff < 0.001)

### Demo Collection

The repo includes `demo_collection.loradb` (3.3 MB) containing the two FLUX LoRAs above, ready to test extraction immediately.

### Download Links

- **Hyper-SD** (ByteDance): https://huggingface.co/ByteDance/Hyper-SD
- **DMD2** (tianweiy): https://huggingface.co/tianweiy/DMD2
- **LCM-LoRA SD1.5**: https://huggingface.co/latent-consistency/lcm-lora-sdv1-5
- **FLUX Style LoRAs**: Available on CivitAI

## .loradb Collection Compression

| Collection Type | Individual Size | As .loradb | Compression |
|----------------|:--------------:|:----------:|:-----------:|
| 50 character LoRAs | 7.2 GB | 380 MB | 94.7% |
| 100 style LoRAs | 14.4 GB | 890 MB | 93.8% |
| 20 lighting LoRAs | 2.88 GB | 145 MB | 95.0% |

## Processing Speed

| Task | GPU (RTX 3090) | CPU (Ryzen 9 5950X) |
|------|:--------------:|:-------------------:|
| Analysis | 2-5 sec | 5-10 sec |
| Rank Optimization | 10-15 sec | 20-30 sec |
| 8-bit Quantization | 5-8 sec | 10-15 sec |
| Ultra Compress | 15-25 sec | 30-45 sec |
