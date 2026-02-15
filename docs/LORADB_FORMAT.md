# .loradb Format Specification

## What is a .loradb?

A LoRA Database (`.loradb`) is a single-file format that stores multiple LoRAs using differential compression. Instead of storing each LoRA independently, it stores a compressed base LoRA and then only the *differences* (deltas) between the base and each subsequent LoRA.

This is especially effective for related LoRAs — character variants, style series, lighting packs — where the underlying weights share significant structure.

## How It Works

```
Traditional Storage:
├─ warrior.safetensors    (144 MB)
├─ mage.safetensors       (144 MB)
├─ rogue.safetensors      (144 MB)
├─ paladin.safetensors    (144 MB)
└─ ranger.safetensors     (144 MB)
Total: 720 MB (5 separate files)

.loradb Storage:
character_pack.loradb     (45 MB)
├─ Base: warrior (compressed)    ~20 MB
├─ Delta: mage (changes only)    ~6 MB
├─ Delta: rogue (changes only)   ~7 MB
├─ Delta: paladin (changes only) ~6 MB
└─ Delta: ranger (changes only)  ~6 MB
Total: 45 MB (1 file, 94% reduction)
```

**Reconstruction:** `base + delta_N = LoRA_N` at original quality.

## File Structure

```
.loradb File Layout:
│
├─ Header (fixed size)
│  ├─ Magic bytes: 'LORA' (4 bytes)
│  ├─ Format version: 1.0 (4 bytes)
│  ├─ LoRA count: N (4 bytes)
│  └─ Metadata length: M bytes (4 bytes)
│
├─ Metadata Block (JSON, M bytes)
│  ├─ Collection info
│  │  ├─ name, creator, description
│  │  ├─ version, creation date
│  │  └─ tags (searchable)
│  └─ LoRA manifest
│     ├─ Names, original sizes
│     ├─ Byte offsets within file
│     └─ Compression settings per entry
│
├─ Base LoRA (full, compressed)
│  └─ First LoRA stored complete after rank optimization
│
├─ Differential LoRA #2
│  └─ Sparse tensor: only non-zero weight deltas from base
│
├─ Differential LoRA #3
│  └─ Sparse tensor: only non-zero weight deltas from base
│
└─ ... (remaining LoRAs as diffs)
```

## Real-World Compression Results

| Collection Type | Individual Size | .loradb Size | Reduction |
|----------------|:--------------:|:------------:|:---------:|
| 50 character LoRAs | 7.2 GB | 380 MB | 94.7% |
| 100 style LoRAs | 14.4 GB | 890 MB | 93.8% |
| 20 lighting LoRAs | 2.88 GB | 145 MB | 95.0% |

Compression ratios depend on how similar the LoRAs are. Character variants from the same base training achieve the highest ratios. Completely unrelated LoRAs still benefit from individual rank optimization but see less delta compression.

## Extraction Quality

All LoRAs extracted from a `.loradb` file are verified against the originals:
- Maximum weight difference: < 0.001
- Quality rating: EXCELLENT for all tested collections
- Extraction speed: milliseconds per LoRA

## Use Cases

**For LoRA Creators:**
- Distribute entire collections as a single download
- 90-95% smaller file sizes for customers
- Embedded metadata (creator info, version, tags, descriptions)

**For Users:**
- One download instead of many
- Extract only the LoRAs you need
- Organized, space-efficient storage

**For Platforms:**
- Massive bandwidth and storage savings
- Better user experience
- Simplified content management

## Version Limits

| Version | Create .loradb | Extract from .loradb | Sell .loradb |
|---------|:--------------:|:-------------------:|:------------:|
| **Free** | Up to 5 LoRAs | ✅ Unlimited | ❌ |
| **Pro** | Up to 50 LoRAs | ✅ Unlimited | ❌ |
| **Studio** | ✅ Unlimited | ✅ Unlimited | ✅ |

All versions can extract from any `.loradb` file without limits. Creation limits apply only to building new databases. The Studio license enables commercial distribution of `.loradb` files on marketplaces like Gumroad, CivitAI, or your own website.
