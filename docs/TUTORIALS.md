# 📚 Tutorials

> Step-by-step guides for every major LoRA Lens workflow.

---

## Tutorial 1: Your First Optimization

Compress a single LoRA using the free SVD rank optimization. Takes about 2 minutes.

### Steps

**1. Launch LoRA Lens**

```bash
cd LoRA-Lens
python run_lens.py
```

Your browser opens to `http://localhost:8501`. (Windows users: double-click `launch_lens.bat` instead.)

**2. Upload a LoRA**

Drag and drop any `.safetensors` LoRA file onto the upload area. LoRA Lens immediately starts analyzing the file.

**3. Check the Dashboard**

After 5–10 seconds, the Dashboard populates with:

- Detected format (SD 1.5, SDXL, or FLUX)
- Current rank and parameter count
- Health score and efficiency rating
- AI-generated optimization recommendation

Look at the **efficiency score**. If it's below 80%, there's significant room for compression.

**4. Optimize**

Click the **Optimize** tab, then click **"Prune to Optimal Rank"**.

LoRA Lens performs SVD on every layer, finds the optimal rank, and reconstructs the LoRA with only the dimensions that matter. This takes 10–30 seconds.

**5. Download**

Your compressed LoRA is ready. Download it and drop it into your ComfyUI/A1111/Forge workflow. No configuration changes needed — it's a standard `.safetensors` file.

### Expected Results

| Format | Typical Reduction |
|--------|-------------------|
| SD 1.5 | 80–90% |
| SDXL | 85–90% |
| FLUX | 30–40% |

---

## Tutorial 2: Ultra Compression *(Pro/Studio)*

Stack rank optimization + 8-bit quantization for maximum compression.

### Steps

**1. Upload and analyze** (same as Tutorial 1, steps 1–3)

**2. Go to the Surgery tab**

Click **Surgery** in the sidebar.

**3. Select Ultra Compress**

Choose **"ULTRA COMPRESS"** from the mode selector. This will run both SVD rank optimization and 8-bit quantization in sequence.

**4. Click Compress**

Watch the real-time quality metrics as compression runs:

- **SNR** should stay above 40 dB (no perceptible quality loss)
- **MSE** should stay below 0.005
- **MAE** should stay below 0.01

If any metric drops into the warning zone, LoRA Lens will flag it before you download.

**5. Review and download**

Compare the before/after:
- Original size vs. compressed size
- Quality metrics summary
- Estimated VRAM savings

Download when satisfied.

### Expected Results

| Format | Typical Total Reduction |
|--------|------------------------|
| SD 1.5 | 90–94% |
| SDXL | 90–94% |
| FLUX | 65–82% |

---

## Tutorial 3: Analyzing a LoRA

Understand what's inside a LoRA before doing anything to it.

### Steps

**1. Upload a LoRA** (same as Tutorial 1)

**2. Explore the Dashboard**

The Dashboard gives you the executive summary:

- **Health Score** (0–100): Overall quality and efficiency rating
- **Format**: What architecture the LoRA was trained for
- **Rank**: Current rank and whether it's inflated
- **AI Insights**: Natural language recommendations

**3. Dig into Analytics**

Click the **Analytics** tab for detailed breakdowns:

- **Layer Table**: Every layer listed with rank, parameters, magnitude stats. Sort by any column to find outliers
- **Weight Distribution**: Histograms showing how weights are spread. A tight bell curve is healthy; heavy tails or bimodal distributions suggest training issues
- **Correlation Heatmap**: How similar different layers are. High correlation = redundancy
- **Sparsity Map**: Which layers have the most near-zero weights

**4. Explore 3D Topology**

Click **3D Topology** for a UMAP projection of the LoRA's weight structure. Rotate and zoom the 3D visualization to see:

- **Clusters**: Layers that behave similarly group together
- **Outliers**: Isolated points may indicate unusual or problematic layers
- **Structure**: The overall "shape" of the LoRA's learned behavior

**5. Get AI Advice**

Click **AI Consultant** and ask questions:

- "Is this LoRA over-trained?"
- "Which layers are least efficient?"
- "What compression can I expect?"
- "Is this LoRA good quality?"

The AI has access to all the analysis data and gives answers specific to your file.

---

## Tutorial 4: Testing Two LoRAs for Compatibility

Use the Conflict Scanner to check whether two LoRAs will merge well.

### Steps

**1. Go to the Conflict Scanner tab**

**2. Upload two LoRA files**

Upload the two LoRAs you're considering merging. Both must be for the same base model (e.g., both SDXL).

**3. Review the conflict report**

LoRA Lens compares the two files layer by layer and shows:

- **Conflict Map**: Visual indicator of which layers overlap (potential interference) and which are complementary (safe to merge)
- **Conflict Score**: Overall compatibility rating (lower is better)
- **Hot Spots**: Specific layers where the two LoRAs compete for the same weight space

**4. Get merge recommendations**

Based on the analysis, LoRA Lens suggests:

- **Merge Ratio**: e.g., "0.7:0.3 recommended" — use the first LoRA at 70% strength and the second at 30%
- **Stability Prediction**: Whether the merge is likely to produce stable, artifact-free results
- **Problem Layers**: Specific layers you might want to exclude from the merge

**5. Apply the recommendations**

Use the suggested merge ratios in your preferred merging tool (ComfyUI LoRA stacking, merge scripts, etc.).

---

## Tutorial 5: Creating a .loradb Collection

Package multiple LoRAs into a single compressed file.

### Steps

**1. Prepare your LoRAs**

For best compression, gather LoRAs that are related — character variants, style series, or any set trained from the same base with similar settings.

You can add LoRAs that are already optimized (recommended for best results) or unoptimized (LoRA Lens will optimize during collection building).

**2. Go to the Export tab**

Click **Export** in the sidebar, then click **"Create .loradb Collection"**.

**3. Add LoRAs**

Drag and drop your LoRA files into the collection builder. You'll see each one listed with its original size.

**Collection limits by version:**

| Version | Max LoRAs |
|---------|-----------|
| Free | 5 |
| Pro | 50 |
| Studio | Unlimited |

**4. Set metadata**

Fill in the collection info:

- **Name**: e.g., "Fantasy Character Pack v2"
- **Creator**: Your name or handle
- **Description**: What's in the collection
- **Tags**: Keywords for organization (e.g., "fantasy", "character", "SDXL")

This metadata is embedded in the `.loradb` file and visible when anyone opens it.

**5. Build the database**

Click **"Build Database"**. LoRA Lens will:

1. Optimize each LoRA (SVD rank optimization)
2. Select the best base LoRA
3. Compute deltas for all others
4. Package everything into a single file

This takes 30–120 seconds depending on collection size.

**6. Download**

Your `.loradb` file is ready. Share it, distribute it, or archive it.

### Compression Expectations

| Collection Type | Typical Reduction |
|----------------|-------------------|
| Character variants (same base training) | 93–95% |
| Style series (related aesthetics) | 90–94% |
| Mixed collection (unrelated LoRAs) | 70–85% |

---

## Tutorial 6: Extracting LoRAs from a .loradb

Pull individual LoRAs out of a collection as standard safetensors files.

### Steps

**1. Go to the Export tab**

Click **Export** then **"Extract from .loradb"**.

**2. Upload the .loradb file**

Drag and drop the `.loradb` file. LoRA Lens reads the metadata and displays the collection contents.

**3. Browse the collection**

You'll see a list of every LoRA in the database:

- Name
- Original size
- Compressed size in the collection
- Format (SD 1.5 / SDXL / FLUX)

**4. Select and extract**

Check the LoRAs you want to extract, then click **"Extract Selected"**.

Each selected LoRA is reconstructed (base + delta) and saved as a standard `.safetensors` file. This takes less than 1 second per LoRA.

**5. Download**

Download the extracted `.safetensors` files and use them in your workflow like any normal LoRA.

### Notes

- Extraction is **unlimited in all versions** (including Free)
- Extracted LoRAs are the optimized versions (SVD rank-optimized)
- Quality is identical to what was stored — extraction adds zero additional quality loss
- You can extract the same LoRA multiple times without degradation

---

## Tutorial 7: Batch Processing

Optimize an entire folder of LoRAs at once.

### Steps

**1. Go to the Optimize tab**

**2. Enable batch mode**

Toggle **"Batch Processing"** on.

**3. Upload multiple LoRAs**

Drag and drop multiple `.safetensors` files. They'll be queued for processing.

**4. Start batch**

Click **"Optimize All"**. LoRA Lens processes each file in sequence:

- Analyze then find optimal rank then SVD optimization then save

A progress indicator shows which file is being processed and estimated time remaining.

**5. Download results**

When complete, download all optimized files as a batch. Each file retains its original name with the optimization applied.

### Batch Timing Estimates

| Batch Size | GPU Time | CPU Time |
|------------|----------|----------|
| 10 LoRAs | ~2 min | ~5 min |
| 25 LoRAs | ~5 min | ~12 min |
| 50 LoRAs | ~10 min | ~25 min |

---

## Workflow Tips

### Optimization Order

For maximum compression, follow this order:

```
1. Analyze (understand what you're working with)
2. SVD Rank Optimization (free, biggest single reduction)
3. 8-Bit Quantization (Pro, additional 50%)
4. Package into .loradb (optional, for collections)
```

### Quality Verification

After optimizing a LoRA, generate a few test images with the same prompt/seed using both the original and optimized versions. Compare side by side. With SNR above 40 dB, you should see no visible difference.

### When to Be Conservative

Some LoRAs are trained very efficiently and don't have much room for compression. If the Dashboard shows a health score above 90% and efficiency above 85%, the LoRA is already well-optimized. You'll still get some compression, but don't expect 90% reduction on an already efficient file.

### When to Be Aggressive

LoRAs with health scores below 60% or efficiency below 50% are prime candidates for aggressive optimization. These typically have massive rank inflation and can see 85–95% reduction with zero quality impact.

---

<p align="center">
  <a href="../README.md">Back to README</a>
</p>
