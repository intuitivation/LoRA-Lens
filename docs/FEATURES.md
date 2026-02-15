# Features

LoRA Lens provides a full Streamlit-based web UI. Launch with `python run_lens.py` and access at `http://localhost:8501`.

## 1. Dashboard

Your starting point after uploading a LoRA. Shows:

- **Format detection** — Automatically identifies SD 1.5, SDXL, or FLUX
- **Health score** — Overall efficiency rating for the LoRA
- **Key metrics** — File size, layer count, rank distribution, precision
- **AI-powered insights** — Recommendations for optimization based on the specific LoRA's structure

## 2. Analytics

Deep analysis of your LoRA's internals:

- **Layer-by-layer table** — Rank, parameter count, and efficiency per layer
- **Weight distribution histograms** — Visualize the spread of weight values
- **Correlation heatmaps** — See relationships between layers
- **Sparsity analysis** — Identify layers with high zero-weight percentages
- **Magnitude visualizations** — Understand which layers carry the most information

## 3. 3D Neural Topology

Interactive visualization of your LoRA's weight patterns:

- **UMAP projection** — Reduces high-dimensional weight data to 3D for visual exploration
- **Cluster analysis** — Identifies groups of similar layers
- **Redundancy detection** — Highlights dimensions that can be safely pruned
- **Interactive controls** — Rotate, zoom, and explore the visualization with Plotly

## 4. Conflict Scanner

Test whether two LoRAs will work well together *before* merging:

- **Upload two LoRAs** — Compare their weight patterns
- **Layer conflict detection** — Identifies which layers overlap significantly
- **Merge ratio recommendations** — Suggests optimal blending weights
- **Stability predictions** — Estimates whether the merged result will be coherent

## 5. AI Consultant

Natural language interface for getting advice about your LoRA:

- Ask questions like "What's the best compression for this LoRA?"
- Get format-specific optimization recommendations
- Troubleshoot issues with plain English
- Understand analysis results without ML expertise

## 6. Optimize Tab

One-click compression:

- **Prune to Optimal Rank** — SVD-based rank optimization
- **Batch processing** — Optimize an entire folder of LoRAs at once
- **Format-aware** — Adapts strategy based on detected format
- **Quality guaranteed** — 99%+ variance retention

Available in all versions (Free, Pro, Studio).

## 7. Surgery Tab *(Pro/Studio)*

Advanced compression options:

- **8-bit Quantization** — Convert weights to Int8 for additional 50% reduction
- **4-bit Experimental Mode** — Aggressive compression for testing
- **Ultra Compress** — Combined rank optimization + quantization in one step
- **Real-time quality metrics** — SNR, MSE, and MAE displayed during compression so you can verify quality before saving

## 8. Export Tab

Download and package your results:

- **Download optimized LoRAs** — Standard `.safetensors` format, compatible with all tools
- **Create .loradb collections** — Package multiple LoRAs into a single compressed file
- **Batch export** — Export multiple optimized LoRAs at once
- **Metadata management** — Set creator info, tags, and descriptions for `.loradb` collections

## 9. Settings

Customize your experience:

- **Pre-compute visualizations** — Toggle to speed up or improve analysis
- **Performance options** — Adjust for your hardware
- **Display preferences** — Customize the UI layout
