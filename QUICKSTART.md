# 🚀 Quick Start Guide - LoRA Lens v1.5

## 5-Minute Getting Started

### Step 1: Install (30 seconds)

```bash
pip install -r requirements.txt
```

### Step 2: Launch (10 seconds)

```bash
python run_lens.py
```

Your browser should open automatically to `http://localhost:8501`

### Step 3: Upload LoRA (15 seconds)

1. Click "Browse files" in the sidebar
2. Select your `.safetensors` LoRA file
3. Wait for analysis to complete (~5-30 seconds depending on size)

### Step 4: Explore! (4 minutes)

#### Dashboard View (1 min)
- See your health score
- Check quick insights
- Review layer breakdown

#### Check AI Consultant (1 min)
- Click "🤖 CONSULT" button
- Read recommendations
- Note training parameters

#### Try Optimization (1 min)
- Click "🔧 OPTIMIZE"
- See potential file size savings
- Click "EXECUTE PRUNING" if savings > 30%
- Download optimized LoRA

#### Compare LoRAs (1 min) - Optional
- Click "⚔️ CONFLICT"
- Upload second LoRA
- Check compatibility score
- Get merge suggestions

---

## Common First Tasks

### Task 1: "Is my LoRA any good?"

1. Upload LoRA
2. Check **Health Score** in Dashboard
   - 75+: Great!
   - 50-75: Good
   - <50: Needs work
3. Read AI Consultant insights
4. If score is low, check recommendations

### Task 2: "Make my LoRA smaller"

1. Upload LoRA
2. Click "🔧 OPTIMIZE"
3. Look at "Estimated file size reduction"
4. If > 20%, click "EXECUTE PRUNING"
5. Download optimized file
6. Test both versions to verify quality

### Task 3: "Can I merge these two LoRAs?"

1. Upload first LoRA (primary)
2. Click "⚔️ CONFLICT"
3. Upload second LoRA
4. Check compatibility score:
   - Green (>0.5): Safe to merge
   - Yellow (0-0.5): Caution
   - Red (<0): Don't merge
5. Use suggested merge ratio

### Task 4: "Why did my training fail?"

1. Upload the failed LoRA
2. Click "🤖 CONSULT"
3. Look for warnings (🔴 or ⚠️)
4. Common issues:
   - High rank usage → Learning rate too high
   - Low sparsity → Overfitting
   - High variance → Unstable training
5. Follow recommendations for next training

---

## Understanding Your First Analysis

### What You'll See

```
Health Score: 68/100
Avg Efficiency: 23.4 of 128 declared
Sparsity: 67.3%
Optimization: 45% potential savings
```

**Translation:**
- Health is "Good" (not excellent, but solid)
- Only using ~23 of 128 rank dimensions (efficient!)
- 67% of weights are near-zero (good sparsity)
- Could save 45% file size with pruning

### Quick Health Check Rules

✅ **Good Signs:**
- Health > 60
- Efficiency < 50% of declared
- Sparsity > 50%
- No red warnings

⚠️ **Warning Signs:**
- Health < 50
- Efficiency > 80% of declared
- Sparsity < 30%
- Multiple red flags

🔴 **Bad Signs:**
- Health < 30
- Efficiency > 90% of declared
- Sparsity < 20%
- "CRITICAL" warnings

---

## Keyboard Shortcuts

**Navigation:**
- `1` - Dashboard
- `2` - Analytics
- `3` - 3D Map
- `4` - Conflict
- `5` - Consultant
- `6` - Optimize
- `7` - Export

**Actions:**
- `Ctrl/Cmd + R` - Refresh analysis
- `Ctrl/Cmd + E` - Export view

---

## Tips for First-Time Users

### Do's ✅
- Start with Dashboard view
- Read AI Consultant recommendations
- Try pruning if savings > 30%
- Test optimized LoRA before deleting original
- Check conflicts before merging
- Export HTML report for records

### Don'ts ❌
- Don't trust health score alone - read insights
- Don't prune if savings < 20% (not worth it)
- Don't merge LoRAs with negative similarity
- Don't ignore red warnings
- Don't delete originals until you test
- Don't skip the conflict check when merging

---

## Next Steps

### After Your First Analysis

1. **Export a report** for your records
2. **Optimize** if potential savings > 30%
3. **Read the full README** for advanced features
4. **Try batch analysis** if you have multiple LoRAs
5. **Experiment** with different views

### Learn More

- **Full Guide:** See README.md
- **Metrics Guide:** Detailed explanation of all metrics
- **Advanced Usage:** Programmatic API usage
- **Troubleshooting:** Common issues and fixes

---

## FAQ

**Q: How long does analysis take?**
A: 5-30 seconds typically, depending on LoRA size.

**Q: Does it modify my original file?**
A: Never! Optimization creates a new file.

**Q: What size LoRAs can it handle?**
A: Up to 2GB comfortably. Larger may be slow.

**Q: Can I analyze multiple LoRAs at once?**
A: Yes! Use `batch_analyzer.py` for CLI batch processing.

**Q: Is the health score accurate?**
A: It's a good indicator, but read the AI insights for full picture.

**Q: What if I get errors?**
A: Check Troubleshooting in README.md or open an issue.

**Q: Can I use this commercially?**
A: Yes! MIT license - free to use anywhere.

---

## Emergency Troubleshooting

### LoRA won't load
- Check it's a valid `.safetensors` file
- Verify file isn't corrupted
- Try smaller LoRA first

### Interface is blank
- Refresh page
- Check terminal for errors
- Restart: `python run_lens.py`

### Analysis stuck
- Wait 60 seconds
- Refresh page
- Check system resources
- Try smaller LoRA

### Can't download optimized file
- Check browser download settings
- Try different browser
- Check disk space

---

**Still stuck?** Check the full README or open an issue on GitHub!

---

**🎉 You're ready to start analyzing LoRAs!**

Remember: The tool is here to help you understand and improve your work. Don't be afraid to experiment!
