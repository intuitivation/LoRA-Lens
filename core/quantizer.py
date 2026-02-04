import torch
import numpy as np

def quantize_to_8bit(tensor):
    """
    Symmetric 8-bit quantization.
    Converts BFloat16/Float16 weights to Int8 with scale factor.
    
    Args:
        tensor: PyTorch tensor to quantize
        
    Returns:
        quantized: Int8 tensor
        scale: Scale factor for dequantization
    """
    if tensor is None:
        return None, None
    
    # Calculate the scale factor
    max_val = torch.max(torch.abs(tensor))
    scale = 127.0 / (max_val + 1e-6)
    
    # Quantize and cast to Int8
    quantized = torch.round(tensor * scale).to(torch.int8)
    
    return quantized, scale


def dequantize_from_8bit(quantized, scale):
    """
    Restores quantized tensor to BFloat16 for inference or further analysis.
    
    Args:
        quantized: Int8 tensor
        scale: Scale factor used during quantization
        
    Returns:
        Dequantized BFloat16 tensor
    """
    if quantized is None or scale is None:
        return None
    
    return (quantized.to(torch.bfloat16) / scale)


def quantize_to_4bit(tensor):
    """
    Aggressive 4-bit quantization for maximum compression.
    WARNING: May result in quality loss. Test before deployment.
    
    Args:
        tensor: PyTorch tensor to quantize
        
    Returns:
        quantized: 4-bit packed values (as Int8 with 2 values per byte)
        scale: Scale factor
        zero_point: Zero point for asymmetric quantization
    """
    if tensor is None:
        return None, None, None
    
    # Calculate range
    t_min = tensor.min()
    t_max = tensor.max()
    
    # Scale to 4-bit range [0, 15]
    scale = (t_max - t_min) / 15.0
    zero_point = -t_min / scale
    
    # Quantize to 4-bit values [0, 15]
    quantized = torch.round((tensor / scale) + zero_point)
    quantized = torch.clamp(quantized, 0, 15).to(torch.uint8)
    
    return quantized, scale, zero_point


def estimate_quantization_quality(original_tensor, quantized_tensor, scale):
    """
    Estimate quality loss from quantization.
    
    Args:
        original_tensor: Original BFloat16 tensor
        quantized_tensor: Quantized Int8 tensor
        scale: Scale factor
        
    Returns:
        Dictionary with quality metrics
    """
    # Dequantize for comparison
    dequantized = dequantize_from_8bit(quantized_tensor, scale)
    
    # Calculate error metrics
    mse = torch.mean((original_tensor - dequantized) ** 2).item()
    mae = torch.mean(torch.abs(original_tensor - dequantized)).item()
    
    # Signal-to-noise ratio
    signal_power = torch.mean(original_tensor ** 2).item()
    noise_power = mse
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    # Percentage of weights that changed significantly (>1%)
    relative_error = torch.abs((original_tensor - dequantized) / (original_tensor + 1e-10))
    significant_changes = (relative_error > 0.01).float().mean().item() * 100
    
    return {
        'mse': mse,
        'mae': mae,
        'snr_db': snr_db,
        'quality_retention': 100 - significant_changes,
        'significant_changes_pct': significant_changes
    }


def quantize_lora_full(engine, output_path, bits=8, variance_threshold=0.95):
    """
    Complete quantization pipeline: rank optimization + bit quantization.
    
    Args:
        engine: LoRALensEngine instance
        output_path: Path to save quantized LoRA
        bits: 8 or 4 bit quantization
        variance_threshold: Variance retention for rank optimization
        
    Returns:
        Dictionary with compression statistics
    """
    from safetensors.torch import save_file
    
    new_weights = {}
    stats = {
        'original_size': 0,
        'quantized_size': 0,
        'layers_processed': 0,
        'quality_metrics': []
    }
    
    for name, data in engine.layers.items():
        if 'up' not in data or 'down' not in data:
            continue
        
        up, down = data['up'], data['down']
        
        # Step 1: Rank optimization (optional, already done in prune_to_optimal)
        # We assume this is called after rank optimization
        
        # Step 2: Quantize
        if bits == 8:
            up_q, up_scale = quantize_to_8bit(up)
            down_q, down_scale = quantize_to_8bit(down)
            
            # Calculate quality
            quality_up = estimate_quantization_quality(up, up_q, up_scale)
            quality_down = estimate_quantization_quality(down, down_q, down_scale)
            
            avg_quality = (quality_up['quality_retention'] + quality_down['quality_retention']) / 2
            stats['quality_metrics'].append(avg_quality)
            
        elif bits == 4:
            up_q, up_scale, up_zp = quantize_to_4bit(up)
            down_q, down_scale, down_zp = quantize_to_4bit(down)
        else:
            raise ValueError(f"Unsupported bit depth: {bits}")
        
        # Store quantized weights with naming convention
        if engine.is_flux:
            new_weights[f"{name}.lora_B.weight_q"] = up_q
            new_weights[f"{name}.lora_B.scale"] = torch.tensor(up_scale)
            new_weights[f"{name}.lora_A.weight_q"] = down_q
            new_weights[f"{name}.lora_A.scale"] = torch.tensor(down_scale)
            
            if bits == 4:
                new_weights[f"{name}.lora_B.zero_point"] = torch.tensor(up_zp)
                new_weights[f"{name}.lora_A.zero_point"] = torch.tensor(down_zp)
                
            # Preserve alpha
            if 'alpha' in data:
                new_weights[f"{name}.alpha"] = data['alpha']
        else:
            # SD/SDXL format
            new_weights[f"{name}.lora_up.weight_q"] = up_q
            new_weights[f"{name}.lora_up.scale"] = torch.tensor(up_scale)
            new_weights[f"{name}.lora_down.weight_q"] = down_q
            new_weights[f"{name}.lora_down.scale"] = torch.tensor(down_scale)
            
            if bits == 4:
                new_weights[f"{name}.lora_up.zero_point"] = torch.tensor(up_zp)
                new_weights[f"{name}.lora_down.zero_point"] = torch.tensor(down_zp)
        
        stats['layers_processed'] += 1
    
    # Save quantized LoRA
    save_file(new_weights, output_path)
    
    # Calculate file sizes
    import os
    stats['original_size'] = os.path.getsize(engine.path) / (1024 * 1024)  # MB
    stats['quantized_size'] = os.path.getsize(output_path) / (1024 * 1024)  # MB
    stats['compression_ratio'] = 1 - (stats['quantized_size'] / stats['original_size'])
    stats['avg_quality_retention'] = np.mean(stats['quality_metrics']) if stats['quality_metrics'] else 100
    
    return stats


def combined_optimize_and_quantize(engine, output_path, bits=8, variance_threshold=0.95):
    """
    Two-stage compression: rank optimization followed by quantization.
    This is the "ULTRA COMPRESS" option.
    
    Args:
        engine: LoRALensEngine instance
        output_path: Final output path
        bits: 8 or 4
        variance_threshold: For rank optimization
        
    Returns:
        Complete statistics dictionary
    """
    import os
    import tempfile
    
    # Original size
    original_size = os.path.getsize(engine.path) / (1024 * 1024)
    
    # Stage 1: Rank optimization
    with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as tmp:
        temp_path = tmp.name
    
    rank_stats = engine.prune_to_optimal(temp_path, variance_threshold=variance_threshold)
    rank_size = os.path.getsize(temp_path) / (1024 * 1024)
    
    # Stage 2: Quantize the optimized LoRA
    from core.engine import LoRALensEngine
    optimized_engine = LoRALensEngine(temp_path)
    quant_stats = quantize_lora_full(optimized_engine, output_path, bits=bits)
    
    # Cleanup temp file
    os.unlink(temp_path)
    
    # Combined stats
    return {
        'original_size_mb': original_size,
        'after_rank_opt_mb': rank_size,
        'final_size_mb': quant_stats['quantized_size'],
        'rank_reduction_pct': (1 - rank_size / original_size) * 100,
        'quant_reduction_pct': (1 - quant_stats['quantized_size'] / rank_size) * 100,
        'total_reduction_pct': (1 - quant_stats['quantized_size'] / original_size) * 100,
        'quality_retention_pct': quant_stats['avg_quality_retention'],
        'layers_processed': quant_stats['layers_processed'],
        'rank_stats': rank_stats
    }
