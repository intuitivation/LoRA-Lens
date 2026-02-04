import torch
import numpy as np

def simulate_merge(engine1, engine2, ratio=0.5):
    """
    Predicts the weight distribution of a merged LoRA.
    FLUX-aware: handles higher dimensionality and stability checks.
    """
    prediction = []
    common_layers = set(engine1.layers.keys()) & set(engine2.layers.keys())
    
    # Detect if we're working with FLUX LoRAs
    is_flux = engine1.is_flux if hasattr(engine1, 'is_flux') else False
    
    for layer in common_layers:
        # Reconstruct deltas with proper A/B orientation
        w1 = torch.matmul(engine1.layers[layer]['up'], engine1.layers[layer]['down'])
        w2 = torch.matmul(engine2.layers[layer]['up'], engine2.layers[layer]['down'])
        
        # Weighted merge simulation
        merged_w = (w1 * ratio) + (w2 * (1 - ratio))
        
        # Calculate resulting magnitude and risk
        mag = torch.norm(merged_w).item()
        w1_norm = torch.norm(w1).item()
        w2_norm = torch.norm(w2).item()
        
        # Risk score: if magnitude is significantly higher than both parents, it's a conflict
        base_mag = max(w1_norm, w2_norm, 1e-6)
        risk = mag / base_mag
        
        # FLUX stability check: values over 2.0 in transformer blocks usually cause 'noise'
        if is_flux and risk > 2.0:
            risk = min(risk * 1.5, 5.0)  # Amplify risk for FLUX conflicts
        
        prediction.append({
            "layer": layer,
            "predicted_magnitude": mag,
            "instability_risk": risk
        })
        
    return prediction
