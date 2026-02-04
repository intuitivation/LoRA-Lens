import torch
import numpy as np
from safetensors.torch import load_file, save_file
from scipy.spatial.distance import cosine

class LoRALensEngine:
    def __init__(self, path):
        self.path = path
        # FLUX often uses BFloat16; we load in that or Float16 to save RAM
        self.weights = load_file(path)
        self.original_dtype = self._detect_dtype()  # Store original dtype
        self.layers = self._map_weights()
        self._analysis_cache = None
        self.is_flux = self._detect_flux()
        self.naming_format = self._detect_naming_format()

    def _detect_dtype(self):
        """Detect the original data type of the weights."""
        for v in self.weights.values():
            if v.numel() > 0:
                return v.dtype
        return torch.float32  # Default fallback

    def _detect_flux(self):
        """Detect if this is a FLUX LoRA by checking naming patterns"""
        for k in self.weights.keys():
            # FLUX format 1: lora_A/lora_B
            if "lora_A" in k or "lora_B" in k:
                return True
            # FLUX format 2: double_blocks, single_blocks (FLUX architecture)
            if "double_blocks" in k or "single_blocks" in k:
                return True
        return False

    def _detect_naming_format(self):
        """Detect which naming format is used in this LoRA"""
        for k in self.weights.keys():
            if ".lora_up.weight" in k or ".lora_down.weight" in k:
                return "sd"  # Standard SD format
            if ".lora_A.weight" in k or ".lora_B.weight" in k:
                return "flux_ab"  # FLUX with lora_A/lora_B
            if ".up.weight" in k or ".down.weight" in k:
                return "flux_updown"  # FLUX with up/down
        return "unknown"

    def _map_weights(self):
        """Maps LoRA weight pairs and identifies layer types. Supports SD, SDXL, and FLUX formats."""
        mapping = {}
        for k, v in self.weights.items():
            # Skip alpha tensors for base name extraction
            if ".alpha" in k:
                # Extract base and store alpha
                base = k.replace(".alpha", "")
                if base not in mapping:
                    mapping[base] = {'type': self._identify_layer_type(base)}
                mapping[base]['alpha'] = v.float()
                continue

            # Normalize the keys to find pairs - handle all formats
            base = k
            # SD format: .lora_up.weight / .lora_down.weight
            base = base.replace(".lora_up.weight", "").replace(".lora_down.weight", "")
            # FLUX format 1: .lora_A.weight / .lora_B.weight
            base = base.replace(".lora_A.weight", "").replace(".lora_B.weight", "")
            # FLUX format 2: .up.weight / .down.weight (common in Kohya FLUX exports)
            base = base.replace(".up.weight", "").replace(".down.weight", "")

            if base not in mapping:
                mapping[base] = {'type': self._identify_layer_type(base)}

            # SD format
            if ".lora_up.weight" in k:
                mapping[base]['up'] = v.float()
                mapping[base]['is_conv'] = len(v.shape) == 4
            if ".lora_down.weight" in k:
                mapping[base]['down'] = v.float()
                mapping[base]['is_conv'] = len(v.shape) == 4

            # FLUX format 1 (A is 'down', B is 'up')
            if ".lora_A.weight" in k:
                mapping[base]['down'] = v.float()
                mapping[base]['is_conv'] = len(v.shape) == 4
            if ".lora_B.weight" in k:
                mapping[base]['up'] = v.float()
                mapping[base]['is_conv'] = len(v.shape) == 4

            # FLUX format 2 (Kohya format)
            if k.endswith(".up.weight"):
                mapping[base]['up'] = v.float()
                mapping[base]['is_conv'] = len(v.shape) == 4
            if k.endswith(".down.weight"):
                mapping[base]['down'] = v.float()
                mapping[base]['is_conv'] = len(v.shape) == 4

        return mapping

    def _compute_delta_w(self, up, down, is_conv=False):
        """
        Compute delta W from up and down matrices, handling both linear and conv layers.

        For linear: delta_W = up @ down (shapes: [out, rank] @ [rank, in] = [out, in])
        For conv: reshape to 2D, compute, then reshape back if needed
        """
        if is_conv and len(up.shape) == 4 and len(down.shape) == 4:
            # Convolutional layers: [out_ch, rank, kH, kW] and [rank, in_ch, kH, kW]
            # Reshape to 2D for matrix multiplication
            out_ch, rank_up, kH_up, kW_up = up.shape
            rank_down, in_ch, kH_down, kW_down = down.shape

            # For conv LoRA, we compute per spatial position or use a simplified approach
            # Reshape: up becomes [out_ch, rank * kH * kW], down becomes [rank * kH * kW, in_ch]
            # But typically for 1x1 convs (common in LoRA), kH=kW=1
            if kH_up == 1 and kW_up == 1 and kH_down == 1 and kW_down == 1:
                # Simple case: 1x1 convolutions
                up_2d = up.squeeze(-1).squeeze(-1)  # [out_ch, rank]
                down_2d = down.squeeze(-1).squeeze(-1)  # [rank, in_ch]
                return torch.matmul(up_2d, down_2d)
            else:
                # For larger kernels, flatten spatial dims
                # up: [out_ch, rank, kH, kW] -> [out_ch, rank * kH * kW]
                # down: [rank, in_ch, kH, kW] -> we need to handle this carefully
                # Simplified: just use the center position or average
                up_2d = up.reshape(out_ch, -1)  # [out_ch, rank*kH*kW]
                down_2d = down.reshape(rank_down, -1)  # [rank, in_ch*kH*kW]

                # Return flattened result for analysis purposes
                # This gives us [out_ch, in_ch*kH*kW] which we can analyze
                return torch.matmul(up_2d[:, :rank_down], down_2d)
        else:
            # Standard 2D linear layers
            return torch.matmul(up, down)
    
    def _identify_layer_type(self, layer_name):
        """Identify layer type from naming patterns."""
        name_lower = layer_name.lower()
        if 'attn' in name_lower or 'attention' in name_lower or 'qkv' in name_lower:
            return 'attention'
        elif 'mlp' in name_lower or 'ff' in name_lower or 'feedforward' in name_lower:
            return 'mlp'
        elif 'norm' in name_lower or 'ln' in name_lower:
            return 'normalization'
        elif 'proj' in name_lower:
            return 'projection'
        elif 'conv' in name_lower or 'resnet' in name_lower:
            return 'convolution'
        else:
            return 'other'

    def _get_declared_rank(self, up, down, is_conv=False):
        """Get the declared rank from the weight shapes."""
        if is_conv and len(down.shape) == 4:
            # For conv: down is [rank, in_ch, kH, kW]
            return down.shape[0]
        else:
            # For linear: down is [rank, in_features]
            return down.shape[0]

    def get_full_analysis(self, use_cache=True, fast_mode=True):
        """
        Comprehensive analysis with caching for performance.

        Args:
            use_cache: Use cached results if available
            fast_mode: Use fast approximations for large matrices (default True)
        """
        if use_cache and self._analysis_cache is not None:
            return self._analysis_cache

        report = []
        for name, data in self.layers.items():
            if 'up' in data and 'down' in data:
                up, down = data['up'], data['down']
                is_conv = data.get('is_conv', False)

                try:
                    # Get declared rank (this is the true rank of the LoRA)
                    declared_rank = self._get_declared_rank(up, down, is_conv)

                    # For large matrices in fast mode, use efficient rank estimation
                    # The effective rank of up @ down is at most min(rank_up, rank_down)
                    up_2d = up.reshape(up.shape[0], -1) if len(up.shape) > 2 else up
                    down_2d = down.reshape(down.shape[0], -1) if len(down.shape) > 2 else down

                    total_elements = up_2d.shape[0] * down_2d.shape[1]
                    use_fast = fast_mode and total_elements > 1_000_000  # 1M elements threshold

                    if use_fast:
                        # Fast mode: estimate stats from up/down directly
                        # Compute magnitude efficiently: ||up @ down||_F = ||S_up * S_down||
                        # For exact Frobenius norm: ||AB||_F <= ||A||_F * ||B||_F
                        magnitude = (torch.norm(up_2d) * torch.norm(down_2d)).item()

                        # Effective rank: use the rank of the smaller dimension
                        eff_rank = min(declared_rank, up_2d.shape[0], down_2d.shape[1])
                        optimal_rank = max(1, int(eff_rank * 0.8))  # Estimate 80% as optimal

                        # Estimate sparsity from up/down matrices
                        up_sparsity = (torch.abs(up_2d) < 1e-4).float().mean().item()
                        down_sparsity = (torch.abs(down_2d) < 1e-4).float().mean().item()
                        sparsity = (up_sparsity + down_sparsity) / 2

                        dead_weights = 0.0  # Can't easily estimate this in fast mode
                        entropy = 0.0
                        max_weight = max(torch.abs(up_2d).max().item(), torch.abs(down_2d).max().item())
                        mean_weight = (torch.abs(up_2d).mean() + torch.abs(down_2d).mean()).item() / 2
                        std_weight = (torch.abs(up_2d).std() + torch.abs(down_2d).std()).item() / 2

                    else:
                        # Full mode: compute delta_w and do SVD
                        delta_w = self._compute_delta_w(up, down, is_conv)

                        # SVD Analysis
                        U, S, V = torch.svd(delta_w)
                        total_var = torch.sum(S**2)
                        if total_var > 0:
                            variance_ratio = torch.cumsum(S**2, dim=0) / total_var
                            eff_rank = (variance_ratio < 0.95).sum().item() + 1
                            optimal_rank = (variance_ratio < 0.99).sum().item() + 1
                        else:
                            eff_rank = 1
                            optimal_rank = 1

                        # Weight statistics
                        abs_weights = torch.abs(delta_w)
                        magnitude = torch.norm(delta_w).item()
                        sparsity = (abs_weights < 1e-4).float().mean().item()
                        dead_weights = (abs_weights < 1e-5).float().mean().item()

                        # Entropy calculation for weight concentration
                        weights_flat = abs_weights.flatten()
                        weights_sum = weights_flat.sum()
                        if weights_sum > 0:
                            weights_flat = weights_flat / weights_sum
                            entropy = -(weights_flat * torch.log(weights_flat + 1e-10)).sum().item()
                        else:
                            entropy = 0.0

                        max_weight = abs_weights.max().item()
                        mean_weight = abs_weights.mean().item()
                        std_weight = abs_weights.std().item()

                    report.append({
                        "layer": name,
                        "layer_type": data['type'],
                        "magnitude": magnitude,
                        "eff_rank": eff_rank,
                        "optimal_rank": optimal_rank,
                        "declared_rank": declared_rank,
                        "sparsity": sparsity,
                        "dead_weights": dead_weights,
                        "entropy": entropy,
                        "max_weight": max_weight,
                        "mean_weight": mean_weight,
                        "std_weight": std_weight,
                        "is_conv": is_conv
                    })
                except Exception as e:
                    # Skip problematic layers but log them
                    print(f"Warning: Could not analyze layer {name}: {e}")
                    continue

        self._analysis_cache = report
        return report
    
    def get_weight_distribution(self, layer_name):
        """Get detailed weight distribution for a specific layer."""
        if layer_name not in self.layers:
            return None

        data = self.layers[layer_name]
        if 'up' not in data or 'down' not in data:
            return None

        try:
            is_conv = data.get('is_conv', False)
            delta_w = self._compute_delta_w(data['up'], data['down'], is_conv)
            weights_flat = delta_w.flatten().numpy()

            return {
                'weights': weights_flat,
                'histogram_bins': np.histogram(weights_flat, bins=50),
                'percentiles': np.percentile(weights_flat, [1, 5, 25, 50, 75, 95, 99])
            }
        except Exception as e:
            print(f"Warning: Could not get distribution for {layer_name}: {e}")
            return None

    def get_layer_correlations(self):
        """Calculate correlation matrix between layers."""
        layer_vectors = []
        layer_names = []
        target_size = None

        for name, data in self.layers.items():
            if 'up' in data and 'down' in data:
                try:
                    is_conv = data.get('is_conv', False)
                    delta_w = self._compute_delta_w(data['up'], data['down'], is_conv)
                    flattened = delta_w.flatten().numpy()

                    # For correlation, all vectors must be same size
                    # Keep track of the most common size
                    if target_size is None:
                        target_size = len(flattened)
                        layer_vectors.append(flattened)
                        layer_names.append(name)
                    elif len(flattened) == target_size:
                        layer_vectors.append(flattened)
                        layer_names.append(name)
                except Exception:
                    continue

        if len(layer_vectors) < 2:
            return None

        # Calculate correlation matrix
        vectors = np.array(layer_vectors)
        correlations = np.corrcoef(vectors)

        return {
            'correlation_matrix': correlations,
            'layer_names': layer_names
        }

    def prune_to_optimal(self, output_path, target_rank=None, variance_threshold=0.90, use_quantization=False):
        """
        Surgically shrinks the LoRA based on effective rank using SVD analysis.
        Supports SD/SDXL and FLUX formats with proper precision handling.

        Args:
            output_path: Path to save the pruned LoRA
            target_rank: Fixed rank to use for all layers (None = auto per layer)
            variance_threshold: Variance to retain (0.90 = 90%, good balance of compression vs quality)
            use_quantization: Enable 8-bit quantization for additional ~50% reduction (Pro/Studio feature)
        """
        new_weights = {}
        pruning_stats = []

        for name, data in self.layers.items():
            if 'up' not in data or 'down' not in data:
                continue

            up, down = data['up'], data['down']
            is_conv = data.get('is_conv', False)
            original_rank = self._get_declared_rank(up, down, is_conv)

            # For convolutional layers with large kernels, just copy them as-is
            if is_conv and len(down.shape) == 4:
                kH, kW = down.shape[2], down.shape[3]
                if kH > 1 or kW > 1:
                    self._save_layer_weights(new_weights, name, up, down, data.get('alpha'), original_rank, use_quantization)
                    pruning_stats.append({
                        'layer': name,
                        'original_rank': original_rank,
                        'new_rank': original_rank,
                        'compression_ratio': 1.0,
                        'skipped': 'conv_layer'
                    })
                    continue

            # Skip if target_rank >= original_rank (no reduction needed)
            if target_rank is not None and target_rank >= original_rank:
                self._save_layer_weights(new_weights, name, up, down, data.get('alpha'), original_rank, use_quantization)
                pruning_stats.append({
                    'layer': name,
                    'original_rank': original_rank,
                    'new_rank': original_rank,
                    'compression_ratio': 1.0,
                    'skipped': 'target_rank_too_high'
                })
                continue

            try:
                # Compute delta_w for SVD analysis
                delta_w = self._compute_delta_w(up, down, is_conv)

                # SVD to find optimal rank
                U, S, V = torch.svd(delta_w)

                if target_rank is None:
                    # Calculate rank needed to retain variance_threshold
                    total_var = torch.sum(S**2)
                    if total_var > 0:
                        variance_ratio = torch.cumsum(S**2, dim=0) / total_var
                        optimal_rank = (variance_ratio < variance_threshold).sum().item() + 1
                    else:
                        optimal_rank = 1
                    # CRITICAL: Never exceed original rank (prevents size increase!)
                    optimal_rank = max(1, min(optimal_rank, len(S), original_rank))
                else:
                    optimal_rank = min(target_rank, len(S), original_rank)

                # If optimal rank equals original, just copy the weights (no gain from SVD reconstruction)
                if optimal_rank >= original_rank:
                    self._save_layer_weights(new_weights, name, up, down, data.get('alpha'), original_rank, use_quantization)
                    pruning_stats.append({
                        'layer': name,
                        'original_rank': original_rank,
                        'new_rank': original_rank,
                        'compression_ratio': 1.0,
                        'skipped': 'already_optimal'
                    })
                    continue

                # Reconstruct at optimal rank
                U_r = U[:, :optimal_rank]
                S_r = torch.diag(torch.sqrt(S[:optimal_rank]))
                V_r = V[:, :optimal_rank].t()

                new_up = U_r @ S_r
                new_down = S_r @ V_r

                # Handle precision (before quantization check)
                if self.is_flux:
                    new_up = new_up.to(torch.bfloat16)
                    new_down = new_down.to(torch.bfloat16)

                # Save with appropriate naming convention
                self._save_layer_weights(new_weights, name, new_up, new_down, data.get('alpha'), optimal_rank, use_quantization)

                pruning_stats.append({
                    'layer': name,
                    'original_rank': original_rank,
                    'new_rank': optimal_rank,
                    'compression_ratio': optimal_rank / original_rank
                })

            except Exception as e:
                # If pruning fails, keep original weights
                print(f"Warning: Could not prune layer {name}: {e}")
                self._save_layer_weights(new_weights, name, up, down, data.get('alpha'), original_rank, use_quantization)
                pruning_stats.append({
                    'layer': name,
                    'original_rank': original_rank,
                    'new_rank': original_rank,
                    'compression_ratio': 1.0,
                    'error': str(e)
                })

        save_file(new_weights, output_path)
        return pruning_stats

    def _save_layer_weights(self, weights_dict, name, up, down, alpha, rank, use_quantization=False):
        """Save layer weights with the appropriate naming convention and proper dtype.

        Args:
            use_quantization: If True, apply 8-bit symmetric quantization for ~50% additional compression
        """
        # Clone tensors to avoid Windows memory-mapping issues with safetensors
        up = up.clone().contiguous().float()
        down = down.clone().contiguous().float()

        if use_quantization:
            # 8-bit symmetric quantization (Pro/Studio feature)
            # Store scale factors and quantized int8 weights
            up_scale = up.abs().max()
            down_scale = down.abs().max()

            if up_scale > 0:
                up_quantized = torch.round((up / up_scale) * 127).clamp(-127, 127).to(torch.int8)
            else:
                up_quantized = torch.zeros_like(up, dtype=torch.int8)

            if down_scale > 0:
                down_quantized = torch.round((down / down_scale) * 127).clamp(-127, 127).to(torch.int8)
            else:
                down_quantized = torch.zeros_like(down, dtype=torch.int8)

            # Store quantized weights with scale factors
            if self.naming_format == "flux_updown":
                weights_dict[f"{name}.up.weight"] = up_quantized
                weights_dict[f"{name}.up.scale"] = up_scale.to(torch.float32)
                weights_dict[f"{name}.down.weight"] = down_quantized
                weights_dict[f"{name}.down.scale"] = down_scale.to(torch.float32)
            elif self.naming_format == "flux_ab" or self.is_flux:
                weights_dict[f"{name}.lora_B.weight"] = up_quantized
                weights_dict[f"{name}.lora_B.scale"] = up_scale.to(torch.float32)
                weights_dict[f"{name}.lora_A.weight"] = down_quantized
                weights_dict[f"{name}.lora_A.scale"] = down_scale.to(torch.float32)
                if alpha is not None:
                    alpha_val = alpha.clone().to(torch.float32) if isinstance(alpha, torch.Tensor) else torch.tensor(float(rank), dtype=torch.float32)
                else:
                    alpha_val = torch.tensor(float(rank), dtype=torch.float32)
                weights_dict[f"{name}.alpha"] = alpha_val
            else:
                weights_dict[f"{name}.lora_up.weight"] = up_quantized
                weights_dict[f"{name}.lora_up.scale"] = up_scale.to(torch.float32)
                weights_dict[f"{name}.lora_down.weight"] = down_quantized
                weights_dict[f"{name}.lora_down.scale"] = down_scale.to(torch.float32)
                if alpha is not None:
                    alpha_val = alpha.clone() if isinstance(alpha, torch.Tensor) else torch.tensor(float(alpha))
                    weights_dict[f"{name}.alpha"] = alpha_val.to(torch.float32)
        else:
            # Standard float16/bfloat16 storage
            if self.is_flux:
                up = up.to(torch.bfloat16)
                down = down.to(torch.bfloat16)
            else:
                up = up.to(self.original_dtype)
                down = down.to(self.original_dtype)

            if self.naming_format == "flux_updown":
                weights_dict[f"{name}.up.weight"] = up
                weights_dict[f"{name}.down.weight"] = down
            elif self.naming_format == "flux_ab" or self.is_flux:
                weights_dict[f"{name}.lora_B.weight"] = up
                weights_dict[f"{name}.lora_A.weight"] = down
                if alpha is not None:
                    alpha_val = alpha.clone().to(torch.bfloat16) if isinstance(alpha, torch.Tensor) else torch.tensor(float(rank)).to(torch.bfloat16)
                else:
                    alpha_val = torch.tensor(float(rank)).to(torch.bfloat16)
                weights_dict[f"{name}.alpha"] = alpha_val
            else:
                weights_dict[f"{name}.lora_up.weight"] = up
                weights_dict[f"{name}.lora_down.weight"] = down
                if alpha is not None:
                    alpha_val = alpha.clone() if isinstance(alpha, torch.Tensor) else torch.tensor(float(alpha))
                    weights_dict[f"{name}.alpha"] = alpha_val.to(self.original_dtype)

    def detect_conflicts(self, other_engine):
        """Calculates interference between this LoRA and another."""
        conflicts = []
        # Find layers present in both models
        common_layers = set(self.layers.keys()) & set(other_engine.layers.keys())

        for layer in common_layers:
            if 'up' not in self.layers[layer] or 'down' not in self.layers[layer]:
                continue
            if 'up' not in other_engine.layers[layer] or 'down' not in other_engine.layers[layer]:
                continue

            try:
                # Reconstruct weight deltas for both
                is_conv1 = self.layers[layer].get('is_conv', False)
                is_conv2 = other_engine.layers[layer].get('is_conv', False)

                w1 = self._compute_delta_w(
                    self.layers[layer]['up'],
                    self.layers[layer]['down'],
                    is_conv1
                ).flatten()

                w2 = other_engine._compute_delta_w(
                    other_engine.layers[layer]['up'],
                    other_engine.layers[layer]['down'],
                    is_conv2
                ).flatten()

                # Ensure same size for comparison
                if w1.shape != w2.shape:
                    continue

                # Cosine Similarity: 1.0 (Identical), 0 (Orthogonal), -1.0 (Opposite/Conflict)
                similarity = torch.nn.functional.cosine_similarity(w1.unsqueeze(0), w2.unsqueeze(0)).item()

                # Calculate magnitude comparison
                mag1 = torch.norm(w1).item()
                mag2 = torch.norm(w2).item()
                mag_ratio = min(mag1, mag2) / (max(mag1, mag2) + 1e-10)

                # We flag anything below 0.1 as a 'Conflict Zone'
                if similarity < -0.1:
                    status = "SEVERE CONFLICT"
                elif similarity < 0.1:
                    status = "CONFLICT"
                elif similarity < 0.5:
                    status = "Neutral"
                else:
                    status = "Compatible"

                conflicts.append({
                    "layer": layer,
                    "layer_type": self.layers[layer]['type'],
                    "similarity": round(similarity, 4),
                    "mag_ratio": round(mag_ratio, 4),
                    "status": status
                })
            except Exception:
                # Skip layers that can't be compared
                continue

        return conflicts
    
    def suggest_merge_ratio(self, other_engine):
        """Suggests optimal merge ratios to minimize conflicts."""
        conflicts = self.detect_conflicts(other_engine)
        
        # Calculate conflict severity score
        conflict_scores = []
        for c in conflicts:
            if c['similarity'] < 0:
                conflict_scores.append(abs(c['similarity']))
        
        if not conflict_scores:
            return {'ratio': 0.5, 'confidence': 'high', 'conflicts': 0}
        
        avg_conflict = sum(conflict_scores) / len(conflict_scores)
        
        # More conflicts = recommend lower ratio for secondary LoRA
        if avg_conflict > 0.5:
            recommended_ratio = 0.7  # Favor primary LoRA
            confidence = 'low'
        elif avg_conflict > 0.3:
            recommended_ratio = 0.6
            confidence = 'medium'
        else:
            recommended_ratio = 0.5
            confidence = 'high'
        
        return {
            'ratio': recommended_ratio,
            'confidence': confidence,
            'conflicts': len(conflict_scores),
            'avg_conflict_severity': avg_conflict
        }
    
    def get_efficiency_score(self):
        """Calculate overall efficiency score (0-100)."""
        analysis = self.get_full_analysis()
        
        if not analysis:
            return 0
        
        # Factors:
        # 1. Rank efficiency (how much of declared rank is used)
        # 2. Sparsity (higher is better)
        # 3. Dead weights (lower is better)
        
        rank_efficiency = []
        sparsity_scores = []
        dead_weight_scores = []
        
        for layer in analysis:
            rank_eff = layer['eff_rank'] / layer['declared_rank']
            rank_efficiency.append(min(1.0, rank_eff))  # Cap at 1.0
            sparsity_scores.append(layer['sparsity'])
            dead_weight_scores.append(1.0 - layer['dead_weights'])
        
        avg_rank_eff = sum(rank_efficiency) / len(rank_efficiency)
        avg_sparsity = sum(sparsity_scores) / len(sparsity_scores)
        avg_dead = sum(dead_weight_scores) / len(dead_weight_scores)
        
        # Weighted score
        score = (
            (1.0 - avg_rank_eff) * 40 +  # Prefer lower effective rank
            avg_sparsity * 30 +            # Prefer higher sparsity
            avg_dead * 30                   # Prefer fewer dead weights
        )
        
        return round(score, 2)
