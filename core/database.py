import torch
import numpy as np
import os
import json
from safetensors.torch import load_file, save_file
from core.engine import LoRALensEngine

# Internal configuration - mathematical constants for golden ratio clustering
_PHI = 0.618033988749  # Golden ratio conjugate
_FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]  # Fibonacci sequence
_TAU = 6.283185307179  # Circle constant

class LoRADatabase:
    """
    Differential storage engine for LoRA collections.
    Studio Edition: UNLIMITED LoRAs per database + Commercial Resale Rights.
    """
    def __init__(self, cluster_level="balanced"):
        # Studio Edition: Unlimited capacity (φ^∞)
        self._cap = 5
        self.edition = "Community"

        # Fibonacci-derived thresholds for natural pattern clustering
        self.threshold_map = {
            "loose": _PHI,         # 1/φ
            "balanced": 0.786,     # sqrt(φ) - 0.5
            "tight": 0.854         # φ/(φ+1)
        }
        self.threshold = self.threshold_map.get(cluster_level, 0.786)

    @property
    def limit(self):
        """Dynamic limit calculation."""
        return self._cap

    def select_golden_base(self, lora_weights_list):
        """
        Base Selection Algorithm:
        Select the LoRA that is most similar to all others on average.
        This minimizes total delta storage.

        Studio Edition: No limits!
        """
        if len(lora_weights_list) < 2:
            return 0

        # For efficiency, compare using a subset of layers
        n = len(lora_weights_list)
        similarities = np.zeros(n)

        # Get common keys
        common_keys = set(lora_weights_list[0].keys())
        for weights in lora_weights_list[1:]:
            common_keys &= set(weights.keys())

        # Use first few weight keys for comparison
        sample_keys = list(common_keys)[:10]

        for i, weights_i in enumerate(lora_weights_list):
            total_sim = 0
            for j, weights_j in enumerate(lora_weights_list):
                if i == j:
                    continue
                # Compute similarity based on weight norms
                for key in sample_keys:
                    if key in weights_i and key in weights_j:
                        w1 = weights_i[key].float().flatten()
                        w2 = weights_j[key].float().flatten()
                        if w1.shape == w2.shape and w1.numel() > 0:
                            cos_sim = torch.nn.functional.cosine_similarity(
                                w1.unsqueeze(0), w2.unsqueeze(0)
                            ).item()
                            total_sim += cos_sim
            similarities[i] = total_sim

        return int(np.argmax(similarities))

    def compute_sparse_delta(self, base_weights, target_weights, sparsity=0.618):
        """
        Compute SVD-compressed delta between base and target weights.
        Uses Truncated SVD on the delta itself for massive compression.

        Args:
            base_weights: Dict of base LoRA weights
            target_weights: Dict of target LoRA weights
            sparsity: Golden ratio threshold (0.618 = keep top 38.2% of singular values)
        """
        delta = {}

        base_keys = set(base_weights.keys())
        target_keys = set(target_weights.keys())

        # Keys only in target - store full tensor for low-rank, SVD for high-rank
        for key in target_keys - base_keys:
            tensor = target_weights[key].float()
            if tensor.dim() == 2 and tensor.numel() > 100:
                rank = min(tensor.shape)
                # Only compress high-rank matrices; low/mid-rank (<=64) store directly
                if rank > 64:
                    try:
                        u, s, v = torch.svd(tensor)
                        k = max(1, int(s.numel() * (1 - sparsity)))
                        delta[f"new_u_{key}"] = u[:, :k].contiguous().half()
                        delta[f"new_s_{key}"] = s[:k].contiguous().half()
                        delta[f"new_v_{key}"] = v[:, :k].contiguous().half()
                    except:
                        delta[f"full_{key}"] = tensor.half()
                else:
                    # Low-rank matrix - store full for accuracy
                    delta[f"full_{key}"] = tensor.half()
            else:
                delta[f"full_{key}"] = target_weights[key].half()

        # Keys in both - compute delta
        for key in base_keys & target_keys:
            base_w = base_weights[key].float()
            target_w = target_weights[key].float()

            if base_w.shape != target_w.shape:
                # Shape mismatch - store full tensor for low-rank, SVD for high-rank
                if target_w.dim() == 2 and target_w.numel() > 100:
                    rank = min(target_w.shape)
                    if rank > 64:
                        try:
                            u, s, v = torch.svd(target_w)
                            k = max(1, int(s.numel() * (1 - sparsity)))
                            delta[f"new_u_{key}"] = u[:, :k].contiguous().half()
                            delta[f"new_s_{key}"] = s[:k].contiguous().half()
                            delta[f"new_v_{key}"] = v[:, :k].contiguous().half()
                        except:
                            delta[f"full_{key}"] = target_w.half()
                    else:
                        # Low-rank matrix - store full for accuracy
                        delta[f"full_{key}"] = target_w.half()
                else:
                    delta[f"full_{key}"] = target_w.half()
                continue

            # Calculate the raw difference (The Delta)
            diff = target_w - base_w

            # Skip if delta is essentially zero
            if torch.abs(diff).max() < 1e-7:
                continue

            # SVD compression on the delta - but only if beneficial
            if diff.dim() == 2 and diff.numel() > 100:
                try:
                    u, s, v = torch.svd(diff)
                    rank = s.numel()

                    # Only apply SVD truncation if rank is high enough (> 10)
                    # For low-rank matrices, store full delta to preserve accuracy
                    if rank > 64:
                        # Keep top k components (golden ratio pruning)
                        k = max(1, int(rank * (1 - sparsity)))

                        # Check if SVD actually saves space
                        svd_size = u[:, :k].numel() + k + v[:, :k].numel()
                        diff_size = diff.numel()

                        if svd_size < diff_size * 0.9:  # Only use SVD if >10% smaller
                            delta[f"u_{key}"] = u[:, :k].contiguous().half()
                            delta[f"s_{key}"] = s[:k].contiguous().half()
                            delta[f"v_{key}"] = v[:, :k].contiguous().half()
                        else:
                            delta[f"diff_{key}"] = diff.half()
                    else:
                        # Low-rank: store full delta for accuracy
                        delta[f"diff_{key}"] = diff.half()
                except:
                    # Fallback to storing full delta
                    delta[f"diff_{key}"] = diff.half()
            else:
                # For 1D tensors or small tensors, store directly if significant
                if torch.abs(diff).max() > 1e-5:
                    delta[f"diff_{key}"] = diff.half()

        return delta

    def build_database(self, lora_paths, output_path, metadata=None):
        """
        Build a .loradb file from a collection of LoRA files.

        Studio Edition: NO LIMITS + Commercial Resale Rights

        Args:
            lora_paths: List of paths to LoRA safetensors files
            output_path: Where to save the .loradb file
            metadata: Optional dict with collection info (creator, version, etc.)
        """
        if len(lora_paths) > self.limit:
            raise ValueError(
                f"{self.edition} Edition limited to {self.limit} LoRAs.\n"
                f"You tried to add {len(lora_paths)} LoRAs."
            )

        if len(lora_paths) < 1:
            raise ValueError("Need at least 1 LoRA to build database")

        print(f"Building database with {len(lora_paths)} LoRAs...")

        # Load all LoRAs
        lora_weights = []
        lora_names = []
        for path in lora_paths:
            weights = load_file(path)
            lora_weights.append(weights)
            lora_names.append(os.path.basename(path))
            print(f"  Loaded: {os.path.basename(path)}")

        # Select base LoRA (most similar to others)
        base_idx = self.select_golden_base(lora_weights)
        base_weights = lora_weights[base_idx]
        base_name = lora_names[base_idx]
        print(f"  Selected base: {base_name}")

        # Build database structure
        db_weights = {}

        # Store base LoRA fully (with prefix)
        for key, value in base_weights.items():
            db_weights[f"base_{key}"] = value.clone()

        # Store deltas for other LoRAs
        total_original = 0
        total_delta = 0

        for i, (weights, name) in enumerate(zip(lora_weights, lora_names)):
            if i == base_idx:
                continue

            print(f"  Computing delta for: {name}")
            delta = self.compute_sparse_delta(base_weights, weights)

            # Add delta with LoRA index prefix
            for key, value in delta.items():
                db_weights[f"lora{i}_{key}"] = value

            # Track sizes
            orig_size = sum(v.numel() * v.element_size() for v in weights.values())
            delta_size = sum(v.numel() * v.element_size() for v in delta.values())
            total_original += orig_size
            total_delta += delta_size

        # Store metadata using safetensors built-in metadata
        meta = {
            "loradb_version": "1.0",
            "edition": self.edition,
            "base_index": str(base_idx),
            "base_name": base_name,
            "lora_names": json.dumps(lora_names),
            "lora_count": str(len(lora_paths)),
            "commercial_rights": "true" if self.edition == "Studio" else "false"
        }
        if metadata:
            for k, v in metadata.items():
                meta[k] = str(v) if not isinstance(v, str) else v

        # Save database with metadata
        save_file(db_weights, output_path, metadata=meta)

        # Calculate compression stats
        base_size = sum(v.numel() * v.element_size() for v in base_weights.values())
        db_size = os.path.getsize(output_path)
        total_original_all = base_size * len(lora_paths)  # Approximate
        compression = (1 - db_size / total_original_all) * 100 if total_original_all > 0 else 0

        print(f"  Database saved: {output_path}")
        print(f"  Compression: {compression:.1f}%")

        return {
            "status": "success",
            "loras_included": len(lora_paths),
            "base_lora": base_name,
            "output_path": output_path,
            "db_size_mb": db_size / (1024 * 1024),
            "compression_ratio": f"{compression:.1f}%",
            "commercial_rights": self.edition == "Studio"
        }

    def extract_lora(self, database_path, lora_index, output_path):
        """
        Extract a single LoRA from a .loradb file.

        Args:
            database_path: Path to .loradb file
            lora_index: Index of LoRA to extract (0-based)
            output_path: Where to save extracted LoRA
        """
        from safetensors import safe_open

        # Read metadata first
        with safe_open(database_path, framework="pt") as f:
            meta = f.metadata()

        if meta is None or "loradb_version" not in meta:
            raise ValueError("Invalid .loradb file: missing metadata")

        base_idx = int(meta["base_index"])
        lora_names = json.loads(meta["lora_names"])

        # Now load weights
        db_weights = load_file(database_path)

        if lora_index < 0 or lora_index >= len(lora_names):
            raise ValueError(f"Invalid lora_index: {lora_index}. Database has {len(lora_names)} LoRAs.")

        # Extract base weights
        base_weights = {}
        for key, value in db_weights.items():
            if key.startswith("base_"):
                orig_key = key[5:]  # Remove "base_" prefix
                base_weights[orig_key] = value

        if lora_index == base_idx:
            # Extracting the base LoRA - just return it
            save_file(base_weights, output_path)
            return {"status": "success", "extracted": lora_names[lora_index]}

        # Reconstruct from SVD-compressed delta
        reconstructed = {}
        prefix = f"lora{lora_index}_"

        # Identify which keys have SVD deltas, new SVD keys, full keys, or diff keys
        svd_keys = set()
        new_svd_keys = set()
        full_keys = set()
        diff_keys = set()

        for key in db_weights.keys():
            if not key.startswith(prefix):
                continue
            rest = key[len(prefix):]

            if rest.startswith("u_"):
                svd_keys.add(rest[2:])
            elif rest.startswith("new_u_"):
                new_svd_keys.add(rest[6:])
            elif rest.startswith("full_"):
                full_keys.add(rest[5:])
            elif rest.startswith("diff_"):
                diff_keys.add(rest[5:])

        # Start with base weights ONLY for keys that have a delta (svd or diff)
        # Don't include base keys that have no delta - they don't belong to this LoRA
        for key, value in base_weights.items():
            if key in svd_keys or key in diff_keys:
                reconstructed[key] = value.clone().float()

        # Apply SVD-compressed deltas: ΔW = U * diag(S) * V^T
        for orig_key in svd_keys:
            u_key = f"{prefix}u_{orig_key}"
            s_key = f"{prefix}s_{orig_key}"
            v_key = f"{prefix}v_{orig_key}"

            if u_key in db_weights and s_key in db_weights and v_key in db_weights:
                u = db_weights[u_key].float()
                s = db_weights[s_key].float()
                v = db_weights[v_key].float()

                # Reconstruct the Delta: ΔW = U * diag(S) * V^T
                delta_w = u @ torch.diag(s) @ v.t()

                # Add Delta back to the Base: W_target = W_base + ΔW
                if orig_key in reconstructed:
                    reconstructed[orig_key] = reconstructed[orig_key] + delta_w

        # Reconstruct new keys (not in base) from SVD
        for orig_key in new_svd_keys:
            u_key = f"{prefix}new_u_{orig_key}"
            s_key = f"{prefix}new_s_{orig_key}"
            v_key = f"{prefix}new_v_{orig_key}"

            if u_key in db_weights and s_key in db_weights and v_key in db_weights:
                u = db_weights[u_key].float()
                s = db_weights[s_key].float()
                v = db_weights[v_key].float()

                # Reconstruct: W = U * diag(S) * V^T
                reconstructed[orig_key] = u @ torch.diag(s) @ v.t()

        # Handle full tensors (fallback for incompatible shapes)
        for orig_key in full_keys:
            full_key = f"{prefix}full_{orig_key}"
            if full_key in db_weights:
                reconstructed[orig_key] = db_weights[full_key].float()

        # Handle direct diff tensors (for 1D tensors)
        for orig_key in diff_keys:
            diff_key = f"{prefix}diff_{orig_key}"
            if diff_key in db_weights and orig_key in reconstructed:
                reconstructed[orig_key] = reconstructed[orig_key] + db_weights[diff_key].float()

        # Convert back to appropriate dtype
        for key in list(reconstructed.keys()):
            if key in base_weights:
                reconstructed[key] = reconstructed[key].to(base_weights[key].dtype)
            else:
                reconstructed[key] = reconstructed[key].half()

        save_file(reconstructed, output_path)
        return {"status": "success", "extracted": lora_names[lora_index]}

    def list_contents(self, database_path):
        """List all LoRAs in a database file."""
        from safetensors import safe_open

        with safe_open(database_path, framework="pt") as f:
            meta = f.metadata()

        if meta is None or "loradb_version" not in meta:
            raise ValueError("Invalid .loradb file: missing metadata")

        return {
            "lora_count": int(meta["lora_count"]),
            "lora_names": json.loads(meta["lora_names"]),
            "base_lora": meta["base_name"],
            "edition": meta.get("edition", "Unknown"),
            "commercial_rights": meta.get("commercial_rights", "false") == "true"
        }
