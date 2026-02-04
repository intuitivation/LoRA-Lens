import os
import pandas as pd
from core.engine import LoRALensEngine

def run_batch_audit(folder_path):
    all_stats = []
    files = [f for f in os.listdir(folder_path) if f.endswith('.safetensors')]
    
    print(f"🔎 Auditing {len(files)} LoRAs...")
    
    for file in files:
        full_path = os.path.join(folder_path, file)
        try:
            engine = LoRALensEngine(full_path)
            layer_data = engine.get_full_analysis()
            
            # Aggregate stats for the whole file
            avg_eff = sum(d['eff_rank'] for d in layer_data) / len(layer_data)
            avg_sparsity = sum(d['sparsity'] for d in layer_data) / len(layer_data)
            total_mag = sum(d['magnitude'] for d in layer_data)
            
            all_stats.append({
                "filename": file,
                "avg_effective_rank": round(avg_eff, 2),
                "sparsity_score": round(avg_sparsity, 4),
                "total_strength": round(total_mag, 2),
                "health_score": round((avg_eff / layer_data[0]['declared_rank']) * 100, 1)
            })
        except Exception as e:
            print(f"❌ Failed to analyze {file}: {e}")

    df = pd.DataFrame(all_stats)
    df.to_csv("lora_audit_report.csv", index=False)
    print("✅ Audit Complete. Results saved to 'lora_audit_report.csv'")

if __name__ == "__main__":
    path = input("Enter path to your LoRA folder: ")
    run_batch_audit(path)