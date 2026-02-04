import pandas as pd

def get_ai_guidance(stats_df, engine=None):
    """Provides automated AI insights based on the dataframe stats and engine."""
    insights = []
    warnings = []
    recommendations = []
    
    # Basic stats
    avg_eff = stats_df['eff_rank'].mean()
    avg_optimal = stats_df['optimal_rank'].mean()
    avg_declared = stats_df['declared_rank'].mean()
    avg_sparsity = stats_df['sparsity'].mean()
    avg_dead = stats_df['dead_weights'].mean()
    max_magnitude = stats_df['magnitude'].max()
    
    # === RANK EFFICIENCY ANALYSIS ===
    rank_efficiency = avg_eff / avg_declared
    
    if rank_efficiency < 0.25:
        insights.append("✅ **EXCELLENT EFFICIENCY**: Your LoRA is using less than 25% of its declared rank.")
        recommendations.append(f"💡 **Action**: Retrain at rank {int(avg_optimal * 1.5)} (currently {int(avg_declared)}) to save {int((1 - avg_optimal/avg_declared) * 100)}% file size without quality loss.")
    elif rank_efficiency < 0.5:
        insights.append("🟢 **GOOD EFFICIENCY**: Your LoRA is reasonably optimized.")
        recommendations.append(f"💡 **Optional**: Consider retraining at rank {int(avg_optimal * 1.2)} for a smaller file.")
    elif rank_efficiency < 0.8:
        insights.append("🟡 **MODERATE EFFICIENCY**: Some optimization possible.")
        recommendations.append(f"💡 **Consider**: Using automatic rank optimization. Current rank could be reduced to {int(avg_optimal)}.")
    else:
        warnings.append("🔴 **HIGH RANK USAGE**: Effective rank is close to declared rank.")
        warnings.append("⚠️ **Risk**: Possible overfitting or memorization of training data.")
        recommendations.append(f"🛠️ **Fix**: Increase dropout rate, reduce learning rate, or add more diverse training images.")
    
    # === SPARSITY ANALYSIS ===
    if avg_sparsity > 0.7:
        insights.append("✅ **OPTIMAL SPARSITY**: High sparsity indicates efficient, targeted learning.")
    elif avg_sparsity > 0.4:
        insights.append("🟢 **GOOD SPARSITY**: Weights are reasonably sparse.")
    elif avg_sparsity > 0.2:
        warnings.append("🟡 **LOW SPARSITY**: Weights are relatively dense.")
        recommendations.append("💡 **Suggestion**: Consider increasing L1 regularization or dropout during training.")
    else:
        warnings.append("🔴 **VERY LOW SPARSITY**: Dense weights everywhere - likely overfitting.")
        recommendations.append("🛠️ **Critical**: Reduce learning rate by 50%, add dropout (0.1-0.2), or increase dataset size.")
    
    # === DEAD WEIGHTS ANALYSIS ===
    if avg_dead > 0.3:
        warnings.append(f"⚠️ **HIGH DEAD WEIGHTS**: {int(avg_dead*100)}% of weights are effectively zero.")
        recommendations.append("💡 **Action**: Use automatic pruning to remove dead weights and shrink file size.")
    elif avg_dead > 0.15:
        insights.append(f"🟡 **MODERATE DEAD WEIGHTS**: {int(avg_dead*100)}% of weights are near-zero.")
        recommendations.append("💡 **Optional**: Pruning could save ~{int(avg_dead*50)}% file size.")
    
    # === LAYER TYPE ANALYSIS ===
    if 'layer_type' in stats_df.columns:
        layer_types = stats_df.groupby('layer_type').agg({
            'magnitude': 'mean',
            'eff_rank': 'mean'
        })
        
        if 'attention' in layer_types.index:
            attn_mag = layer_types.loc['attention', 'magnitude']
            if attn_mag > stats_df['magnitude'].mean() * 1.5:
                insights.append("📊 **ATTENTION LAYERS DOMINANT**: Strong modifications in attention layers.")
                insights.append("ℹ️ This often indicates the LoRA is learning composition/style rather than objects.")
        
        if 'mlp' in layer_types.index:
            mlp_mag = layer_types.loc['mlp', 'magnitude']
            if mlp_mag > stats_df['magnitude'].mean() * 1.5:
                insights.append("📊 **MLP LAYERS DOMINANT**: Strong modifications in feedforward layers.")
                insights.append("ℹ️ This often indicates the LoRA is learning specific objects/concepts.")
    
    # === MAGNITUDE ANALYSIS ===
    high_magnitude_layers = stats_df[stats_df['magnitude'] > stats_df['magnitude'].mean() * 2]
    if len(high_magnitude_layers) > 0:
        warnings.append(f"⚠️ **{len(high_magnitude_layers)} OUTLIER LAYERS**: Some layers have extremely high magnitude.")
        recommendations.append("🔍 **Investigate**: Check if training images were too similar or if learning rate was too high.")
        
        # List the specific layers
        outlier_names = high_magnitude_layers['layer'].head(3).tolist()
        insights.append(f"📍 **Top outliers**: {', '.join([n.split('.')[-1] for n in outlier_names])}")
    
    # === TRAINING QUALITY INFERENCE ===
    if 'std_weight' in stats_df.columns:
        avg_std = stats_df['std_weight'].mean()
        max_std = stats_df['std_weight'].max()
        
        if max_std > avg_std * 5:
            warnings.append("🔴 **HIGH VARIANCE DETECTED**: Unstable training signature found.")
            recommendations.append("🛠️ **Fix**: Learning rate was likely too high. Reduce by 50% and retrain.")
    
    # === OVERALL HEALTH SCORE ===
    if engine:
        health_score = engine.get_efficiency_score()
        if health_score > 75:
            insights.append(f"🏆 **OVERALL HEALTH**: {health_score}/100 - Excellent!")
        elif health_score > 50:
            insights.append(f"✅ **OVERALL HEALTH**: {health_score}/100 - Good")
        elif health_score > 30:
            warnings.append(f"🟡 **OVERALL HEALTH**: {health_score}/100 - Needs improvement")
        else:
            warnings.append(f"🔴 **OVERALL HEALTH**: {health_score}/100 - Poor quality")
    
    # === COMBINE ALL INSIGHTS ===
    all_messages = []
    
    if insights:
        all_messages.append("### 📊 Key Insights")
        all_messages.extend(insights)
    
    if warnings:
        all_messages.append("")
        all_messages.append("### ⚠️ Warnings")
        all_messages.extend(warnings)
    
    if recommendations:
        all_messages.append("")
        all_messages.append("### 💡 Recommendations")
        all_messages.extend(recommendations)
    
    return all_messages


def get_training_recommendations(stats_df):
    """Generate specific training parameter recommendations."""
    avg_eff = stats_df['eff_rank'].mean()
    avg_declared = stats_df['declared_rank'].mean()
    avg_sparsity = stats_df['sparsity'].mean()
    
    params = {
        'recommended_rank': int(stats_df['optimal_rank'].mean() * 1.2),
        'network_alpha': None,
        'learning_rate': None,
        'dropout': None
    }
    
    # Learning rate recommendations
    if avg_sparsity < 0.2:
        params['learning_rate'] = "Reduce to 5e-5 or lower (currently likely too high)"
    elif avg_eff / avg_declared > 0.8:
        params['learning_rate'] = "Reduce to 7e-5 (signs of overfitting)"
    else:
        params['learning_rate'] = "Current is likely fine (1e-4 range)"
    
    # Dropout recommendations
    if avg_sparsity < 0.3:
        params['dropout'] = "Add or increase to 0.1-0.2"
    elif avg_sparsity > 0.7:
        params['dropout'] = "Current is good (0.1 or off)"
    else:
        params['dropout'] = "Consider adding 0.1"
    
    # Alpha recommendations
    params['network_alpha'] = params['recommended_rank'] // 2
    
    return params


def get_merge_guidance(conflicts_df):
    """Provide guidance on LoRA merging based on conflicts."""
    if conflicts_df is None or len(conflicts_df) == 0:
        return ["No conflict data available"]
    
    guidance = []
    
    severe_conflicts = conflicts_df[conflicts_df['status'] == 'SEVERE CONFLICT']
    conflicts = conflicts_df[conflicts_df['status'] == 'CONFLICT']
    compatible = conflicts_df[conflicts_df['status'] == 'Compatible']
    
    if len(severe_conflicts) > 0:
        guidance.append(f"🚨 **{len(severe_conflicts)} SEVERE CONFLICTS** detected")
        guidance.append("⛔ **NOT RECOMMENDED** to merge these LoRAs")
        guidance.append(f"📍 Affected layers: {', '.join(severe_conflicts['layer'].head(3).tolist())}")
    elif len(conflicts) > len(conflicts_df) * 0.5:
        guidance.append(f"⚠️ **{len(conflicts)} CONFLICTS** detected ({int(len(conflicts)/len(conflicts_df)*100)}% of layers)")
        guidance.append("🟡 **CAUTION**: Merging may produce artifacts")
        guidance.append("💡 **Suggestion**: Use lower weights (0.3-0.5) for secondary LoRA")
    elif len(conflicts) > 0:
        guidance.append(f"🟡 **{len(conflicts)} minor conflicts** detected")
        guidance.append("✅ **SAFE TO MERGE** with standard ratios (0.5-0.7)")
    else:
        guidance.append("✅ **HIGHLY COMPATIBLE** - No conflicts detected")
        guidance.append("🎯 **RECOMMENDED**: Safe to merge at any ratio")
    
    # Calculate average similarity
    avg_sim = conflicts_df['similarity'].mean()
    guidance.append(f"📊 **Average Similarity**: {avg_sim:.3f}")
    
    if avg_sim > 0.7:
        guidance.append("ℹ️ These LoRAs learn very similar concepts - merging may not add much value")
    elif avg_sim < 0.2:
        guidance.append("ℹ️ These LoRAs learn different concepts - merging may combine their strengths")
    
    return guidance
