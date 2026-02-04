import umap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import torch

def generate_3d_concept_map(engine):
    """Projections of weight patterns into 3D space to identify clusters."""
    weight_samples = []
    layer_names = []
    magnitudes = []

    for name, data in engine.layers.items():
        if 'up' not in data or 'down' not in data:
            continue
            
        # Flatten the delta weight matrix to a vector
        delta_w = torch.matmul(data['up'], data['down']).flatten()
        
        weight_samples.append(delta_w.numpy())
        layer_names.append(name.split('.')[-1])  # Shortened name
        magnitudes.append(torch.norm(delta_w).item())

    if len(weight_samples) < 2:
        return None
        
    # UMAP Projection
    reducer = umap.UMAP(n_components=3, random_state=42)
    projections = reducer.fit_transform(weight_samples)

    df = pd.DataFrame(projections, columns=['x', 'y', 'z'])
    df['layer'] = layer_names
    df['magnitude'] = magnitudes
    
    fig = px.scatter_3d(
        df, x='x', y='y', z='z', 
        color='magnitude',
        hover_data=['layer'],
        title="LoRA Concept Clusters (3D UMAP Projection)",
        labels={'color': 'Magnitude', 'magnitude': 'Weight Strength'},
        color_continuous_scale='Viridis'
    )
    
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='#00ffcc')))
    
    return fig


def generate_weight_histogram(engine, layer_name=None):
    """
    Generate histogram of weight distributions.
    
    Args:
        engine: LoRALensEngine instance
        layer_name: Specific layer to analyze (None = all layers combined)
    """
    if layer_name:
        # Single layer histogram
        dist = engine.get_weight_distribution(layer_name)
        if dist is None:
            return None
            
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=dist['weights'],
            nbinsx=100,
            name=layer_name,
            marker_color='#00ffcc',
            opacity=0.75
        ))
        
        fig.update_layout(
            title=f"Weight Distribution: {layer_name}",
            xaxis_title="Weight Value",
            yaxis_title="Frequency",
            template="plotly_dark",
            showlegend=False
        )
        
        # Add percentile lines
        percentiles = dist['percentiles']
        for i, p in enumerate([1, 25, 50, 75, 99]):
            fig.add_vline(
                x=percentiles[i if p != 50 else 3],
                line_dash="dash",
                line_color="red" if p == 50 else "orange",
                annotation_text=f"p{p}"
            )
        
    else:
        # Multi-layer comparison
        fig = go.Figure()
        
        for name, data in list(engine.layers.items())[:5]:  # Limit to 5 for readability
            if 'up' not in data or 'down' not in data:
                continue
                
            dist = engine.get_weight_distribution(name)
            if dist:
                fig.add_trace(go.Histogram(
                    x=dist['weights'],
                    nbinsx=50,
                    name=name.split('.')[-1],
                    opacity=0.6
                ))
        
        fig.update_layout(
            title="Weight Distributions Across Layers",
            xaxis_title="Weight Value",
            yaxis_title="Frequency",
            template="plotly_dark",
            barmode='overlay'
        )
    
    return fig


def generate_correlation_heatmap(engine):
    """Generate correlation matrix heatmap showing layer relationships."""
    corr_data = engine.get_layer_correlations()
    
    if corr_data is None:
        return None
    
    correlation_matrix = corr_data['correlation_matrix']
    layer_names = [name.split('.')[-1] for name in corr_data['layer_names']]
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=layer_names,
        y=layer_names,
        colorscale='RdBu',
        zmid=0,
        text=np.round(correlation_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Layer Correlation Matrix",
        xaxis_title="Layer",
        yaxis_title="Layer",
        template="plotly_dark",
        height=600,
        width=800
    )
    
    return fig


def generate_layer_strength_chart(df):
    """Generate bar chart showing layer strength rankings."""
    # Sort by magnitude
    df_sorted = df.sort_values('magnitude', ascending=True).tail(20)
    
    fig = go.Figure()
    
    # Color by layer type if available
    if 'layer_type' in df_sorted.columns:
        colors = {
            'attention': '#00ffcc',
            'mlp': '#ff00cc',
            'normalization': '#ffcc00',
            'projection': '#00ccff',
            'other': '#cccccc'
        }
        bar_colors = [colors.get(t, '#cccccc') for t in df_sorted['layer_type']]
    else:
        bar_colors = '#00ffcc'
    
    fig.add_trace(go.Bar(
        y=[name.split('.')[-1] for name in df_sorted['layer']],
        x=df_sorted['magnitude'],
        orientation='h',
        marker_color=bar_colors,
        text=df_sorted['magnitude'].round(2),
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Top 20 Layers by Magnitude",
        xaxis_title="Magnitude (L2 Norm)",
        yaxis_title="Layer",
        template="plotly_dark",
        height=600,
        showlegend=False
    )
    
    return fig


def generate_efficiency_dashboard(df, engine):
    """Generate comprehensive efficiency dashboard with multiple subplots."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Rank Utilization',
            'Sparsity Distribution',
            'Dead Weights per Layer',
            'Magnitude vs Rank'
        ),
        specs=[[{'type': 'bar'}, {'type': 'histogram'}],
               [{'type': 'bar'}, {'type': 'scatter'}]]
    )
    
    # 1. Rank Utilization
    df['rank_util'] = df['eff_rank'] / df['declared_rank'] * 100
    fig.add_trace(
        go.Bar(
            x=df['rank_util'],
            y=[name.split('.')[-1] for name in df['layer']],
            orientation='h',
            marker_color='#00ffcc',
            name='Rank Utilization %'
        ),
        row=1, col=1
    )
    
    # 2. Sparsity Distribution
    fig.add_trace(
        go.Histogram(
            x=df['sparsity'] * 100,
            nbinsx=20,
            marker_color='#ff00cc',
            name='Sparsity %'
        ),
        row=1, col=2
    )
    
    # 3. Dead Weights per Layer
    fig.add_trace(
        go.Bar(
            x=df['dead_weights'] * 100,
            y=[name.split('.')[-1] for name in df['layer']],
            orientation='h',
            marker_color='#ffcc00',
            name='Dead Weights %'
        ),
        row=2, col=1
    )
    
    # 4. Magnitude vs Rank scatter
    fig.add_trace(
        go.Scatter(
            x=df['eff_rank'],
            y=df['magnitude'],
            mode='markers',
            marker=dict(
                size=10,
                color=df['sparsity'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Sparsity", x=1.15)
            ),
            text=[name.split('.')[-1] for name in df['layer']],
            name='Layers'
        ),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Utilization %", row=1, col=1)
    fig.update_xaxes(title_text="Sparsity %", row=1, col=2)
    fig.update_xaxes(title_text="Dead Weights %", row=2, col=1)
    fig.update_xaxes(title_text="Effective Rank", row=2, col=2)
    
    fig.update_yaxes(title_text="Layer", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Layer", row=2, col=1)
    fig.update_yaxes(title_text="Magnitude", row=2, col=2)
    
    fig.update_layout(
        height=800,
        showlegend=False,
        template="plotly_dark",
        title_text="Comprehensive Efficiency Analysis"
    )
    
    return fig


def generate_comparison_chart(df1, df2, label1="LoRA 1", label2="LoRA 2"):
    """Generate side-by-side comparison of two LoRAs."""
    # Create comparison dataframe
    comparison = pd.DataFrame({
        'Metric': ['Avg Eff Rank', 'Avg Sparsity', 'Avg Magnitude', 'Avg Dead Weights'],
        label1: [
            df1['eff_rank'].mean(),
            df1['sparsity'].mean() * 100,
            df1['magnitude'].mean(),
            df1['dead_weights'].mean() * 100
        ],
        label2: [
            df2['eff_rank'].mean(),
            df2['sparsity'].mean() * 100,
            df2['magnitude'].mean(),
            df2['dead_weights'].mean() * 100
        ]
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name=label1,
        x=comparison['Metric'],
        y=comparison[label1],
        marker_color='#00ffcc'
    ))
    
    fig.add_trace(go.Bar(
        name=label2,
        x=comparison['Metric'],
        y=comparison[label2],
        marker_color='#ff00cc'
    ))
    
    fig.update_layout(
        title="LoRA Comparison",
        xaxis_title="Metric",
        yaxis_title="Value",
        template="plotly_dark",
        barmode='group'
    )
    
    return fig


def generate_layer_type_breakdown(df):
    """Generate pie chart showing layer type distribution."""
    if 'layer_type' not in df.columns:
        return None
    
    type_counts = df['layer_type'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=type_counts.index,
        values=type_counts.values,
        hole=0.3,
        marker_colors=['#00ffcc', '#ff00cc', '#ffcc00', '#00ccff', '#cccccc']
    )])
    
    fig.update_layout(
        title="Layer Type Distribution",
        template="plotly_dark"
    )
    
    return fig


def generate_optimization_potential(df):
    """Visualize optimization potential per layer."""
    df['size_savings'] = (df['declared_rank'] - df['optimal_rank']) / df['declared_rank'] * 100
    df_sorted = df.sort_values('size_savings', ascending=True).tail(20)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=[name.split('.')[-1] for name in df_sorted['layer']],
        x=df_sorted['size_savings'],
        orientation='h',
        marker=dict(
            color=df_sorted['size_savings'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Savings %")
        ),
        text=df_sorted['size_savings'].round(1),
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Optimization Potential by Layer (Top 20)",
        xaxis_title="Potential Size Reduction %",
        yaxis_title="Layer",
        template="plotly_dark",
        height=600
    )
    
    return fig
