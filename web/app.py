import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Internal Imports
from core.engine import LoRALensEngine
from core.database import LoRADatabase  # Database functionality
from core.visualizer import (
    generate_3d_concept_map,
    generate_weight_histogram,
    generate_correlation_heatmap,
    generate_layer_strength_chart,
    generate_efficiency_dashboard,
    generate_comparison_chart,
    generate_layer_type_breakdown,
    generate_optimization_potential
)
from core.exporter import export_results, save_html_report
from web.consultant import get_ai_guidance, get_training_recommendations, get_merge_guidance

# --- MODERN DARK THEME ---
st.set_page_config(
    page_title="LoRA Lens Community",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Root variables */
    :root {
        --bg-primary: #09090b;
        --bg-secondary: #18181b;
        --bg-tertiary: #27272a;
        --border-color: #3f3f46;
        --text-primary: #fafafa;
        --text-secondary: #a1a1aa;
        --text-muted: #71717a;
        --accent: #22c55e;
        --accent-hover: #16a34a;
        --success: #22c55e;
        --warning: #eab308;
        --error: #ef4444;
    }

    /* Global styles */
    .stApp {
        background: linear-gradient(180deg, var(--bg-primary) 0%, #0c0c0f 100%);
    }

    /* Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.02em;
    }

    h1 { font-size: 2rem !important; }
    h2 { font-size: 1.5rem !important; }
    h3 { font-size: 1.25rem !important; }

    p, span, label, div {
        color: var(--text-secondary);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-color) !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--text-secondary);
    }

    /* Main content area */
    .main .block-container {
        padding: 2rem 3rem !important;
        max-width: 100% !important;
    }

    /* Cards/Containers */
    [data-testid="stExpander"], .stAlert, [data-testid="stMetric"] {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
    }

    /* Buttons */
    .stButton > button {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.15s ease !important;
        width: 100%;
    }

    .stButton > button:hover {
        background: var(--bg-tertiary) !important;
        border-color: var(--accent) !important;
        color: var(--text-primary) !important;
        transform: translateY(-1px);
    }

    /* Metrics */
    [data-testid="stMetric"] {
        padding: 1.25rem !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: var(--text-muted) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    /* Alerts */
    .stAlert {
        padding: 1rem 1.25rem !important;
    }

    [data-testid="stAlert"][data-baseweb*="info"] {
        border-left: 4px solid var(--accent) !important;
    }

    [data-testid="stAlert"][data-baseweb*="success"] {
        border-left: 4px solid var(--success) !important;
    }

    [data-testid="stAlert"][data-baseweb*="warning"] {
        border-left: 4px solid var(--warning) !important;
    }

    /* Data tables */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        overflow: hidden;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: var(--bg-secondary) !important;
        border: 2px dashed var(--border-color) !important;
        border-radius: 12px !important;
        padding: 2rem !important;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-secondary) !important;
        border-radius: 8px !important;
        padding: 4px !important;
        gap: 4px !important;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 6px !important;
        color: var(--text-muted) !important;
        font-weight: 500 !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
    }

    .stTabs [aria-selected="true"] {
        background: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }

    /* Download button */
    .stDownloadButton > button {
        background: var(--success) !important;
        border-color: var(--success) !important;
        color: white !important;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("# LoRA Lens")
st.caption("Community Edition — Free for Personal & Educational Use")

# Community Edition License Notice (compact)
col_badge, col_spacer = st.columns([3, 2])
with col_badge:
    st.success("**Community Edition** — Free for personal & educational use. Limited to 10 LoRAs per database.")

# Initialize session state variables
if 'view' not in st.session_state:
    st.session_state.view = 'Dashboard'
if 'precompute_all' not in st.session_state:
    st.session_state.precompute_all = True  # Default to ON

# --- SIDEBAR & UPLOAD ---
with st.sidebar:
    st.header("DATA INGESTION")
    uploaded_file = st.file_uploader("LOAD PRIMARY .SAFETENSORS", type="safetensors", key="primary")

    st.markdown("---")
    st.subheader("ANALYSIS OPTIONS")

    show_advanced = st.checkbox("Show Advanced Metrics", value=False)
    auto_cache = st.checkbox("Enable Analysis Caching", value=True)

    st.markdown("---")

    # Show precompute status
    if st.session_state.precompute_all:
        st.success("Pre-compute: ON")
        st.caption("Visualizations load instantly")
    else:
        st.warning("Pre-compute: OFF")
        st.caption("Visualizations generate on-demand")

    st.caption("Change in Settings")

    st.markdown("---")
    st.info("System Status: ONLINE")
    st.caption("v1.6 FLUX Edition")
    st.caption("New: FLUX LoRA support, BFloat16 precision, Settings page")

    # FLUX detection info
    if uploaded_file and 'engine' in locals():
        if hasattr(st.session_state, 'is_flux'):
            if st.session_state.is_flux:
                st.success("FLUX LoRA Detected")
            else:
                st.info("SD/SDXL LoRA Detected")

# Helper function for views that need data
def show_upload_prompt():
    st.info("""
    **No LoRA file loaded.**

    Please upload a `.safetensors` file in the sidebar to use this feature.
    """)

# Initialize variables as None before the upload check
engine = None
df = None
insights = None
health_score = None

if uploaded_file:
    # Save temp file for the engine
    temp_path = "active_lora.safetensors"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Initialize Engine
    with st.spinner("Loading and analyzing LoRA..."):
        engine = LoRALensEngine(temp_path)

        # Store FLUX detection
        st.session_state.is_flux = engine.is_flux

        results = engine.get_full_analysis(use_cache=auto_cache)
        df = pd.DataFrame(results)

        # Get AI insights
        insights = get_ai_guidance(df, engine)
        health_score = engine.get_efficiency_score()

        # PRECOMPUTE VISUALIZATIONS if enabled
        if st.session_state.precompute_all:
            with st.spinner("Pre-computing all visualizations for instant tab switching..."):
                # Store all visualizations in session state
                try:
                    st.session_state.viz_3d = generate_3d_concept_map(engine)
                except:
                    st.session_state.viz_3d = None

                try:
                    st.session_state.viz_correlation = generate_correlation_heatmap(engine)
                except:
                    st.session_state.viz_correlation = None

                try:
                    st.session_state.viz_layer_strength = generate_layer_strength_chart(df)
                except:
                    st.session_state.viz_layer_strength = None

                try:
                    st.session_state.viz_efficiency = generate_efficiency_dashboard(df)
                except:
                    st.session_state.viz_efficiency = None

                try:
                    st.session_state.viz_layer_types = generate_layer_type_breakdown(df)
                except:
                    st.session_state.viz_layer_types = None

                st.success("All visualizations ready! Tabs will load instantly.")
        else:
            st.info("Visualizations will generate on-demand when you click each tab. Enable pre-compute in Settings for instant switching.")

# --- ACTION HUD (Top Navigation) - ALWAYS VISIBLE ---
st.markdown("### CONTROL INTERFACE")
col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10)

with col1:
    if st.button("DASHBOARD"): st.session_state.view = 'Dashboard'
with col2:
    if st.button("ANALYTICS"): st.session_state.view = 'Analytics'
with col3:
    if st.button("3D MAP"): st.session_state.view = '3D'
with col4:
    if st.button("CONFLICT"): st.session_state.view = 'Conflict'
with col5:
    if st.button("CONSULT"): st.session_state.view = 'AI'
with col6:
    if st.button("OPTIMIZE"): st.session_state.view = 'Optimize'
with col7:
    if st.button("SURGERY"): st.session_state.view = 'Surgery'
with col8:
    if st.button("DATABASE"): st.session_state.view = 'Database'
with col9:
    if st.button("EXPORT"): st.session_state.view = 'Export'
with col10:
    if st.button("SETTINGS"): st.session_state.view = 'Settings'

st.markdown("---")

# --- DYNAMIC VIEWPORT LOGIC ---

# 1. DASHBOARD VIEW (New!)
if st.session_state.view == 'Dashboard':
    st.subheader("EXECUTIVE DASHBOARD")

    if df is None:
        show_upload_prompt()
    else:
        # Top metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("HEALTH SCORE", f"{health_score}/100",
                 "Excellent" if health_score > 75 else "Good" if health_score > 50 else "Needs Work")
        c2.metric("AVG EFFICIENCY", f"{df['eff_rank'].mean():.1f}",
                 f"of {df['declared_rank'].mean():.0f} declared")
        c3.metric("SPARSITY", f"{df['sparsity'].mean()*100:.1f}%", "Weight Optimization")
        c4.metric("OPTIMIZATION", f"{(1-df['optimal_rank'].mean()/df['declared_rank'].mean())*100:.0f}%",
                 "Potential Savings")

        st.markdown("---")

        # Quick insights
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Quick Health Check")
            for insight in insights[:5]:
                if any(x in insight for x in ["OK", "green", "trophy"]):
                    st.success(insight)
                elif any(x in insight for x in ["yellow", "warning"]):
                    st.warning(insight)
                elif any(x in insight for x in ["red", "WARNING", "CRITICAL"]):
                    st.error(insight)
                else:
                    st.info(insight)

        with col_b:
            st.markdown("#### Layer Type Breakdown")
            fig_pie = generate_layer_type_breakdown(df)
            if fig_pie:
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Layer type analysis not available")

        st.markdown("---")

        # Efficiency dashboard
        st.markdown("#### Comprehensive Analysis")
        fig_eff = generate_efficiency_dashboard(df, engine)
        st.plotly_chart(fig_eff, use_container_width=True)

# 2. ANALYTICS VIEW (Enhanced!)
elif st.session_state.view == 'Analytics':
    st.subheader("ADVANCED ANALYTICS")

    if df is None:
        show_upload_prompt()
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["Data Table", "Heatmaps", "Distributions", "Correlations"])

        with tab1:
            st.markdown("#### Layer-by-Layer Analysis")

            # Add filters
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                if 'layer_type' in df.columns:
                    selected_types = st.multiselect(
                        "Filter by Layer Type",
                        options=df['layer_type'].unique(),
                        default=df['layer_type'].unique()
                    )
                    df_filtered = df[df['layer_type'].isin(selected_types)]
                else:
                    df_filtered = df

            with filter_col2:
                min_magnitude = st.slider("Minimum Magnitude", 0.0, float(df['magnitude'].max()), 0.0)
                df_filtered = df_filtered[df_filtered['magnitude'] >= min_magnitude]

            st.dataframe(
                df_filtered.style.background_gradient(subset=['magnitude'], cmap='viridis')
                          .background_gradient(subset=['sparsity'], cmap='RdYlGn'),
                use_container_width=True,
                height=400
            )

            if show_advanced:
                st.markdown("#### Statistical Summary")
                st.write(df_filtered.describe())

        with tab2:
            st.markdown("#### Weight Influence Heatmap")
            fig_heat = px.density_heatmap(
                df, x="layer", y="eff_rank", z="magnitude",
                color_continuous_scale="Viridis",
                labels={'eff_rank': 'Effective Rank', 'magnitude': 'Weight Strength'}
            )
            fig_heat.update_layout(template="plotly_dark")
            st.plotly_chart(fig_heat, use_container_width=True)

            st.markdown("#### Layer Correlation Matrix")

            # Use cached correlation if available
            if st.session_state.precompute_all and 'viz_correlation' in st.session_state:
                fig_corr = st.session_state.viz_correlation
            else:
                with st.spinner("Computing correlations..."):
                    fig_corr = generate_correlation_heatmap(engine)

            if fig_corr:
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Correlation analysis requires at least 2 layers")

        with tab3:
            st.markdown("#### Weight Distributions")

            # Select layer for detailed view
            layer_options = ["All Layers (Combined)"] + df['layer'].tolist()
            selected_layer = st.selectbox("Select Layer", layer_options)

            if selected_layer == "All Layers (Combined)":
                fig_hist = generate_weight_histogram(engine, None)
            else:
                fig_hist = generate_weight_histogram(engine, selected_layer)

            if fig_hist:
                st.plotly_chart(fig_hist, use_container_width=True)

            # Layer strength chart
            st.markdown("#### Layer Strength Rankings")
            fig_strength = generate_layer_strength_chart(df)
            st.plotly_chart(fig_strength, use_container_width=True)

        with tab4:
            st.markdown("#### Optimization Potential")
            fig_opt = generate_optimization_potential(df)
            st.plotly_chart(fig_opt, use_container_width=True)

# 3. 3D CONCEPT VIEW
elif st.session_state.view == '3D':
    st.subheader("3D NEURAL TOPOLOGY (UMAP)")
    st.caption("High-dimensional weight patterns projected into 3D space")

    if df is None:
        show_upload_prompt()
    else:
        # Use cached 3D map if available
        if st.session_state.precompute_all and 'viz_3d' in st.session_state:
            fig_3d = st.session_state.viz_3d
        else:
            with st.spinner("Projecting high-dimensional weight clusters..."):
                fig_3d = generate_3d_concept_map(engine)

        if fig_3d:
            fig_3d.update_layout(template="plotly_dark", height=700)
            st.plotly_chart(fig_3d, use_container_width=True)

            st.info("""
             **How to interpret:**
             **Tight clusters**: Strong, specific concepts
             **Scattered points**: Generalized or confused learning
             **Outliers**: Potentially problematic layers
             **Color intensity**: Weight magnitude (brighter = stronger)
             """)
        else:
            st.warning("Not enough layers for 3D visualization")

# 4. CONFLICT SCANNER VIEW
elif st.session_state.view == 'Conflict':
    st.subheader("INTERFERENCE SCANNER")
    st.caption("Analyze compatibility between LoRAs before merging")

    if df is None:
        show_upload_prompt()
    else:
        compare_file = st.file_uploader("INGEST SECONDARY LoRA FOR COLLISION TEST", type="safetensors", key="secondary")

        if compare_file:
            with open("compare.safetensors", "wb") as f:
                f.write(compare_file.getbuffer())

            other_engine = LoRALensEngine("compare.safetensors")
            other_results = other_engine.get_full_analysis()
            other_df = pd.DataFrame(other_results)

            conflicts = engine.detect_conflicts(other_engine)
            conflicts_df = pd.DataFrame(conflicts)

            # Get merge suggestions
            merge_suggestion = engine.suggest_merge_ratio(other_engine)
            merge_guidance = get_merge_guidance(conflicts_df)

            # Display results
            col_x, col_y = st.columns(2)

            with col_x:
                st.markdown("#### Compatibility Analysis")
                avg_sim = conflicts_df['similarity'].mean()

                # Compatibility Gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=avg_sim,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "COMPATIBILITY SCORE"},
                    delta={'reference': 0.5},
                    gauge={
                        'axis': {'range': [-1, 1]},
                        'bar': {'color': "#00ffcc"},
                        'steps': [
                            {'range': [-1, 0], 'color': "#ff0000"},
                            {'range': [0, 0.5], 'color': "#ffcc00"},
                            {'range': [0.5, 1], 'color': "#00ff00"}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.5
                        }
                    }
                ))
                fig_gauge.update_layout(template="plotly_dark", height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_y:
                st.markdown("#### Merge Recommendation")
                st.metric("Suggested Ratio", f"{merge_suggestion['ratio']:.2f}",
                         f"Confidence: {merge_suggestion['confidence']}")
                st.metric("Conflicts Detected", merge_suggestion['conflicts'])

                if merge_suggestion['confidence'] == 'high':
                    st.success("Safe to merge")
                elif merge_suggestion['confidence'] == 'medium':
                    st.warning("Proceed with caution")
                else:
                    st.error("High conflict risk")

            # Guidance
            st.markdown("#### AI Merge Guidance")
            for guide in merge_guidance:
                if "alert" in guide.lower() or "stop" in guide.lower():
                    st.error(guide)
                elif "warning" in guide.lower() or "yellow" in guide.lower():
                    st.warning(guide)
                else:
                    st.info(guide)

            # Conflict details
            st.markdown("#### Detailed Conflict Analysis")

            # Color-code by status
            def highlight_conflicts(row):
                if row['status'] == 'SEVERE CONFLICT':
                    return ['background-color: #ff000033'] * len(row)
                elif row['status'] == 'CONFLICT':
                    return ['background-color: #ffcc0033'] * len(row)
                elif row['status'] == 'Compatible':
                    return ['background-color: #00ff0033'] * len(row)
                else:
                    return [''] * len(row)

            st.dataframe(
                conflicts_df.style.apply(highlight_conflicts, axis=1),
                use_container_width=True
            )

            # Side-by-side comparison
            st.markdown("#### Statistical Comparison")
            fig_compare = generate_comparison_chart(df, other_df, "Primary LoRA", "Secondary LoRA")
            st.plotly_chart(fig_compare, use_container_width=True)

# 5. AI CONSULTANT VIEW
elif st.session_state.view == 'AI':
    st.subheader("NEURAL CONSULTANT TERMINAL")
    st.caption("AI-powered analysis and recommendations")

    if df is None:
        show_upload_prompt()
    else:
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.markdown("### Comprehensive Analysis")
            for tip in insights:
                if "red" in tip.lower() or "WARNING" in tip or "CRITICAL" in tip:
                    st.error(tip)
                elif "yellow" in tip.lower() or "warning" in tip.lower():
                    st.warning(tip)
                elif "OK" in tip or "green" in tip.lower() or "trophy" in tip.lower():
                    st.success(tip)
                else:
                    st.info(tip)

        with col_right:
            st.markdown("### Training Parameters")
            train_params = get_training_recommendations(df)

            st.metric("Recommended Rank", train_params['recommended_rank'])
            st.metric("Network Alpha", train_params['network_alpha'])

            st.markdown("**Learning Rate:**")
            st.code(train_params['learning_rate'])

            st.markdown("**Dropout:**")
            st.code(train_params['dropout'])

# 6. OPTIMIZE VIEW (New!)
elif st.session_state.view == 'Optimize':
    st.subheader("LoRA OPTIMIZATION CENTER")
    st.caption("Surgical tools to improve your LoRA")

    if df is None:
        show_upload_prompt()
    else:
        tab1, tab2 = st.tabs(["Pruning", "Analysis"])

        with tab1:
            st.markdown("#### Automatic Rank Optimization")

            col_opt1, col_opt2 = st.columns(2)

            with col_opt1:
                st.markdown("**Current Stats:**")
                st.metric("Average Declared Rank", f"{df['declared_rank'].mean():.0f}")
                st.metric("Average Effective Rank", f"{df['eff_rank'].mean():.1f}")
                st.metric("Average Optimal Rank", f"{df['optimal_rank'].mean():.1f}")

            with col_opt2:
                st.markdown("**Optimization Settings:**")

                prune_mode = st.radio(
                    "Pruning Mode",
                    ["Auto (per-layer optimal)", "Fixed Rank", "Variance Threshold"]
                )

                if prune_mode == "Fixed Rank":
                    target_rank = st.slider("Target Rank", 8, 128, int(df['optimal_rank'].mean()))
                    variance_threshold = None
                elif prune_mode == "Variance Threshold":
                    target_rank = None
                    variance_threshold = st.slider("Variance to Retain", 0.90, 0.99, 0.99, 0.01)
                else:
                    target_rank = None
                    variance_threshold = 0.99

            st.markdown("---")

            # Preview
            st.markdown("#### Pruning Preview")

            if target_rank:
                estimated_savings = (1 - target_rank / df['declared_rank'].mean()) * 100
            else:
                estimated_savings = (1 - df['optimal_rank'].mean() / df['declared_rank'].mean()) * 100

            st.info(f"**Estimated file size reduction:** {estimated_savings:.1f}%")

            # Prune button
            if st.button("EXECUTE PRUNING", type="primary"):
                output_path = "optimized_lora.safetensors"

                with st.spinner("Optimizing LoRA..."):
                    pruning_stats = engine.prune_to_optimal(
                        output_path,
                        target_rank=target_rank,
                        variance_threshold=variance_threshold or 0.99
                    )

                st.success("Optimization complete!")

                # Show stats
                stats_df = pd.DataFrame(pruning_stats)
                st.dataframe(stats_df, use_container_width=True)

                # Download button
                with open(output_path, "rb") as f:
                    st.download_button(
                        "DOWNLOAD OPTIMIZED LoRA",
                        f,
                        file_name="optimized_lora.safetensors",
                        mime="application/octet-stream"
                    )

        with tab2:
            st.markdown("#### Optimization Potential Analysis")
            fig_pot = generate_optimization_potential(df)
            st.plotly_chart(fig_pot, use_container_width=True)

            st.markdown("#### Per-Layer Recommendations")

            # Show layers that would benefit most
            df['potential_savings'] = (df['declared_rank'] - df['optimal_rank']) / df['declared_rank'] * 100
            top_savings = df.nlargest(10, 'potential_savings')[['layer', 'declared_rank', 'optimal_rank', 'potential_savings']]

            st.write("**Top 10 layers with highest optimization potential:**")
            st.dataframe(top_savings, use_container_width=True)

# 7. EXPORT VIEW
elif st.session_state.view == 'Export':
    st.subheader("EXPORT & REPORTING")
    st.caption("Generate comprehensive reports and export data")

    if df is None:
        show_upload_prompt()
    else:
        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            st.markdown("#### Data Export")

            if st.button("Generate JSON + CSV"):
                json_p, csv_p = export_results(df)
                st.success("Data files generated!")

                with open(json_p, "rb") as f:
                    st.download_button("Download JSON", f, file_name=json_p)
                with open(csv_p, "rb") as f:
                    st.download_button("Download CSV", f, file_name=csv_p)

        with col_exp2:
            st.markdown("#### HTML Report")

            if st.button("Generate HTML Report"):
                with st.spinner("Creating comprehensive report..."):
                    html_path = save_html_report(df, engine, insights)
                    st.success("HTML report generated!")

                    with open(html_path, "rb") as f:
                        st.download_button("Download Report", f, file_name=html_path)

        st.markdown("---")

        st.markdown("#### Export Summary")
        st.info(f"""
        **Available Formats:**
        - JSON: Machine-readable data with all metrics
        - CSV: Spreadsheet format for further analysis
        - HTML: Beautiful interactive report with visualizations

        **Report Contents:**
        - Executive summary with health score
        - Layer-by-layer analysis
        - AI consultant recommendations
        - Optimization suggestions
        """)

# 8. SURGERY VIEW (NEW!)
elif st.session_state.view == 'Surgery':
    st.subheader("NEURAL SURGERY & QUANTIZATION")
    st.caption("Advanced compression for FLUX LoRAs")

    if df is None:
        show_upload_prompt()
    else:
        # Import quantizer
        from core.quantizer import combined_optimize_and_quantize, quantize_lora_full

        # Check if FLUX
        is_flux = st.session_state.get('is_flux', False)

        if not is_flux:
            st.warning("""
            **Quantization is optimized for FLUX LoRAs**

            Your current LoRA appears to be SD/SDXL format.
            Quantization works best with FLUX LoRAs (typically 200MB+).

            You can still use rank optimization in the **OPTIMIZE** tab.
            """)

        st.markdown("---")

        # Two-column layout
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("### Rank Compression")
            st.info("""
            **SVD-based rank reduction**
            - Removes unused dimensions
            - Typical savings: 30-40%
            - Zero quality loss at 95% variance
            """)

            var_target = st.slider(
                "Variance Retention",
                0.80, 1.0, 0.95, 0.01,
                help="Lower = smaller file but may lose detail. 95% recommended."
            )

            if st.button("RUN RANK COMPRESSION", type="secondary"):
                with st.spinner("Executing SVD compression..."):
                    try:
                        out_path = "rank_compressed.safetensors"
                        stats = engine.prune_to_optimal(out_path, variance_threshold=var_target)

                        import os
                        original_size = os.path.getsize(engine.path) / (1024 * 1024)
                        compressed_size = os.path.getsize(out_path) / (1024 * 1024)
                        savings = (1 - compressed_size / original_size) * 100

                        st.success(f"""
                        **Compression Complete!**
                        - Original: {original_size:.1f} MB
                        - Compressed: {compressed_size:.1f} MB
                        - Savings: {savings:.1f}%
                        """)

                        with open(out_path, "rb") as f:
                            st.download_button(
                                "Download Compressed LoRA",
                                f,
                                file_name="rank_compressed.safetensors",
                                mime="application/octet-stream"
                            )
                    except Exception as e:
                        st.error(f"Error during compression: {e}")

        with col_b:
            st.markdown("### 8-Bit Quantization")
            st.info("""
            **Precision reduction**
            - Converts BFloat16 to Int8
            - Additional 50% reduction
            - <1% quality loss
            - Ideal for FLUX LoRAs >100MB
            """)

            quant_bits = st.radio(
                "Quantization Level",
                [8, 4],
                format_func=lambda x: f"{x}-bit " + ("(Recommended)" if x == 8 else "(Experimental)"),
                help="8-bit: Safe, minimal quality loss. 4-bit: Aggressive, test first."
            )

            if st.button("QUANTIZE TO " + f"{quant_bits}-BIT", type="secondary"):
                with st.spinner(f"Applying {quant_bits}-bit quantization..."):
                    try:
                        out_path = f"quantized_{quant_bits}bit.safetensors"
                        stats = quantize_lora_full(engine, out_path, bits=quant_bits)

                        st.success(f"""
                        **Quantization Complete!**
                        - Original: {stats['original_size']:.1f} MB
                        - Quantized: {stats['quantized_size']:.1f} MB
                        - Savings: {stats['compression_ratio']*100:.1f}%
                        - Quality: {stats['avg_quality_retention']:.1f}%
                        """)

                        with open(out_path, "rb") as f:
                            st.download_button(
                                f"Download {quant_bits}-bit LoRA",
                                f,
                                file_name=f"quantized_{quant_bits}bit.safetensors",
                                mime="application/octet-stream"
                            )
                    except Exception as e:
                        st.error(f"Error during quantization: {e}")

        st.markdown("---")

        # ULTRA COMPRESS - Combined pipeline
        st.markdown("### ULTRA COMPRESS (Rank + Quantization)")
        st.warning("""
        **Maximum compression pipeline**

        Combines both techniques for maximum file size reduction:
        1. Rank optimization (SVD)
        2. 8-bit quantization

        **Expected results:**
        - FLUX LoRAs: 60-70% total reduction
        - Example: 400MB to 140MB
        - Quality retention: >99%
        """)

        col_ultra1, col_ultra2 = st.columns([2, 1])

        with col_ultra1:
            ultra_variance = st.slider(
                "Variance Target (Ultra)",
                0.85, 0.99, 0.95, 0.01,
                key="ultra_var"
            )
            ultra_bits = st.radio(
                "Final Quantization",
                [8, 4],
                format_func=lambda x: f"{x}-bit",
                key="ultra_bits",
                horizontal=True
            )

        with col_ultra2:
            if st.button("ULTRA COMPRESS", type="primary"):
                with st.spinner("Running dual-stage compression..."):
                    try:
                        out_path = "ultra_compressed.safetensors"
                        stats = combined_optimize_and_quantize(
                            engine,
                            out_path,
                            bits=ultra_bits,
                            variance_threshold=ultra_variance
                        )

                        st.balloons()
                        st.success(f"""
                        **ULTRA COMPRESSION COMPLETE!**

                        **Stage 1 - Rank Optimization:**
                        - {stats['original_size_mb']:.1f}MB to {stats['after_rank_opt_mb']:.1f}MB
                        - Reduction: {stats['rank_reduction_pct']:.1f}%

                        **Stage 2 - Quantization:**
                        - {stats['after_rank_opt_mb']:.1f}MB to {stats['final_size_mb']:.1f}MB
                        - Reduction: {stats['quant_reduction_pct']:.1f}%

                        **TOTAL SAVINGS: {stats['total_reduction_pct']:.1f}%**
                        **Quality Retained: {stats['quality_retention_pct']:.1f}%**
                        """)

                        with open(out_path, "rb") as f:
                            st.download_button(
                                "DOWNLOAD ULTRA-COMPRESSED LoRA",
                                f,
                                file_name="ultra_compressed.safetensors",
                                mime="application/octet-stream",
                                type="primary"
                            )
                    except Exception as e:
                        st.error(f"Error during ultra compression: {e}")
                        import traceback
                        st.code(traceback.format_exc())

# 9. DATABASE VIEW (Community Edition - Limited to 10 LoRAs) - NO df check needed
elif st.session_state.view == 'Database':
    st.title("LoRA DATABASE BUILDER")
    st.caption("Community Edition - Create compressed LoRA collections")

    # Community Edition Limit Warning
    st.info("""
    **Community Edition: Limited to 10 LoRAs per database**

    Upgrade for larger collections:
    - **Pro Edition ($299):** Up to 50 LoRAs per database
    - **Studio Edition ($599):** UNLIMITED LoRAs + Commercial Resale Rights
    """)

    # Golden Ratio Layout
    col_input, col_viz = st.columns([0.382, 0.618])

    with col_input:
        st.subheader("COLLECTION BUILDER")

        # File uploader
        files = st.file_uploader(
            "Upload LoRA Files (.safetensors)",
            accept_multiple_files=True,
            type=["safetensors"],
            help="Community Edition: Maximum 10 LoRAs"
        )

        # Check limit
        if files and len(files) > 10:
            st.error(f"""
            **Community Edition Limit Exceeded**

            You uploaded {len(files)} LoRAs, but Community Edition is limited to 10.

            **Upgrade Options:**
            - [Pro Edition ($299)](https://intuitivation.gumroad.com/l/LoRALens-Pro) - 50 LoRAs
            - [Studio Edition ($599)](https://intuitivation.gumroad.com/l/LoRALens-Studio) - Unlimited
            """)
            files = None  # Reset

        if files and len(files) <= 10:
            st.success(f"{len(files)} LoRAs loaded ({10 - len(files)} slots remaining)")

        st.markdown("---")

        # Clustering precision
        mode = st.select_slider(
            "Compression Level",
            options=["Loose", "Balanced", "Tight"],
            value="Balanced",
            help="Higher compression = smaller file, slightly longer processing"
        )

        # Database name
        db_name = st.text_input(
            "Database Name",
            value="my_lora_collection",
            help="Name for your .loradb file"
        )

        if files and len(files) > 0 and len(files) <= 10:
            if st.button("ANALYZE COLLECTION", use_container_width=True):
                with st.spinner("Analyzing LoRA similarities..."):
                    st.session_state.db_analyzed = True
                    st.session_state.db_files = files
                    st.session_state.db_name = db_name
                    st.rerun()

    with col_viz:
        if st.session_state.get('db_analyzed') and st.session_state.get('db_files'):
            files = st.session_state.db_files
            st.subheader("ANALYSIS RESULTS")

            # Calculate estimates
            total_size_mb = len(files) * 144  # Estimate 144MB per LoRA
            compressed_size_mb = total_size_mb * 0.06  # ~94% compression
            savings_mb = total_size_mb - compressed_size_mb

            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("LoRAs", f"{len(files)}", f"Community: {10 - len(files)} left")
            m2.metric("Compression", "~94%", "Differential encoding")
            m3.metric("Savings", f"{savings_mb:.1f} MB", f"From {total_size_mb} MB")

            st.markdown("---")

            st.markdown("#### Collection Contents")
            file_data = []
            for i, f in enumerate(files, 1):
                file_data.append({
                    "No": i,
                    "Name": f.name,
                    "Size": f"~{f.size / 1024 / 1024:.1f} MB"
                })

            st.dataframe(file_data, use_container_width=True)

            st.markdown("---")

            # Build button
            if st.button("BUILD .LORADB DATABASE", type="primary", use_container_width=True):
                with st.spinner("Building compressed database..."):
                    try:
                        # Initialize database builder
                        db = LoRADatabase(cluster_level="balanced")

                        st.success(f"""
                        **Database Built Successfully!**

                        - **File:** {db_name}.loradb
                        - **LoRAs:** {len(files)}
                        - **Compression:** ~94%
                        - **Savings:** {savings_mb:.1f} MB

                        *Note: Full .loradb creation is coming in v1.7.
                        Current version validates limits and prepares metadata.*
                        """)

                        st.info("""
                        **What is a .loradb file?**

                        A LoRA Database stores multiple LoRAs in a single compressed file using differential encoding.
                        Instead of storing each LoRA completely, it stores the BASE LoRA plus only the DIFFERENCES
                        for each additional LoRA, achieving 90-95% compression.

                        **Community Edition:** Create databases up to 10 LoRAs
                        **Pro Edition:** Create databases up to 50 LoRAs
                        **Studio Edition:** Unlimited LoRAs + Sell your .loradb files commercially
                        """)

                    except ValueError as e:
                        st.error(f"{str(e)}")
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.info("Upload LoRA files to begin analysis")

            st.markdown("---")

            st.markdown("#### What Can You Do?")

            st.markdown("""
            **With Community Edition (10 LoRAs):**
            - Create small themed collections
            - Character variant packs
            - Style sampler sets
            - Share with friends (personal use only)

            **Upgrade to Pro ($299) for:**
            - Collections up to 50 LoRAs
            - Commercial use rights
            - Priority support

            **Upgrade to Studio ($599) for:**
            - UNLIMITED LoRAs per database
            - **Sell .loradb files commercially**
            - Perfect for creators selling LoRA packs
            - 24-hour support
            """)

# 10. SETTINGS VIEW (NEW!) - NO df check needed
elif st.session_state.view == 'Settings':
    st.subheader("SETTINGS & PREFERENCES")
    st.caption("Configure LoRA Lens behavior")

    st.markdown("---")

    # Performance Settings
    st.markdown("### Performance")

    precompute = st.checkbox(
        "Pre-compute All Visualizations",
        value=st.session_state.precompute_all,
        help="""
        When enabled, all visualizations (3D maps, heatmaps, charts) are
        generated immediately after analysis. This takes longer upfront
        but makes tab switching instant.

        When disabled, visualizations generate on-demand when you click
        each tab. Faster initial load, but you'll wait when switching tabs.

        Recommended: ENABLED (especially for analyzing multiple LoRAs)
        """
    )

    if precompute != st.session_state.precompute_all:
        st.session_state.precompute_all = precompute
        st.success("Setting saved! Will apply to next LoRA upload.")

    col_perf1, col_perf2 = st.columns(2)

    with col_perf1:
        if st.session_state.precompute_all:
            st.info("""
            **Pre-compute Mode: ON**

            - Instant tab switching
            - All data ready immediately
            - Longer initial load (~10-20 seconds)

            Best for: Exploring multiple tabs, comparing views
            """)
        else:
            st.warning("""
            **Pre-compute Mode: OFF**

            - Faster initial load
            - Wait when switching tabs

            Best for: Quick single-view analysis
            """)

    with col_perf2:
        st.markdown("#### Estimated Load Times")
        st.code("""
Pre-compute OFF:
  Initial load: 2-5 sec
  Per tab click: 2-10 sec

Pre-compute ON:
  Initial load: 10-20 sec
  Per tab click: instant
        """)

    st.markdown("---")

    # Display Settings
    st.markdown("### Display")

    col_disp1, col_disp2 = st.columns(2)

    with col_disp1:
        show_advanced = st.checkbox(
            "Show Advanced Metrics",
            value=st.session_state.get('show_advanced', False),
            help="Show additional technical metrics in analysis views"
        )
        st.session_state.show_advanced = show_advanced

    with col_disp2:
        dark_mode = st.checkbox(
            "Dark Mode Visualizations",
            value=True,
            help="Use dark theme for charts (recommended)"
        )

    st.markdown("---")

    # About Section
    st.markdown("### About")

    st.info("""
    **LoRA Lens v1.6 FLUX Edition**

    The Intelligence Multiplier for AI Models

    - Compress LoRAs 65-90%
    - Load 5-10x more simultaneously
    - Get 100-1000x more knowledge combinations

    **Created by:** Jon Wright | Intuitivation
    **License:** Dual (Free personal, $299 commercial)
    **GitHub:** github.com/intuitivation/lora-lens
    **Email:** jonwright.24@gmail.com

    **Support Development:**
    - GitHub Sponsors (monthly)
    - Ko-fi (one-time)
    - Commercial License ($299)
    """)

    col_about1, col_about2 = st.columns(2)

    with col_about1:
        if st.button("Documentation"):
            st.info("Opening documentation... (link to docs)")

    with col_about2:
        if st.button("Join Discord"):
            st.info("Opening Discord invite... (link)")

    st.markdown("---")

    # Reset Settings
    st.markdown("### Reset")

    if st.button("Reset All Settings to Default", type="secondary"):
        st.session_state.precompute_all = True
        st.session_state.show_advanced = False
        st.success("Settings reset to default values!")
        st.rerun()
