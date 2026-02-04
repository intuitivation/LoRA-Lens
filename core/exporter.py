import json
import pandas as pd
import datetime
from io import BytesIO
import base64

def export_results(df, filename="analysis_report"):
    """Export analysis results in multiple formats."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    
    # Export JSON (Full Forensic Data)
    json_path = f"{filename}_{timestamp}.json"
    df.to_json(json_path, orient="records", indent=4)
    
    # Export CSV (Spreadsheet for logging)
    csv_path = f"{filename}_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    return json_path, csv_path


def generate_html_report(df, engine, insights=None):
    """
    Generate comprehensive HTML report with embedded charts.
    
    Args:
        df: Analysis dataframe
        engine: LoRALensEngine instance
        insights: List of AI insights (optional)
    
    Returns:
        HTML string
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate summary statistics
    avg_eff = df['eff_rank'].mean()
    avg_declared = df['declared_rank'].mean()
    avg_sparsity = df['sparsity'].mean()
    health_score = engine.get_efficiency_score() if engine else "N/A"
    
    # Build HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>LoRA Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #0b0e14 0%, #1a1c23 100%);
            color: #00ffcc;
        }}
        .header {{
            text-align: center;
            padding: 30px;
            background: #1a1c23;
            border: 2px solid #00ffcc;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 0 20px #00ffcc33;
        }}
        h1 {{
            color: #00ffcc;
            text-shadow: 0 0 10px #00ffcc88;
            margin: 0;
        }}
        .timestamp {{
            color: #888;
            font-size: 0.9em;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #1a1c23;
            border: 1px solid #00ffcc44;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #00ffcc;
            margin: 10px 0;
        }}
        .metric-label {{
            color: #888;
            font-size: 0.9em;
        }}
        .section {{
            background: #1a1c23;
            border: 1px solid #00ffcc44;
            border-radius: 8px;
            padding: 25px;
            margin-bottom: 25px;
        }}
        h2 {{
            color: #00ffcc;
            border-bottom: 2px solid #00ffcc44;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th {{
            background: #00ffcc22;
            color: #00ffcc;
            padding: 12px;
            text-align: left;
            border: 1px solid #00ffcc44;
        }}
        td {{
            padding: 10px;
            border: 1px solid #00ffcc22;
        }}
        tr:nth-child(even) {{
            background: #ffffff05;
        }}
        .insight {{
            background: #00ffcc11;
            border-left: 4px solid #00ffcc;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        .warning {{
            background: #ffcc0011;
            border-left: 4px solid #ffcc00;
        }}
        .footer {{
            text-align: center;
            color: #666;
            padding: 20px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🛰️ LoRA LENS // FORENSIC REPORT</h1>
        <p class="timestamp">Generated: {timestamp}</p>
        <p>File: {engine.path if engine else 'N/A'}</p>
    </div>
    
    <div class="summary-grid">
        <div class="metric-card">
            <div class="metric-label">HEALTH SCORE</div>
            <div class="metric-value">{health_score}</div>
            <div class="metric-label">out of 100</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">AVG EFFICIENCY</div>
            <div class="metric-value">{avg_eff:.1f}</div>
            <div class="metric-label">of {avg_declared:.0f} declared</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">SPARSITY</div>
            <div class="metric-value">{avg_sparsity*100:.1f}%</div>
            <div class="metric-label">weight optimization</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">TOTAL LAYERS</div>
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">analyzed</div>
        </div>
    </div>
    
    <div class="section">
        <h2>📊 Executive Summary</h2>
        <p>This LoRA has been analyzed across {len(df)} layers. The effective rank indicates how much of the declared rank is actually being utilized by the model.</p>
        <ul>
            <li><strong>Rank Efficiency:</strong> {(avg_eff/avg_declared)*100:.1f}% - {'Excellent' if avg_eff/avg_declared < 0.4 else 'Good' if avg_eff/avg_declared < 0.7 else 'Needs Optimization'}</li>
            <li><strong>File Optimization Potential:</strong> {(1 - df['optimal_rank'].mean()/avg_declared)*100:.1f}% size reduction possible</li>
            <li><strong>Training Quality:</strong> {'High sparsity indicates good training' if avg_sparsity > 0.5 else 'Low sparsity may indicate overfitting'}</li>
        </ul>
    </div>
"""
    
    # Add AI insights if provided
    if insights and len(insights) > 0:
        html += """
    <div class="section">
        <h2>🤖 AI Consultant Analysis</h2>
"""
        for insight in insights:
            css_class = "warning" if any(x in insight for x in ["🔴", "⚠️", "WARNING"]) else "insight"
            html += f'        <div class="insight {css_class}">{insight}</div>\n'
        html += "    </div>\n"
    
    # Layer details table
    html += """
    <div class="section">
        <h2>🔬 Layer-by-Layer Analysis</h2>
        <table>
            <thead>
                <tr>
                    <th>Layer</th>
                    <th>Type</th>
                    <th>Magnitude</th>
                    <th>Eff. Rank</th>
                    <th>Optimal Rank</th>
                    <th>Declared Rank</th>
                    <th>Sparsity</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for _, row in df.iterrows():
        layer_name = row['layer'].split('.')[-1] if '.' in row['layer'] else row['layer']
        html += f"""
                <tr>
                    <td>{layer_name}</td>
                    <td>{row.get('layer_type', 'N/A')}</td>
                    <td>{row['magnitude']:.2f}</td>
                    <td>{row['eff_rank']}</td>
                    <td>{row.get('optimal_rank', 'N/A')}</td>
                    <td>{row['declared_rank']}</td>
                    <td>{row['sparsity']*100:.1f}%</td>
                </tr>
"""
    
    html += """
            </tbody>
        </table>
    </div>
    
    <div class="section">
        <h2>💡 Recommendations</h2>
        <div class="insight">
            <strong>Optimal Rank for Retraining:</strong> {optimal_rank}
        </div>
        <div class="insight">
            <strong>Expected File Size Reduction:</strong> {reduction}%
        </div>
        <div class="insight">
            <strong>Training Quality:</strong> {quality}
        </div>
    </div>
    
    <div class="footer">
        <p>Generated by LoRA Lens v1.5 // Neural Forensics Platform</p>
        <p>For questions or issues, refer to the documentation</p>
    </div>
</body>
</html>
""".format(
        optimal_rank=int(df['optimal_rank'].mean()),
        reduction=int((1 - df['optimal_rank'].mean()/avg_declared)*100),
        quality="Excellent - High sparsity and efficiency" if avg_sparsity > 0.5 and avg_eff/avg_declared < 0.5 else "Good" if avg_sparsity > 0.3 else "Needs Improvement - Consider adjusting training parameters"
    )
    
    return html


def save_html_report(df, engine, insights=None, filename="lora_report"):
    """Save HTML report to file."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    html_path = f"{filename}_{timestamp}.html"
    
    html_content = generate_html_report(df, engine, insights)
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return html_path
