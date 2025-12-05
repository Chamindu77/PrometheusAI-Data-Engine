"""
Report generation module
Creates HTML and Markdown reports for EDA
"""

import pandas as pd
from typing import Dict, Any, List
from datetime import datetime
import os
import base64


def escape_html(text: str) -> str:
    """Escape HTML special characters"""
    return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def get_health_score_color(score: float) -> str:
    """Get color based on health score"""
    if score >= 80:
        return '#28a745'  # Green
    elif score >= 60:
        return '#ffc107'  # Yellow
    elif score >= 40:
        return '#fd7e14'  # Orange
    else:
        return '#dc3545'  # Red


def format_number(num: float, decimals: int = 2) -> str:
    """Format number for display"""
    if abs(num) >= 1_000_000:
        return f"{num/1_000_000:.{decimals}f}M"
    elif abs(num) >= 1_000:
        return f"{num/1_000:.{decimals}f}K"
    else:
        return f"{num:.{decimals}f}"


def generate_html_report(
    overview: Dict[str, Any],
    schema: Dict[str, Any],
    column_analysis: Dict[str, Dict[str, Any]],
    cleaning_log: Dict[str, Any],
    plot_metadata: List[Dict[str, str]],
    correlation_insights: Dict[str, Any] = None
) -> str:
    """
    Generate comprehensive HTML report
    
    Args:
        overview: Dataset overview from overview.py
        schema: Schema information from schema_infer.py
        column_analysis: Column-wise analysis
        cleaning_log: Cleaning operations log
        plot_metadata: List of plot metadata
        correlation_insights: Correlation analysis results
        
    Returns:
        HTML report string
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    health_score = overview.get('health_score', 0)
    health_color = get_health_score_color(health_score)
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>EDA Report - {timestamp}</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                background: #f5f5f5;
                padding: 20px;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }}
            
            h2 {{
                color: #34495e;
                margin-top: 30px;
                margin-bottom: 15px;
                padding-left: 10px;
                border-left: 4px solid #3498db;
            }}
            
            h3 {{
                color: #555;
                margin-top: 20px;
                margin-bottom: 10px;
            }}
            
            .header {{
                text-align: center;
                margin-bottom: 30px;
            }}
            
            .health-score {{
                display: inline-block;
                padding: 15px 30px;
                background: {health_color};
                color: white;
                border-radius: 50px;
                font-size: 24px;
                font-weight: bold;
                margin: 20px 0;
            }}
            
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            
            .stat-card {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #3498db;
            }}
            
            .stat-label {{
                color: #666;
                font-size: 14px;
                margin-bottom: 5px;
            }}
            
            .stat-value {{
                color: #2c3e50;
                font-size: 28px;
                font-weight: bold;
            }}
            
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background: white;
            }}
            
            th {{
                background: #3498db;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: 600;
            }}
            
            td {{
                padding: 10px 12px;
                border-bottom: 1px solid #ddd;
            }}
            
            tr:hover {{
                background: #f8f9fa;
            }}
            
            .warning {{
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 15px;
                margin: 10px 0;
                border-radius: 4px;
            }}
            
            .warning.high {{
                background: #f8d7da;
                border-left-color: #dc3545;
            }}
            
            .warning.critical {{
                background: #f8d7da;
                border-left-color: #721c24;
                color: #721c24;
            }}
            
            .action-badge {{
                display: inline-block;
                padding: 5px 10px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: 600;
                margin: 2px;
            }}
            
            .action-badge.success {{
                background: #d4edda;
                color: #155724;
            }}
            
            .plot-container {{
                margin: 20px 0;
                text-align: center;
            }}
            
            .plot-container img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 8px;
                margin: 10px 0;
            }}
            
            .timestamp {{
                color: #888;
                font-size: 14px;
                text-align: right;
                margin-top: 30px;
            }}
            
            .badge {{
                display: inline-block;
                padding: 3px 8px;
                border-radius: 3px;
                font-size: 11px;
                font-weight: 600;
                text-transform: uppercase;
            }}
            
            .badge-numeric {{ background: #e3f2fd; color: #1976d2; }}
            .badge-categorical {{ background: #f3e5f5; color: #7b1fa2; }}
            .badge-datetime {{ background: #e8f5e9; color: #388e3c; }}
            .badge-text {{ background: #fff3e0; color: #f57c00; }}
            
            ul {{
                margin: 10px 0;
                padding-left: 25px;
            }}
            
            li {{
                margin: 5px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Exploratory Data Analysis Report</h1>
                <p style="color: #666; font-size: 16px;">Generated on {timestamp}</p>
                <div class="health-score">
                    Dataset Health Score: {health_score:.1f}/100
                </div>
            </div>
            
            <h2>1. Executive Summary</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Total Rows</div>
                    <div class="stat-value">{overview['basic_info']['rows']:,}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total Columns</div>
                    <div class="stat-value">{overview['basic_info']['columns']}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Memory Usage</div>
                    <div class="stat-value">{overview['memory_usage']['total_mb']:.1f} MB</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Missing Values</div>
                    <div class="stat-value">{overview['missing_values']['total_missing_pct']:.1f}%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Duplicate Rows</div>
                    <div class="stat-value">{overview['duplicates']['total_duplicates']:,}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Numeric Features</div>
                    <div class="stat-value">{schema['summary']['numeric_columns']}</div>
                </div>
            </div>
    """
    
    # Warnings section
    if overview.get('warnings'):
        html += "<h2>2. Data Quality Warnings</h2>"
        for warning in overview['warnings']:
            severity_class = warning.get('severity', 'medium')
            html += f'<div class="warning {severity_class}">'
            html += f'<strong>{warning["severity"].upper()}:</strong> {escape_html(warning["message"])}'
            if 'columns' in warning:
                html += f'<br><small>Affected columns: {", ".join(warning["columns"][:5])}'
                if len(warning['columns']) > 5:
                    html += f' and {len(warning["columns"])-5} more'
                html += '</small>'
            html += '</div>'
    
    # Schema summary
    html += """
        <h2>3. Schema Overview</h2>
        <table>
            <tr>
                <th>Data Type</th>
                <th>Count</th>
                <th>Percentage</th>
            </tr>
    """
    
    total_cols = schema['summary']['total_columns']
    for dtype, count in schema['summary'].items():
        if dtype != 'total_columns' and isinstance(count, int):
            pct = (count / total_cols * 100) if total_cols > 0 else 0
            html += f"<tr><td>{dtype.replace('_', ' ').title()}</td><td>{count}</td><td>{pct:.1f}%</td></tr>"
    
    html += "</table>"
    
    # Column-wise analysis (top issues)
    html += "<h2>4. Column Analysis Summary</h2>"
    html += "<table><tr><th>Column</th><th>Type</th><th>Missing %</th><th>Unique Values</th><th>Key Issues</th></tr>"
    
    for col_name, col_data in list(column_analysis.items())[:30]:  # Limit to 30 columns
        detected_type = col_data.get('detected_type', 'unknown')
        analysis = col_data.get('analysis', {})
        missing_pct = analysis.get('missing_pct', 0)
        
        # Get unique count
        if 'unique_count' in analysis:
            unique = analysis['unique_count']
        else:
            unique = 'N/A'
        
        # Get key recommendations
        recommendations = col_data.get('recommendations', [])
        issues = '<br>'.join([f'• {r}' for r in recommendations[:2]])
        
        badge_class = f'badge-{detected_type}' if detected_type in ['numeric', 'categorical', 'datetime', 'text'] else ''
        
        html += f"""
        <tr>
            <td><strong>{escape_html(col_name)}</strong></td>
            <td><span class="badge {badge_class}">{detected_type}</span></td>
            <td>{missing_pct:.1f}%</td>
            <td>{unique}</td>
            <td style="font-size: 11px;">{issues if issues else '✓ No issues'}</td>
        </tr>
        """
    
    html += "</table>"
    
    # Cleaning actions
    if cleaning_log and cleaning_log.get('actions'):
        html += "<h2>5. Cleaning Actions Applied</h2>"
        html += "<table><tr><th>#</th><th>Action</th><th>Details</th><th>Shape After</th></tr>"
        
        for idx, action in enumerate(cleaning_log['actions'], 1):
            action_name = action.get('action', 'Unknown')
            details = action.get('details', {})
            shape = action.get('shape_after', (0, 0))
            
            details_str = ', '.join([f"{k}: {v}" for k, v in details.items() if k not in ['columns']])
            if 'columns' in details:
                cols = details['columns']
                if isinstance(cols, list) and len(cols) <= 3:
                    details_str += f", columns: {', '.join(map(str, cols))}"
                elif isinstance(cols, list):
                    details_str += f", columns: {len(cols)} affected"
            
            html += f"""
            <tr>
                <td>{idx}</td>
                <td><span class="action-badge success">{escape_html(action_name)}</span></td>
                <td style="font-size: 12px;">{escape_html(details_str)}</td>
                <td>{shape[0]:,} × {shape[1]}</td>
            </tr>
            """
        
        html += "</table>"
        
        # Statistics
        stats = cleaning_log.get('statistics', {})
        if stats:
            html += "<h3>Cleaning Statistics</h3><ul>"
            html += f"<li>Original Shape: {stats.get('original_shape', (0,0))[0]:,} rows × {stats.get('original_shape', (0,0))[1]} columns</li>"
            html += f"<li>Final Shape: {stats.get('final_shape', (0,0))[0]:,} rows × {stats.get('final_shape', (0,0))[1]} columns</li>"
            html += f"<li>Rows Removed: {stats.get('rows_removed', 0):,}</li>"
            html += f"<li>Columns Removed: {stats.get('columns_removed', 0)}</li>"
            html += f"<li>Memory Saved: {stats.get('memory_reduction_mb', 0):.2f} MB</li>"
            html += "</ul>"
    
    # Correlation insights
    if correlation_insights and not correlation_insights.get('error'):
        html += "<h2>6. Correlation Insights</h2>"
        html += f"<p>Found {correlation_insights['high_correlations_count']} high correlations (|r| ≥ {correlation_insights['threshold']})</p>"
        
        if correlation_insights.get('high_correlations'):
            html += "<table><tr><th>Feature 1</th><th>Feature 2</th><th>Correlation</th><th>Strength</th></tr>"
            for corr in correlation_insights['high_correlations'][:15]:
                html += f"""
                <tr>
                    <td>{escape_html(corr['feature_1'])}</td>
                    <td>{escape_html(corr['feature_2'])}</td>
                    <td>{corr['correlation']:.3f}</td>
                    <td>{corr['strength']}</td>
                </tr>
                """
            html += "</table>"
    
    # Visualizations
    if plot_metadata:
        html += "<h2>7. Visual Exploratory Analysis</h2>"
        
        # Group plots by type
        plots_by_type = {}
        for plot in plot_metadata:
            plot_type = plot.get('type', 'other')
            if plot_type not in plots_by_type:
                plots_by_type[plot_type] = []
            plots_by_type[plot_type].append(plot)
        
        # Show correlation and missing values first
        priority_types = ['correlation', 'missing_values', 'outliers', 'pairplot']
        
        for plot_type in priority_types:
            if plot_type in plots_by_type:
                html += f"<h3>{plot_type.replace('_', ' ').title()}</h3>"
                for plot in plots_by_type[plot_type]:
                    html += f"""
                    <div class="plot-container">
                        <h4>{escape_html(plot['title'])}</h4>
                        <img src="{escape_html(plot['filename'])}" alt="{escape_html(plot['title'])}">
                    </div>
                    """
        
        # Show sample of other plots
        for plot_type in ['distribution', 'categorical']:
            if plot_type in plots_by_type:
                html += f"<h3>{plot_type.title()} Plots (Sample)</h3>"
                for plot in plots_by_type[plot_type][:5]:  # Show first 5
                    html += f"""
                    <div class="plot-container">
                        <h4>{escape_html(plot['title'])}</h4>
                        <img src="{escape_html(plot['filename'])}" alt="{escape_html(plot['title'])}">
                    </div>
                    """
                
                remaining = len(plots_by_type[plot_type]) - 5
                if remaining > 0:
                    html += f"<p><em>... and {remaining} more {plot_type} plots (see output folder)</em></p>"
    
    # Footer
    html += f"""
            <div class="timestamp">
                <p>Report generated by EDA Engine on {timestamp}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html


def generate_markdown_report(
    overview: Dict[str, Any],
    schema: Dict[str, Any],
    column_analysis: Dict[str, Dict[str, Any]],
    cleaning_log: Dict[str, Any]
) -> str:
    """
    Generate Markdown report (GitHub-friendly)
    
    Args:
        overview: Dataset overview
        schema: Schema information
        column_analysis: Column-wise analysis
        cleaning_log: Cleaning operations log
        
    Returns:
        Markdown report string
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    md = f"""# 📊 Exploratory Data Analysis Report

**Generated:** {timestamp}  
**Dataset Health Score:** {overview.get('health_score', 0):.1f}/100

---

## 1. Executive Summary

| Metric | Value |
|--------|-------|
| Total Rows | {overview['basic_info']['rows']:,} |
| Total Columns | {overview['basic_info']['columns']} |
| Memory Usage | {overview['memory_usage']['total_mb']:.1f} MB |
| Missing Values | {overview['missing_values']['total_missing_pct']:.1f}% |
| Duplicate Rows | {overview['duplicates']['total_duplicates']:,} |
| Numeric Features | {schema['summary']['numeric_columns']} |
| Categorical Features | {schema['summary']['categorical_columns']} |

---

## 2. Data Quality Warnings

"""
    
    if overview.get('warnings'):
        for warning in overview['warnings']:
            severity_emoji = '🔴' if warning['severity'] == 'critical' else '🟡' if warning['severity'] == 'high' else '🔵'
            md += f"{severity_emoji} **{warning['severity'].upper()}**: {warning['message']}\n\n"
    else:
        md += "✅ No major warnings\n\n"
    
    md += """
---

## 3. Column Analysis

| Column | Type | Missing % | Unique | Key Issues |
|--------|------|-----------|---------|------------|
"""
    
    for col_name, col_data in list(column_analysis.items())[:20]:
        detected_type = col_data.get('detected_type', 'unknown')
        analysis = col_data.get('analysis', {})
        missing_pct = analysis.get('missing_pct', 0)
        unique = analysis.get('unique_count', 'N/A')
        
        recommendations = col_data.get('recommendations', [])
        issues = ' / '.join(recommendations[:2]) if recommendations else '✓ No issues'
        
        md += f"| {col_name} | {detected_type} | {missing_pct:.1f}% | {unique} | {issues} |\n"
    
    if cleaning_log and cleaning_log.get('actions'):
        md += "\n---\n\n## 4. Cleaning Actions Applied\n\n"
        for idx, action in enumerate(cleaning_log['actions'], 1):
            action_name = action.get('action', 'Unknown')
            md += f"{idx}. **{action_name}**\n"
    
    md += f"\n---\n\n*Report generated by EDA Engine on {timestamp}*\n"
    
    return md
