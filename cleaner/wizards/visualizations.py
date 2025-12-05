"""
Visualization utilities for data cleaning wizard
Generates plots and charts for data analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple
import io
import base64


def generate_missing_heatmap(df: pd.DataFrame, output_path: Optional[str] = None) -> str:
    """
    Generate heatmap of missing values
    
    Args:
        df: DataFrame to analyze
        output_path: Optional path to save figure
    
    Returns:
        Path to saved figure or base64 encoded image
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create missing value matrix
    missing_matrix = df.isnull()
    
    # Plot heatmap
    sns.heatmap(missing_matrix, cmap='RdYlGn_r', cbar=True, 
                yticklabels=False, ax=ax)
    ax.set_title('Missing Values Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Columns')
    ax.set_ylabel('Rows')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        # Return as base64 for direct display
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf


def generate_distribution_plot(
    series: pd.Series,
    mean_val: Optional[float] = None,
    median_val: Optional[float] = None,
    mode_val: Optional[float] = None,
    title: str = "Distribution Plot"
) -> io.BytesIO:
    """
    Generate distribution plot with central tendency markers
    
    Args:
        series: Data series to plot
        mean_val: Mean value to mark
        median_val: Median value to mark
        mode_val: Mode value to mark
        title: Plot title
    
    Returns:
        BytesIO buffer with plot image
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot histogram
    ax.hist(series.dropna(), bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Add central tendency lines
    if mean_val is not None:
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    if median_val is not None:
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
    if mode_val is not None:
        ax.axvline(mode_val, color='orange', linestyle='--', linewidth=2, label=f'Mode: {mode_val:.2f}')
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf


def generate_boxplot(series: pd.Series, title: str = "Box Plot", outliers_highlighted: bool = False) -> io.BytesIO:
    """
    Generate box plot for outlier visualization
    
    Args:
        series: Data series
        title: Plot title
        outliers_highlighted: Whether to highlight outliers in red
    
    Returns:
        BytesIO buffer with plot image
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Create boxplot
    bp = ax.boxplot(series.dropna(), vert=False, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='blue'),
                    whiskerprops=dict(color='blue'),
                    capprops=dict(color='blue'),
                    medianprops=dict(color='red', linewidth=2))
    
    if outliers_highlighted:
        for flier in bp['fliers']:
            flier.set(marker='o', color='red', alpha=0.7, markersize=6)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Value')
    ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf


def generate_bar_chart(data: pd.Series, title: str = "Bar Chart", 
                       xlabel: str = "Category", ylabel: str = "Count",
                       top_n: int = None) -> io.BytesIO:
    """
    Generate bar chart for categorical data
    
    Args:
        data: Data series (categorical)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        top_n: Show only top N categories
    
    Returns:
        BytesIO buffer with plot image
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get value counts
    value_counts = data.value_counts()
    if top_n:
        value_counts = value_counts.head(top_n)
    
    # Create bar plot
    value_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf


def generate_comparison_chart(before: int, after: int, 
                              labels: Tuple[str, str] = ("Before", "After"),
                              title: str = "Comparison") -> io.BytesIO:
    """
    Generate before/after comparison bar chart
    
    Args:
        before: Before value
        after: After value
        labels: Labels for before and after
        title: Chart title
    
    Returns:
        BytesIO buffer with plot image
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    categories = list(labels)
    values = [before, after]
    colors = ['#ff6b6b' if before > after else '#51cf66', '#51cf66' if before > after else '#ff6b6b']
    
    bars = ax.bar(categories, values, color=colors, edgecolor='black', alpha=0.7)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:,}',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Count')
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf
