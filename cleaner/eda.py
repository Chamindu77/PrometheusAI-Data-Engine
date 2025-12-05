"""
Exploratory Data Analysis (EDA) visualization module
Generates comprehensive visualizations for data analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class EDAVisualizer:
    """Generate and save EDA visualizations"""
    
    def __init__(self, df: pd.DataFrame, output_dir: str = 'output'):
        """
        Initialize visualizer
        
        Args:
            df: DataFrame to visualize
            output_dir: Directory to save plots
        """
        self.df = df
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.plot_paths = []
        self.plot_metadata = []
    
    def save_plot(self, filename: str, title: str, plot_type: str):
        """Save current plot and record metadata"""
        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        
        self.plot_paths.append(filepath)
        self.plot_metadata.append({
            'filename': filename,
            'filepath': filepath,
            'title': title,
            'type': plot_type
        })
    
    def plot_numeric_distributions(self, max_cols: int = 20) -> List[str]:
        """
        Plot histograms for numeric columns
        
        Args:
            max_cols: Maximum number of columns to plot
            
        Returns:
            List of plot filenames
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = numeric_cols[:max_cols]
        
        if not numeric_cols:
            return []
        
        plots = []
        
        for col in numeric_cols:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Histogram
            axes[0].hist(self.df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
            axes[0].set_title(f'Distribution of {col}')
            axes[0].set_xlabel(col)
            axes[0].set_ylabel('Frequency')
            axes[0].grid(True, alpha=0.3)
            
            # Box plot
            axes[1].boxplot(self.df[col].dropna(), vert=True)
            axes[1].set_title(f'Box Plot of {col}')
            axes[1].set_ylabel(col)
            axes[1].grid(True, alpha=0.3)
            
            filename = f'dist_{col}.png'
            self.save_plot(filename, f'Distribution Analysis: {col}', 'distribution')
            plots.append(filename)
        
        return plots
    
    def plot_categorical_distributions(self, max_cols: int = 15, max_categories: int = 10) -> List[str]:
        """
        Plot bar charts for categorical columns
        
        Args:
            max_cols: Maximum number of columns to plot
            max_categories: Maximum categories per plot
            
        Returns:
            List of plot filenames
        """
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_cols = categorical_cols[:max_cols]
        
        if not categorical_cols:
            return []
        
        plots = []
        
        for col in categorical_cols:
            value_counts = self.df[col].value_counts().head(max_categories)
            
            if len(value_counts) == 0:
                continue
            
            plt.figure(figsize=(10, 6))
            value_counts.plot(kind='bar', edgecolor='black', alpha=0.7)
            plt.title(f'Top {len(value_counts)} Categories in {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            
            filename = f'cat_{col}.png'
            self.save_plot(filename, f'Category Distribution: {col}', 'categorical')
            plots.append(filename)
        
        return plots
    
    def plot_correlation_matrix(self, method: str = 'pearson', min_cols: int = 2) -> Optional[str]:
        """
        Plot correlation heatmap
        
        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall')
            min_cols: Minimum numeric columns required
            
        Returns:
            Plot filename or None
        """
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < min_cols:
            return None
        
        # Limit to prevent overcrowding
        if len(numeric_df.columns) > 20:
            numeric_df = numeric_df.iloc[:, :20]
        
        corr_matrix = numeric_df.corr(method=method)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )
        plt.title(f'Correlation Matrix ({method.capitalize()})')
        plt.tight_layout()
        
        filename = 'correlation_matrix.png'
        self.save_plot(filename, f'Correlation Matrix ({method})', 'correlation')
        
        return filename
    
    def plot_missing_values_heatmap(self) -> Optional[str]:
        """
        Plot missing values heatmap
        
        Returns:
            Plot filename or None
        """
        missing_data = self.df.isna()
        
        if not missing_data.any().any():
            return None  # No missing values
        
        # Only show columns with missing values
        cols_with_missing = missing_data.columns[missing_data.any()].tolist()
        
        if not cols_with_missing:
            return None
        
        # Limit rows for visualization
        sample_size = min(100, len(self.df))
        missing_sample = missing_data[cols_with_missing].head(sample_size)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            missing_sample.T,
            cmap='RdYlGn_r',
            cbar_kws={'label': 'Missing'},
            yticklabels=True,
            xticklabels=False
        )
        plt.title(f'Missing Values Pattern (First {sample_size} rows)')
        plt.xlabel('Rows')
        plt.ylabel('Columns')
        
        filename = 'missing_values_heatmap.png'
        self.save_plot(filename, 'Missing Values Pattern', 'missing_values')
        
        return filename
    
    def plot_missing_values_bar(self) -> Optional[str]:
        """
        Plot missing values as bar chart
        
        Returns:
            Plot filename or None
        """
        missing_counts = self.df.isna().sum()
        missing_pcts = (missing_counts / len(self.df)) * 100
        
        cols_with_missing = missing_pcts[missing_pcts > 0].sort_values(ascending=False)
        
        if len(cols_with_missing) == 0:
            return None
        
        plt.figure(figsize=(10, 6))
        cols_with_missing.plot(kind='barh', edgecolor='black', alpha=0.7)
        plt.title('Missing Values by Column (%)')
        plt.xlabel('Missing Percentage')
        plt.ylabel('Columns')
        plt.grid(True, alpha=0.3, axis='x')
        
        filename = 'missing_values_bar.png'
        self.save_plot(filename, 'Missing Values Summary', 'missing_values')
        
        return filename
    
    def plot_pairplot(self, max_features: int = 5, sample_size: int = 1000) -> Optional[str]:
        """
        Plot pairplot for numeric features
        
        Args:
            max_features: Maximum features to include
            sample_size: Maximum rows to sample
            
        Returns:
            Plot filename or None
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return None
        
        # Select top features by variance
        numeric_df = self.df[numeric_cols].dropna()
        variances = numeric_df.var().sort_values(ascending=False)
        top_features = variances.head(max_features).index.tolist()
        
        # Sample data if too large
        if len(numeric_df) > sample_size:
            numeric_df = numeric_df.sample(n=sample_size, random_state=42)
        
        subset = numeric_df[top_features]
        
        pairplot = sns.pairplot(subset, diag_kind='kde', plot_kws={'alpha': 0.6})
        pairplot.fig.suptitle(f'Pair Plot (Top {len(top_features)} Features)', y=1.01)
        
        filename = 'pairplot.png'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        
        self.plot_paths.append(filepath)
        self.plot_metadata.append({
            'filename': filename,
            'filepath': filepath,
            'title': 'Pair Plot',
            'type': 'pairplot'
        })
        
        return filename
    
    def plot_outliers_summary(self, method: str = 'iqr', max_cols: int = 10) -> Optional[str]:
        """
        Plot outlier counts for numeric columns
        
        Args:
            method: Outlier detection method ('iqr' or 'zscore')
            max_cols: Maximum columns to show
            
        Returns:
            Plot filename or None
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return None
        
        outlier_counts = {}
        
        for col in numeric_cols:
            if method == 'iqr':
                q1 = self.df[col].quantile(0.25)
                q3 = self.df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
            else:  # zscore
                from scipy import stats
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                outliers = (z_scores > 3).sum()
            
            if outliers > 0:
                outlier_counts[col] = outliers
        
        if not outlier_counts:
            return None
        
        # Sort and limit
        outlier_series = pd.Series(outlier_counts).sort_values(ascending=False).head(max_cols)
        
        plt.figure(figsize=(10, 6))
        outlier_series.plot(kind='barh', edgecolor='black', color='coral', alpha=0.7)
        plt.title(f'Outlier Counts by Column ({method.upper()} method)')
        plt.xlabel('Number of Outliers')
        plt.ylabel('Columns')
        plt.grid(True, alpha=0.3, axis='x')
        
        filename = f'outliers_summary_{method}.png'
        self.save_plot(filename, f'Outlier Summary ({method})', 'outliers')
        
        return filename
    
    def generate_all_plots(self) -> Dict[str, Any]:
        """
        Generate all available plots
        
        Returns:
            Dictionary with plot metadata
        """
        report = {
            'total_plots': 0,
            'plots_by_type': {},
            'plot_metadata': []
        }
        
        print("Generating numeric distributions...")
        self.plot_numeric_distributions()
        
        print("Generating categorical distributions...")
        self.plot_categorical_distributions()
        
        print("Generating correlation matrix...")
        self.plot_correlation_matrix()
        
        print("Generating missing values visualizations...")
        self.plot_missing_values_heatmap()
        self.plot_missing_values_bar()
        
        print("Generating outlier summary...")
        self.plot_outliers_summary()
        
        print("Generating pairplot...")
        self.plot_pairplot()
        
        report['total_plots'] = len(self.plot_metadata)
        report['plot_metadata'] = self.plot_metadata
        
        # Group by type
        for plot in self.plot_metadata:
            plot_type = plot['type']
            if plot_type not in report['plots_by_type']:
                report['plots_by_type'][plot_type] = []
            report['plots_by_type'][plot_type].append(plot['filename'])
        
        print(f"\nGenerated {report['total_plots']} plots in {self.output_dir}")
        
        return report


def get_correlation_insights(df: pd.DataFrame, threshold: float = 0.7) -> Dict[str, Any]:
    """
    Get insights from correlation analysis
    
    Args:
        df: DataFrame to analyze
        threshold: Correlation threshold for highlighting
        
    Returns:
        Dictionary with correlation insights
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        return {'error': 'Not enough numeric columns'}
    
    corr_matrix = numeric_df.corr()
    
    # Find high correlations
    high_correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:
                high_correlations.append({
                    'feature_1': corr_matrix.columns[i],
                    'feature_2': corr_matrix.columns[j],
                    'correlation': float(corr_value),
                    'strength': 'strong positive' if corr_value >= threshold else 'strong negative'
                })
    
    # Sort by absolute correlation
    high_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    return {
        'total_numeric_features': len(numeric_df.columns),
        'high_correlations_count': len(high_correlations),
        'threshold': threshold,
        'high_correlations': high_correlations[:20]  # Top 20
    }
