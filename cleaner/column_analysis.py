"""
Per-column analysis module
Detailed statistical analysis for each column
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from scipy import stats


def analyze_numeric_column(series: pd.Series) -> Dict[str, Any]:
    """
    Analyze numeric column
    
    Args:
        series: Numeric series to analyze
        
    Returns:
        Dictionary with numeric statistics
    """
    non_null = series.dropna()
    
    if len(non_null) == 0:
        return {'error': 'No non-null values'}
    
    analysis = {
        'count': len(non_null),
        'missing': int(series.isna().sum()),
        'missing_pct': (series.isna().sum() / len(series)) * 100,
        'mean': float(non_null.mean()),
        'median': float(non_null.median()),
        'std': float(non_null.std()),
        'min': float(non_null.min()),
        'max': float(non_null.max()),
        'q1': float(non_null.quantile(0.25)),
        'q3': float(non_null.quantile(0.75)),
        'iqr': float(non_null.quantile(0.75) - non_null.quantile(0.25)),
        'range': float(non_null.max() - non_null.min()),
        'variance': float(non_null.var()),
        'skewness': float(non_null.skew()) if len(non_null) > 2 else None,
        'kurtosis': float(non_null.kurtosis()) if len(non_null) > 3 else None,
    }
    
    # Outlier detection (IQR method)
    q1 = analysis['q1']
    q3 = analysis['q3']
    iqr = analysis['iqr']
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = non_null[(non_null < lower_bound) | (non_null > upper_bound)]
    analysis['outliers_count'] = len(outliers)
    analysis['outliers_pct'] = (len(outliers) / len(non_null)) * 100
    analysis['outlier_bounds'] = {'lower': lower_bound, 'upper': upper_bound}
    
    # Z-score outliers
    if len(non_null) > 2:
        z_scores = np.abs(stats.zscore(non_null))
        z_outliers = non_null[z_scores > 3]
        analysis['outliers_zscore_count'] = len(z_outliers)
        analysis['outliers_zscore_pct'] = (len(z_outliers) / len(non_null)) * 100
    
    # Distribution characteristics
    if analysis['skewness'] is not None:
        if abs(analysis['skewness']) < 0.5:
            analysis['distribution'] = 'approximately symmetric'
        elif analysis['skewness'] > 0:
            analysis['distribution'] = 'right-skewed (positive skew)'
        else:
            analysis['distribution'] = 'left-skewed (negative skew)'
    
    # Coefficient of variation
    if analysis['mean'] != 0:
        analysis['coefficient_of_variation'] = abs(analysis['std'] / analysis['mean'])
    
    return analysis


def analyze_categorical_column(series: pd.Series) -> Dict[str, Any]:
    """
    Analyze categorical column
    
    Args:
        series: Categorical series to analyze
        
    Returns:
        Dictionary with categorical statistics
    """
    non_null = series.dropna()
    
    if len(non_null) == 0:
        return {'error': 'No non-null values'}
    
    value_counts = non_null.value_counts()
    
    analysis = {
        'count': len(non_null),
        'missing': int(series.isna().sum()),
        'missing_pct': (series.isna().sum() / len(series)) * 100,
        'unique_count': non_null.nunique(),
        'cardinality_ratio': non_null.nunique() / len(non_null),
        'mode': str(value_counts.index[0]) if len(value_counts) > 0 else None,
        'mode_frequency': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
        'mode_percentage': (value_counts.iloc[0] / len(non_null)) * 100 if len(value_counts) > 0 else 0,
    }
    
    # Top 5 categories
    top_5 = value_counts.head(5)
    analysis['top_5_categories'] = [
        {
            'value': str(val),
            'count': int(count),
            'percentage': (count / len(non_null)) * 100
        }
        for val, count in top_5.items()
    ]
    
    # Entropy (measure of diversity)
    proportions = value_counts / len(non_null)
    entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
    analysis['entropy'] = float(entropy)
    
    # Concentration metrics
    analysis['top_1_concentration'] = analysis['mode_percentage']
    analysis['top_5_concentration'] = (top_5.sum() / len(non_null)) * 100
    
    # Is this effectively constant?
    analysis['is_constant'] = analysis['unique_count'] == 1
    
    return analysis


def analyze_datetime_column(series: pd.Series) -> Dict[str, Any]:
    """
    Analyze datetime column
    
    Args:
        series: Datetime series to analyze
        
    Returns:
        Dictionary with datetime statistics
    """
    non_null = series.dropna()
    
    if len(non_null) == 0:
        return {'error': 'No non-null values'}
    
    analysis = {
        'count': len(non_null),
        'missing': int(series.isna().sum()),
        'missing_pct': (series.isna().sum() / len(series)) * 100,
        'min_date': str(non_null.min()),
        'max_date': str(non_null.max()),
        'range_days': (non_null.max() - non_null.min()).days,
    }
    
    # Try to infer frequency
    if len(non_null) > 1:
        sorted_dates = non_null.sort_values()
        diffs = sorted_dates.diff().dropna()
        
        if len(diffs) > 0:
            median_diff = diffs.median()
            analysis['median_interval_days'] = median_diff.days
            
            # Guess frequency
            if median_diff.days < 1:
                analysis['inferred_frequency'] = 'hourly or sub-hourly'
            elif median_diff.days == 1:
                analysis['inferred_frequency'] = 'daily'
            elif 6 <= median_diff.days <= 8:
                analysis['inferred_frequency'] = 'weekly'
            elif 28 <= median_diff.days <= 31:
                analysis['inferred_frequency'] = 'monthly'
            elif 365 <= median_diff.days <= 366:
                analysis['inferred_frequency'] = 'yearly'
            else:
                analysis['inferred_frequency'] = 'irregular'
    
    # Extract date components if useful
    analysis['unique_years'] = non_null.dt.year.nunique()
    analysis['unique_months'] = non_null.dt.month.nunique()
    analysis['unique_days'] = non_null.dt.day.nunique()
    
    return analysis


def analyze_text_column(series: pd.Series) -> Dict[str, Any]:
    """
    Analyze text column
    
    Args:
        series: Text series to analyze
        
    Returns:
        Dictionary with text statistics
    """
    non_null = series.dropna().astype(str)
    
    if len(non_null) == 0:
        return {'error': 'No non-null values'}
    
    lengths = non_null.str.len()
    
    analysis = {
        'count': len(non_null),
        'missing': int(series.isna().sum()),
        'missing_pct': (series.isna().sum() / len(series)) * 100,
        'unique_count': non_null.nunique(),
        'avg_length': float(lengths.mean()),
        'min_length': int(lengths.min()),
        'max_length': int(lengths.max()),
        'median_length': float(lengths.median()),
    }
    
    # Word count (approximate)
    word_counts = non_null.str.split().str.len()
    analysis['avg_word_count'] = float(word_counts.mean())
    analysis['max_word_count'] = int(word_counts.max())
    
    # Check for common patterns
    has_numbers = non_null.str.contains(r'\d', regex=True, na=False).sum()
    has_special_chars = non_null.str.contains(r'[^a-zA-Z0-9\s]', regex=True, na=False).sum()
    
    analysis['contains_numbers_pct'] = (has_numbers / len(non_null)) * 100
    analysis['contains_special_chars_pct'] = (has_special_chars / len(non_null)) * 100
    
    return analysis


def analyze_column(series: pd.Series, column_type: str = None) -> Dict[str, Any]:
    """
    Analyze a single column with appropriate method based on type
    
    Args:
        series: Column to analyze
        column_type: Type hint ('numeric', 'categorical', 'datetime', 'text')
        
    Returns:
        Dictionary with analysis results
    """
    result = {
        'column_name': series.name,
        'dtype': str(series.dtype),
        'total_rows': len(series),
        'analysis': {}
    }
    
    # Auto-detect type if not provided
    if column_type is None:
        if pd.api.types.is_numeric_dtype(series):
            column_type = 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(series):
            column_type = 'datetime'
        elif pd.api.types.is_categorical_dtype(series) or pd.api.types.is_object_dtype(series):
            # Distinguish between categorical and text
            unique_ratio = series.nunique() / len(series)
            column_type = 'categorical' if unique_ratio < 0.5 else 'text'
    
    result['detected_type'] = column_type
    
    # Perform type-specific analysis
    if column_type == 'numeric':
        result['analysis'] = analyze_numeric_column(series)
    elif column_type == 'categorical':
        result['analysis'] = analyze_categorical_column(series)
    elif column_type == 'datetime':
        result['analysis'] = analyze_datetime_column(series)
    elif column_type == 'text':
        result['analysis'] = analyze_text_column(series)
    else:
        result['analysis'] = {'error': f'Unknown column type: {column_type}'}
    
    # Add recommendations
    result['recommendations'] = generate_column_recommendations(result)
    
    return result


def generate_column_recommendations(column_analysis: Dict[str, Any]) -> List[str]:
    """
    Generate cleaning recommendations for a column
    
    Args:
        column_analysis: Analysis results from analyze_column
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    analysis = column_analysis.get('analysis', {})
    col_type = column_analysis.get('detected_type')
    
    # Missing values
    missing_pct = analysis.get('missing_pct', 0)
    if missing_pct > 50:
        recommendations.append(f"⚠️ High missing values ({missing_pct:.1f}%) - consider dropping column")
    elif missing_pct > 10:
        recommendations.append(f"Consider imputing {missing_pct:.1f}% missing values")
    
    # Type-specific recommendations
    if col_type == 'numeric':
        outliers_pct = analysis.get('outliers_pct', 0)
        if outliers_pct > 5:
            recommendations.append(f"⚠️ {outliers_pct:.1f}% outliers detected - review and handle")
        
        skew = analysis.get('skewness')
        if skew and abs(skew) > 1:
            recommendations.append(f"Consider log/sqrt transformation (skewness: {skew:.2f})")
    
    elif col_type == 'categorical':
        unique_count = analysis.get('unique_count', 0)
        if analysis.get('is_constant'):
            recommendations.append("⚠️ Column has only one unique value - consider dropping")
        elif unique_count > 100:
            recommendations.append(f"High cardinality ({unique_count} categories) - consider grouping rare categories")
        
        top_1_conc = analysis.get('top_1_concentration', 0)
        if top_1_conc > 95:
            recommendations.append(f"⚠️ Heavily concentrated ({top_1_conc:.1f}% in top category)")
    
    elif col_type == 'datetime':
        if 'inferred_frequency' in analysis and analysis['inferred_frequency'] == 'irregular':
            recommendations.append("Irregular time intervals detected")
    
    elif col_type == 'text':
        unique_count = analysis.get('unique_count', 0)
        total = column_analysis.get('total_rows', 1)
        if unique_count / total > 0.95:
            recommendations.append("Nearly all values are unique - may be an identifier")
    
    if not recommendations:
        recommendations.append("✓ No major issues detected")
    
    return recommendations


def analyze_all_columns(df: pd.DataFrame, column_types: Dict[str, str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Analyze all columns in DataFrame
    
    Args:
        df: DataFrame to analyze
        column_types: Optional dict mapping column names to types
        
    Returns:
        Dictionary mapping column names to their analysis
    """
    column_types = column_types or {}
    
    results = {}
    for col in df.columns:
        col_type = column_types.get(col)
        results[col] = analyze_column(df[col], col_type)
    
    return results
