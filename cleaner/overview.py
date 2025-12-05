"""
Dataset overview and summary statistics
Generates comprehensive dataset health report
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


def get_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive dataset overview
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with overview statistics
    """
    overview = {
        'basic_info': {},
        'memory_usage': {},
        'missing_values': {},
        'duplicates': {},
        'data_types': {},
        'warnings': [],
        'health_score': 0.0
    }
    
    # Basic info
    overview['basic_info'] = {
        'rows': len(df),
        'columns': len(df.columns),
        'total_cells': len(df) * len(df.columns),
        'column_names': list(df.columns)
    }
    
    # Memory usage
    memory_bytes = df.memory_usage(deep=True)
    overview['memory_usage'] = {
        'total_mb': memory_bytes.sum() / (1024 * 1024),
        'total_bytes': int(memory_bytes.sum()),
        'per_column': {
            col: {
                'mb': memory_bytes[col] / (1024 * 1024),
                'bytes': int(memory_bytes[col])
            }
            for col in df.columns
        }
    }
    
    # Missing values
    missing_counts = df.isna().sum()
    missing_pcts = (missing_counts / len(df)) * 100
    
    overview['missing_values'] = {
        'total_missing': int(missing_counts.sum()),
        'total_missing_pct': (missing_counts.sum() / overview['basic_info']['total_cells']) * 100,
        'columns_with_missing': int((missing_counts > 0).sum()),
        'per_column': {
            col: {
                'count': int(missing_counts[col]),
                'percentage': float(missing_pcts[col])
            }
            for col in df.columns if missing_counts[col] > 0
        }
    }
    
    # Duplicates
    duplicate_rows = df.duplicated()
    overview['duplicates'] = {
        'total_duplicates': int(duplicate_rows.sum()),
        'duplicate_percentage': (duplicate_rows.sum() / len(df)) * 100,
        'unique_rows': len(df) - int(duplicate_rows.sum())
    }
    
    # Data types distribution
    dtype_counts = df.dtypes.value_counts()
    overview['data_types'] = {
        'distribution': {str(dtype): int(count) for dtype, count in dtype_counts.items()},
        'per_column': {col: str(dtype) for col, dtype in df.dtypes.items()}
    }
    
    # Generate warnings
    warnings = []
    
    # High missing value columns
    high_missing_cols = [col for col, info in overview['missing_values']['per_column'].items() 
                         if info['percentage'] > 50]
    if high_missing_cols:
        warnings.append({
            'severity': 'high',
            'category': 'missing_values',
            'message': f"{len(high_missing_cols)} column(s) have >50% missing values",
            'columns': high_missing_cols
        })
    
    # Very high missing value columns (potentially unusable)
    very_high_missing_cols = [col for col, info in overview['missing_values']['per_column'].items() 
                              if info['percentage'] > 90]
    if very_high_missing_cols:
        warnings.append({
            'severity': 'critical',
            'category': 'missing_values',
            'message': f"{len(very_high_missing_cols)} column(s) have >90% missing values (consider dropping)",
            'columns': very_high_missing_cols
        })
    
    # High duplicate percentage
    if overview['duplicates']['duplicate_percentage'] > 10:
        warnings.append({
            'severity': 'medium',
            'category': 'duplicates',
            'message': f"{overview['duplicates']['duplicate_percentage']:.1f}% of rows are duplicates"
        })
    
    # Large dataset warning
    if overview['memory_usage']['total_mb'] > 1000:
        warnings.append({
            'severity': 'medium',
            'category': 'performance',
            'message': f"Large dataset ({overview['memory_usage']['total_mb']:.1f} MB) - consider optimization"
        })
    
    # Empty columns
    empty_cols = [col for col in df.columns if df[col].isna().all()]
    if empty_cols:
        warnings.append({
            'severity': 'high',
            'category': 'data_quality',
            'message': f"{len(empty_cols)} column(s) are completely empty",
            'columns': empty_cols
        })
    
    # Single value columns
    single_value_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if single_value_cols:
        warnings.append({
            'severity': 'low',
            'category': 'data_quality',
            'message': f"{len(single_value_cols)} column(s) have only one unique value",
            'columns': single_value_cols
        })
    
    overview['warnings'] = warnings
    
    # Calculate health score (0-100)
    health_score = 100.0
    
    # Deduct for missing values
    health_score -= min(overview['missing_values']['total_missing_pct'], 30)
    
    # Deduct for duplicates
    health_score -= min(overview['duplicates']['duplicate_percentage'], 20)
    
    # Deduct for critical warnings
    critical_warnings = [w for w in warnings if w['severity'] == 'critical']
    health_score -= len(critical_warnings) * 10
    
    # Deduct for high warnings
    high_warnings = [w for w in warnings if w['severity'] == 'high']
    health_score -= len(high_warnings) * 5
    
    overview['health_score'] = max(0, min(100, health_score))
    
    return overview


def get_column_cardinality(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Get cardinality information for all columns
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with cardinality info per column
    """
    cardinality = {}
    
    for col in df.columns:
        unique_count = df[col].nunique()
        total_count = len(df[col].dropna())
        
        cardinality[col] = {
            'unique_count': unique_count,
            'total_count': total_count,
            'cardinality_ratio': unique_count / total_count if total_count > 0 else 0,
            'is_unique': unique_count == total_count,
            'is_constant': unique_count <= 1
        }
    
    return cardinality


def identify_id_columns(df: pd.DataFrame) -> List[str]:
    """
    Identify potential ID columns (unique identifiers)
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        List of column names that appear to be IDs
    """
    id_columns = []
    cardinality = get_column_cardinality(df)
    
    for col, info in cardinality.items():
        # Column is likely an ID if:
        # 1. It's unique (or nearly unique)
        # 2. Column name suggests it (contains 'id', 'key', 'index')
        name_suggests_id = any(keyword in col.lower() for keyword in ['id', 'key', 'index', 'code'])
        is_unique = info['cardinality_ratio'] > 0.95
        
        if is_unique and name_suggests_id:
            id_columns.append(col)
        elif info['is_unique'] and len(df) > 10:  # Unique but not obvious from name
            id_columns.append(col)
    
    return id_columns


def suggest_dataset_actions(overview: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Suggest actions based on overview
    
    Args:
        overview: Overview dictionary from get_dataset_overview
        
    Returns:
        List of suggested actions
    """
    suggestions = []
    
    # Missing values
    if overview['missing_values']['total_missing'] > 0:
        if overview['missing_values']['total_missing_pct'] > 30:
            suggestions.append({
                'priority': 'high',
                'action': 'Handle missing values',
                'description': f"{overview['missing_values']['total_missing_pct']:.1f}% of data is missing"
            })
        else:
            suggestions.append({
                'priority': 'medium',
                'action': 'Handle missing values',
                'description': f"{overview['missing_values']['columns_with_missing']} columns have missing values"
            })
    
    # Duplicates
    if overview['duplicates']['total_duplicates'] > 0:
        suggestions.append({
            'priority': 'high' if overview['duplicates']['duplicate_percentage'] > 5 else 'medium',
            'action': 'Remove duplicates',
            'description': f"{overview['duplicates']['total_duplicates']} duplicate rows found"
        })
    
    # Memory optimization
    if overview['memory_usage']['total_mb'] > 100:
        suggestions.append({
            'priority': 'medium',
            'action': 'Optimize memory usage',
            'description': f"Dataset uses {overview['memory_usage']['total_mb']:.1f} MB"
        })
    
    # Empty columns
    empty_col_warnings = [w for w in overview['warnings'] 
                          if w['category'] == 'data_quality' and 'empty' in w['message'].lower()]
    if empty_col_warnings:
        suggestions.append({
            'priority': 'high',
            'action': 'Drop empty columns',
            'description': empty_col_warnings[0]['message']
        })
    
    return suggestions
