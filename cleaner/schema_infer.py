"""
Schema inference and data type detection
Intelligently detects and converts column types
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from datetime import datetime


def infer_column_type(series: pd.Series, cardinality_threshold: float = 0.05) -> Dict[str, Any]:
    """
    Infer the best data type for a column
    
    Args:
        series: Pandas Series to analyze
        cardinality_threshold: Threshold for categorical detection (unique/total ratio)
        
    Returns:
        Dictionary with inferred type and metadata
    """
    result = {
        'column_name': series.name,
        'current_dtype': str(series.dtype),
        'inferred_type': None,
        'suggested_dtype': None,
        'confidence': 0.0,
        'conversion_possible': False,
        'metadata': {}
    }
    
    # Skip if all null
    if series.isna().all():
        result['inferred_type'] = 'null'
        result['confidence'] = 1.0
        return result
    
    # Get non-null subset
    non_null = series.dropna()
    n_unique = non_null.nunique()
    n_total = len(non_null)
    cardinality_ratio = n_unique / n_total if n_total > 0 else 0
    
    result['metadata']['unique_count'] = n_unique
    result['metadata']['non_null_count'] = n_total
    result['metadata']['cardinality_ratio'] = cardinality_ratio
    
    # Boolean detection
    if n_unique <= 2:
        unique_values = set(non_null.astype(str).str.lower().unique())
        bool_values = {'true', 'false', '1', '0', 'yes', 'no', 't', 'f', 'y', 'n'}
        if unique_values.issubset(bool_values):
            result['inferred_type'] = 'boolean'
            result['suggested_dtype'] = 'bool'
            result['confidence'] = 0.95
            result['conversion_possible'] = True
            return result
    
    # Try numeric conversion
    try:
        numeric = pd.to_numeric(non_null, errors='coerce')
        non_null_numeric = numeric.dropna()
        conversion_rate = len(non_null_numeric) / len(non_null) if len(non_null) > 0 else 0
        
        if conversion_rate > 0.95:  # 95% convertible
            # Check if integer
            if (non_null_numeric == non_null_numeric.astype(int)).all():
                result['inferred_type'] = 'integer'
                result['suggested_dtype'] = 'int64'
                result['confidence'] = conversion_rate
                result['conversion_possible'] = True
                result['metadata']['min'] = int(non_null_numeric.min())
                result['metadata']['max'] = int(non_null_numeric.max())
            else:
                result['inferred_type'] = 'float'
                result['suggested_dtype'] = 'float64'
                result['confidence'] = conversion_rate
                result['conversion_possible'] = True
                result['metadata']['min'] = float(non_null_numeric.min())
                result['metadata']['max'] = float(non_null_numeric.max())
            return result
    except:
        pass
    
    # Try datetime conversion
    try:
        # Try multiple datetime formats
        datetime_series = pd.to_datetime(non_null, errors='coerce', infer_datetime_format=True)
        non_null_datetime = datetime_series.dropna()
        conversion_rate = len(non_null_datetime) / len(non_null) if len(non_null) > 0 else 0
        
        if conversion_rate > 0.90:  # 90% convertible
            result['inferred_type'] = 'datetime'
            result['suggested_dtype'] = 'datetime64[ns]'
            result['confidence'] = conversion_rate
            result['conversion_possible'] = True
            result['metadata']['min_date'] = str(non_null_datetime.min())
            result['metadata']['max_date'] = str(non_null_datetime.max())
            return result
    except:
        pass
    
    # Categorical vs Text
    if cardinality_ratio < cardinality_threshold:
        result['inferred_type'] = 'categorical'
        result['suggested_dtype'] = 'category'
        result['confidence'] = 0.85
        result['conversion_possible'] = True
        result['metadata']['top_values'] = non_null.value_counts().head(5).to_dict()
    else:
        # Check if text (string length analysis)
        avg_length = non_null.astype(str).str.len().mean()
        result['inferred_type'] = 'text'
        result['suggested_dtype'] = 'object'
        result['confidence'] = 0.80
        result['conversion_possible'] = False
        result['metadata']['avg_length'] = avg_length
    
    return result


def infer_schema(df: pd.DataFrame, cardinality_threshold: float = 0.05) -> Dict[str, Any]:
    """
    Infer schema for entire DataFrame
    
    Args:
        df: DataFrame to analyze
        cardinality_threshold: Threshold for categorical detection
        
    Returns:
        Dictionary with schema information
    """
    schema = {
        'columns': {},
        'summary': {
            'total_columns': len(df.columns),
            'numeric_columns': 0,
            'categorical_columns': 0,
            'datetime_columns': 0,
            'text_columns': 0,
            'boolean_columns': 0,
            'null_columns': 0
        },
        'conversion_recommendations': []
    }
    
    for col in df.columns:
        col_info = infer_column_type(df[col], cardinality_threshold)
        schema['columns'][col] = col_info
        
        # Update summary
        inferred_type = col_info['inferred_type']
        if inferred_type == 'integer' or inferred_type == 'float':
            schema['summary']['numeric_columns'] += 1
        elif inferred_type == 'categorical':
            schema['summary']['categorical_columns'] += 1
        elif inferred_type == 'datetime':
            schema['summary']['datetime_columns'] += 1
        elif inferred_type == 'text':
            schema['summary']['text_columns'] += 1
        elif inferred_type == 'boolean':
            schema['summary']['boolean_columns'] += 1
        elif inferred_type == 'null':
            schema['summary']['null_columns'] += 1
        
        # Add conversion recommendation
        if col_info['conversion_possible'] and col_info['confidence'] > 0.85:
            schema['conversion_recommendations'].append({
                'column': col,
                'from': col_info['current_dtype'],
                'to': col_info['suggested_dtype'],
                'confidence': col_info['confidence']
            })
    
    return schema


def apply_schema_conversions(df: pd.DataFrame, schema: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply recommended schema conversions
    
    Args:
        df: DataFrame to convert
        schema: Schema information from infer_schema
        
    Returns:
        Tuple of (converted DataFrame, list of conversion logs)
    """
    df_converted = df.copy()
    conversion_log = []
    
    for col, col_info in schema['columns'].items():
        if not col_info['conversion_possible']:
            continue
        
        try:
            inferred_type = col_info['inferred_type']
            
            if inferred_type == 'integer':
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce').astype('Int64')
                conversion_log.append(f"✓ Converted '{col}' to integer")
                
            elif inferred_type == 'float':
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
                conversion_log.append(f"✓ Converted '{col}' to float")
                
            elif inferred_type == 'datetime':
                df_converted[col] = pd.to_datetime(df_converted[col], errors='coerce')
                conversion_log.append(f"✓ Converted '{col}' to datetime")
                
            elif inferred_type == 'boolean':
                # Map common boolean values
                bool_map = {
                    'true': True, 'false': False,
                    '1': True, '0': False,
                    'yes': True, 'no': False,
                    't': True, 'f': False,
                    'y': True, 'n': False
                }
                df_converted[col] = df_converted[col].astype(str).str.lower().map(bool_map)
                conversion_log.append(f"✓ Converted '{col}' to boolean")
                
            elif inferred_type == 'categorical':
                df_converted[col] = df_converted[col].astype('category')
                conversion_log.append(f"✓ Converted '{col}' to category")
                
        except Exception as e:
            conversion_log.append(f"✗ Failed to convert '{col}': {str(e)}")
    
    return df_converted, conversion_log


def optimize_dtypes(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Optimize DataFrame dtypes to reduce memory usage
    
    Args:
        df: DataFrame to optimize
        
    Returns:
        Tuple of (optimized DataFrame, optimization report)
    """
    df_optimized = df.copy()
    report = {
        'original_memory_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
        'optimized_memory_mb': 0,
        'memory_reduction_mb': 0,
        'memory_reduction_pct': 0,
        'optimizations': []
    }
    
    for col in df_optimized.columns:
        col_type = df_optimized[col].dtype
        
        # Optimize integers
        if pd.api.types.is_integer_dtype(col_type):
            c_min = df_optimized[col].min()
            c_max = df_optimized[col].max()
            
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df_optimized[col] = df_optimized[col].astype(np.int8)
                report['optimizations'].append(f"{col}: int64 → int8")
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df_optimized[col] = df_optimized[col].astype(np.int16)
                report['optimizations'].append(f"{col}: int64 → int16")
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df_optimized[col] = df_optimized[col].astype(np.int32)
                report['optimizations'].append(f"{col}: int64 → int32")
        
        # Optimize floats
        elif pd.api.types.is_float_dtype(col_type):
            c_min = df_optimized[col].min()
            c_max = df_optimized[col].max()
            
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df_optimized[col] = df_optimized[col].astype(np.float32)
                report['optimizations'].append(f"{col}: float64 → float32")
        
        # Convert object to category if beneficial
        elif col_type == 'object':
            num_unique = df_optimized[col].nunique()
            num_total = len(df_optimized[col])
            if num_unique / num_total < 0.5:  # Less than 50% unique values
                df_optimized[col] = df_optimized[col].astype('category')
                report['optimizations'].append(f"{col}: object → category")
    
    report['optimized_memory_mb'] = df_optimized.memory_usage(deep=True).sum() / (1024 * 1024)
    report['memory_reduction_mb'] = report['original_memory_mb'] - report['optimized_memory_mb']
    report['memory_reduction_pct'] = (report['memory_reduction_mb'] / report['original_memory_mb']) * 100
    
    return df_optimized, report
