"""
I/O utilities for EDA Engine
Handles file reading, writing, and format detection
"""

import pandas as pd
import json
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from datetime import datetime


def detect_delimiter(file_path: str, sample_size: int = 5) -> str:
    """
    Auto-detect CSV delimiter by analyzing first few rows
    
    Args:
        file_path: Path to CSV file
        sample_size: Number of rows to sample
        
    Returns:
        Detected delimiter character
    """
    delimiters = [',', ';', '\t', '|']
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        sample_lines = [f.readline() for _ in range(sample_size)]
    
    delimiter_counts = {}
    for delimiter in delimiters:
        counts = [line.count(delimiter) for line in sample_lines if line.strip()]
        if counts and len(set(counts)) == 1 and counts[0] > 0:
            delimiter_counts[delimiter] = counts[0]
    
    if delimiter_counts:
        return max(delimiter_counts.items(), key=lambda x: x[1])[0]
    
    return ','  # default


def read_csv_smart(
    file_path: str,
    delimiter: Optional[str] = None,
    encoding: Optional[str] = None,
    preview_rows: int = 5
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Smart CSV reader with auto-detection and error handling
    
    Args:
        file_path: Path to CSV file
        delimiter: CSV delimiter (auto-detected if None)
        encoding: File encoding (auto-detected if None)
        preview_rows: Number of rows to preview
        
    Returns:
        Tuple of (DataFrame, metadata_dict)
    """
    metadata = {
        'file_path': file_path,
        'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
        'read_timestamp': datetime.now().isoformat(),
        'errors': []
    }
    
    # Auto-detect delimiter
    if delimiter is None:
        try:
            delimiter = detect_delimiter(file_path)
            metadata['delimiter'] = delimiter
            metadata['delimiter_auto_detected'] = True
        except Exception as e:
            delimiter = ','
            metadata['delimiter'] = delimiter
            metadata['delimiter_auto_detected'] = False
            metadata['errors'].append(f"Delimiter detection failed: {str(e)}")
    else:
        metadata['delimiter'] = delimiter
        metadata['delimiter_auto_detected'] = False
    
    # Try different encodings
    encodings = [encoding] if encoding else ['utf-8', 'latin-1', 'windows-1252', 'iso-8859-1']
    
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(
                file_path,
                delimiter=delimiter,
                encoding=enc,
                low_memory=False
            )
            metadata['encoding'] = enc
            metadata['encoding_auto_detected'] = (encoding is None)
            break
        except Exception as e:
            metadata['errors'].append(f"Failed with encoding {enc}: {str(e)}")
            continue
    
    if df is None:
        raise ValueError(f"Failed to read CSV with any encoding. Errors: {metadata['errors']}")
    
    # Add metadata
    metadata['rows'] = len(df)
    metadata['columns'] = len(df.columns)
    metadata['preview'] = df.head(preview_rows).to_dict('records')
    
    # Standardize column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()
    
    return df, metadata


def save_dataframe(
    df: pd.DataFrame,
    output_dir: str = 'output',
    filename: Optional[str] = None,
    add_timestamp: bool = True
) -> str:
    """
    Save DataFrame to CSV with optional timestamp
    
    Args:
        df: DataFrame to save
        output_dir: Output directory
        filename: Output filename (auto-generated if None)
        add_timestamp: Whether to add timestamp to filename
        
    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if filename is None:
        filename = 'cleaned.csv'
    
    if add_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{timestamp}{ext}"
    
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    
    return output_path


def save_json(
    data: Dict[str, Any],
    output_dir: str = 'output',
    filename: str = 'cleaning_log.json',
    add_timestamp: bool = True
) -> str:
    """
    Save dictionary to JSON file
    
    Args:
        data: Dictionary to save
        output_dir: Output directory
        filename: Output filename
        add_timestamp: Whether to add timestamp to filename
        
    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if add_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{timestamp}{ext}"
    
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)
    
    return output_path


def save_report(
    content: str,
    output_dir: str = 'output',
    filename: str = 'report.html',
    add_timestamp: bool = True
) -> str:
    """
    Save report (HTML or Markdown) to file
    
    Args:
        content: Report content
        output_dir: Output directory
        filename: Output filename
        add_timestamp: Whether to add timestamp to filename
        
    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if add_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{timestamp}{ext}"
    
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return output_path


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get file information
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file information
    """
    stat = os.stat(file_path)
    
    return {
        'path': file_path,
        'name': os.path.basename(file_path),
        'size_bytes': stat.st_size,
        'size_mb': stat.st_size / (1024 * 1024),
        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'exists': os.path.exists(file_path)
    }
