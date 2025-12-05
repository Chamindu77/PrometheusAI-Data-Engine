"""
Data transformation and cleaning module
Implements all cleaning operations with logging
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats


class DataCleaner:
    """Main data cleaning class with comprehensive transformation methods"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize cleaner with DataFrame
        
        Args:
            df: DataFrame to clean
        """
        self.df = df.copy()
        self.original_df = df.copy()
        self.cleaning_log = {
            'actions': [],
            'warnings': [],
            'statistics': {
                'original_shape': df.shape,
                'original_memory_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
            }
        }
    
    def log_action(self, action: str, details: Dict[str, Any] = None):
        """Log a cleaning action"""
        self.cleaning_log['actions'].append({
            'action': action,
            'details': details or {},
            'shape_after': self.df.shape
        })
    
    def log_warning(self, warning: str):
        """Log a warning"""
        self.cleaning_log['warnings'].append(warning)
    
    def standardize_column_names(self) -> 'DataCleaner':
        """
        Standardize column names (lowercase, underscores, no special chars)
        
        Returns:
            Self for chaining
        """
        original_cols = list(self.df.columns)
        
        # Clean column names
        new_cols = []
        for col in self.df.columns:
            # Convert to lowercase
            new_col = str(col).lower()
            # Replace spaces and special chars with underscores
            new_col = new_col.replace(' ', '_').replace('-', '_')
            # Remove other special characters
            new_col = ''.join(c if c.isalnum() or c == '_' else '_' for c in new_col)
            # Remove consecutive underscores
            while '__' in new_col:
                new_col = new_col.replace('__', '_')
            # Remove leading/trailing underscores
            new_col = new_col.strip('_')
            new_cols.append(new_col)
        
        self.df.columns = new_cols
        
        changes = [f"{old} → {new}" for old, new in zip(original_cols, new_cols) if old != new]
        if changes:
            self.log_action('standardize_column_names', {'changes': changes})
        
        return self
    
    def remove_duplicates(self, subset: Optional[List[str]] = None, keep: str = 'first') -> 'DataCleaner':
        """
        Remove duplicate rows
        
        Args:
            subset: Columns to consider for duplicates (None = all columns)
            keep: Which duplicates to keep ('first', 'last', False)
            
        Returns:
            Self for chaining
        """
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        removed = initial_rows - len(self.df)
        
        if removed > 0:
            self.log_action('remove_duplicates', {
                'rows_removed': removed,
                'subset': subset,
                'keep': keep
            })
        
        return self
    
    def drop_columns(self, columns: List[str], reason: str = 'manual') -> 'DataCleaner':
        """
        Drop specified columns
        
        Args:
            columns: List of column names to drop
            reason: Reason for dropping
            
        Returns:
            Self for chaining
        """
        existing_cols = [col for col in columns if col in self.df.columns]
        if existing_cols:
            self.df = self.df.drop(columns=existing_cols)
            self.log_action('drop_columns', {
                'columns': existing_cols,
                'reason': reason
            })
        
        return self
    
    def drop_high_missing_columns(self, threshold: float = 0.9) -> 'DataCleaner':
        """
        Drop columns with missing values above threshold
        
        Args:
            threshold: Missing value ratio threshold (0-1)
            
        Returns:
            Self for chaining
        """
        missing_ratios = self.df.isna().mean()
        cols_to_drop = missing_ratios[missing_ratios > threshold].index.tolist()
        
        if cols_to_drop:
            self.drop_columns(cols_to_drop, reason=f'missing > {threshold*100}%')
        
        return self
    
    def drop_constant_columns(self) -> 'DataCleaner':
        """
        Drop columns with only one unique value
        
        Returns:
            Self for chaining
        """
        constant_cols = [col for col in self.df.columns if self.df[col].nunique() <= 1]
        
        if constant_cols:
            self.drop_columns(constant_cols, reason='constant_value')
        
        return self
    
    def impute_missing_numeric(
        self,
        columns: Optional[List[str]] = None,
        strategy: str = 'median'
    ) -> 'DataCleaner':
        """
        Impute missing values in numeric columns
        
        Args:
            columns: List of columns to impute (None = all numeric)
            strategy: 'mean', 'median', 'mode', 'forward_fill', 'backward_fill', or numeric value
            
        Returns:
            Self for chaining
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        imputed_cols = []
        for col in columns:
            if col not in self.df.columns:
                continue
            
            missing_count = self.df[col].isna().sum()
            if missing_count == 0:
                continue
            
            if strategy == 'mean':
                fill_value = self.df[col].mean()
            elif strategy == 'median':
                fill_value = self.df[col].median()
            elif strategy == 'mode':
                fill_value = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 0
            elif strategy == 'forward_fill':
                self.df[col] = self.df[col].fillna(method='ffill')
                imputed_cols.append(col)
                continue
            elif strategy == 'backward_fill':
                self.df[col] = self.df[col].fillna(method='bfill')
                imputed_cols.append(col)
                continue
            elif isinstance(strategy, (int, float)):
                fill_value = strategy
            else:
                self.log_warning(f"Unknown imputation strategy: {strategy}")
                continue
            
            self.df[col] = self.df[col].fillna(fill_value)
            imputed_cols.append(col)
        
        if imputed_cols:
            self.log_action('impute_missing_numeric', {
                'columns': imputed_cols,
                'strategy': strategy
            })
        
        return self
    
    def impute_missing_categorical(
        self,
        columns: Optional[List[str]] = None,
        strategy: str = 'mode',
        fill_value: str = 'MISSING'
    ) -> 'DataCleaner':
        """
        Impute missing values in categorical columns
        
        Args:
            columns: List of columns to impute (None = all object/category)
            strategy: 'mode' or 'constant'
            fill_value: Value to use for 'constant' strategy
            
        Returns:
            Self for chaining
        """
        if columns is None:
            columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        imputed_cols = []
        for col in columns:
            if col not in self.df.columns:
                continue
            
            missing_count = self.df[col].isna().sum()
            if missing_count == 0:
                continue
            
            if strategy == 'mode':
                mode_values = self.df[col].mode()
                if len(mode_values) > 0:
                    self.df[col] = self.df[col].fillna(mode_values[0])
                    imputed_cols.append(col)
            elif strategy == 'constant':
                self.df[col] = self.df[col].fillna(fill_value)
                imputed_cols.append(col)
        
        if imputed_cols:
            self.log_action('impute_missing_categorical', {
                'columns': imputed_cols,
                'strategy': strategy
            })
        
        return self
    
    def create_missing_indicators(self, columns: List[str]) -> 'DataCleaner':
        """
        Create binary indicator columns for missing values
        
        Args:
            columns: Columns to create indicators for
            
        Returns:
            Self for chaining
        """
        created_cols = []
        for col in columns:
            if col in self.df.columns:
                indicator_col = f"{col}_missing"
                self.df[indicator_col] = self.df[col].isna().astype(int)
                created_cols.append(indicator_col)
        
        if created_cols:
            self.log_action('create_missing_indicators', {'columns': created_cols})
        
        return self
    
    def detect_outliers_iqr(
        self,
        columns: Optional[List[str]] = None,
        multiplier: float = 1.5
    ) -> Dict[str, np.ndarray]:
        """
        Detect outliers using IQR method
        
        Args:
            columns: Columns to check (None = all numeric)
            multiplier: IQR multiplier (default 1.5)
            
        Returns:
            Dictionary mapping column names to boolean arrays (True = outlier)
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_masks = {}
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr
            
            mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outlier_masks[col] = mask
        
        return outlier_masks
    
    def detect_outliers_zscore(
        self,
        columns: Optional[List[str]] = None,
        threshold: float = 3.0
    ) -> Dict[str, np.ndarray]:
        """
        Detect outliers using Z-score method
        
        Args:
            columns: Columns to check (None = all numeric)
            threshold: Z-score threshold (default 3.0)
            
        Returns:
            Dictionary mapping column names to boolean arrays (True = outlier)
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_masks = {}
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            z_scores = np.abs(stats.zscore(self.df[col].dropna()))
            mask = pd.Series(False, index=self.df.index)
            mask.loc[self.df[col].notna()] = z_scores > threshold
            outlier_masks[col] = mask
        
        return outlier_masks
    
    def remove_outliers(
        self,
        columns: Optional[List[str]] = None,
        method: str = 'iqr',
        **kwargs
    ) -> 'DataCleaner':
        """
        Remove outlier rows
        
        Args:
            columns: Columns to check
            method: 'iqr' or 'zscore'
            **kwargs: Additional arguments for outlier detection
            
        Returns:
            Self for chaining
        """
        if method == 'iqr':
            outlier_masks = self.detect_outliers_iqr(columns, **kwargs)
        elif method == 'zscore':
            outlier_masks = self.detect_outliers_zscore(columns, **kwargs)
        else:
            self.log_warning(f"Unknown outlier method: {method}")
            return self
        
        # Combine masks (row is outlier if ANY column is outlier)
        combined_mask = pd.Series(False, index=self.df.index)
        for mask in outlier_masks.values():
            combined_mask |= mask
        
        initial_rows = len(self.df)
        self.df = self.df[~combined_mask]
        removed = initial_rows - len(self.df)
        
        if removed > 0:
            self.log_action('remove_outliers', {
                'method': method,
                'rows_removed': removed,
                'columns': columns
            })
        
        return self
    
    def cap_outliers(
        self,
        columns: Optional[List[str]] = None,
        method: str = 'iqr',
        **kwargs
    ) -> 'DataCleaner':
        """
        Cap outliers at boundaries (winsorization)
        
        Args:
            columns: Columns to cap
            method: 'iqr' or 'zscore'
            **kwargs: Additional arguments
            
        Returns:
            Self for chaining
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        capped_cols = []
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            if method == 'iqr':
                q1 = self.df[col].quantile(0.25)
                q3 = self.df[col].quantile(0.75)
                iqr = q3 - q1
                multiplier = kwargs.get('multiplier', 1.5)
                lower_bound = q1 - multiplier * iqr
                upper_bound = q3 + multiplier * iqr
            elif method == 'percentile':
                lower_pct = kwargs.get('lower', 1)
                upper_pct = kwargs.get('upper', 99)
                lower_bound = self.df[col].quantile(lower_pct / 100)
                upper_bound = self.df[col].quantile(upper_pct / 100)
            else:
                continue
            
            # Cap values
            original_outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            if original_outliers > 0:
                self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
                capped_cols.append(col)
        
        if capped_cols:
            self.log_action('cap_outliers', {
                'method': method,
                'columns': capped_cols
            })
        
        return self
    
    def add_outlier_flags(
        self,
        columns: Optional[List[str]] = None,
        method: str = 'iqr',
        **kwargs
    ) -> 'DataCleaner':
        """
        Add binary flag columns for outliers
        
        Args:
            columns: Columns to flag
            method: 'iqr' or 'zscore'
            **kwargs: Additional arguments
            
        Returns:
            Self for chaining
        """
        if method == 'iqr':
            outlier_masks = self.detect_outliers_iqr(columns, **kwargs)
        elif method == 'zscore':
            outlier_masks = self.detect_outliers_zscore(columns, **kwargs)
        else:
            return self
        
        created_cols = []
        for col, mask in outlier_masks.items():
            flag_col = f"{col}_outlier"
            self.df[flag_col] = mask.astype(int)
            created_cols.append(flag_col)
        
        if created_cols:
            self.log_action('add_outlier_flags', {
                'method': method,
                'columns': created_cols
            })
        
        return self
    
    def get_cleaned_data(self) -> pd.DataFrame:
        """Get cleaned DataFrame"""
        return self.df.copy()
    
    def get_cleaning_log(self) -> Dict[str, Any]:
        """Get cleaning log with final statistics"""
        self.cleaning_log['statistics']['final_shape'] = self.df.shape
        self.cleaning_log['statistics']['final_memory_mb'] = self.df.memory_usage(deep=True).sum() / (1024 * 1024)
        self.cleaning_log['statistics']['memory_reduction_mb'] = (
            self.cleaning_log['statistics']['original_memory_mb'] - 
            self.cleaning_log['statistics']['final_memory_mb']
        )
        self.cleaning_log['statistics']['rows_removed'] = (
            self.cleaning_log['statistics']['original_shape'][0] - 
            self.cleaning_log['statistics']['final_shape'][0]
        )
        self.cleaning_log['statistics']['columns_removed'] = (
            self.cleaning_log['statistics']['original_shape'][1] - 
            self.cleaning_log['statistics']['final_shape'][1]
        )
        
        return self.cleaning_log
