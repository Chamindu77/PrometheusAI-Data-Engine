"""
Basic unit tests for EDA Engine cleaner module
"""

import pytest
import pandas as pd
import numpy as np
from cleaner import schema_infer, overview, column_analysis, transformers


class TestSchemaInfer:
    """Tests for schema inference module"""
    
    def test_infer_numeric_type(self):
        """Test numeric type detection"""
        series = pd.Series([1, 2, 3, 4, 5])
        result = schema_infer.infer_column_type(series)
        assert result['inferred_type'] == 'integer'
    
    def test_infer_categorical_type(self):
        """Test categorical type detection"""
        series = pd.Series(['A', 'B', 'A', 'C', 'B', 'A'])
        result = schema_infer.infer_column_type(series)
        assert result['inferred_type'] == 'categorical'
    
    def test_infer_boolean_type(self):
        """Test boolean type detection"""
        series = pd.Series([True, False, True, False])
        result = schema_infer.infer_column_type(series)
        assert result['inferred_type'] == 'boolean'
    
    def test_optimize_dtypes(self):
        """Test dtype optimization"""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.0, 2.0, 3.0]
        })
        df['int_col'] = df['int_col'].astype('int64')
        df['float_col'] = df['float_col'].astype('float64')
        
        df_opt, report = schema_infer.optimize_dtypes(df)
        
        assert report['original_memory_mb'] > report['optimized_memory_mb']
        assert len(report['optimizations']) > 0


class TestOverview:
    """Tests for overview module"""
    
    def test_get_dataset_overview(self):
        """Test dataset overview generation"""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': ['x', 'y', 'x', 'y'],
            'C': [1, 1, 1, 1]  # constant
        })
        
        overview_data = overview.get_dataset_overview(df)
        
        assert overview_data['basic_info']['rows'] == 4
        assert overview_data['basic_info']['columns'] == 3
        assert overview_data['missing_values']['total_missing'] == 1
        assert 'health_score' in overview_data
    
    def test_get_column_cardinality(self):
        """Test cardinality calculation"""
        df = pd.DataFrame({
            'unique_col': [1, 2, 3, 4],
            'constant_col': [1, 1, 1, 1]
        })
        
        cardinality = overview.get_column_cardinality(df)
        
        assert cardinality['unique_col']['is_unique'] == True
        assert cardinality['constant_col']['is_constant'] == True


class TestColumnAnalysis:
    """Tests for column analysis module"""
    
    def test_analyze_numeric_column(self):
        """Test numeric column analysis"""
        series = pd.Series([1, 2, 3, 4, 5, 100], name='test_col')  # 100 is outlier
        
        result = column_analysis.analyze_numeric_column(series)
        
        assert 'mean' in result
        assert 'median' in result
        assert 'outliers_count' in result
        assert result['outliers_count'] > 0  # Should detect 100 as outlier
    
    def test_analyze_categorical_column(self):
        """Test categorical column analysis"""
        series = pd.Series(['A', 'B', 'A', 'C', 'A'], name='test_col')
        
        result = column_analysis.analyze_categorical_column(series)
        
        assert result['unique_count'] == 3
        assert result['mode'] == 'A'
        assert len(result['top_5_categories']) <= 5


class TestTransformers:
    """Tests for transformers module"""
    
    def test_remove_duplicates(self):
        """Test duplicate removal"""
        df = pd.DataFrame({
            'A': [1, 2, 2, 3],
            'B': ['x', 'y', 'y', 'z']
        })
        
        cleaner = transformers.DataCleaner(df)
        cleaner.remove_duplicates()
        
        result = cleaner.get_cleaned_data()
        assert len(result) == 3  # One duplicate removed
        assert len(cleaner.cleaning_log['actions']) == 1
    
    def test_impute_missing_numeric(self):
        """Test numeric imputation"""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4]
        })
        
        cleaner = transformers.DataCleaner(df)
        cleaner.impute_missing_numeric(strategy='median')
        
        result = cleaner.get_cleaned_data()
        assert result['A'].isna().sum() == 0
    
    def test_drop_constant_columns(self):
        """Test constant column removal"""
        df = pd.DataFrame({
            'constant': [1, 1, 1, 1],
            'varying': [1, 2, 3, 4]
        })
        
        cleaner = transformers.DataCleaner(df)
        cleaner.drop_constant_columns()
        
        result = cleaner.get_cleaned_data()
        assert 'constant' not in result.columns
        assert 'varying' in result.columns
    
    def test_standardize_column_names(self):
        """Test column name standardization"""
        df = pd.DataFrame({
            'Column Name': [1, 2],
            'UPPERCASE': [3, 4],
            'with-dash': [5, 6]
        })
        
        cleaner = transformers.DataCleaner(df)
        cleaner.standardize_column_names()
        
        result = cleaner.get_cleaned_data()
        assert 'column_name' in result.columns
        assert 'uppercase' in result.columns
        assert 'with_dash' in result.columns


class TestIntegration:
    """Integration tests for full pipeline"""
    
    def test_full_pipeline(self):
        """Test complete cleaning pipeline"""
        # Create sample dataset with issues
        df = pd.DataFrame({
            'ID': [1, 2, 3, 4, 4],  # Duplicate
            'Value': [10, 20, np.nan, 30, 30],  # Missing value
            'Category': ['A', 'B', 'A', 'B', 'B'],
            'Constant': [1, 1, 1, 1, 1]  # Constant column
        })
        
        # Run cleaning pipeline
        cleaner = transformers.DataCleaner(df)
        cleaner.standardize_column_names()
        cleaner.drop_constant_columns()
        cleaner.remove_duplicates()
        cleaner.impute_missing_numeric(strategy='median')
        
        result = cleaner.get_cleaned_data()
        log = cleaner.get_cleaning_log()
        
        # Assertions
        assert len(result) == 4  # One duplicate removed
        assert 'constant' not in result.columns  # Constant column dropped
        assert result['value'].isna().sum() == 0  # Missing values imputed
        assert len(log['actions']) >= 3  # At least 3 actions applied


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
