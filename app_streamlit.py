"""
Streamlit UI for EDA Engine
Interactive web interface for automated data cleaning and EDA
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os
import json

# Import cleaner modules
from cleaner import schema_infer, overview, column_analysis, transformers, eda, report
from utils import io as utils_io


# Page configuration
st.set_page_config(
    page_title="PrometheusAI Data Engine",
    page_icon="🕵🏻‍♀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 18px;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background: #fff3cd;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
    }
    .success-box {
        background: #d4edda;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 10px 0;
    }
    
    /* Darken scrollbar thumb */
    ::-webkit-scrollbar-thumb {
        background: #888 !important;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555 !important;
    }
    
    /* Remove scrollbar highlight/selection in dataframes */
    [data-testid="stDataFrame"] {
        scrollbar-color: #888 transparent !important;
    }
    
    [data-testid="stDataFrame"]::-webkit-scrollbar-thumb {
        background: #888 !important;
        border: none !important;
    }
    
    [data-testid="stDataFrame"]::-webkit-scrollbar-thumb:active {
        background: #555 !important;
    }
    
    [data-testid="stDataFrame"]::-webkit-scrollbar-track {
        background: transparent !important;
    }
    
    /* Professional button hover and focus styling */
    button[kind="secondary"], button[kind="primary"] {
        transition: all 0.3s ease !important;
    }
    
    button[kind="secondary"]:hover, button[kind="primary"]:hover {
        border-color: #6b7280 !important;
        color: white !important;
    }
    
    button[kind="secondary"]:focus, button[kind="primary"]:focus {
        border-color: #6b7280 !important;
        box-shadow: 0 0 0 0.2rem rgba(107, 114, 128, 0.4) !important;
        outline: none !important;
    }
    
    button[kind="secondary"]:active, button[kind="primary"]:active {
        border-color: #4b5563 !important;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_cleaned' not in st.session_state:
    st.session_state.df_cleaned = None
if 'overview_data' not in st.session_state:
    st.session_state.overview_data = None
if 'schema_data' not in st.session_state:
    st.session_state.schema_data = None
if 'column_analysis_data' not in st.session_state:
    st.session_state.column_analysis_data = None
if 'cleaning_log' not in st.session_state:
    st.session_state.cleaning_log = None
if 'plot_metadata' not in st.session_state:
    st.session_state.plot_metadata = []
if 'correlation_insights' not in st.session_state:
    st.session_state.correlation_insights = None
if 'selected_conversions' not in st.session_state:
    st.session_state.selected_conversions = set()


# Header
st.markdown('<div class="main-header">📊 PrometheusAI Data Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Automated Data Preparation, EDA & AI-Powered Insight Engine</div>',
    unsafe_allow_html=True
)


# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=['csv'],
        help="Upload a CSV file to begin analysis"
    )
    
    if uploaded_file is not None:
        # Load file
        if st.session_state.df_original is None or st.button("🔄 Reload File"):
            with st.spinner("Loading CSV..."):
                try:
                    # Save temp file
                    temp_path = f"temp/{uploaded_file.name}"
                    os.makedirs("temp", exist_ok=True)
                    
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Load with smart reader
                    df, metadata = utils_io.read_csv_smart(temp_path)
                    st.session_state.df_original = df
                    st.session_state.file_metadata = metadata
                    
                    # Reset downstream data
                    st.session_state.df_cleaned = None
                    st.session_state.overview_data = None
                    st.session_state.schema_data = None
                    st.session_state.column_analysis_data = None
                    st.session_state.cleaning_log = None
                    
                    st.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
                    
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
        
        # Quick stats
        if st.session_state.df_original is not None:
            st.divider()
            st.metric("Rows", f"{len(st.session_state.df_original):,}")
            st.metric("Columns", f"{len(st.session_state.df_original.columns)}")
            st.metric("Memory", f"{st.session_state.df_original.memory_usage(deep=True).sum() / (1024*1024):.1f} MB")
    
    st.divider()
    
    # Auto mode
    auto_mode = st.checkbox(
        "🤖 Full Auto Mode",
        help="Execute entire pipeline with one click"
    )
    
    if auto_mode and st.button("▶️ Run Full Pipeline", type="primary"):
        if st.session_state.df_original is not None:
            with st.spinner("Running full pipeline..."):
                try:
                    # Overview
                    st.session_state.overview_data = overview.get_dataset_overview(
                        st.session_state.df_original
                    )
                    
                    # Schema
                    st.session_state.schema_data = schema_infer.infer_schema(
                        st.session_state.df_original
                    )
                    
                    # Column analysis
                    st.session_state.column_analysis_data = column_analysis.analyze_all_columns(
                        st.session_state.df_original
                    )
                    
                    # Cleaning
                    cleaner = transformers.DataCleaner(st.session_state.df_original)
                    cleaner.standardize_column_names()
                    cleaner.drop_constant_columns()
                    cleaner.drop_high_missing_columns(0.9)
                    cleaner.remove_duplicates()
                    cleaner.impute_missing_numeric(strategy='median')
                    cleaner.impute_missing_categorical(strategy='mode')
                    cleaner.add_outlier_flags(method='iqr')
                    
                    st.session_state.df_cleaned = cleaner.get_cleaned_data()
                    st.session_state.cleaning_log = cleaner.get_cleaning_log()
                    
                    # Memory optimization
                    st.session_state.df_cleaned, opt_report = schema_infer.optimize_dtypes(
                        st.session_state.df_cleaned
                    )
                    st.session_state.cleaning_log['memory_optimization'] = opt_report
                    
                    # EDA
                    visualizer = eda.EDAVisualizer(st.session_state.df_cleaned, 'output')
                    plot_report = visualizer.generate_all_plots()
                    st.session_state.plot_metadata = plot_report['plot_metadata']
                    
                    # Correlation
                    st.session_state.correlation_insights = eda.get_correlation_insights(
                        st.session_state.df_cleaned
                    )
                    
                    st.success("✅ Full pipeline completed!")
                    
                except Exception as e:
                    st.error(f"Pipeline error: {str(e)}")


# Main tabs
if st.session_state.df_original is not None:
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📁 Upload & Preview",
        "📊 Dataset Overview",
        "🔍 Schema Detection",
        "📋 Column Analysis",
        "🧹 Data Cleaning",
        "📈 EDA Visualizations",
        "📄 Report",
        "💾 Downloads"
    ])
    
    # Tab 1: Upload & Preview
    with tab1:
        st.header("Dataset Preview")
        
        # Dataset Stats - Card Layout (Single Row)
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%); 
                    padding: 25px; border-radius: 12px; color: white; margin: 15px 0;
                    box-shadow: 0 4px 15px rgba(15, 118, 110, 0.2);'>
            <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 25px;'>
                <div style='text-align: center;'>
                    <div style='font-size: 14px; opacity: 0.9; margin-bottom: 8px;'>📊 Total Rows</div>
                    <div style='font-size: 32px; font-weight: bold;'>{len(st.session_state.df_original):,}</div>
                </div>
                <div style='text-align: center; border-left: 2px solid rgba(255,255,255,0.3); padding-left: 15px;'>
                    <div style='font-size: 14px; opacity: 0.9; margin-bottom: 8px;'>📝 Total Columns</div>
                    <div style='font-size: 32px; font-weight: bold;'>{len(st.session_state.df_original.columns)}</div>
                </div>
                <div style='text-align: center; border-left: 2px solid rgba(255,255,255,0.3); padding-left: 15px;'>
                    <div style='font-size: 14px; opacity: 0.9; margin-bottom: 8px;'>💾 Memory Size</div>
                    <div style='font-size: 32px; font-weight: bold;'>{st.session_state.df_original.memory_usage(deep=True).sum() / (1024*1024):.1f} MB</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("First 10 Rows")
        st.dataframe(st.session_state.df_original.head(10), use_container_width=True)
        
        st.subheader("Column Data Types")
        dtype_df = pd.DataFrame({
            'Column': st.session_state.df_original.columns,
            'Data Type': st.session_state.df_original.dtypes.values
        })
        st.dataframe(dtype_df, use_container_width=True)
    
    # Tab 2: Dataset Overview
    with tab2:
        st.header("Dataset Overview")
        
        if st.button("🔍 Generate Overview") or st.session_state.overview_data:
            if st.session_state.overview_data is None:
                with st.spinner("Analyzing dataset..."):
                    st.session_state.overview_data = overview.get_dataset_overview(
                        st.session_state.df_original
                    )
            
            overview_data = st.session_state.overview_data
            
            # Health score
            health_score = overview_data.get('health_score', 0)
            score_color = "🟢" if health_score >= 80 else "🟡" if health_score >= 60 else "🔴"
            
            st.markdown(f"### Dataset Health Score: {score_color} {health_score:.1f}/100")
            st.progress(health_score / 100)
            
            # Basic info - Card Layout (Single Row)
            st.subheader("📋 Basic Information")
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%); 
                        padding: 25px; border-radius: 12px; color: white; margin: 15px 0;
                        box-shadow: 0 4px 15px rgba(15, 118, 110, 0.2);'>
                <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 25px;'>
                    <div style='text-align: center;'>
                        <div style='font-size: 14px; opacity: 0.9; margin-bottom: 8px;'>📊 Total Rows</div>
                        <div style='font-size: 32px; font-weight: bold;'>{overview_data['basic_info']['rows']:,}</div>
                    </div>
                    <div style='text-align: center; border-left: 2px solid rgba(255,255,255,0.3); padding-left: 15px;'>
                        <div style='font-size: 14px; opacity: 0.9; margin-bottom: 8px;'>📝 Total Columns</div>
                        <div style='font-size: 32px; font-weight: bold;'>{overview_data['basic_info']['columns']}</div>
                    </div>
                    <div style='text-align: center; border-left: 2px solid rgba(255,255,255,0.3); padding-left: 15px;'>
                        <div style='font-size: 14px; opacity: 0.9; margin-bottom: 8px;'>🔢 Total Cells</div>
                        <div style='font-size: 32px; font-weight: bold;'>{overview_data['basic_info']['total_cells']:,}</div>
                    </div>
                    <div style='text-align: center; border-left: 2px solid rgba(255,255,255,0.3); padding-left: 15px;'>
                        <div style='font-size: 14px; opacity: 0.9; margin-bottom: 8px;'>💾 Memory Usage</div>
                        <div style='font-size: 32px; font-weight: bold;'>{overview_data['memory_usage']['total_mb']:.1f} MB</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Missing values - Card Layout with dynamic colors (Single Row)
            st.subheader("⚠️ Missing Values Summary")
            
            # Determine color based on missing percentage
            missing_pct = overview_data['missing_values']['total_missing_pct']
            if missing_pct <= 5.0:
                # Low missing values - Green (Good)
                gradient_color = "linear-gradient(135deg, #16a34a 0%, #22c55e 100%)"
                shadow_color = "rgba(22, 163, 74, 0.2)"
                status_icon = "✅"
            elif missing_pct <= 20.0:
                # Moderate missing values - Amber (Warning)
                gradient_color = "linear-gradient(135deg, #d97706 0%, #f59e0b 100%)"
                shadow_color = "rgba(217, 119, 6, 0.2)"
                status_icon = "⚠️"
            else:
                # High missing values - Red (Critical)
                gradient_color = "linear-gradient(135deg, #dc2626 0%, #ef4444 100%)"
                shadow_color = "rgba(220, 38, 38, 0.2)"
                status_icon = "🔴"
            
            st.markdown(f"""
            <div style='background: {gradient_color}; 
                        padding: 25px; border-radius: 12px; color: white; margin: 15px 0;
                        box-shadow: 0 4px 15px {shadow_color};'>
                <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 25px;'>
                    <div style='text-align: center;'>
                        <div style='font-size: 14px; opacity: 0.9; margin-bottom: 8px;'>{status_icon} Total Missing Values</div>
                        <div style='font-size: 32px; font-weight: bold;'>{overview_data['missing_values']['total_missing']:,}</div>
                    </div>
                    <div style='text-align: center; border-left: 2px solid rgba(255,255,255,0.3); padding-left: 15px;'>
                        <div style='font-size: 14px; opacity: 0.9; margin-bottom: 8px;'>📊 Missing Percentage</div>
                        <div style='font-size: 32px; font-weight: bold;'>{overview_data['missing_values']['total_missing_pct']:.2f}%</div>
                    </div>
                    <div style='text-align: center; border-left: 2px solid rgba(255,255,255,0.3); padding-left: 15px;'>
                        <div style='font-size: 14px; opacity: 0.9; margin-bottom: 8px;'>📋 Columns Affected</div>
                        <div style='font-size: 32px; font-weight: bold;'>{overview_data['missing_values']['columns_with_missing']}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if overview_data['missing_values']['per_column']:
                missing_df = pd.DataFrame([
                    {'Column': col, 'Missing Count': info['count'], 'Missing %': info['percentage']}
                    for col, info in overview_data['missing_values']['per_column'].items()
                ]).sort_values('Missing %', ascending=False)
                st.dataframe(missing_df, use_container_width=True)
            
            # Duplicates
            st.subheader("Duplicate Rows")
            col1, col2 = st.columns(2)
            col1.metric("Duplicates", f"{overview_data['duplicates']['total_duplicates']:,}")
            col2.metric("Duplicate %", f"{overview_data['duplicates']['duplicate_percentage']:.2f}%")
            
            # Warnings
            if overview_data.get('warnings'):
                st.subheader("⚠️ Warnings")
                for warning in overview_data['warnings']:
                    severity_emoji = "🔴" if warning['severity'] == 'critical' else "🟡" if warning['severity'] == 'high' else "🔵"
                    st.warning(f"{severity_emoji} **{warning['severity'].upper()}**: {warning['message']}")
    
    # Tab 3: Schema Detection
    with tab3:
        st.header("Schema Detection & Type Inference")
        
        if st.button("🔍 Infer Schema") or st.session_state.schema_data:
            if st.session_state.schema_data is None:
                with st.spinner("Inferring schema..."):
                    st.session_state.schema_data = schema_infer.infer_schema(
                        st.session_state.df_original
                    )
            
            schema_data = st.session_state.schema_data
            
            # Type Distribution - Card Layout (Single Row)
            st.subheader("📊 Type Distribution")
            summary = schema_data['summary']
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%); 
                        padding: 25px; border-radius: 12px; color: white; margin: 15px 0;
                        box-shadow: 0 4px 15px rgba(15, 118, 110, 0.2);'>
                <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 25px;'>
                    <div style='text-align: center;'>
                        <div style='font-size: 14px; opacity: 0.9; margin-bottom: 8px;'>🔢 Numeric</div>
                        <div style='font-size: 32px; font-weight: bold;'>{summary['numeric_columns']}</div>
                    </div>
                    <div style='text-align: center; border-left: 2px solid rgba(255,255,255,0.3); padding-left: 15px;'>
                        <div style='font-size: 14px; opacity: 0.9; margin-bottom: 8px;'>📝 Categorical</div>
                        <div style='font-size: 32px; font-weight: bold;'>{summary['categorical_columns']}</div>
                    </div>
                    <div style='text-align: center; border-left: 2px solid rgba(255,255,255,0.3); padding-left: 15px;'>
                        <div style='font-size: 14px; opacity: 0.9; margin-bottom: 8px;'>📅 Datetime</div>
                        <div style='font-size: 32px; font-weight: bold;'>{summary['datetime_columns']}</div>
                    </div>
                    <div style='text-align: center; border-left: 2px solid rgba(255,255,255,0.3); padding-left: 15px;'>
                        <div style='font-size: 14px; opacity: 0.9; margin-bottom: 8px;'>📄 Text</div>
                        <div style='font-size: 32px; font-weight: bold;'>{summary['text_columns']}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed schema - MOVED TO TOP
            st.subheader("Detailed Schema")
            schema_records = []
            for col, info in schema_data['columns'].items():
                schema_records.append({
                    'Column': col,
                    'Current Type': info['current_dtype'],
                    'Inferred Type': info['inferred_type'],
                    'Suggested Type': info.get('suggested_dtype', 'N/A'),
                    'Confidence': f"{info['confidence']:.2%}" if info['confidence'] else 'N/A'
                })
            
            schema_df = pd.DataFrame(schema_records)
            st.dataframe(schema_df, use_container_width=True)
            
            # Recommendations
            if schema_data.get('conversion_recommendations'):
                st.subheader("Recommended Conversions")
                conv_df = pd.DataFrame(schema_data['conversion_recommendations'])
                st.dataframe(conv_df, use_container_width=True)
                
                st.info("💡 Select which column type conversions you want to apply:")
                
                # Initialize selected conversions if needed
                all_columns = [rec['column'] for rec in schema_data['conversion_recommendations']]
                
                # Create checkboxes for each conversion recommendation
                st.write("---")
                
                # Track changes in checkboxes
                temp_selections = set()
                
                for rec in schema_data['conversion_recommendations']:
                    col_name = rec['column']
                    from_type = rec['from']
                    to_type = rec['to']
                    confidence = rec['confidence']
                    
                    # Check if this column should be checked by default
                    is_selected = col_name in st.session_state.selected_conversions
                    
                    checkbox_col, info_col = st.columns([3, 1])
                    with checkbox_col:
                        checked = st.checkbox(
                            f"**{col_name}**: `{from_type}` → `{to_type}`",
                            value=is_selected,
                            key=f"conv_{col_name}"
                        )
                        
                        # Add to temp selections if checked
                        if checked:
                            temp_selections.add(col_name)
                    
                    with info_col:
                        st.caption(f"Confidence: {confidence:.1%}")
                
                # Update session state with current selections
                st.session_state.selected_conversions = temp_selections
                
                st.write("---")
                
                # Apply button with selected count
                selected_count = len(st.session_state.selected_conversions)
                if selected_count > 0:
                    if st.button(f"✨ Apply Selected Conversions ({selected_count})", type="primary", key="btn_apply_conversions"):
                        with st.spinner(f"Applying {selected_count} conversions..."):
                            # Create a filtered schema with only selected columns
                            filtered_schema = schema_data.copy()
                            filtered_schema['columns'] = {
                                col: info for col, info in schema_data['columns'].items()
                                if col in st.session_state.selected_conversions
                            }
                            
                            df_converted, logs = schema_infer.apply_schema_conversions(
                                st.session_state.df_original,
                                filtered_schema
                            )
                            st.session_state.df_original = df_converted
                            
                            # Show success messages
                            for log in logs:
                                st.success(log)
                            
                            # Re-infer schema to get updated information
                            updated_schema = schema_infer.infer_schema(st.session_state.df_original)
                            st.session_state.schema_data = updated_schema
                            st.session_state.selected_conversions = set()
                            
                            st.success("✅ Conversions applied successfully!")
                            
                            # Show updated schema below
                            st.subheader("📊 Updated Schema After Conversions")
                            updated_records = []
                            for col, info in updated_schema['columns'].items():
                                updated_records.append({
                                    'Column': col,
                                    'Current Type': info['current_dtype'],
                                    'Inferred Type': info['inferred_type'],
                                    'Suggested Type': info.get('suggested_dtype', 'N/A'),
                                    'Confidence': f"{info['confidence']:.2%}" if info['confidence'] else 'N/A'
                                })
                            
                            updated_df = pd.DataFrame(updated_records)
                            st.dataframe(updated_df, use_container_width=True)

                else:
                    st.warning("⚠️ No columns selected. Please select at least one column to apply conversions.")
    
    # Tab 4: Column Analysis
    with tab4:
        st.header("Column-wise Analysis")
        
        if st.button("📊 Analyze Columns") or st.session_state.column_analysis_data:
            if st.session_state.column_analysis_data is None:
                with st.spinner("Analyzing columns..."):
                    st.session_state.column_analysis_data = column_analysis.analyze_all_columns(
                        st.session_state.df_original
                    )
            
            col_data = st.session_state.column_analysis_data
            
            # Column selector
            selected_col = st.selectbox(
                "Select Column to View Details",
                options=list(col_data.keys())
            )
            
            if selected_col:
                col_info = col_data[selected_col]
                
                st.subheader(f"Analysis: {selected_col}")
                
                # Type badge
                col_type = col_info.get('detected_type', 'unknown')
                st.markdown(f"**Type:** `{col_type}`")
                
                # Analysis details
                analysis = col_info.get('analysis', {})
                
                if col_type == 'numeric':
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Mean", f"{analysis.get('mean', 0):.2f}")
                    col2.metric("Median", f"{analysis.get('median', 0):.2f}")
                    col3.metric("Std Dev", f"{analysis.get('std', 0):.2f}")
                    col4.metric("Missing", f"{analysis.get('missing_pct', 0):.1f}%")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Min", f"{analysis.get('min', 0):.2f}")
                    col2.metric("Q1", f"{analysis.get('q1', 0):.2f}")
                    col3.metric("Q3", f"{analysis.get('q3', 0):.2f}")
                    col4.metric("Max", f"{analysis.get('max', 0):.2f}")
                    
                    st.metric("Outliers (IQR)", f"{analysis.get('outliers_count', 0)} ({analysis.get('outliers_pct', 0):.1f}%)")
                
                elif col_type == 'categorical':
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Unique Values", analysis.get('unique_count', 0))
                    col2.metric("Mode", str(analysis.get('mode', 'N/A')))
                    col3.metric("Missing", f"{analysis.get('missing_pct', 0):.1f}%")
                    
                    if 'top_5_categories' in analysis:
                        st.subheader("Top 5 Categories")
                        top_df = pd.DataFrame(analysis['top_5_categories'])
                        st.dataframe(top_df, use_container_width=True)
                
                # Recommendations
                if col_info.get('recommendations'):
                    st.subheader("Recommendations")
                    for rec in col_info['recommendations']:
                        if '⚠️' in rec or 'High' in rec or 'consider dropping' in rec:
                            st.warning(rec)
                        elif '✓' in rec:
                            st.success(rec)
                        else:
                            st.info(rec)
    
    # Tab 5: Data Cleaning Wizard
    with tab5:
        st.header("🧹 Data Cleaning Wizard")
        
        # Import wizard manager
        from cleaner.wizards.step_manager import WizardStepManager
        from cleaner.wizards import visualizations as viz
        
        # Initialize wizard
        WizardStepManager.initialize()
        
        # Mode Selector
        if st.session_state.wizard_mode is None:
            st.markdown("### Choose Your Cleaning Mode")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 20px; border-radius: 10px; color: white;'>
                    <h3>⚡ Quick Mode</h3>
                    <p>One-click automated cleaning with recommended defaults</p>
                    <ul>
                        <li>Fast & Simple</li>
                        <li>Best practices applied</li>
                        <li>Ideal for rapid prototypes</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("🚀 Start Quick Mode", use_container_width=True, type="primary"):
                    WizardStepManager.start_wizard(st.session_state.df_original, mode='quick')
                    st.rerun()
            
            with col2:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                            padding: 20px; border-radius: 10px; color: white;'>
                    <h3>🎯 Advanced Mode</h3>
                    <p>Step-by-step wizard with full control & previews</p>
                    <ul>
                        <li>Column-by-column control</li>
                        <li>Detailed visualizations</li>
                        <li>Ideal for data scientists</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("🔬 Start Advanced Mode", use_container_width=True):
                    WizardStepManager.start_wizard(st.session_state.df_original, mode='advanced')
                    st.rerun()
        
        # Quick Mode Execution
        elif st.session_state.wizard_mode == 'quick':
            st.info("🚀 Running Quick Mode - Automated Cleaning with Best Practices")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            if st.button("↩️ Back to Mode Selection"):
                WizardStepManager.reset_wizard()
                st.rerun()
            
            with st.spinner("Executing cleaning pipeline..."):
                try:
                    cleaner = transformers.DataCleaner(st.session_state.df_original)
                    
                    # Step 1: Standardize names
                    status_text.text("Step 1/8: Standardizing column names...")
                    progress_bar.progress(12)
                    cleaner.standardize_column_names()
                    
                    # Step 2: Drop constant columns
                    status_text.text("Step 2/8: Dropping constant columns...")
                    progress_bar.progress(25)
                    cleaner.drop_constant_columns()
                    
                    # Step 3: Drop high missing
                    status_text.text("Step 3/8: Dropping high missing columns...")
                    progress_bar.progress(37)
                    cleaner.drop_high_missing_columns(0.9)
                    
                    # Step 4: Remove duplicates
                    status_text.text("Step 4/8: Removing duplicates...")
                    progress_bar.progress(50)
                    cleaner.remove_duplicates()
                    
                    # Step 5: Impute numeric
                    status_text.text("Step 5/8: Imputing numeric values...")
                    progress_bar.progress(62)
                    cleaner.impute_missing_numeric(strategy='median')
                    
                    # Step 6: Impute categorical
                    status_text.text("Step 6/8: Imputing categorical values...")
                    progress_bar.progress(75)
                    cleaner.impute_missing_categorical(strategy='mode')
                    
                    # Step 7: Handle outliers
                    status_text.text("Step 7/8: Flagging outliers...")
                    progress_bar.progress(87)
                    cleaner.add_outlier_flags(method='iqr')
                    
                    # Get cleaned data
                    st.session_state.df_cleaned = cleaner.get_cleaned_data()
                    st.session_state.cleaning_log = cleaner.get_cleaning_log()
                    
                    # Step 8: Optimize memory
                    status_text.text("Step 8/8: Optimizing memory...")
                    progress_bar.progress(100)
                    st.session_state.df_cleaned, opt_report = schema_infer.optimize_dtypes(
                        st.session_state.df_cleaned
                    )
                    st.session_state.cleaning_log['memory_optimization'] = opt_report
                    
                    status_text.empty()
                    st.success("✅ Quick Mode Cleaning Completed!")
                    
                    # Show summary
                    st.subheader("📊 Cleaning Summary")
                    log = st.session_state.cleaning_log
                    stats = log.get('statistics', {})
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Original Shape", 
                               f"{stats.get('original_shape', (0,0))[0]:,} × {stats.get('original_shape', (0,0))[1]}")
                    col2.metric("Final Shape", 
                               f"{stats.get('final_shape', (0,0))[0]:,} × {stats.get('final_shape', (0,0))[1]}")
                    col3.metric("Memory Saved", 
                               f"{stats.get('memory_reduction_mb', 0):.2f} MB")
                    
                    with st.expander("📋 View Detailed Actions"):
                        for idx, action in enumerate(log.get('actions', []), 1):
                            st.markdown(f"**{idx}. {action.get('action', 'Unknown')}**")
                            details = action.get('details', {})
                            
                            # Display details in a user-friendly format
                            if details:
                                detail_items = []
                                for key, value in details.items():
                                    # Format key to be more readable
                                    readable_key = key.replace('_', ' ').title()
                                    
                                    # Format value based on type
                                    if isinstance(value, (int, float)):
                                        if isinstance(value, float):
                                            formatted_value = f"{value:,.2f}" if value >= 1 else f"{value:.4f}"
                                        else:
                                            formatted_value = f"{value:,}"
                                    elif isinstance(value, list):
                                        if len(value) <= 5:
                                            formatted_value = ", ".join(str(v) for v in value)
                                        else:
                                            formatted_value = f"{', '.join(str(v) for v in value[:5])}... ({len(value)} total)"
                                    elif isinstance(value, dict):
                                        formatted_value = ", ".join(f"{k}: {v}" for k, v in list(value.items())[:3])
                                        if len(value) > 3:
                                            formatted_value += f"... ({len(value)} items)"
                                    else:
                                        formatted_value = str(value)
                                    
                                    detail_items.append(f"• **{readable_key}:** {formatted_value}")
                                
                                for item in detail_items:
                                    st.markdown(item)
                            else:
                                st.caption("No additional details")
                            
                            st.divider()
                    
                except Exception as e:
                    st.error(f"❌ Cleaning error: {str(e)}")
        
        # Advanced Mode - Step-by-Step Wizard
        elif st.session_state.wizard_mode == 'advanced':
            # Progress bar
            progress = WizardStepManager.get_progress_percentage()
            st.progress(progress / 100)
            
            current_step = WizardStepManager.get_current_step()
            step_info = WizardStepManager.get_step_info(current_step)
            
            st.markdown(f"### Step {current_step} of 8: {step_info['name']}")
            
            # Navigation buttons
            nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 1, 3])
            
            with nav_col1:
                if current_step > 1:
                    if st.button("◀ Back"):
                        WizardStepManager.previous_step()
                        st.rerun()
            
            with nav_col2:
                if step_info['skippable']:
                    if st.button("⏭️ Skip"):
                        WizardStepManager.skip_step()
                        st.rerun()
            
            with nav_col3:
                if st.button("↩️ Reset"):
                    WizardStepManager.reset_wizard()
                    st.rerun()
            
            st.divider()
            
            # STEP 1: Standardize Column Names
            if current_step == 1:
                st.subheader("📝 Standardize Column Names")
                st.info("Clean and standardize column names: lowercase, underscores, no special characters")
                
                df = st.session_state.wizard_data if st.session_state.wizard_data is not None else st.session_state.df_original
                
                # Preview changes
                original_cols = list(df.columns)
                new_cols = []
                for col in df.columns:
                    new_col = str(col).lower()
                    new_col = new_col.replace(' ', '_').replace('-', '_')
                    new_col = ''.join(c if c.isalnum() or c == '_' else '_' for c in new_col)
                    while '__' in new_col:
                        new_col = new_col.replace('__', '_')
                    new_col = new_col.strip('_')
                    new_cols.append(new_col)
                
                changes = [(old, new) for old, new in zip(original_cols, new_cols) if old != new]
                
                if changes:
                    st.warning(f"⚠️ {len(changes)} column names will be changed")
                    
                    change_df = pd.DataFrame(changes, columns=['Original Name', 'Standardized Name'])
                    st.dataframe(change_df, use_container_width=True)
                else:
                    st.success("✓ All column names are already standardized!")
                
                st.metric("Total Columns", len(df.columns))
                
                if st.button("✅ Apply Standardization & Continue", type="primary"):
                    # Apply changes
                    cleaner = transformers.DataCleaner(df)
                    cleaner.standardize_column_names()
                    st.session_state.wizard_data = cleaner.get_cleaned_data()
                    
                    # Log action
                    WizardStepManager.log_step_action(1, "standardize_column_names", 
                                                     {"changes": len(changes)})
                    WizardStepManager.next_step()
                    st.rerun()
            
            # STEP 2: Remove Duplicates
            elif current_step == 2:
                st.subheader("🔄 Remove Duplicate Rows")
                st.info("Identify and remove duplicate rows from the dataset")
                
                df = st.session_state.wizard_data
                
                # Detect duplicates
                duplicates = df.duplicated()
                dup_count = duplicates.sum()
                total_rows = len(df)
                dup_percentage = (dup_count / total_rows) * 100 if total_rows > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Rows", f"{total_rows:,}")
                col2.metric("Duplicate Rows", f"{dup_count:,}")
                col3.metric("Duplicate %", f"{dup_percentage:.2f}%")
                
                if dup_count > 0:
                    st.warning(f"⚠️ Found {dup_count:,} duplicate rows")
                    
                    # Show sample duplicates
                    with st.expander("📋 View Sample Duplicate Rows"):
                        dup_samples = df[duplicates].head(10)
                        st.dataframe(dup_samples, use_container_width=True)
                    
                    # Visualization
                    buf = viz.generate_comparison_chart(
                        total_rows - dup_count, dup_count,
                        ("Unique Rows", "Duplicate Rows"),
                        "Duplicate Analysis"
                    )
                    st.image(buf, use_container_width=True)
                    
                    if st.button("✅ Remove Duplicates & Continue", type="primary"):
                        cleaner = transformers.DataCleaner(df)
                        cleaner.remove_duplicates()
                        st.session_state.wizard_data = cleaner.get_cleaned_data()
                        
                        WizardStepManager.log_step_action(2, "remove_duplicates", 
                                                         {"removed": dup_count})
                        WizardStepManager.next_step()
                        st.rerun()
                else:
                    st.success("✓ No duplicate rows found!")
                    
                    if st.button("➡️ Continue to Next Step", type="primary"):
                        WizardStepManager.next_step()
                        st.rerun()
            
            # STEP 3: Drop Constant Columns
            elif current_step == 3:
                st.subheader("🚫 Drop Constant Columns")
                st.info("Remove columns with only one unique value (provide no information)")
                
                df = st.session_state.wizard_data
                
                # Identify constant columns
                constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
                
                col1, col2 = st.columns(2)
                col1.metric("Total Columns", len(df.columns))
                col2.metric("Constant Columns", len(constant_cols))
                
                if constant_cols:
                    st.warning(f"⚠️ Found {len(constant_cols)} constant columns")
                    
                    # Show constant columns
                    const_data = []
                    for col in constant_cols:
                        unique_val = df[col].dropna().unique()
                        val = unique_val[0] if len(unique_val) > 0 else "All NaN"
                        const_data.append({
                            "Column": col,
                            "Unique Values": df[col].nunique(),
                            "Value": str(val)[:50]
                        })
                    
                    const_df = pd.DataFrame(const_data)
                    st.dataframe(const_df, use_container_width=True)
                    
                    # Visualization
                    buf = viz.generate_comparison_chart(
                        len(df.columns) - len(constant_cols), len(constant_cols),
                        ("Variable Columns", "Constant Columns"),
                        "Column Analysis"
                    )
                    st.image(buf, use_container_width=True)
                    
                    if st.button("✅ Drop Constant Columns & Continue", type="primary"):
                        cleaner = transformers.DataCleaner(df)
                        cleaner.drop_constant_columns()
                        st.session_state.wizard_data = cleaner.get_cleaned_data()
                        
                        WizardStepManager.log_step_action(3, "drop_constant_columns", 
                                                         {"dropped": len(constant_cols)})
                        WizardStepManager.next_step()
                        st.rerun()
                else:
                    st.success("✓ No constant columns found!")
                    
                    if st.button("➡️ Continue to Next Step", type="primary"):
                        WizardStepManager.next_step()
                        st.rerun()
            
            # STEP 4: Drop High Missing Columns
            elif current_step == 4:
                st.subheader("❌ Drop High Missing Columns")
                st.info("Remove columns with excessive missing values")
                
                df = st.session_state.wizard_data
                
                # Threshold slider
                threshold = st.slider("Missing Value Threshold", 
                                     min_value=0.5, max_value=1.0, 
                                     value=0.9, step=0.05,
                                     help="Drop columns with missing % above this threshold")
                
                # Calculate missing percentages
                missing_pcts = df.isnull().mean().sort_values(ascending=False)
                high_missing_cols = missing_pcts[missing_pcts > threshold].index.tolist()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Columns", len(df.columns))
                col2.metric(f"Above {threshold*100:.0f}% Missing", len(high_missing_cols))
                col3.metric("Will Keep", len(df.columns) - len(high_missing_cols))
                
                # Show all columns with missing data
                if len(missing_pcts[missing_pcts > 0]) > 0:
                    missing_data = []
                    for col in missing_pcts[missing_pcts > 0].index:
                        pct = missing_pcts[col]
                        count = df[col].isnull().sum()
                        above_threshold = "✓ Drop" if pct > threshold else "✗ Keep"
                        missing_data.append({
                            "Column": col,
                            "Missing %": f"{pct*100:.1f}%",
                            "Missing Count": f"{count:,}",
                            "Total": len(df),
                            "Action": above_threshold
                        })
                    
                    missing_df = pd.DataFrame(missing_data)
                    st.dataframe(missing_df, use_container_width=True)
                    
                    # Visualization: Bar chart
                    st.write("**Missing Value Percentages:**")
                    buf = viz.generate_bar_chart(
                        missing_pcts[missing_pcts > 0] * 100,
                        title=f"Missing Values by Column (Threshold: {threshold*100:.0f}%)",
                        xlabel="Columns",
                        ylabel="Missing Percentage (%)",
                        top_n=20
                    )
                    st.image(buf, use_container_width=True)
                    
                    if high_missing_cols:
                        st.warning(f"⚠️ {len(high_missing_cols)} columns will be dropped")
                        
                        if st.button("✅ Drop High Missing Columns & Continue", type="primary"):
                            cleaner = transformers.DataCleaner(df)
                            cleaner.drop_high_missing_columns(threshold)
                            st.session_state.wizard_data = cleaner.get_cleaned_data()
                            
                            WizardStepManager.log_step_action(4, "drop_high_missing_columns", 
                                                             {"threshold": threshold, 
                                                              "dropped": len(high_missing_cols)})
                            WizardStepManager.next_step()
                            st.rerun()
                    else:
                        st.success(f"✓ No columns above {threshold*100:.0f}% missing threshold!")
                        
                        if st.button("➡️ Continue to Next Step", type="primary"):
                            WizardStepManager.next_step()
                            st.rerun()
                else:
                    st.success("✓ No missing values in dataset!")
                    
                    if st.button("➡️ Continue to Next Step", type="primary"):
                        WizardStepManager.next_step()
                        st.rerun()
            
            # STEP 5: Impute Numeric Missing Values
            elif current_step == 5:
                st.subheader("🔢 Impute Numeric Missing Values")
                st.info("Fill missing values in numeric columns using statistical methods")
                
                df = st.session_state.wizard_data
                
                # Get numeric columns with missing values
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                numeric_missing = [col for col in numeric_cols if df[col].isnull().sum() > 0]
                
                # Store original list for progress tracking (only on first visit)
                if 'wizard_numeric_missing_original' not in st.session_state:
                    st.session_state.wizard_numeric_missing_original = numeric_missing.copy()
                
                # Use original list for progress, current list for processing
                numeric_missing_original = st.session_state.wizard_numeric_missing_original
                
                if not numeric_missing:
                    st.success("✓ No missing values in numeric columns!")
                    
                    if st.button("➡️ Continue to Next Step", type="primary"):
                        WizardStepManager.next_step()
                        st.rerun()
                else:
                    st.write(f"**Found {len(numeric_missing)} numeric columns with missing values**")
                    
                    # Quick Apply All option
                    with st.expander("⚡ Quick: Apply to All Columns"):
                        quick_strategy = st.selectbox(
                            "Select strategy for all numeric columns",
                            options=['median', 'mean', 'mode'],
                            index=0,
                            key="quick_numeric"
                        )
                        
                        if st.button("Apply to All Columns", type="primary", key="apply_all_numeric"):
                            progress_placeholder = st.empty()
                            status_placeholder = st.empty()
                            
                            cleaner = transformers.DataCleaner(df)
                            total_cols = len(numeric_missing)
                            
                            for idx, col in enumerate(numeric_missing, 1):
                                status_placeholder.text(f"Processing {idx}/{total_cols}: {col}")
                                progress_placeholder.progress(idx / total_cols)
                            
                            cleaner.impute_missing_numeric(columns=numeric_missing, strategy=quick_strategy)
                            st.session_state.wizard_data = cleaner.get_cleaned_data()
                            
                            for col in numeric_missing:
                                st.session_state.wizard_numeric_imputation[col] = quick_strategy
                            
                            status_placeholder.empty()
                            WizardStepManager.log_step_action(5, "impute_numeric", 
                                                             {"columns": numeric_missing, 
                                                              "strategy": quick_strategy})
                            WizardStepManager.next_step()
                            st.rerun()
                    
                    st.divider()
                    
                    # Column-by-column processing
                    st.write("**Column-by-Column Imputation:**")
                    
                    # Show only unprocessed columns in dropdown
                    unprocessed_cols = [col for col in numeric_missing if col not in st.session_state.wizard_numeric_imputation]
                    
                    if not unprocessed_cols:
                        st.success("✅ All numeric columns have been imputed!")
                        if st.button("➡️ Continue to Next Step", type="primary"):
                            WizardStepManager.next_step()
                            st.rerun()
                    else:
                        st.info(f"📊 {len(unprocessed_cols)} column(s) remaining to process")
                        
                        selected_col = st.selectbox(
                            "Select column to impute",
                            options=unprocessed_cols,
                            index=0,  # Always select first unprocessed
                            key="select_numeric_col"
                        )
                    
                    if selected_col:
                        col_data = df[selected_col]
                        missing_count = col_data.isnull().sum()
                        missing_pct = (missing_count / len(df)) * 100
                        
                        # Statistics
                        st.write(f"**Column: {selected_col}**")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Values", len(df))
                        col2.metric("Missing", f"{missing_count:,}")
                        col3.metric("Missing %", f"{missing_pct:.1f}%")
                        col4.metric("Non-null", f"{col_data.count():,}")
                        
                        # Calculate statistics
                        mean_val = col_data.mean()
                        median_val = col_data.median()
                        mode_vals = col_data.mode()
                        mode_val = mode_vals[0] if len(mode_vals) > 0 else None
                        std_val = col_data.std()
                        
                        # Statistics table
                        stats_df = pd.DataFrame({
                            'Measure': ['Mean', 'Median', 'Mode', 'Std Dev', 'Min', 'Max'],
                            'Value': [
                                f"{mean_val:.2f}" if pd.notna(mean_val) else "N/A",
                                f"{median_val:.2f}" if pd.notna(median_val) else "N/A",
                                f"{mode_val:.2f}" if mode_val is not None and pd.notna(mode_val) else "N/A",
                                f"{std_val:.2f}" if pd.notna(std_val) else "N/A",
                                f"{col_data.min():.2f}" if pd.notna(col_data.min()) else "N/A",
                                f"{col_data.max():.2f}" if pd.notna(col_data.max()) else "N/A"
                            ]
                        })
                        st.dataframe(stats_df, use_container_width=True)
                        
                        # Visualization
                        buf = viz.generate_distribution_plot(
                            col_data,
                            mean_val=mean_val,
                            median_val=median_val,
                            mode_val=mode_val,
                            title=f"Distribution: {selected_col}"
                        )
                        st.image(buf, use_container_width=True)
                        
                        # Strategy selection (pre-selected: median)
                        strategy = st.radio(
                            "Select imputation strategy",
                            options=['median', 'mean', 'mode', 'custom'],
                            index=0,  # Default to median
                            horizontal=True,
                            key=f"strategy_{selected_col}"
                        )
                        
                        fill_value = None
                        if strategy == 'custom':
                            fill_value = st.number_input(
                                "Custom value",
                                value=float(median_val) if pd.notna(median_val) else 0.0,
                                key=f"custom_{selected_col}"
                            )
                        
                        # Apply button
                        if st.button(f"✅ Apply to '{selected_col}'", type="primary", key=f"apply_{selected_col}"):
                            # Show processing status
                            with st.spinner(f"Processing '{selected_col}'..."):
                                cleaner = transformers.DataCleaner(df)
                                
                                if strategy == 'custom' and fill_value is not None:
                                    cleaner.impute_missing_numeric(columns=[selected_col], strategy=fill_value)
                                else:
                                    cleaner.impute_missing_numeric(columns=[selected_col], strategy=strategy)
                                
                                st.session_state.wizard_data = cleaner.get_cleaned_data()
                                st.session_state.wizard_numeric_imputation[selected_col] = strategy
                            
                            # Check if all columns done
                            remaining = [c for c in numeric_missing if c not in st.session_state.wizard_numeric_imputation]
                            if not remaining:
                                WizardStepManager.log_step_action(5, "impute_numeric", 
                                                                 {"imputed": st.session_state.wizard_numeric_imputation})
                                WizardStepManager.next_step()
                            
                            st.rerun()
                        
                        # Show progress - use original list for accurate tracking
                        completed = len([c for c in numeric_missing_original if c in st.session_state.wizard_numeric_imputation])
                        total_cols = len(numeric_missing_original)
                        progress_value = completed / total_cols if total_cols > 0 else 0
                        
                        st.progress(progress_value)
                        st.caption(f"Progress: {completed}/{total_cols} columns completed")
            
            # STEP 6: Impute Categorical Missing Values
            elif current_step == 6:
                st.subheader("📝 Impute Categorical Missing Values")
                st.info("Fill missing values in categorical columns")
                
                df = st.session_state.wizard_data
                
                # Get categorical columns with missing values
                cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                cat_missing = [col for col in cat_cols if df[col].isnull().sum() > 0]
                
                # Store original list for progress tracking (only on first visit)
                if 'wizard_categorical_missing_original' not in st.session_state:
                    st.session_state.wizard_categorical_missing_original = cat_missing.copy()
                
                # Use original list for progress, current list for processing
                cat_missing_original = st.session_state.wizard_categorical_missing_original
                
                if not cat_missing:
                    st.success("✓ No missing values in categorical columns!")
                    
                    if st.button("➡️ Continue to Next Step", type="primary"):
                        WizardStepManager.next_step()
                        st.rerun()
                else:
                    st.write(f"**Found {len(cat_missing)} categorical columns with missing values**")
                    
                    # Quick Apply All option
                    with st.expander("⚡ Quick: Apply to All Columns"):
                        if st.button("Apply Mode to All Columns", type="primary", key="apply_all_cat"):
                            progress_placeholder = st.empty()
                            status_placeholder = st.empty()
                            
                            cleaner = transformers.DataCleaner(df)
                            total_cols = len(cat_missing)
                            
                            for idx, col in enumerate(cat_missing, 1):
                                status_placeholder.text(f"Processing {idx}/{total_cols}: {col}")
                                progress_placeholder.progress(idx / total_cols)
                            
                            cleaner.impute_missing_categorical(columns=cat_missing, strategy='mode')
                            st.session_state.wizard_data = cleaner.get_cleaned_data()
                            
                            for col in cat_missing:
                                st.session_state.wizard_categorical_imputation[col] = 'mode'
                            
                            status_placeholder.empty()
                            WizardStepManager.log_step_action(6, "impute_categorical", 
                                                             {"columns": cat_missing, "strategy": "mode"})
                            WizardStepManager.next_step()
                            st.rerun()
                    
                    st.divider()
                    
                    # Column-by-column processing
                    st.write("**Column-by-Column Imputation:**")
                    
                    # Show only unprocessed columns in dropdown
                    unprocessed_cols = [col for col in cat_missing if col not in st.session_state.wizard_categorical_imputation]
                    
                    if not unprocessed_cols:
                        st.success("✅ All categorical columns have been imputed!")
                        if st.button("➡️ Continue to Next Step", type="primary"):
                            WizardStepManager.next_step()
                            st.rerun()
                    else:
                        st.info(f"📊 {len(unprocessed_cols)} column(s) remaining to process")
                        
                        selected_col = st.selectbox(
                            "Select column to impute",
                            options=unprocessed_cols,
                            index=0,  # Always select first unprocessed
                            key="select_cat_col"
                        )
                    
                    if selected_col:
                        col_data = df[selected_col]
                        missing_count = col_data.isnull().sum()
                        missing_pct = (missing_count / len(df)) * 100
                        
                        # Statistics
                        st.write(f"**Column: {selected_col}**")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Values", len(df))
                        col2.metric("Missing", f"{missing_count:,}")
                        col3.metric("Missing %", f"{missing_pct:.1f}%")
                        
                        # Value frequency table
                        value_counts = col_data.value_counts().head(10)
                        freq_data = []
                        cumsum = 0
                        for val, count in value_counts.items():
                            pct = (count / col_data.count()) * 100
                            cumsum += pct
                            freq_data.append({
                                'Category': str(val),
                                'Count': f"{count:,}",
                                'Percentage': f"{pct:.1f}%",
                                'Cumulative %': f"{cumsum:.1f}%"
                            })
                        
                        freq_df = pd.DataFrame(freq_data)
                        st.dataframe(freq_df, use_container_width=True)
                        
                        # Visualization
                        buf = viz.generate_bar_chart(
                            col_data,
                            title=f"Category Distribution: {selected_col}",
                            xlabel="Category",
                            ylabel="Count",
                            top_n=10
                        )
                        st.image(buf, use_container_width=True)
                        
                        # Strategy selection (pre-selected: mode)
                        strategy = st.radio(
                            "Select imputation strategy",
                            options=['mode', 'custom'],
                            index=0,  # Default to mode
                            horizontal=True,
                            key=f"cat_strategy_{selected_col}"
                        )
                        
                        fill_value = None
                        if strategy == 'custom':
                            fill_value = st.text_input(
                                "Custom value",
                                value="MISSING",
                                key=f"cat_custom_{selected_col}"
                            )
                        
                        # Apply button
                        if st.button(f"✅ Apply to '{selected_col}'", type="primary", key=f"cat_apply_{selected_col}"):
                            # Show processing status
                            with st.spinner(f"Processing '{selected_col}'..."):
                                cleaner = transformers.DataCleaner(df)
                                
                                if strategy == 'custom':
                                    cleaner.impute_missing_categorical(columns=[selected_col], 
                                                                       strategy='constant', 
                                                                       fill_value=fill_value)
                                else:
                                    cleaner.impute_missing_categorical(columns=[selected_col], strategy='mode')
                                
                                st.session_state.wizard_data = cleaner.get_cleaned_data()
                                st.session_state.wizard_categorical_imputation[selected_col] = strategy
                            
                            # Check if all columns done
                            remaining = [c for c in cat_missing if c not in st.session_state.wizard_categorical_imputation]
                            if not remaining:
                                WizardStepManager.log_step_action(6, "impute_categorical", 
                                                                 {"imputed": st.session_state.wizard_categorical_imputation})
                                WizardStepManager.next_step()
                            
                            st.rerun()
                        
                        # Show progress - use original list for accurate tracking
                        completed = len([c for c in cat_missing_original if c in st.session_state.wizard_categorical_imputation])
                        total_cols = len(cat_missing_original)
                        progress_value = completed / total_cols if total_cols > 0 else 0
                        
                        st.progress(progress_value)
                        st.caption(f"Progress: {completed}/{total_cols} columns completed")
            
            # STEP 7: Handle Outliers
            elif current_step == 7:
                st.subheader("📊 Handle Outliers")
                st.info("Detect and handle outliers in numeric columns")
                
                df = st.session_state.wizard_data
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                # Filter out outlier flag columns
                numeric_cols = [col for col in numeric_cols if not col.endswith('_outlier')]
                
                if not numeric_cols:
                    st.warning("No numeric columns available for outlier detection")
                    if st.button("⏭️ Skip to Next Step"):
                        WizardStepManager.skip_step()
                        st.rerun()
                else:
                    # Quick Apply All option with pre-selected defaults
                    with st.expander("⚡ Quick: Apply to All Columns"):
                        col1, col2 = st.columns(2)
                        with col1:
                            quick_method = st.selectbox(
                                "Detection Method",
                                options=['iqr', 'zscore'],
                                index=0,  # Default to IQR
                                key="quick_out_method"
                            )
                        with col2:
                            quick_action = st.selectbox(
                                "Handling Strategy",
                                options=['flag', 'remove', 'cap', 'none'],
                                index=0,  # Default to flag
                                key="quick_out_action"
                            )
                        
                        if st.button("Apply to All Columns", type="primary", key="apply_all_outliers"):
                            cleaner = transformers.DataCleaner(df)
                            
                            if quick_action == 'flag':
                                cleaner.add_outlier_flags(columns=numeric_cols, method=quick_method)
                            elif quick_action == 'remove':
                                cleaner.remove_outliers(columns=numeric_cols, method=quick_method)
                            elif quick_action == 'cap':
                                cleaner.cap_outliers(columns=numeric_cols, method=quick_method)
                            
                            st.session_state.wizard_data = cleaner.get_cleaned_data()
                            
                            for col in numeric_cols:
                                st.session_state.wizard_outlier_handling[col] = {
                                    'method': quick_method,
                                    'action': quick_action
                                }
                            
                            WizardStepManager.log_step_action(7, "handle_outliers", 
                                                             {"columns": numeric_cols, 
                                                              "method": quick_method,
                                                              "action": quick_action})
                            WizardStepManager.next_step()
                            st.rerun()
                    
                    st.divider()
                    
                    # Column-by-column processing
                    st.write("**Column-by-Column Outlier Detection:**")
                    
                    # Show only unprocessed columns in dropdown
                    unprocessed_cols = [col for col in numeric_cols if col not in st.session_state.wizard_outlier_handling]
                    
                    if not unprocessed_cols:
                        st.success("✅ All columns have been processed!")
                        if st.button("➡️ Continue to Next Step", type="primary"):
                            WizardStepManager.next_step()
                            st.rerun()
                    else:
                        st.info(f"📊 {len(unprocessed_cols)} column(s) remaining to process")
                        
                        selected_col = st.selectbox(
                            "Select column",
                            options=unprocessed_cols,
                            index=0,  # Always select first unprocessed
                            key="select_outlier_col"
                        )
                    
                    if selected_col:
                        col_data = df[selected_col].dropna()
                        
                        st.write(f"**Column: {selected_col}**")
                        
                        # Calculate outliers with both methods
                        q1 = col_data.quantile(0.25)
                        q3 = col_data.quantile(0.75)
                        iqr = q3 - q1
                        iqr_lower = q1 - 1.5 * iqr
                        iqr_upper = q3 + 1.5 * iqr
                        iqr_outliers = ((col_data < iqr_lower) | (col_data > iqr_upper)).sum()
                        iqr_pct = (iqr_outliers / len(col_data)) * 100 if len(col_data) > 0 else 0
                        
                        from scipy import stats
                        z_scores = np.abs(stats.zscore(col_data))
                        zscore_outliers = (z_scores > 3.0).sum()
                        zscore_pct = (zscore_outliers / len(col_data)) * 100 if len(col_data) > 0 else 0
                        
                        # Calculate Z-score bounds
                        mean_val = col_data.mean()
                        std_val = col_data.std()
                        zscore_lower = mean_val - 3 * std_val
                        zscore_upper = mean_val + 3 * std_val
                        
                        # Method comparison table
                        comparison_df = pd.DataFrame({
                            'Method': ['IQR', 'Z-Score'],
                            'Outliers Detected': [f"{iqr_outliers:,}", f"{zscore_outliers:,}"],
                            'Percentage': [f"{iqr_pct:.2f}%", f"{zscore_pct:.2f}%"],
                            'Lower Bound': [f"{iqr_lower:.2f}", f"{zscore_lower:.2f}"],
                            'Upper Bound': [f"{iqr_upper:.2f}", f"{zscore_upper:.2f}"]
                        })
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Visualization: Box plot
                        buf = viz.generate_boxplot(
                            df[selected_col],
                            title=f"Box Plot: {selected_col}",
                            outliers_highlighted=True
                        )
                        st.image(buf, use_container_width=True)
                        
                        # Selection
                        col1, col2 = st.columns(2)
                        with col1:
                            method = st.radio(
                                "Detection Method",
                                options=['iqr', 'zscore'],
                                index=0,  # Default to IQR
                                key=f"method_{selected_col}"
                            )
                        with col2:
                            action = st.radio(
                                "Handling Strategy",
                                options=['flag', 'remove', 'cap', 'none'],
                                index=0,  # Default to flag
                                key=f"action_{selected_col}",
                                help="flag: add indicator column, remove: delete rows, cap: winsorize"
                            )
                        
                        # Apply button
                        if st.button(f"✅ Apply to '{selected_col}'", type="primary", key=f"outlier_apply_{selected_col}"):
                            cleaner = transformers.DataCleaner(df)
                            
                            if action == 'flag':
                                cleaner.add_outlier_flags(columns=[selected_col], method=method)
                            elif action == 'remove':
                                cleaner.remove_outliers(columns=[selected_col], method=method)
                            elif action == 'cap':
                                cleaner.cap_outliers(columns=[selected_col], method=method)
                            
                            st.session_state.wizard_data = cleaner.get_cleaned_data()
                            st.session_state.wizard_outlier_handling[selected_col] = {
                                'method': method,
                                'action': action
                            }
                            
                            # Check if all columns done
                            remaining = [c for c in numeric_cols if c not in st.session_state.wizard_outlier_handling]
                            if not remaining:
                                WizardStepManager.log_step_action(7, "handle_outliers", 
                                                                 {"handled": st.session_state.wizard_outlier_handling})
                                WizardStepManager.next_step()
                            
                            st.rerun()
                        
                        # Show progress
                        completed = len([c for c in numeric_cols if c in st.session_state.wizard_outlier_handling])
                        st.progress(completed / len(numeric_cols))
                        st.caption(f"Progress: {completed}/{len(numeric_cols)} columns processed")
                        
                        if completed == len(numeric_cols):
                            if st.button("➡️ All Columns Processed - Continue", type="primary"):
                                WizardStepManager.next_step()
                                st.rerun()
            
            # STEP 8: Optimize Memory
            elif current_step == 8:
                st.subheader("💾 Optimize Memory")
                st.info("Reduce memory usage by downcasting data types")
                
                df = st.session_state.wizard_data
                
                # Analyze optimization opportunities
                original_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
                
                # Calculate potential optimizations
                opt_opportunities = []
                test_df = df.copy()
                
                for col in df.columns:
                    col_type = df[col].dtype
                    current_mem = df[col].memory_usage(deep=True) / (1024 * 1024)
                    
                    suggested_type = None
                    potential_mem = current_mem
                    savings_pct = 0
                    
                    # Check integer optimization
                    if pd.api.types.is_integer_dtype(col_type):
                        c_min = df[col].min()
                        c_max = df[col].max()
                        
                        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                            suggested_type = 'int8'
                            test_df[col] = df[col].astype(np.int8)
                        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                            suggested_type = 'int16'
                            test_df[col] = df[col].astype(np.int16)
                        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                            suggested_type = 'int32'
                            test_df[col] = df[col].astype(np.int32)
                        
                        if suggested_type:
                            potential_mem = test_df[col].memory_usage(deep=True) / (1024 * 1024)
                            savings_pct = ((current_mem - potential_mem) / current_mem) * 100
                    
                    # Check float optimization
                    elif pd.api.types.is_float_dtype(col_type):
                        c_min = df[col].min()
                        c_max = df[col].max()
                        
                        if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                            suggested_type = 'float32'
                            test_df[col] = df[col].astype(np.float32)
                            potential_mem = test_df[col].memory_usage(deep=True) / (1024 * 1024)
                            savings_pct = ((current_mem - potential_mem) / current_mem) * 100
                    
                    # Check category optimization
                    elif col_type == 'object':
                        num_unique = df[col].nunique()
                        num_total = len(df[col])
                        if num_unique / num_total < 0.5:
                            suggested_type = 'category'
                            test_df[col] = df[col].astype('category')
                            potential_mem = test_df[col].memory_usage(deep=True) / (1024 * 1024)
                            savings_pct = ((current_mem - potential_mem) / current_mem) * 100
                    
                    if suggested_type and savings_pct > 1:  # Only show if > 1% savings
                        opt_opportunities.append({
                            'Column': col,
                            'Current Type': str(col_type),
                            'Suggested Type': suggested_type,
                            'Current Memory (MB)': f"{current_mem:.3f}",
                            'Optimized Memory (MB)': f"{potential_mem:.3f}",
                            'Savings': f"{savings_pct:.1f}%"
                        })
                
                # Calculate total potential savings
                optimized_df, opt_report = schema_infer.optimize_dtypes(df)
                optimized_memory = optimized_df.memory_usage(deep=True).sum() / (1024 * 1024)
                total_savings = original_memory - optimized_memory
                savings_pct = (total_savings / original_memory) * 100 if original_memory > 0 else 0
                
                # Display summary
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Memory", f"{original_memory:.2f} MB")
                col2.metric("After Optimization", f"{optimized_memory:.2f} MB")
                col3.metric("Savings", f"{total_savings:.2f} MB ({savings_pct:.1f}%)")
                
                if opt_opportunities:
                    st.write(f"**Found {len(opt_opportunities)} optimization opportunities:**")
                    opt_df = pd.DataFrame(opt_opportunities)
                    st.dataframe(opt_df, use_container_width=True)
                    
                    # Visualization
                    buf = viz.generate_comparison_chart(
                        original_memory, optimized_memory,
                        ("Before Optimization", "After Optimization"),
                        "Memory Usage Comparison (MB)"
                    )
                    st.image(buf, use_container_width=True)
                    
                    if st.button("✅ Apply Memory Optimization & Complete Wizard", type="primary"):
                        st.session_state.wizard_data = optimized_df
                        st.session_state.df_cleaned = optimized_df
                        
                        # Create comprehensive cleaning log
                        cleaning_log = {
                            'actions': [],
                            'statistics': {
                                'original_shape': st.session_state.wizard_original_data.shape,
                                'final_shape': optimized_df.shape,
                                'original_memory_mb': st.session_state.wizard_original_data.memory_usage(deep=True).sum() / (1024*1024),
                                'final_memory_mb': optimized_df.memory_usage(deep=True).sum() / (1024*1024)
                            },
                            'memory_optimization': opt_report
                        }
                        
                        # Add all step logs
                        for step_id in range(1, 9):
                            logs = WizardStepManager.get_step_logs(step_id)
                            for log in logs:
                                cleaning_log['actions'].append(log)
                        
                        st.session_state.cleaning_log = cleaning_log
                        
                        WizardStepManager.log_step_action(8, "optimize_memory", opt_report)
                        WizardStepManager.next_step()
                        st.rerun()
                else:
                    st.success("✓ No significant memory optimization opportunities found!")
                    
                    if st.button("✅ Complete Wizard", type="primary"):
                        st.session_state.df_cleaned = df
                        WizardStepManager.next_step()
                        st.rerun()
            
            # Wizard Complete
            elif WizardStepManager.is_wizard_complete():
                st.success("🎉 Data Cleaning Wizard Complete!")
                
                st.subheader("📊 Final Summary")
                
                original_df = st.session_state.wizard_original_data
                final_df = st.session_state.wizard_data
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Original Shape", f"{original_df.shape[0]:,} × {original_df.shape[1]}")
                col2.metric("Final Shape", f"{final_df.shape[0]:,} × {final_df.shape[1]}")
                col3.metric("Memory Saved", 
                           f"{(original_df.memory_usage(deep=True).sum() - final_df.memory_usage(deep=True).sum()) / (1024*1024):.2f} MB")
                
                # Show completed steps
                st.subheader("✅ Completed Steps")
                for step_id in WizardStepManager.STEPS:
                    if WizardStepManager.is_step_completed(step_id['id']):
                        st.success(f"✓ Step {step_id['id']}: {step_id['name']}")
                    elif WizardStepManager.is_step_skipped(step_id['id']):
                        st.info(f"⏭️ Step {step_id['id']}: {step_id['name']} (Skipped)")
                
                if st.button("🔄 Start New Wizard"):
                    WizardStepManager.reset_wizard()
                    st.rerun()
    
    # Tab 6: EDA Visualizations
    with tab6:
        st.header("Exploratory Data Analysis")
        
        if st.button("📈 Generate Visualizations") or st.session_state.plot_metadata:
            df_to_viz = st.session_state.df_cleaned if st.session_state.df_cleaned is not None else st.session_state.df_original
            
            with st.spinner("Generating visualizations..."):
                try:
                    visualizer = eda.EDAVisualizer(df_to_viz, 'output')
                    plot_report = visualizer.generate_all_plots()
                    st.session_state.plot_metadata = plot_report['plot_metadata']
                    
                    st.session_state.correlation_insights = eda.get_correlation_insights(df_to_viz)
                    
                    st.success(f"✅ Generated {plot_report['total_plots']} plots")
                    
                except Exception as e:
                    st.error(f"Visualization error: {str(e)}")
        
        if st.session_state.plot_metadata:
            # Group plots
            plots_by_type = {}
            for plot in st.session_state.plot_metadata:
                plot_type = plot.get('type', 'other')
                if plot_type not in plots_by_type:
                    plots_by_type[plot_type] = []
                plots_by_type[plot_type].append(plot)
            
            # Show correlation matrix
            if 'correlation' in plots_by_type:
                st.subheader("Correlation Analysis")
                for plot in plots_by_type['correlation']:
                    st.image(plot['filepath'], caption=plot['title'])
                
                # Show correlation insights
                if st.session_state.correlation_insights:
                    insights = st.session_state.correlation_insights
                    if not insights.get('error'):
                        st.write(f"**High Correlations Found:** {insights['high_correlations_count']}")
                        
                        if insights.get('high_correlations'):
                            corr_df = pd.DataFrame(insights['high_correlations'][:10])
                            st.dataframe(corr_df, use_container_width=True)
            
            # Show missing values
            if 'missing_values' in plots_by_type:
                st.subheader("Missing Values Analysis")
                for plot in plots_by_type['missing_values']:
                    st.image(plot['filepath'], caption=plot['title'])
            
            # Show outliers
            if 'outliers' in plots_by_type:
                st.subheader("Outlier Analysis")
                for plot in plots_by_type['outliers']:
                    st.image(plot['filepath'], caption=plot['title'])
            
            # Sample distributions
            if 'distribution' in plots_by_type:
                st.subheader("Distribution Plots (Sample)")
                for plot in plots_by_type['distribution'][:5]:
                    st.image(plot['filepath'], caption=plot['title'])
                
                remaining = len(plots_by_type['distribution']) - 5
                if remaining > 0:
                    st.info(f"... and {remaining} more distribution plots available in output folder")
    
    # Tab 7: Report
    with tab7:
        st.header("Generate Report")
        
        report_format = st.radio(
            "Report Format",
            options=['HTML', 'Markdown'],
            index=0,
            horizontal=True
        )
        
        if st.button("📄 Generate Report", type="primary"):
            # Ensure all data is generated
            if st.session_state.overview_data is None:
                st.session_state.overview_data = overview.get_dataset_overview(
                    st.session_state.df_original
                )
            
            if st.session_state.schema_data is None:
                st.session_state.schema_data = schema_infer.infer_schema(
                    st.session_state.df_original
                )
            
            if st.session_state.column_analysis_data is None:
                df_to_analyze = st.session_state.df_cleaned if st.session_state.df_cleaned is not None else st.session_state.df_original
                st.session_state.column_analysis_data = column_analysis.analyze_all_columns(
                    df_to_analyze
                )
            
            with st.spinner("Generating report..."):
                try:
                    if report_format == 'HTML':
                        html_content = report.generate_html_report(
                            st.session_state.overview_data,
                            st.session_state.schema_data,
                            st.session_state.column_analysis_data,
                            st.session_state.cleaning_log or {},
                            st.session_state.plot_metadata,
                            st.session_state.correlation_insights
                        )
                        
                        report_path = utils_io.save_report(html_content, filename='eda_report.html')
                        st.success(f"✅ HTML report saved: {report_path}")
                        
                        # Show preview
                        with st.expander("📄 Preview HTML Report"):
                            st.components.v1.html(html_content, height=600, scrolling=True)
                    
                    else:  # Markdown
                        md_content = report.generate_markdown_report(
                            st.session_state.overview_data,
                            st.session_state.schema_data,
                            st.session_state.column_analysis_data,
                            st.session_state.cleaning_log or {}
                        )
                        
                        report_path = utils_io.save_report(md_content, filename='eda_report.md')
                        st.success(f"✅ Markdown report saved: {report_path}")
                        
                        # Show preview
                        with st.expander("📄 Preview Markdown Report"):
                            st.markdown(md_content)
                
                except Exception as e:
                    st.error(f"Report generation error: {str(e)}")
    
    # Tab 8: Downloads
    with tab8:
        st.header("Download Outputs")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Cleaned Dataset")
            if st.session_state.df_cleaned is not None:
                csv = st.session_state.df_cleaned.to_csv(index=False)
                st.download_button(
                    label="💾 Download Cleaned CSV",
                    data=csv,
                    file_name="cleaned_data.csv",
                    mime="text/csv"
                )
                
                st.metric("Cleaned Rows", f"{len(st.session_state.df_cleaned):,}")
                st.metric("Cleaned Columns", len(st.session_state.df_cleaned.columns))
            else:
                st.info("Run cleaning pipeline first")
        
        with col2:
            st.subheader("📋 Cleaning Log")
            if st.session_state.cleaning_log:
                log_json = json.dumps(st.session_state.cleaning_log, indent=2, default=str)
                st.download_button(
                    label="💾 Download Cleaning Log (JSON)",
                    data=log_json,
                    file_name="cleaning_log.json",
                    mime="application/json"
                )
                
                st.metric("Actions Applied", len(st.session_state.cleaning_log.get('actions', [])))
            else:
                st.info("Run cleaning pipeline first")
        
        st.divider()
        
        # Bulk download info
        st.subheader("📁 All Outputs")
        st.info("""
        All generated files are saved in the `output/` directory:
        - Cleaned CSV
        - EDA visualizations (PNG)
        - HTML/Markdown reports
        - Cleaning log (JSON)
        """)

else:
    # Landing page
    st.info("👈 Upload a CSV file in the sidebar to get started")
    
    st.markdown("""
    ### 🚀 Features
    
    - **Automated Data Cleaning**: Handle missing values, duplicates, outliers
    - **Schema Inference**: Intelligent type detection and conversion
    - **Column Analysis**: Detailed statistics for every feature
    - **Visual EDA**: Comprehensive visualizations (distributions, correlations, outliers)
    - **Professional Reports**: Generate HTML/Markdown reports
    - **Memory Optimization**: Reduce dataset size through dtype optimization
    - **Full Auto Mode**: Execute entire pipeline with one click
    
    ### 📊 Supported Analysis
    
    1. Dataset Overview & Health Scoring
    2. Missing Value Analysis
    3. Duplicate Detection
    4. Outlier Detection (IQR & Z-score)
    5. Correlation Analysis
    6. Distribution Analysis
    7. Data Type Optimization
    
    ### 🎯 Quick Start
    
    1. Upload your CSV file
    2. Choose between manual configuration or Full Auto Mode
    3. View analysis results in interactive tabs
    4. Download cleaned data and reports
    """)


# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    <p>PrometheusAI Data Engine v1.0.0 | Automated Data Preparation, EDA & AI-Powered Insight Engine </p>
</div>
""", unsafe_allow_html=True)
