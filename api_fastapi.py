"""
FastAPI Backend for EDA Engine
REST API endpoints for data analysis automation
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import pandas as pd
import os
import uuid
import shutil
from datetime import datetime

# Import cleaner modules
from cleaner import schema_infer, overview, column_analysis, transformers, eda, report
from utils import io as utils_io


app = FastAPI(
    title="EDA Engine API",
    description="Automated CSV cleaning and EDA API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage (use Redis/DB for production)
sessions = {}
MAX_FILE_SIZE_MB = 500


class CleaningConfig(BaseModel):
    """Configuration for cleaning pipeline"""
    session_id: str
    remove_duplicates: bool = True
    drop_high_missing_threshold: float = 0.9
    drop_constant_columns: bool = True
    impute_numeric_strategy: str = "median"
    impute_categorical_strategy: str = "mode"
    handle_outliers: str = "flag"  # "remove", "cap", "flag", or "none"
    outlier_method: str = "iqr"  # "iqr" or "zscore"
    standardize_column_names: bool = True
    optimize_memory: bool = True


class SessionData:
    """Session data container"""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.df_original = None
        self.df_cleaned = None
        self.file_metadata = {}
        self.overview_data = {}
        self.schema_data = {}
        self.column_analysis_data = {}
        self.cleaning_log = {}
        self.plot_metadata = []
        self.correlation_insights = {}


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "EDA Engine API",
        "version": "1.0.0",
        "endpoints": [
            "/upload",
            "/overview/{session_id}",
            "/schema/{session_id}",
            "/column-analysis/{session_id}",
            "/clean",
            "/eda/{session_id}",
            "/report/{session_id}",
            "/download-cleaned/{session_id}",
            "/download-log/{session_id}"
        ]
    }


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload CSV file and create session
    
    Returns session_id and basic file info
    """
    # Validate file
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    # Create session
    session_id = str(uuid.uuid4())
    
    # Save uploaded file temporarily
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{session_id}.csv")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Check file size
        file_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            os.remove(temp_path)
            raise HTTPException(
                status_code=400,
                detail=f"File too large ({file_size_mb:.1f}MB). Maximum: {MAX_FILE_SIZE_MB}MB"
            )
        
        # Load CSV
        df, metadata = utils_io.read_csv_smart(temp_path)
        
        # Create session data
        session_data = SessionData(session_id)
        session_data.df_original = df
        session_data.file_metadata = metadata
        session_data.file_metadata['original_filename'] = file.filename
        
        sessions[session_id] = session_data
        
        return {
            "session_id": session_id,
            "filename": file.filename,
            "rows": len(df),
            "columns": len(df.columns),
            "size_mb": file_size_mb,
            "message": "File uploaded successfully"
        }
    
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/overview/{session_id}")
async def get_overview(session_id: str):
    """Get dataset overview"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = sessions[session_id]
    
    if not session_data.overview_data:
        # Generate overview
        session_data.overview_data = overview.get_dataset_overview(session_data.df_original)
    
    return session_data.overview_data


@app.get("/schema/{session_id}")
async def get_schema(session_id: str):
    """Get inferred schema"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = sessions[session_id]
    
    if not session_data.schema_data:
        # Infer schema
        session_data.schema_data = schema_infer.infer_schema(session_data.df_original)
    
    return session_data.schema_data


@app.get("/column-analysis/{session_id}")
async def get_column_analysis(session_id: str):
    """Get per-column analysis"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = sessions[session_id]
    
    if not session_data.column_analysis_data:
        # Analyze columns
        session_data.column_analysis_data = column_analysis.analyze_all_columns(
            session_data.df_original
        )
    
    return session_data.column_analysis_data


@app.post("/clean")
async def clean_data(config: CleaningConfig):
    """Execute cleaning pipeline with specified configuration"""
    if config.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = sessions[config.session_id]
    
    try:
        # Create cleaner instance
        cleaner = transformers.DataCleaner(session_data.df_original)
        
        # Apply transformations based on config
        if config.standardize_column_names:
            cleaner.standardize_column_names()
        
        if config.drop_constant_columns:
            cleaner.drop_constant_columns()
        
        if config.drop_high_missing_threshold > 0:
            cleaner.drop_high_missing_columns(config.drop_high_missing_threshold)
        
        if config.remove_duplicates:
            cleaner.remove_duplicates()
        
        # Impute missing values
        cleaner.impute_missing_numeric(strategy=config.impute_numeric_strategy)
        cleaner.impute_missing_categorical(strategy=config.impute_categorical_strategy)
        
        # Handle outliers
        if config.handle_outliers == "remove":
            cleaner.remove_outliers(method=config.outlier_method)
        elif config.handle_outliers == "cap":
            cleaner.cap_outliers(method=config.outlier_method)
        elif config.handle_outliers == "flag":
            cleaner.add_outlier_flags(method=config.outlier_method)
        
        # Get cleaned data
        session_data.df_cleaned = cleaner.get_cleaned_data()
        session_data.cleaning_log = cleaner.get_cleaning_log()
        
        # Optimize memory if requested
        if config.optimize_memory:
            session_data.df_cleaned, opt_report = schema_infer.optimize_dtypes(
                session_data.df_cleaned
            )
            session_data.cleaning_log['memory_optimization'] = opt_report
        
        return {
            "message": "Cleaning completed successfully",
            "original_shape": session_data.cleaning_log['statistics']['original_shape'],
            "final_shape": session_data.cleaning_log['statistics']['final_shape'],
            "actions_applied": len(session_data.cleaning_log['actions']),
            "memory_saved_mb": session_data.cleaning_log['statistics'].get('memory_reduction_mb', 0)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleaning error: {str(e)}")


@app.get("/eda/{session_id}")
async def generate_eda(session_id: str):
    """Generate EDA visualizations"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = sessions[session_id]
    
    # Use cleaned data if available, otherwise original
    df_to_analyze = session_data.df_cleaned if session_data.df_cleaned is not None else session_data.df_original
    
    try:
        # Create visualizer
        output_dir = f"output/{session_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        visualizer = eda.EDAVisualizer(df_to_analyze, output_dir)
        plot_report = visualizer.generate_all_plots()
        
        session_data.plot_metadata = plot_report['plot_metadata']
        
        # Get correlation insights
        session_data.correlation_insights = eda.get_correlation_insights(df_to_analyze)
        
        return {
            "message": "EDA visualizations generated",
            "total_plots": plot_report['total_plots'],
            "plots_by_type": plot_report['plots_by_type'],
            "output_directory": output_dir
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EDA error: {str(e)}")


@app.get("/report/{session_id}")
async def generate_report(session_id: str, format: str = "html"):
    """Generate and download report"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = sessions[session_id]
    
    # Ensure we have all data
    if not session_data.overview_data:
        session_data.overview_data = overview.get_dataset_overview(session_data.df_original)
    
    if not session_data.schema_data:
        session_data.schema_data = schema_infer.infer_schema(session_data.df_original)
    
    if not session_data.column_analysis_data:
        df_to_analyze = session_data.df_cleaned if session_data.df_cleaned is not None else session_data.df_original
        session_data.column_analysis_data = column_analysis.analyze_all_columns(df_to_analyze)
    
    try:
        output_dir = f"output/{session_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        if format == "html":
            html_content = report.generate_html_report(
                session_data.overview_data,
                session_data.schema_data,
                session_data.column_analysis_data,
                session_data.cleaning_log,
                session_data.plot_metadata,
                session_data.correlation_insights
            )
            
            report_path = utils_io.save_report(
                html_content,
                output_dir=output_dir,
                filename="eda_report.html"
            )
            
            return FileResponse(
                report_path,
                media_type="text/html",
                filename=f"eda_report_{session_id}.html"
            )
        
        elif format == "markdown":
            md_content = report.generate_markdown_report(
                session_data.overview_data,
                session_data.schema_data,
                session_data.column_analysis_data,
                session_data.cleaning_log
            )
            
            report_path = utils_io.save_report(
                md_content,
                output_dir=output_dir,
                filename="eda_report.md"
            )
            
            return FileResponse(
                report_path,
                media_type="text/markdown",
                filename=f"eda_report_{session_id}.md"
            )
        
        else:
            raise HTTPException(status_code=400, detail="Format must be 'html' or 'markdown'")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation error: {str(e)}")


@app.get("/download-cleaned/{session_id}")
async def download_cleaned(session_id: str):
    """Download cleaned CSV"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = sessions[session_id]
    
    if session_data.df_cleaned is None:
        raise HTTPException(status_code=400, detail="No cleaned data available. Run /clean first.")
    
    try:
        output_dir = f"output/{session_id}"
        cleaned_path = utils_io.save_dataframe(
            session_data.df_cleaned,
            output_dir=output_dir,
            filename="cleaned.csv"
        )
        
        original_filename = session_data.file_metadata.get('original_filename', 'data.csv')
        download_name = original_filename.replace('.csv', '_cleaned.csv')
        
        return FileResponse(
            cleaned_path,
            media_type="text/csv",
            filename=download_name
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading cleaned data: {str(e)}")


@app.get("/download-log/{session_id}")
async def download_log(session_id: str):
    """Download cleaning log JSON"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = sessions[session_id]
    
    if not session_data.cleaning_log:
        raise HTTPException(status_code=400, detail="No cleaning log available. Run /clean first.")
    
    try:
        output_dir = f"output/{session_id}"
        log_path = utils_io.save_json(
            session_data.cleaning_log,
            output_dir=output_dir,
            filename="cleaning_log.json"
        )
        
        return FileResponse(
            log_path,
            media_type="application/json",
            filename=f"cleaning_log_{session_id}.json"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading log: {str(e)}")


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete session and cleanup files"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Cleanup
    temp_file = f"temp/{session_id}.csv"
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    output_dir = f"output/{session_id}"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    del sessions[session_id]
    
    return {"message": "Session deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
