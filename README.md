# 📊 PrometheusAI Data Engine

**Fully Automated CSV → Cleaned Dataset → EDA Report**

PrometheusAI Data Engine is a comprehensive data cleaning and exploratory data analysis automation tool. Upload any CSV file and get instant, professional-grade analysis with cleaned datasets and beautiful visualizations.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688)](https://fastapi.tiangolo.com/)

---

## ✨ Features

### 🌟 Workflow Modes

Choose your preferred interaction style:

- **🤖 AI Agent**: Chat with your data in natural language and get instant insights with auto-generated visualizations
- **⚡ Quick Wizard**: One-click automated cleaning with best practices
- **🎯 Advanced Wizard**: Step-by-step control with column-by-column customization
- **🔧 Manual Mode**: Full customization of every analysis step

### 🎯 Core Capabilities

- **AI-Powered Chat Agent**: Ask questions in natural language and get instant insights with auto-generated visualizations
- **Data Cleaning Wizard**: Interactive step-by-step cleaning with Quick and Advanced modes
- **Automated Data Validation**: Instant dataset health scoring
- **Smart Type Detection**: Intelligent inference of numeric, categorical, datetime, boolean, and text columns
- **Missing Value Handling**: Multiple imputation strategies (mean, median, mode, forward-fill) with column-by-column control
- **Duplicate Detection**: Automatic identification and removal
- **Outlier Management**: IQR and Z-score methods with removal/capping/flagging options
- **Memory Optimization**: Automatic dtype downcasting to reduce memory footprint by 30-70%
- **Correlation Analysis**: Pearson correlation with heatmaps and insights
- **Professional Reports**: Beautiful HTML and Markdown reports with embedded visualizations
- **Interactive Visualizations**: Real-time charts and plots for all analysis types

### 📊 Analysis Components

1. **Dataset Overview**
   - Row/column counts and memory usage
   - Missing value summary
   - Duplicate detection
   - Data type distribution
   - Health scoring system

2. **Schema Inference**
   - Automatic type detection with confidence scores
   - Recommended type conversions
   - Cardinality analysis

3. **Column-wise Analysis**
   - **Numeric**: mean, median, std, quartiles, skewness, kurtosis, outliers
   - **Categorical**: unique counts, top categories, entropy
   - **Datetime**: range, frequency inference, temporal patterns
   - **Text**: length statistics, word counts, pattern detection

4. **Data Cleaning Wizard**
   - **Quick Mode**: One-click automated cleaning with best practices
   - **Advanced Mode**: 8-step interactive wizard
     - Step 1: Column name standardization
     - Step 2: Constant column removal
     - Step 3: High missing value column removal
     - Step 4: Duplicate row removal
     - Step 5: Missing value imputation (column-by-column)
     - Step 6: Outlier handling (column-by-column)
     - Step 7: Type conversion
     - Step 8: Memory optimization
   - Real-time progress tracking
   - Preview before/after for each step
   - Detailed statistics and visualizations

5. **AI-Powered Chat Agent**
   - Natural language queries about your data
   - Auto-generated Python code for visualizations
   - Interactive chat history
   - Support for matplotlib and seaborn plots
   - Intelligent dataframe context awareness

6. **Visual EDA**
   - Distribution plots (histograms, KDE, box plots)
   - Categorical bar charts
   - Correlation heatmaps
   - Missing value heatmaps
   - Outlier visualizations
   - Pair plots

7. **Downloadable Outputs**
   - ✅ Cleaned CSV
   - ✅ HTML/Markdown EDA reports
   - ✅ JSON cleaning log (reproducible pipeline)
   - ✅ All visualizations (PNG)


---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/PrometheusAI-Data-Engine.git
cd PrometheusAI-Data-Engine

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Option 1: Streamlit UI (Interactive)

```bash
streamlit run app_streamlit.py
```

Then open your browser to `http://localhost:8501`

**Features:**
- **Professional Welcome Screen**: Choose from 4 workflow modes
  - 🤖 **AI Agent**: Chat with your data using natural language
  - ⚡ **Quick Wizard**: One-click automated cleaning
  - 🎯 **Advanced Wizard**: Step-by-step interactive cleaning
  - 🔧 **Manual Mode**: Full control over every parameter
- **AI-Powered Chat**: Ask questions and get instant visualizations
- **Interactive Wizard**: 8-step guided cleaning process with real-time previews
- **Drag-and-drop CSV upload**
- **Interactive tabs** for each analysis step
- **Real-time visualizations** with matplotlib and seaborn
- **Download cleaned data and reports**

#### Option 2: FastAPI Backend (REST API)

```bash
uvicorn api_fastapi:app --reload
```

API documentation available at `http://localhost:8000/docs`

**Example API Usage:**

```python
import requests

# Upload CSV
files = {'file': open('data.csv', 'rb')}
response = requests.post('http://localhost:8000/upload', files=files)
session_id = response.json()['session_id']

# Get overview
overview = requests.get(f'http://localhost:8000/overview/{session_id}').json()

# Clean data
cleaning_config = {
    'session_id': session_id,
    'remove_duplicates': True,
    'impute_numeric_strategy': 'median',
    'handle_outliers': 'flag'
}
requests.post('http://localhost:8000/clean', json=cleaning_config)

# Generate EDA
requests.get(f'http://localhost:8000/eda/{session_id}')

# Download report
report = requests.get(f'http://localhost:8000/report/{session_id}?format=html')
```

---

## 📁 Project Structure

```
PrometheusAI-Data-Engine/
├── app_streamlit.py        # Streamlit UI with AI Agent & Wizard
├── api_fastapi.py          # FastAPI backend
├── agent/
│   ├── core.py             # AI agent logic & LLM integration
│   └── ui.py               # AI chat interface
├── cleaner/
│   ├── schema_infer.py     # Type detection & inference
│   ├── overview.py         # Dataset overview & health scoring
│   ├── column_analysis.py  # Per-column detailed analysis
│   ├── transformers.py     # Data cleaning operations
│   ├── eda.py              # Visualization generation
│   └── report.py           # HTML/Markdown report generation
├── utils/
│   └── io.py               # File I/O utilities
├── output/                 # Generated files
│   ├── cleaned.csv
│   ├── report.html
│   ├── cleaning_log.json
│   └── *.png               # Visualizations
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🔧 Configuration

### Cleaning Pipeline Options

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `remove_duplicates` | `True`/`False` | `True` | Remove duplicate rows |
| `drop_high_missing_threshold` | `0.0-1.0` | `0.9` | Drop columns with missing % above threshold |
| `drop_constant_columns` | `True`/`False` | `True` | Drop columns with single unique value |
| `impute_numeric_strategy` | `median`, `mean`, `mode` | `median` | Numeric imputation method |
| `impute_categorical_strategy` | `mode`, `constant` | `mode` | Categorical imputation method |
| `handle_outliers` | `remove`, `cap`, `flag`, `none` | `flag` | Outlier handling strategy |
| `outlier_method` | `iqr`, `zscore` | `iqr` | Outlier detection method |
| `standardize_column_names` | `True`/`False` | `True` | Clean column names |
| `optimize_memory` | `True`/`False` | `True` | Downcast dtypes |

---

## 📊 Output Files

### 1. Cleaned CSV (`cleaned.csv`)
Your dataset after all cleaning operations with:
- Standardized column names
- Missing values imputed
- Duplicates removed
- Outliers handled
- Types optimized

### 2. EDA Report (`report.html`)
Professional HTML report with:
- Executive summary with health score
- Data quality warnings
- Schema overview
- Column analysis tables
- Cleaning action log
- Correlation insights
- Embedded visualizations

### 3. Cleaning Log (`cleaning_log.json`)
Reproducible pipeline configuration:
```json
{
  "actions": [
    {"action": "standardize_column_names", "details": {...}},
    {"action": "remove_duplicates", "details": {...}}
  ],
  "statistics": {
    "original_shape": [1000, 20],
    "final_shape": [950, 18],
    "memory_reduction_mb": 5.2
  }
}
```

### 4. Visualizations (`*.png`)
- Distribution plots for numeric columns
- Bar charts for categorical columns
- Correlation heatmap
- Missing value patterns
- Outlier summaries
- Pair plots

---

## 🎓 Example Workflows

### AI Agent Mode (Recommended for Quick Insights)

1. Launch Streamlit: `streamlit run app_streamlit.py`
2. Upload your CSV file
3. Navigate to the **PrometheusAI Agent** tab
4. Configure your OpenRouter API key (or use other LLM providers)
5. Ask questions in natural language:
   - "Show me the distribution of the price column"
   - "What's the correlation between age and salary?"
   - "Create a scatter plot of x vs y"
6. Get instant visualizations and insights

### Quick Wizard Mode (One-Click Cleaning)

1. Launch Streamlit: `streamlit run app_streamlit.py`
2. Upload your CSV file
3. Navigate to the **Data Cleaning Wizard** tab
4. Select **Quick Mode**
5. Click "Run Quick Clean"
6. Download cleaned data from the Downloads tab

### Advanced Wizard Mode (Step-by-Step Control)

1. Launch Streamlit: `streamlit run app_streamlit.py`
2. Upload your CSV file
3. Navigate to the **Data Cleaning Wizard** tab
4. Select **Advanced Mode**
5. Go through each step:
   - Step 1: Standardize column names
   - Step 2: Remove constant columns
   - Step 3: Handle high missing columns
   - Step 4: Remove duplicates
   - Step 5: Impute missing values (column-by-column)
   - Step 6: Handle outliers (column-by-column)
   - Step 7: Convert data types
   - Step 8: Optimize memory
6. Review previews and statistics at each step
7. Download cleaned data and reports

### Manual Configuration (Full Control)

1. Upload CSV → View preview
2. Dataset Overview → Check health score and warnings
3. Schema Detection → Review and apply type conversions
4. Column Analysis → Examine detailed statistics
5. Data Cleaning → Configure and execute pipeline
6. EDA Visualizations → Generate and review plots
7. Report → Generate HTML/Markdown report
8. Downloads → Get cleaned CSV and logs

### API Workflow

See the FastAPI example in the Quick Start section above.

---

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Test specific module
python -m pytest tests/test_cleaner.py
```

---

## 📈 Performance

- Handles datasets up to 500MB (configurable)
- Memory optimization typically reduces size by 30-70%
- Automatic dtype downcasting (int64→int8/16/32, float64→float32)
- Efficient outlier detection using vectorized operations

---

## 🛠️ Advanced Usage

### Custom Cleaning Pipeline

```python
from cleaner.transformers import DataCleaner
import pandas as pd

df = pd.read_csv('data.csv')
cleaner = DataCleaner(df)

# Chain operations
cleaner \
    .standardize_column_names() \
    .drop_constant_columns() \
    .remove_duplicates() \
    .impute_missing_numeric(strategy='median') \
    .cap_outliers(method='iqr', multiplier=1.5)

cleaned_df = cleaner.get_cleaned_data()
log = cleaner.get_cleaning_log()
```

### Schema Inference

```python
from cleaner.schema_infer import infer_schema, apply_schema_conversions

schema = infer_schema(df, cardinality_threshold=0.05)
df_converted, logs = apply_schema_conversions(df, schema)
```

### Custom Visualizations

```python
from cleaner.eda import EDAVisualizer, get_correlation_insights

visualizer = EDAVisualizer(df, output_dir='my_plots')
visualizer.plot_correlation_matrix()
visualizer.plot_missing_values_heatmap()
visualizer.plot_numeric_distributions()

insights = get_correlation_insights(df, threshold=0.7)
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) and [FastAPI](https://fastapi.tiangolo.com/)
- Visualizations powered by [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/)
- Data processing with [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/)

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Happy Data Cleaning! 🎉**
