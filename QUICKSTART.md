# 🚀 Quick Start Guide - PrometheusPrep EDA Engine

Get up and running in under 2 minutes!

---

## Step 1: Install Dependencies

```bash
cd d:\Project\ML\PrometheusPrep-EDA-Engine
pip install -r requirements.txt
```

**Installation time**: ~1-2 minutes

---

## Step 2: Choose Your Interface

### Option A: Streamlit UI (Recommended for Beginners)

```bash
streamlit run app_streamlit.py
```

Then open your browser to: **http://localhost:8501**

### Option B: FastAPI Backend (For Developers)

```bash
uvicorn api_fastapi:app --reload
```

Then open: **http://localhost:8000/docs** for interactive API documentation

---

## Step 3: Try It Out!

### Using Streamlit (Full Auto Mode)

1. **Upload** the provided `sample_data.csv` file
2. **Enable** "🤖 Full Auto Mode" in the sidebar
3. **Click** "▶️ Run Full Pipeline"
4. **Wait** ~5-10 seconds for processing
5. **Explore** results in the tabs:
   - 📊 Dataset Overview → See health score
   - 🧹 Data Cleaning → View applied actions
   - 📈 EDA Visualizations → Browse charts
   - 📄 Report → Generate HTML report
   - 💾 Downloads → Get cleaned CSV

### Using FastAPI

**Python Example**:
```python
import requests

# Upload file
with open('sample_data.csv', 'rb') as f:
    files = {'file': f}
    res = requests.post('http://localhost:8000/upload', files=files)
    session_id = res.json()['session_id']

# Get overview
overview = requests.get(f'http://localhost:8000/overview/{session_id}').json()
print(f"Health Score: {overview['health_score']}")

# Run cleaning
config = {
    'session_id': session_id,
    'remove_duplicates': True,
    'impute_numeric_strategy': 'median',
    'handle_outliers': 'flag'
}
requests.post('http://localhost:8000/clean', json=config)

# Generate EDA
requests.get(f'http://localhost:8000/eda/{session_id}')

# Download report
report = requests.get(f'http://localhost:8000/report/{session_id}?format=html')
with open('eda_report.html', 'wb') as f:
    f.write(report.content)
```

---

## Expected Outputs

After running the pipeline on `sample_data.csv`, you should see:

### In `output/` directory:
- ✅ `cleaned_TIMESTAMP.csv` - Cleaned dataset
- ✅ `eda_report_TIMESTAMP.html` - Professional HTML report
- ✅ `cleaning_log_TIMESTAMP.json` - Reproducible pipeline config
- ✅ Multiple `.png` files - All visualizations

### Key Improvements from Sample Data:
- 🔧 10 duplicate rows removed
- 🔧 Column names standardized (lowercase, underscores)
- 🔧 1 constant column dropped
- 🔧 Missing values imputed
- 🔧 20 income outliers flagged
- 🔧 Memory optimized (30-70% reduction)

---

## What Sample Data Contains

The `sample_data.csv` file includes **intentional issues** to demonstrate cleaning capabilities:

| Issue | Count | EDA Engine Solution |
|-------|-------|---------------------|
| Duplicate rows | 10 | Automatically removed |
| Missing values (70%) | 1 column | Flagged for review/dropping |
| Missing values (15%) | 1 column | Imputed with mode |
| Income outliers | 20 rows | Flagged with indicator column |
| Invalid credit scores | 5 rows | Detected in analysis |
| Constant column | 1 | Automatically dropped |
| All-null row | 1 | Handled during cleaning |
| Spaces in column names | All | Standardized to snake_case |

---

## Troubleshooting

### Import Errors
```bash
# Make sure you're in the project directory
cd d:\Project\ML\PrometheusPrep-EDA-Engine

# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Port Already in Use
```bash
# Streamlit: Use different port
streamlit run app_streamlit.py --server.port 8502

# FastAPI: Use different port
uvicorn api_fastapi:app --port 8001
```

### Memory Issues
If working with large datasets (>500MB), increase limits in `api_fastapi.py`:
```python
MAX_FILE_SIZE_MB = 1000  # Change from 500 to 1000
```

---

## Next Steps

1. **Try Your Own Data**: Upload your own CSV files
2. **Customize Settings**: Adjust cleaning parameters in UI or API
3. **Explore Reports**: Review the generated HTML reports
4. **Read Full Docs**: Check [README.md](README.md) for advanced features

---

## Support

- 📖 Full Documentation: [README.md](README.md)
- 🔍 Implementation Details: See `walkthrough.md` in artifacts
- 🐛 Issues: Check error messages and logs
- 💡 Examples: See usage examples in README

---

**Happy Data Cleaning! 🎉**
