import pandas as pd
import openai
import io
import json

class DataAgent:
    def __init__(self, api_key: str, model: str = "google/gemma-3-27b-it:free"):
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model

    def _get_dataframe_context(self, df: pd.DataFrame) -> str:
        """extracts metadata from the dataframe for the LLM."""
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        
        head_str = df.head(5).to_markdown(index=False)
        
        columns_str = ", ".join(df.columns.tolist())
        
        context = f"""
Dataset Metadata:
Columns: {columns_str}

Data Info:
{info_str}

First 5 Rows:
{head_str}
        """
        return context

    def chat(self, df: pd.DataFrame, messages: list) -> str:
        """Sends chat history to LLM with dataframe context."""
        
        context = self._get_dataframe_context(df)
        
        system_prompt = f"""You are PrometheusAI, an intelligent data assistant. 
You have access to a dataframe with the following details:
{context}

Your capabilities:
1. Answer questions about the dataset structure, content, and statistics.
2. Generate Python code to visualize data using matplotlib or seaborn.
3. If the user asks for a graph/plot, you MUST generate Python code.
   - Wrap the code in ```python ... ``` blocks.
   - Initialise the figure using `fig, ax = plt.subplots()` or similar.
   - Assume the dataframe is available as variable `df`.
   - Do NOT show the plot with `plt.show()`. instead, the code should create a figure object.
   - If using plotly, create a `fig` object.
   
   Example format for plot:
   ```python
   import matplotlib.pyplot as plt
   import seaborn as sns
   
   fig, ax = plt.subplots(figsize=(5, 3))
   sns.histplot(df['column_name'], ax=ax)
   plt.title('Distribution of column_name')
   ```

Refuse to answer questions unrelated to data analysis or the provided dataset.
Keep answers concise and professional.
"""
        
        # Prepend system prompt to messages
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                extra_headers={
                    "HTTP-Referer": "https://prometheus-ai.local", 
                    "X-Title": "PrometheusAI Data Engine",
                },
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error communicating with OpenRouter: {str(e)}"
