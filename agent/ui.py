import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from agent.core import DataAgent

def extract_code(text):
    """Extracts python code from markdown code blocks."""
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]

def render_agent_tab(df: pd.DataFrame):
    st.header("🤖 PrometheusAI Agent")
    
    # --- Configuration Section ---
    with st.expander("⚙️ Agent Configuration", expanded=not st.session_state.get("openrouter_api_key")):
        st.markdown("Configure your AI assistant settings.")
        
        # API Key Handling
        if "openrouter_api_key" not in st.session_state:
            if "OPENROUTER_API_KEY" in st.secrets.get("general", {}):
                st.session_state.openrouter_api_key = st.secrets["general"]["OPENROUTER_API_KEY"]
            elif "OPENROUTER_API_KEY" in st.secrets:
                 st.session_state.openrouter_api_key = st.secrets["OPENROUTER_API_KEY"]
            else:
                st.session_state.openrouter_api_key = ""

        api_key = st.text_input("OpenRouter API Key", value=st.session_state.openrouter_api_key, type="password", placeholder="sk-or-...", help="Required to access LLM models")
        if api_key:
            st.session_state.openrouter_api_key = api_key
        
        # Model Selection
        available_models = [
            "google/gemma-3-27b-it:free",
            "tngtech/deepseek-r1t2-chimera:free",
            "kwaipilot/kat-coder-pro:free",
            "tngtech/deepseek-r1t-chimera:free",
            "z-ai/glm-4.5-air:free",
            "nvidia/nemotron-nano-12b-v2-vl:free",
            "tngtech/tng-r1t-chimera:free",
            "amazon/nova-2-lite-v1:free"
        ]
        
        selected_model = st.selectbox(
            "Select Model", 
            options=available_models,
            index=0,
            help="Free models available via OpenRouter"
        )
        
        if not st.session_state.openrouter_api_key:
            st.warning("⚠️ Please enter your OpenRouter API Key to continue.")
            st.markdown("[Get your key here](https://openrouter.ai/keys)")
            return

    # Initialize Agent
    agent = DataAgent(api_key=st.session_state.openrouter_api_key, model=selected_model)
    
    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Toolbar ---
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🗑️ Clear", help="Clear chat history", type="secondary"):
            st.session_state.messages = []
            st.rerun()

    # --- Input Area (Top) ---
    st.markdown("### 💬 Chat")
    
    # View-only chips
    st.caption("Try asking: _Show me the first 5 rows_ • _What are the column names?_ • _Generate a correlation heatmap_")

    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask a question about your data...", placeholder="Type your message here...")
        submit_button = st.form_submit_button("Send 🚀")

    if submit_button and user_input:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Generate Response immediately
        with st.spinner("Thinking..."):
            api_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            response_text = agent.chat(df, api_messages)
            
            # Check for code/plot
            code_blocks = extract_code(response_text)
            has_plot = False
            
            # Save assistant message
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_text,
                "code_blocks": code_blocks
            })
            
        st.rerun()

    # --- Chat Interface ---
    
    st.divider()
    
    # Group messages into conversations (User + Assistant pairs)
    # Structure: [[msg1, msg2], [msg3], ...]
    conversations = []
    current_group = []
    
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            if current_group:
                conversations.append(current_group)
            current_group = [msg]
        else:
            current_group.append(msg)
    if current_group:
        conversations.append(current_group)
        
    # Iterate groups in reverse (Newest conversation top)
    for group in reversed(conversations):
        # Iterate messages in group (Oldest to Newest within group: Q -> A)
        for message in group:
            avatar = "🤖" if message["role"] == "assistant" else "👤"
            with st.chat_message(message["role"], avatar=avatar):
                
                # For assistant, strip code blocks from main text to avoid duplication
                display_text = message["content"]
                if message["role"] == "assistant" and "code_blocks" in message and message["code_blocks"]:
                    # Remove code blocks from display text
                    for code in message["code_blocks"]:
                        display_text = display_text.replace(f"```python\n{code}\n```", "(Code Executed Below)")
                        display_text = display_text.replace(f"```python{code}```", "(Code Executed Below)") # Fallback
                
                st.markdown(display_text)
                
                # Display Code and Result if present
                if message["role"] == "assistant" and "code_blocks" in message and message["code_blocks"]:
                    for code in message["code_blocks"]:
                        try:
                            with st.expander("📝 Executed Code", expanded=False):
                                st.code(code, language='python')
                            
                            # Execute code
                            local_scope = {
                                "df": df,
                                "pd": pd,
                                "plt": plt,
                                "sns": sns,
                                "st": st
                            }
                            
                            exec(code, {}, local_scope)
                            
                            if "fig" in local_scope:
                                st.pyplot(local_scope["fig"], use_container_width=False) # Reduced width
                                
                        except Exception as e:
                            st.error(f"Error executing code: {e}") 
        
        # Add subtle divider between conversations
        st.markdown("---")
