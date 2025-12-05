"""
Wizard Step Manager
Handles state management and navigation for the data cleaning wizard
"""

import streamlit as st
from typing import Dict, Any, Optional, List


class WizardStepManager:
    """Manages wizard state, navigation, and step tracking"""
    
    # Step definitions
    STEPS = [
        {"id": 1, "name": "Standardize Column Names", "skippable": False},
        {"id": 2, "name": "Remove Duplicates", "skippable": True},
        {"id": 3, "name": "Drop Constant Columns", "skippable": True},
        {"id": 4, "name": "Drop High Missing Columns", "skippable": True},
        {"id": 5, "name": "Impute Numeric Values", "skippable": True},
        {"id": 6, "name": "Impute Categorical Values", "skippable": True},
        {"id": 7, "name": "Handle Outliers", "skippable": True},
        {"id": 8, "name": "Optimize Memory", "skippable": True}
    ]
    
    @staticmethod
    def initialize():
        """Initialize wizard session state"""
        if 'wizard_mode' not in st.session_state:
            st.session_state.wizard_mode = None  # 'quick' or 'advanced'
        
        if 'wizard_current_step' not in st.session_state:
            st.session_state.wizard_current_step = 1
        
        if 'wizard_completed_steps' not in st.session_state:
            st.session_state.wizard_completed_steps = set()
        
        if 'wizard_skipped_steps' not in st.session_state:
            st.session_state.wizard_skipped_steps = set()
        
        if 'wizard_data' not in st.session_state:
            st.session_state.wizard_data = None
        
        if 'wizard_original_data' not in st.session_state:
            st.session_state.wizard_original_data = None
        
        if 'wizard_step_logs' not in st.session_state:
            st.session_state.wizard_step_logs = {}
        
        # Step-specific selections
        if 'wizard_numeric_imputation' not in st.session_state:
            st.session_state.wizard_numeric_imputation = {}  # {column: strategy}
        
        if 'wizard_categorical_imputation' not in st.session_state:
            st.session_state.wizard_categorical_imputation = {}
        
        if 'wizard_outlier_handling' not in st.session_state:
            st.session_state.wizard_outlier_handling = {}  # {column: {method, action}}
    
    @staticmethod
    def start_wizard(df, mode: str = 'advanced'):
        """Start the wizard with a dataset"""
        st.session_state.wizard_mode = mode
        st.session_state.wizard_current_step = 1
        st.session_state.wizard_completed_steps = set()
        st.session_state.wizard_skipped_steps = set()
        st.session_state.wizard_data = df.copy()
        st.session_state.wizard_original_data = df.copy()
        st.session_state.wizard_step_logs = {}
        st.session_state.wizard_numeric_imputation = {}
        st.session_state.wizard_categorical_imputation = {}
        st.session_state.wizard_outlier_handling = {}
    
    @staticmethod
    def reset_wizard():
        """Reset wizard to initial state"""
        st.session_state.wizard_mode = None
        st.session_state.wizard_current_step = 1
        st.session_state.wizard_completed_steps = set()
        st.session_state.wizard_skipped_steps = set()
        st.session_state.wizard_data = None
        st.session_state.wizard_original_data = None
        st.session_state.wizard_step_logs = {}
        st.session_state.wizard_numeric_imputation = {}
        st.session_state.wizard_categorical_imputation = {}
        st.session_state.wizard_outlier_handling = {}
    
    @staticmethod
    def get_current_step() -> int:
        """Get current step number"""
        return st.session_state.wizard_current_step
    
    @staticmethod
    def get_step_info(step_id: int) -> Dict[str, Any]:
        """Get information about a specific step"""
        for step in WizardStepManager.STEPS:
            if step['id'] == step_id:
                return step
        return None
    
    @staticmethod
    def next_step():
        """Move to next step"""
        current = st.session_state.wizard_current_step
        if current < len(WizardStepManager.STEPS):
            st.session_state.wizard_completed_steps.add(current)
            st.session_state.wizard_current_step = current + 1
    
    @staticmethod
    def previous_step():
        """Move to previous step"""
        current = st.session_state.wizard_current_step
        if current > 1:
            st.session_state.wizard_current_step = current - 1
    
    @staticmethod
    def skip_step():
        """Skip current step"""
        current = st.session_state.wizard_current_step
        st.session_state.wizard_skipped_steps.add(current)
        if current < len(WizardStepManager.STEPS):
            st.session_state.wizard_current_step = current + 1
    
    @staticmethod
    def is_step_completed(step_id: int) -> bool:
        """Check if a step is completed"""
        return step_id in st.session_state.wizard_completed_steps
    
    @staticmethod
    def is_step_skipped(step_id: int) -> bool:
        """Check if a step is skipped"""
        return step_id in st.session_state.wizard_skipped_steps
    
    @staticmethod
    def is_step_current(step_id: int) -> bool:
        """Check if this is the current step"""
        return step_id == st.session_state.wizard_current_step
    
    @staticmethod
    def get_progress_percentage() -> float:
        """Get overall progress percentage"""
        total_steps = len(WizardStepManager.STEPS)
        completed = len(st.session_state.wizard_completed_steps)
        skipped = len(st.session_state.wizard_skipped_steps)
        return ((completed + skipped) / total_steps) * 100
    
    @staticmethod
    def log_step_action(step_id: int, action: str, details: Dict[str, Any]):
        """Log an action for a step"""
        if step_id not in st.session_state.wizard_step_logs:
            st.session_state.wizard_step_logs[step_id] = []
        
        st.session_state.wizard_step_logs[step_id].append({
            'action': action,
            'details': details
        })
    
    @staticmethod
    def get_step_logs(step_id: int) -> List[Dict[str, Any]]:
        """Get logs for a specific step"""
        return st.session_state.wizard_step_logs.get(step_id, [])
    
    @staticmethod
    def is_wizard_complete() -> bool:
        """Check if wizard is complete"""
        total_steps = len(WizardStepManager.STEPS)
        completed = len(st.session_state.wizard_completed_steps)
        skipped = len(st.session_state.wizard_skipped_steps)
        return (completed + skipped) >= total_steps
