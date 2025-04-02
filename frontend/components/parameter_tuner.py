# components/parameter_tuner.py
import streamlit as st
from typing import Dict, Optional, Callable

class ParameterTuner:
    """
    A reusable component for tuning LLM parameters in Streamlit apps.
    
    Features:
    - Temperature adjustment
    - Max tokens adjustment
    - Additional custom parameters
    - Parameter presets
    """
    
    def __init__(
        self,
        session_key: str = "model_params",
        default_temperature: float = 0.7,
        default_max_tokens: int = 1000,
        min_temperature: float = 0.0,
        max_temperature: float = 1.0,
        min_tokens: int = 100,
        max_tokens: int = 4000
    ):
        """
        Initialize the parameter tuner component.
        
        Args:
            session_key: Key to use for storing parameters in session state
            default_temperature: Default temperature value
            default_max_tokens: Default max tokens value
            min_temperature: Minimum temperature value
            max_temperature: Maximum temperature value
            min_tokens: Minimum tokens value
            max_tokens: Maximum tokens value
        """
        self.session_key = session_key
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        
        # Initialize session state if needed
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = {
                "temperature": default_temperature,
                "max_tokens": default_max_tokens,
                "preset": "balanced"
            }
    
    def _apply_preset(self, preset: str):
        """Apply a parameter preset"""
        if preset == "creative":
            st.session_state[self.session_key]["temperature"] = min(0.9, self.max_temperature)
            st.session_state[self.session_key]["max_tokens"] = min(2000, self.max_tokens)
        elif preset == "precise":
            st.session_state[self.session_key]["temperature"] = max(0.2, self.min_temperature)
            st.session_state[self.session_key]["max_tokens"] = min(1500, self.max_tokens)
        elif preset == "balanced":
            st.session_state[self.session_key]["temperature"] = self.default_temperature
            st.session_state[self.session_key]["max_tokens"] = self.default_max_tokens
        elif preset == "concise":
            st.session_state[self.session_key]["temperature"] = max(0.3, self.min_temperature)
            st.session_state[self.session_key]["max_tokens"] = max(500, self.min_tokens)
        
        st.session_state[self.session_key]["preset"] = preset
    
    def render(
        self,
        on_change: Optional[Callable] = None,
        show_presets: bool = True,
        show_advanced: bool = True
    ) -> Dict[str, any]:
        """
        Render the parameter tuner component.
        
        Args:
            on_change: Optional callback when parameters change
            show_presets: Whether to show parameter presets
            show_advanced: Whether to show advanced options
            
        Returns:
            Dictionary of parameter values
        """
        params = st.session_state[self.session_key]
        
        st.header("Parameter Settings")
        # Parameter presets
        if show_presets:
            col1, col2 = st.columns(2)
            with col1:
                st.caption("Parameter Presets:")
            with col2:
                current_preset = params.get("preset", "balanced")
                new_preset = st.selectbox(
                    "Preset",
                    options=["balanced", "creative", "precise", "concise"],
                    index=["balanced", "creative", "precise", "concise"].index(current_preset),
                    label_visibility="collapsed"
                )
                
                if new_preset != current_preset:
                    self._apply_preset(new_preset)
                    if on_change:
                        on_change()
        
        # Temperature slider
        temperature = st.slider(
            "Temperature", 
            min_value=self.min_temperature, 
            max_value=self.max_temperature, 
            value=params["temperature"], 
            step=0.1,
            help="Higher values produce more creative responses",
            on_change=on_change if on_change else None,
            key=f"{self.session_key}_temperature"
        )
        params["temperature"] = temperature
        
        # Max tokens slider
        max_tokens = st.slider(
            "Max Tokens", 
            min_value=self.min_tokens, 
            max_value=self.max_tokens, 
            value=params["max_tokens"], 
            step=100,
            help="Maximum length of the response",
            on_change=on_change if on_change else None,
            key=f"{self.session_key}_max_tokens"
        )
        params["max_tokens"] = max_tokens
        
        # Advanced parameters
        if show_advanced:
            with st.expander("Advanced Parameters"):
                # Top-p sampling
                top_p = st.slider(
                    "Top-P", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=params.get("top_p", 0.9), 
                    step=0.05,
                    help="Controls diversity via nucleus sampling",
                    on_change=on_change if on_change else None,
                    key=f"{self.session_key}_top_p"
                )
                params["top_p"] = top_p
                
                # Frequency penalty
                freq_penalty = st.slider(
                    "Frequency Penalty", 
                    min_value=0.0, 
                    max_value=2.0, 
                    value=params.get("frequency_penalty", 0.0), 
                    step=0.1,
                    help="Reduces repetition of token sequences",
                    on_change=on_change if on_change else None,
                    key=f"{self.session_key}_freq_penalty"
                )
                params["frequency_penalty"] = freq_penalty
                
                # Presence penalty
                presence_penalty = st.slider(
                    "Presence Penalty", 
                    min_value=0.0, 
                    max_value=2.0, 
                    value=params.get("presence_penalty", 0.0), 
                    step=0.1,
                    help="Reduces repetition of topics",
                    on_change=on_change if on_change else None,
                    key=f"{self.session_key}_presence_penalty"
                )
                params["presence_penalty"] = presence_penalty
        
        # Update session state
        st.session_state[self.session_key] = params
        
        return params