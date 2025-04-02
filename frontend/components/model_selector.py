# components/model_selector.py
import streamlit as st
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable


class ModelSelector:
    """
    A reusable component for selecting LLM models in Streamlit apps.

    Features:
    - Fetches models from API
    - Displays model details
    - Handles model selection
    - Caches model data in session state
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8000/models",
        session_key: str = "models_data",
        default_model: str = "llama3.2",
    ):
        """
        Initialize the model selector component.

        Args:
            api_url: URL to fetch models from
            session_key: Key to use for storing models in session state
            default_model: Fallback model if none are available
        """
        self.api_url = api_url
        self.session_key = session_key
        self.default_model = default_model

        # Initialize session state if needed
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = []

    def fetch_models(self) -> List[Dict]:
        """Fetch models from the API and update session state"""
        try:
            response = requests.get(self.api_url)
            if response.status_code == 200:
                models = response.json()
                if isinstance(models, list) and len(models) > 0:
                    st.session_state[self.session_key] = models
                    return models
            return []
        except Exception as e:
            st.error(f"Failed to fetch models: {e}")
            return []

    def _format_model_display(self, model: Dict) -> str:
        """Format a model for display in the dropdown"""
        details = model.get("details", {})
        name = model["name"]
        param_size = details.get("parameter_size", "")
        quant = details.get("quantization_level", "")

        # Format the display string
        if param_size or quant:
            display = f"{name} ({param_size}{', ' + quant if quant else ''})"
        else:
            display = name

        return display

    def _display_model_details(self, model: Dict):
        """Display detailed information about a model"""
        details = model.get("details", {})

        st.caption("Model Details:")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Family:** {details.get('family', 'N/A')}")
            st.write(f"**Parameters:** {details.get('parameter_size', 'N/A')}")
        with col2:
            st.write(f"**Format:** {details.get('format', 'N/A')}")
            st.write(f"**Quantization:** {details.get('quantization_level', 'N/A')}")

        # Format the modified date
        if "modified_at" in model:
            try:
                modified_date = datetime.fromisoformat(model["modified_at"].replace("Z", "+00:00"))
                st.write(f"**Last Updated:** {modified_date.strftime('%Y-%m-%d')}")
            except (ValueError, TypeError):
                pass

    def render(
        self, on_change: Optional[Callable] = None, show_refresh: bool = True, show_details: bool = True
    ) -> Tuple[str, Dict]:
        """
        Render the model selector component.

        Args:
            on_change: Optional callback when model selection changes
            show_refresh: Whether to show the refresh button
            show_details: Whether to show model details

        Returns:
            Tuple containing (selected_model_name, selected_model_data)
        """
        col1, col2 = st.columns([3, 1])
        with col1:
            st.header("Model Selection:")
        with col2:
            # Add a button to refresh model list if requested
            if show_refresh:
                if st.button("🔄"):
                    if self.fetch_models():
                        st.success("Models refreshed successfully!")
                        st.rerun()

        # Fetch models if not already loaded
        if not st.session_state[self.session_key]:
            self.fetch_models()

        models_data = st.session_state[self.session_key]

        if models_data:
            # Create a list of model names for the selectbox
            model_names = [model["name"] for model in models_data]

            # Create more descriptive display names
            model_display = [self._format_model_display(model) for model in models_data]

            # Display model selection dropdown with descriptive names
            selected_display = st.selectbox(
                "Model",
                options=model_display,
                index=0,
                on_change=on_change if on_change else None,
                label_visibility="collapsed",
            )

            # Get the actual model name from the display name
            selected_index = model_display.index(selected_display)
            selected_model = model_names[selected_index]
            selected_model_data = models_data[selected_index]

            # Show model details if requested
            if show_details:
                self._display_model_details(selected_model_data)

        else:
            st.warning("No models available. Please check your API connection.")
            selected_model = self.default_model
            selected_model_data = {}

        return selected_model, selected_model_data
