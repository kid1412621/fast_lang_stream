# chatbot.py
import streamlit as st
from components.api_status_indicator import APIStatusIndicator
from components.model_selector import ModelSelector
from components.parameter_tuner import ParameterTuner
from components.sse_streamer import SSEStreamer

st.set_page_config(page_title="Chat with Ollama", page_icon="💬")
st.title("💬 Chat with Ollama")

col1, col2 = st.columns([3, 1])
with col1:
    # Add API status indicator in the header
    status_indicator = APIStatusIndicator(
        api_url="http://localhost:8000/health", service_name="Ollama API"
    )
    status_indicator.render()

with col2:
    # Add a button to clear chat history
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Sidebar for model parameters
with st.sidebar:
    # Use the ModelSelector component
    model_selector = ModelSelector()
    selected_model, model_data = model_selector.render()

    st.divider()

    # Use the ParameterTuner component
    parameter_tuner = ParameterTuner()
    params = parameter_tuner.render()


# Initialize the SSE streamer
streamer = SSEStreamer(api_url="http://localhost:8000/stream")

# Handle the conversation
if prompt := st.chat_input("Say something..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare the request data
    request_data = {
        "prompt": prompt,
        "messages": [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.messages
            if msg["role"] in ["user", "assistant"]
        ],
        "model": selected_model,
        "temperature": params["temperature"],
        "max_tokens": params["max_tokens"],
    }

    # Add advanced parameters if they exist
    for param_name in ["top_p", "frequency_penalty", "presence_penalty"]:
        if param_name in params:
            request_data[param_name] = params[param_name]

    # Create a chat message container for the assistant's response
    with st.chat_message("assistant"):
        # Define callback for when streaming is complete
        def on_complete(response):
            # Add assistant response to chat history
            if not response.startswith("Error:"):
                st.session_state.messages.append({"role": "assistant", "content": response})

        # Stream the response directly using st.write_stream
        spinner_text = f"Connecting to {selected_model}..."
        streamer.stream(request_data=request_data, on_complete=on_complete, spinner_text=spinner_text)
