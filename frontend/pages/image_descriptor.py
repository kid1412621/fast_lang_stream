# image_chatbot.py
import streamlit as st
from PIL import Image
from components.api_status_indicator import APIStatusIndicator
from components.model_selector import ModelSelector
from components.parameter_tuner import ParameterTuner
from components.sse_streamer import SSEStreamer

st.set_page_config(page_title="Image Chat with Ollama", page_icon="🖼️")
st.title("🖼️ Image Chat with Ollama")

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
        st.session_state.image_history = []
        st.rerun()

# Add image upload button
uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# Show preview if image is uploaded
current_image = None
# Show a hint about image upload if no image is present
if uploaded_image is None:
    st.caption("💡 Upload an image to ask questions about it")
else:
    current_image = Image.open(uploaded_image)
    st.image(current_image, caption="Preview", use_container_width=True)
    
    # Add a button to remove the image
    if st.button("❌ Remove"):
        uploaded_image = None
        current_image = None
        st.rerun()

st.divider()

# Initialize chat history and image history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "image_history" not in st.session_state:
    st.session_state.image_history = []

# Display chat history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        # Check if this message has an associated image
        if message["role"] == "user" and i < len(st.session_state.image_history) and st.session_state.image_history[i] is not None:
            # Display the image
            st.image(st.session_state.image_history[i], caption="Uploaded Image")
        
        # Display the message content
        st.markdown(message["content"])

# Sidebar for model parameters
with st.sidebar:
    # Use the ModelSelector component with multimodal models filter
    model_selector = ModelSelector(multimodal_only=True)
    selected_model, model_data = model_selector.render()

    st.divider()

    # Use the ParameterTuner component
    parameter_tuner = ParameterTuner()
    params = parameter_tuner.render()

# Initialize the SSE streamer
streamer = SSEStreamer(api_url="http://localhost:8000/stream")

# Create a container for the input area
input_container = st.container()

# Chat input
prompt_placeholder = "Ask about the image or type a message..."
prompt = st.chat_input(prompt_placeholder)

# Handle the conversation
if prompt:
    # Process uploaded image if available
    if uploaded_image is not None:
        current_image = Image.open(uploaded_image)
        # Store the image in history
        st.session_state.image_history.append(current_image)
    else:
        # No image for this message
        st.session_state.image_history.append(None)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message and image if available
    with st.chat_message("user"):
        if current_image is not None:
            st.image(current_image, caption="Uploaded Image")
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

        # Stream the response using the image if available
        spinner_text = f"Analyzing image with {selected_model}..." if current_image else f"Connecting to {selected_model}..."
        
        if current_image:
            # Pass the image to the streamer
            streamer.stream(
                request_data=request_data, 
                images=[current_image],
                on_complete=on_complete, 
                spinner_text=spinner_text
            )
        else:
            # No image, just stream the text response
            streamer.stream(
                request_data=request_data,
                on_complete=on_complete,
                spinner_text=spinner_text
            )
    
    # Clear the uploaded image after sending
    uploaded_image = None