# streamlit_app.py
import streamlit as st
import requests
import json

st.set_page_config(page_title="Chat with API", page_icon="💬")
st.title("💬 Chat with Streaming API")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to stream response using SSE
def stream_response_sse(prompt, message_placeholder):
    url = "http://localhost:8000/stream"
    
    try:
        # Send the request
        with requests.post(
            url, 
            json={"prompt": prompt},
            stream=True,
            headers={"Accept": "text/event-stream"}
        ) as response:
            
            # Check if the request was successful
            response.raise_for_status()
            
            full_response = ""
            
            # Process the streaming response
            for line in response.iter_lines():
                if line:
                    # SSE format: lines starting with "data: "
                    if line.startswith(b"data: "):
                        # Parse the JSON data
                        json_str = line[6:].decode('utf-8')  # Remove "data: " prefix
                        try:
                            data = json.loads(json_str)
                            
                            if data.get("type") == "chunk" and "data" in data:
                                chunk = data["data"]
                                full_response += chunk
                                # Update the placeholder with the current response
                                message_placeholder.markdown(full_response + "▌")
                            
                            elif data.get("type") == "end":
                                # Final update without cursor
                                message_placeholder.markdown(full_response)
                                break
                                
                            elif data.get("type") == "error":
                                error_msg = data.get("data", "Unknown error")
                                message_placeholder.markdown(f"Error: {error_msg}")
                                return f"Error: {error_msg}"
                                
                        except json.JSONDecodeError:
                            continue
            
            return full_response
            
    except Exception as e:
        error_message = f"Error: {e}"
        message_placeholder.markdown(error_message)
        return error_message

# Chat input
prompt = st.chat_input("Say something...")

# Handle the conversation
if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Create a placeholder for the assistant's response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
    
    # Get the streaming response
    response = stream_response_sse(prompt, message_placeholder)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})