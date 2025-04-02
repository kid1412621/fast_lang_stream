# components/sse_streamer.py
import streamlit as st
import requests
import json
from typing import Dict, Optional, Callable, Generator

class SSEStreamer:
    """
    A reusable component for streaming Server-Sent Events (SSE) in Streamlit.
    
    Features:
    - Handles SSE format parsing
    - Uses st.write_stream for efficient streaming
    - Provides error handling
    """
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000/stream",
    ):
        """
        Initialize the SSE streamer component.
        
        Args:
            api_url: URL for the streaming API endpoint
            cursor_char: Character to display as a cursor during streaming
        """
        self.api_url = api_url
    
    def _create_sse_generator(self, response) -> Generator[str, None, None]:
        """Create a generator that yields content from SSE events"""
        buffer = ""
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                buffer += chunk.decode('utf-8')
                while '\n\n' in buffer:
                    event, buffer = buffer.split('\n\n', 1)
                    if event.startswith('data: '):
                        try:
                            data = json.loads(event[6:])  # Remove 'data: ' prefix
                            
                            # Only yield content for chunk events
                            if data.get("type") == "chunk" and "data" in data:
                                yield data["data"]
                            elif data.get("type") == "error" and "data" in data:
                                yield f"Error: {data['data']}"
                                
                        except json.JSONDecodeError:
                            continue
    
    def stream(
        self,
        request_data: Dict,
        on_complete: Optional[Callable[[str], None]] = None,
        spinner_text: str = "Connecting..."
    ) -> str:
        """
        Stream a response from the API.
        
        Args:
            request_data: Data to send in the request
            on_complete: Callback to execute when streaming is complete
            spinner_text: Text to show in the spinner while connecting
            
        Returns:
            The full response text
        """
        try:
            # Create the request
            with st.spinner(spinner_text):
                response = requests.post(
                    self.api_url, 
                    json=request_data,
                    stream=True,
                    headers={"Accept": "text/event-stream"}
                )
            
            # Check if the request was successful
            response.raise_for_status()
            
            # Create a generator for st.write_stream
            sse_generator = self._create_sse_generator(response)
            
            # Use st.write_stream to display the streaming response
            full_response = ""
            
            # This is where st.write_stream displays the content
            for chunk in st.write_stream(sse_generator):
                full_response += chunk
            
            # Call the completion callback if provided
            if on_complete:
                on_complete(full_response)
                
            return full_response
                
        except requests.exceptions.ConnectionError:
            error_message = "Error: Could not connect to the API server. Make sure it's running."
            st.error(error_message)
            return error_message
        except Exception as e:
            error_message = f"Error: {str(e)}"
            st.error(error_message)
            return error_message