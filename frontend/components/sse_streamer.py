# components/sse_streamer.py
import streamlit as st
import requests
import json
import base64
from PIL import Image
from io import BytesIO
from typing import Dict, Optional, Callable, Generator, List, Union

class SSEStreamer:
    """
    A reusable component for streaming Server-Sent Events (SSE) in Streamlit.
    
    Features:
    - Handles SSE format parsing
    - Uses st.write_stream for efficient streaming
    - Provides error handling
    - Supports image data in requests
    """
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000/stream",
    ):
        """
        Initialize the SSE streamer component.
        
        Args:
            api_url: URL for the streaming API endpoint
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
    
    def _encode_image(self, image: Union[Image.Image, str, bytes]) -> str:
        """
        Encode an image to base64 for API transmission.
        
        Args:
            image: PIL Image object, file path, or bytes
            
        Returns:
            Base64 encoded image string
        """
        # If image is already a string, assume it's a path
        if isinstance(image, str):
            with open(image, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode("utf-8")
                
        # If image is bytes, encode directly
        elif isinstance(image, bytes):
            return base64.b64encode(image).decode("utf-8")
            
        # If image is a PIL Image, convert to bytes first
        elif isinstance(image, Image.Image):
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
            
        else:
            raise TypeError("Image must be a PIL Image, file path, or bytes")
    
    def _prepare_request_with_images(self, request_data: Dict, images: List[Union[Image.Image, str, bytes]]) -> Dict:
        """
        Prepare request data with encoded images.
        
        Args:
            request_data: Original request data
            images: List of images to include
            
        Returns:
            Updated request data with images
        """
        # Create a copy of the request data
        updated_request = request_data.copy()
        
        # Encode images
        encoded_images = [self._encode_image(img) for img in images]
        
        # Add images to request based on API format
        # This format may need adjustment based on your specific API requirements
        updated_request["images"] = encoded_images
        
        return updated_request
    
    def stream(
        self,
        request_data: Dict,
        images: Optional[List[Union[Image.Image, str, bytes]]] = None,
        on_complete: Optional[Callable[[str], None]] = None,
        spinner_text: str = "Connecting..."
    ) -> str:
        """
        Stream a response from the API.
        
        Args:
            request_data: Data to send in the request
            images: Optional list of images to include in the request
            on_complete: Callback to execute when streaming is complete
            spinner_text: Text to show in the spinner while connecting
            
        Returns:
            The full response text
        """
        try:
            # Prepare request data with images if provided
            if images:
                request_data = self._prepare_request_with_images(request_data, images)
            
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