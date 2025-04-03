# components/api_status_indicator.py

import streamlit as st
import requests
from typing import Optional, Dict, Any

class APIStatusIndicator:
    """
    A component to check and display the status of an API endpoint.
    """
    
    def __init__(self, 
                 api_url: str = "http://localhost:8000/health", 
                 service_name: str = "API server",
                 timeout: int = 3,
                 custom_messages: Optional[Dict[str, str]] = None):
        """
        Initialize the API Status Indicator component.
        
        Args:
            api_url: The URL to check for API health status
            service_name: The name of the service to display in status messages
            timeout: Request timeout in seconds
            custom_messages: Optional dictionary of custom status messages
        """
        self.api_url = api_url
        self.service_name = service_name
        self.timeout = timeout
        
        # Default status messages with service_name placeholder
        self.messages = {
            "healthy": "✅ {service_name} is online",
            "unhealthy": "⚠️ {service_name} status: {status}",
            "error": "❌ {service_name} is experiencing issues",
            "offline": "❌ {service_name} is offline"
        }
        
        # Override with any custom messages
        if custom_messages:
            self.messages.update(custom_messages)
    
    def check_status(self) -> Dict[str, Any]:
        """
        Check the API status and return the result.
        
        Returns:
            Dictionary with status information
        """
        try:
            health_response = requests.get(self.api_url, timeout=self.timeout)
            
            if health_response.status_code == 200:
                data = health_response.json()
                status = data.get("status", "unknown")
                
                if status == "healthy":
                    return {
                        "is_online": True,
                        "status": status,
                        "message": self.messages["healthy"].format(service_name=self.service_name),
                        "raw_data": data
                    }
                else:
                    return {
                        "is_online": True,
                        "status": status,
                        "message": self.messages["unhealthy"].format(
                            service_name=self.service_name,
                            status=status
                        ),
                        "raw_data": data
                    }
            else:
                return {
                    "is_online": False,
                    "status": "error",
                    "message": self.messages["error"].format(service_name=self.service_name),
                    "status_code": health_response.status_code
                }
                
        except Exception as e:
            return {
                "is_online": False,
                "status": "offline",
                "message": self.messages["offline"].format(service_name=self.service_name),
                "error": str(e)
            }
    
    def render(self) -> Dict[str, Any]:
        """
        Check the API status and render the status message.
        
        Returns:
            Dictionary with status information
        """
        status_data = self.check_status()
        st.markdown(status_data["message"])
        return status_data