# core/llm.py
import logging
from typing import Dict, Tuple, Annotated, List, Optional, Any
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import SystemMessage
from langchain_core.messages import HumanMessage
from fastapi import Depends

logger = logging.getLogger("llm")

class LLMManager:
    def __init__(self, cache_size: int = 10):
        self.cache_size = cache_size
        self.llm_cache: Dict[Tuple[str, float, int], ChatOllama] = {}
    
    def get_llm(self, model: str, temperature: float, max_tokens: int) -> ChatOllama:
        """Get LLM instance with caching"""
        # Create a cache key from the parameters
        cache_key = (model, temperature, max_tokens)
        
        # Check if we have a cached instance
        if cache_key not in self.llm_cache:
            logger.info(f"Creating new LLM instance for model: {model}")
            
            # Create a new LLM instance
            llm = ChatOllama(
                model=model,
                temperature=temperature,
                num_predict=max_tokens,
                streaming=True,
            )
            
            # Cache the instance
            self.llm_cache[cache_key] = llm
            
            # If cache is too large, remove the oldest entry
            if len(self.llm_cache) > self.cache_size:
                # Get the first key (oldest entry)
                oldest_key = next(iter(self.llm_cache))
                del self.llm_cache[oldest_key]
                
        return self.llm_cache[cache_key]
    
    def create_chain(self, llm: ChatOllama) -> ChatPromptTemplate:
        """Create a LangChain chain with the given LLM"""
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="You are a helpful AI assistant. Provide clear and concise responses to user queries."
                ),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        )
        return prompt | llm | StrOutputParser()
    
    def create_multimodal_message(
        self, 
        text_content: str, 
        image_data: Optional[List[str]] = None
    ) -> HumanMessage:
        """
        Create a multimodal message with text and optional images.
        
        Args:
            text_content: The text content of the message
            image_data: List of base64-encoded image strings
            
        Returns:
            A HumanMessage with multimodal content
        """
        # If no images, return a simple text message
        if not image_data:
            return HumanMessage(content=text_content)
        
        # Create multimodal content structure
        content: List[Dict[str, Any]] = []
        
        # Add text content if provided
        if text_content:
            content.append({
                "type": "text",
                "text": text_content
            })
        
        # Add image content
        for img_base64 in image_data:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}"
                }
            })
        
        # Return multimodal message
        return HumanMessage(content=content)
    
    def is_multimodal_model(self, model_name: str) -> bool:
        """
        Check if a model has multimodal capabilities based on its name.
        
        Args:
            model_name: The name of the model to check
            
        Returns:
            True if the model is likely multimodal, False otherwise
        """
        model_name = model_name.lower()
        return any([
            "gemma3" in model_name,
            "llava" in model_name,
            "vision" in model_name,
            "clip" in model_name,
            "multimodal" in model_name,
            "visual" in model_name,
            "image" in model_name,
            "bakllava" in model_name
        ])


# Create a global instance
llm_manager = None

def get_llm_manager(cache_size: int = 10) -> LLMManager:
    """Dependency for getting the LLM manager"""
    global llm_manager
    if llm_manager is None:
        llm_manager = LLMManager(cache_size=cache_size)
    return llm_manager

# Define annotated dependencies
LLMManagerDep = Annotated[LLMManager, Depends(get_llm_manager)]