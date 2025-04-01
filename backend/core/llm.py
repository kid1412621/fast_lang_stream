# utils/llm.py
from functools import lru_cache
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import SystemMessage
import logging

logger = logging.getLogger("llm")

class LLMManager:
    def __init__(self, cache_size: int = 10):
        self.cache_size = cache_size
    
    @lru_cache(maxsize=10)
    def get_llm(self, model: str, temperature: float, max_tokens: int) -> ChatOllama:
        """Cached LLM instance"""
        logger.info(f"Creating new LLM instance for model: {model}")
        return ChatOllama(
            model=model,
            temperature=temperature,
            num_predict=max_tokens,
            streaming=True,
        )
    
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