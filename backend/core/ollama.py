from typing import Annotated, List
from fastapi import Depends
import httpx
from core.config import settings
from utils.logging import get_logger

logger = get_logger("ollamaAPI")


class OllamaAPIClient:
    
    async def version(self):
        async with httpx.AsyncClient() as client:
            version_response = await client.get(f"{settings.OLLAMA_URL}/api/version")
        version_response.raise_for_status()
        version_info = version_response.json()
        version = version_info["version"]
        logger.info(f"Ollama API version: {version}")
        return version

    async def models(self):
        async with httpx.AsyncClient() as client:
            models_response = await client.get(f"{settings.OLLAMA_URL}/api/tags")
        models_response.raise_for_status()
        models_info = models_response.json()
        models = models_info["models"]
        logger.info(f"Ollama API models: {models}")
        return models

    # ollama just updated its API to check model capability to check if is multimodal or not, it requires 0.6.4
    # see: https://github.com/ollama/ollama/pull/10066
    async def capabilities(self, model_name: str) -> List[str]:
        async with httpx.AsyncClient() as client:
            capabilities_response = await client.post(
                f"{settings.OLLAMA_URL}/api/show",
                json={"model": model_name}
            )
        capabilities_response.raise_for_status()
        capabilities_info = capabilities_response.json()
        capabilities = capabilities_info["capabilities"]
        logger.info(f"Ollama API capabilities: {capabilities}")
        return capabilities

    async def is_multimodal_model(
        self, model_name: str, capabilities: List[str] = []
    ) -> bool:
        """
        Check if a model has multimodal capabilities based on its capabilities or name.

        Args:
            model_name: The name of the model to check
            capabilities: The capabilities of the model

        Returns:
            True if the model is likely multimodal, False otherwise
        """
        model_name = model_name.lower()

        if len(capabilities) > 0:
            # capabilities = await self.capabilities(model_name)

            return any(
                [
                    "multimodal" in capabilities,
                    "vision" in capabilities,
                    "image" in capabilities,
                ]
            )

        # fallback to name-based detection
        return any(
            [
                "gemma3" in model_name,
                "llava" in model_name,
                "vision" in model_name,
                "clip" in model_name,
                "multimodal" in model_name,
                "visual" in model_name,
                "image" in model_name,
                "bakllava" in model_name,
            ]
        )

ollama_api_client = None

def get_ollama_api_client() -> OllamaAPIClient:
    """Dependency for getting the Ollama API client"""
    global ollama_api_client
    if ollama_api_client is None:
        ollama_api_client = OllamaAPIClient()
    return ollama_api_client


OllamaAIPDep = Annotated[OllamaAPIClient, Depends(get_ollama_api_client)]
