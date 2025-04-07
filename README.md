# Welcome to Ollama + FastAPI + LangChain + Streamlit examples! 👋

Why another template project while bunch of similar ones out there?

Yes, lots of `LangChain` and `Streamlit` demos on Github. But almost all of them are direct integration, no API provided.

This project aims to implement modern python solution(like, uv and FastAPI) for UI, API around LLM use cases.

## Examples

1. [chatbot](chatbot) ([code](./frontend/pages/chatbot.py))
  General purpose chatbot integrated with Ollama, features:
  - Model selection
  - Model parameters setting
  - Ollama health check
2. [image descriptor](image_descriptor) ([code](./frontend/pages/image_descriptor.py))
  - describe user uploaded image

## Stack

llm: [Ollama](http://ollama.com)

backend: [uv](https://docs.astral.sh/uv/) + [FastAPI](https://fastapi.tiangolo.com/) + [LangChain](http://langchain.com)

frontend: [uv](https://docs.astral.sh/uv/) + [Streamlit](https://streamlit.io)

## Dev

### Install Ollama & uv

```bash
# other platforms pls check https://ollama.com/download/linux
brew install ollama
# lightweight and multimodal capabilities for local dev
ollama pull gemma3
# other platforms pls check https://docs.astral.sh/uv/getting-started/installation/
brew install uv
```

### Run API app

```bash
cd backend
uv sync --active
uv run main.py
```

For debug in VSCode, add this to `.vscode/launch.json`, then hit `F5`.

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "API Debugger",
            "type": "debugpy",
            "request": "launch",
            "program": "main.py",
            "console": "integratedTerminal"
        }
    ]
}
```

### Run UI app

```bash
cd frontend
uv sync --active
source .venv/bin/activate
streamlit run intro.py
```

For debug in VSCode, add this to `.vscode/launch.json`, then hit `F5`.

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Streamlit Debug",
            "type": "debugpy",
            "request": "launch",
            "module": "streamlit",
            "args": [
                "run",
                "intro.py",
                "--server.port=8501",
                "--server.headless=true",
                "--server.runOnSave=true"
            ],
            "justMyCode": true,
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            }
        }
    ]
}
```

## Other integration examples
- [LangChain + Streamlit](https://github.com/streamlit/llm-examples)
- [LangChain(RAG) + Streamlit](https://github.com/streamlit/example-app-langchain-rag)
- [LangChain + LangGraph + Next.js](https://github.com/langchain-ai/chat-langchain)
- [LangChain.js + Next.js](https://github.com/langchain-ai/langchain-nextjs-template)
- [LangChain + Neo4j + Svelte.js](https://github.com/docker/genai-stack)
- [FastAPI + React.js](https://github.com/fastapi/full-stack-fastapi-template)

## TODO

- LangGraph integration
- RouteLLM integration
- Vector DB integration
