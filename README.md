# fast_lang_stream

Why another template project while bunch of similar ones out there?

Yes, lots of `langChain` and `streamlit` demos on Github. But almost all of them are direct integration, no REST API provided.

This project aims to implement modern python solution(like, uv and fastapi) for UI, API around LLM use cases.

## Stack

llm: ollama

backend: uv + fastapi + langChain

frontend: uv + streamlit

## Dev

```bash
brew install ollama
ollama pull gemma3
```

## Other integration examples
- [langChain + streamlit](https://github.com/streamlit/llm-examples)
- [langChain(RAG) + streamlit](https://github.com/streamlit/example-app-langchain-rag)
- [langChain + langGraph + Next.js](https://github.com/langchain-ai/chat-langchain)
- [langChain.js + Next.js](https://github.com/langchain-ai/langchain-nextjs-template)
- [langChain + Neo4j + svelte.js](https://github.com/docker/genai-stack)
- [FastAPI + React.js](https://github.com/fastapi/full-stack-fastapi-template)

## TODO

[] langGraph integration
[] RouteLLM integration
[] Vector DB integration
