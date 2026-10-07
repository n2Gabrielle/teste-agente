"""Compatibilidade para código legado; a implementação está em agent/ e services/."""

import os

from agent.llm import get_llm


def get_response_from_openai(messages, api_key: str | None = None):
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Configure OPENAI_API_KEY antes de chamar o modelo.")
    return get_llm(api_key).invoke(messages)