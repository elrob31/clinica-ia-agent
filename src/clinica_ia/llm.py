"""Fábrica de modelos: el mismo código funciona con OpenAI, Anthropic o Google.

El proveedor se elige con la variable LLM_PROVIDER (o el argumento ``proveedor``),
así que cambiar de modelo no obliga a tocar el resto del código.
"""

from __future__ import annotations

import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

# Modelos por defecto pensados para tareas de extracción: rápidos y baratos.
# Ajústalos a los modelos vigentes de cada proveedor.
MODELOS_POR_DEFECTO: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
    "google": "gemini-2.5-flash",
}

# Nombre del proveedor en LangChain para cada alias que aceptamos.
_PROVEEDOR_LANGCHAIN: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google_genai",
}


def crear_llm(
    proveedor: str | None = None,
    modelo: str | None = None,
    temperatura: float = 0.0,
) -> BaseChatModel:
    """Devuelve un chat model del proveedor indicado.

    Temperatura 0 por defecto: en extracción y RAG queremos respuestas
    reproducibles, no creativas.
    """
    proveedor = (proveedor or os.getenv("LLM_PROVIDER") or "openai").lower()
    if proveedor not in _PROVEEDOR_LANGCHAIN:
        opciones = ", ".join(sorted(_PROVEEDOR_LANGCHAIN))
        raise ValueError(f"Proveedor '{proveedor}' no soportado. Opciones: {opciones}")

    modelo = modelo or os.getenv("LLM_MODEL") or MODELOS_POR_DEFECTO[proveedor]
    return init_chat_model(
        modelo,
        model_provider=_PROVEEDOR_LANGCHAIN[proveedor],
        temperature=temperatura,
    )
