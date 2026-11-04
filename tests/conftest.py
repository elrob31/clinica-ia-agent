"""Modelos falsos para probar sin claves de API ni coste.

Los tests verifican nuestra lógica (prompts, parseo, validación, recuperación,
orquestación), no la calidad del modelo, así que sustituimos el LLM y los
embeddings por versiones deterministas.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

RAIZ = Path(__file__).resolve().parents[1]
DOCS = RAIZ / "data" / "docs"
FACTURAS = RAIZ / "data" / "facturas"


class FakeLLMConHerramientas(GenericFakeChatModel):
    """GenericFakeChatModel que acepta bind_tools, para probar el agente."""

    def bind_tools(self, tools, **kwargs):
        return self


def llm_falso(*respuestas: str | AIMessage) -> GenericFakeChatModel:
    mensajes = [r if isinstance(r, AIMessage) else AIMessage(content=r) for r in respuestas]
    return GenericFakeChatModel(messages=iter(mensajes))


def json_factura(**cambios) -> str:
    datos = {
        "proveedor": "Suministros Sanitarios Teide S.L.",
        "nif": "B38123456",
        "numero": "2026-0417",
        "fecha": "2026-10-14",
        "lineas": [
            {"concepto": "Vendas neuromusculares 5cm", "cantidad": "20", "importe": "90.00"},
            {"concepto": "Gel conductor 5L", "cantidad": "3", "importe": "36.00"},
            {"concepto": "Camilla plegable aluminio", "cantidad": "1", "importe": "189.00"},
        ],
        "base_imponible": "315.00",
        "impuesto": "IGIC",
        "porcentaje_impuesto": "7",
        "cuota_impuesto": "22.05",
        "total": "337.05",
    }
    datos.update(cambios)
    return json.dumps(datos, ensure_ascii=False)


@pytest.fixture
def embeddings():
    return DeterministicFakeEmbedding(size=256)
