from decimal import Decimal

import pytest
from langchain_core.messages import AIMessage

from clinica_ia.agente import (
    calcular_penalizacion_cancelacion,
    crear_agente,
    crear_herramientas,
    preguntar_agente,
)
from clinica_ia.rag import AsistenteRAG

from conftest import DOCS, FACTURAS, FakeLLMConHerramientas, json_factura, llm_falso


@pytest.mark.parametrize(
    "horas, aviso, esperado",
    [(48, True, "0"), (24, True, "0"), (10, True, "22.50"), (0, False, "45")],
)
def test_politica_de_cancelacion(horas, aviso, esperado):
    assert calcular_penalizacion_cancelacion(Decimal("45"), horas, aviso) == Decimal(esperado)


def _herramientas(embeddings, llm=None):
    llm = llm or llm_falso("sin uso")
    asistente = AsistenteRAG.desde_carpeta(DOCS, llm, embeddings)
    return {t.name: t for t in crear_herramientas(asistente, llm, FACTURAS)}


def test_leer_factura_no_sale_de_la_carpeta(embeddings):
    herramientas = _herramientas(embeddings)
    salida = herramientas["leer_factura"].invoke({"nombre_archivo": "../../pyproject.toml"})
    assert salida.startswith("Archivo no válido")


def test_leer_factura_devuelve_datos_validados(embeddings):
    herramientas = _herramientas(embeddings, llm_falso(json_factura()))
    salida = herramientas["leer_factura"].invoke({"nombre_archivo": "factura-001.txt"})
    assert "337.05" in salida and "Avisos de validación: Ninguno" in salida


def test_agente_usa_la_calculadora_y_responde(embeddings):
    llamada = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "calcular_penalizacion",
                "args": {"precio_sesion": 45, "horas_antelacion": 10},
                "id": "llamada_1",
            }
        ],
    )
    final = AIMessage(content="Pagarías 22.50 € [politica-cancelacion.md]")
    llm = FakeLLMConHerramientas(messages=iter([llamada, final]))
    asistente = AsistenteRAG.desde_carpeta(DOCS, llm, embeddings)
    agente = crear_agente(asistente, llm, FACTURAS)

    estado = agente.invoke({"messages": [{"role": "user", "content": "Cancelo con 10 h"}]})

    resultado_herramienta = [m for m in estado["messages"] if m.type == "tool"]
    assert resultado_herramienta[0].content == "Penalización: 22.50 €"
    assert estado["messages"][-1].content.startswith("Pagarías 22.50 €")
