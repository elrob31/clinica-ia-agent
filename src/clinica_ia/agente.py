"""Agente con herramientas: decide qué hacer según la petición.

Herramientas:
- buscar_documentacion: RAG sobre los documentos internos.
- leer_factura: extrae y valida una factura de la carpeta configurada.
- calcular_penalizacion: regla de negocio determinista (no se la dejamos al LLM).
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool

from .extraccion import extraer_factura
from .rag import AsistenteRAG, formatear_contexto

SISTEMA_AGENTE = (
    "Eres el asistente interno de una clínica. Usa las herramientas en lugar de "
    "suponer: consulta la documentación para normas y precios, y usa la "
    "calculadora para importes. Responde en español, breve y citando la fuente."
)


def calcular_penalizacion_cancelacion(precio: Decimal, horas_antelacion: float, se_presento_aviso: bool = True) -> Decimal:
    """Aplica la política: >=24 h gratis, <24 h el 50 %, sin aviso el 100 %."""
    if not se_presento_aviso:
        return precio
    if horas_antelacion >= 24:
        return Decimal("0")
    return (precio * Decimal("0.5")).quantize(Decimal("0.01"))


def crear_herramientas(
    asistente: AsistenteRAG, llm: BaseChatModel, carpeta_facturas: str | Path
) -> list[BaseTool]:
    carpeta = Path(carpeta_facturas).resolve()

    @tool
    def buscar_documentacion(consulta: str) -> str:
        """Busca en la documentación interna de la clínica (horarios, tarifas, políticas)."""
        fragmentos = asistente.recuperar(consulta)
        return formatear_contexto(fragmentos) if fragmentos else "Sin resultados."

    @tool
    def leer_factura(nombre_archivo: str) -> str:
        """Extrae y valida los datos de una factura. Recibe solo el nombre del archivo."""
        ruta = (carpeta / nombre_archivo).resolve()
        # Evita que el modelo lea archivos fuera de la carpeta permitida.
        if carpeta not in ruta.parents or not ruta.is_file():
            disponibles = ", ".join(p.name for p in sorted(carpeta.glob("*.txt")))
            return f"Archivo no válido. Disponibles: {disponibles}"
        resultado = extraer_factura(ruta.read_text(encoding="utf-8"), llm)
        datos = resultado.factura.model_dump_json(indent=2)
        avisos = "\n".join(resultado.avisos) or "Ninguno"
        return f"{datos}\n\nAvisos de validación: {avisos}"

    @tool
    def calcular_penalizacion(precio_sesion: float, horas_antelacion: float, aviso: bool = True) -> str:
        """Calcula cuánto se cobra por cancelar una cita según la política de la clínica."""
        importe = calcular_penalizacion_cancelacion(
            Decimal(str(precio_sesion)), horas_antelacion, aviso
        )
        return f"Penalización: {importe} €"

    return [buscar_documentacion, leer_factura, calcular_penalizacion]


def crear_agente(asistente: AsistenteRAG, llm: BaseChatModel, carpeta_facturas: str | Path):
    return create_agent(
        llm,
        tools=crear_herramientas(asistente, llm, carpeta_facturas),
        system_prompt=SISTEMA_AGENTE,
    )


def preguntar_agente(agente, pregunta: str) -> str:
    estado = agente.invoke({"messages": [{"role": "user", "content": pregunta}]})
    return estado["messages"][-1].content
