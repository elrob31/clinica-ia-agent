"""Extracción estructurada de facturas con validación.

El LLM convierte texto libre en un objeto ``Factura`` validado por Pydantic.
Después comprobamos la coherencia de los importes en Python: los modelos
leen bien, pero sumar es mejor dejarlo al código.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal, ROUND_HALF_UP

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

TOLERANCIA = Decimal("0.02")  # céntimos de diferencia que aceptamos por redondeo


class LineaFactura(BaseModel):
    concepto: str
    cantidad: Decimal = Field(default=Decimal("1"))
    importe: Decimal = Field(description="Importe total de la línea, sin impuestos")


class Factura(BaseModel):
    proveedor: str = Field(description="Nombre o razón social de quien emite la factura")
    nif: str | None = Field(default=None, description="NIF o CIF del emisor")
    numero: str = Field(description="Número de factura")
    fecha: date = Field(description="Fecha de emisión en formato ISO (AAAA-MM-DD)")
    lineas: list[LineaFactura] = Field(default_factory=list)
    base_imponible: Decimal
    impuesto: str = Field(default="IGIC", description="IGIC, IVA o EXENTO")
    porcentaje_impuesto: Decimal = Field(default=Decimal("0"))
    cuota_impuesto: Decimal = Field(default=Decimal("0"))
    total: Decimal


class ResultadoExtraccion(BaseModel):
    factura: Factura
    avisos: list[str] = Field(default_factory=list)

    @property
    def es_valida(self) -> bool:
        return not self.avisos


SISTEMA = (
    "Eres un asistente contable que extrae datos de facturas españolas, "
    "incluidas las de Canarias, que usan IGIC en lugar de IVA. "
    "Devuelve los importes como números con punto decimal (1234.56), "
    "las fechas en formato ISO y no inventes campos que no aparezcan."
)


def _redondear(valor: Decimal) -> Decimal:
    return valor.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def validar_factura(factura: Factura) -> list[str]:
    """Comprueba que los importes cuadran. Devuelve una lista de avisos (vacía = OK)."""
    avisos: list[str] = []

    if factura.lineas:
        suma_lineas = sum((l.importe for l in factura.lineas), Decimal("0"))
        if abs(suma_lineas - factura.base_imponible) > TOLERANCIA:
            avisos.append(
                f"La suma de líneas ({suma_lineas}) no coincide con la base imponible "
                f"({factura.base_imponible})."
            )

    cuota_esperada = _redondear(factura.base_imponible * factura.porcentaje_impuesto / 100)
    if abs(cuota_esperada - factura.cuota_impuesto) > TOLERANCIA:
        avisos.append(
            f"La cuota de {factura.impuesto} ({factura.cuota_impuesto}) no corresponde al "
            f"{factura.porcentaje_impuesto} % de la base (esperado {cuota_esperada})."
        )

    total_esperado = factura.base_imponible + factura.cuota_impuesto
    if abs(total_esperado - factura.total) > TOLERANCIA:
        avisos.append(
            f"El total ({factura.total}) no es base + impuesto (esperado {total_esperado})."
        )

    return avisos


def crear_cadena_extraccion(llm: BaseChatModel) -> Runnable:
    """Construye la cadena texto -> Factura.

    Si el modelo soporta salida estructurada nativa (function calling) la usamos.
    Si no, recurrimos a instrucciones de formato + PydanticOutputParser, que
    funciona con cualquier modelo de chat.
    """
    try:
        estructurado = llm.with_structured_output(Factura)
        prompt = ChatPromptTemplate.from_messages(
            [("system", SISTEMA), ("human", "Factura:\n\n{texto}")]
        )
        return prompt | estructurado
    except NotImplementedError:
        parser = PydanticOutputParser(pydantic_object=Factura)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SISTEMA + "\n\n{formato}"),
                ("human", "Factura:\n\n{texto}"),
            ]
        ).partial(formato=parser.get_format_instructions())
        return prompt | llm | parser


def extraer_factura(texto: str, llm: BaseChatModel) -> ResultadoExtraccion:
    factura = crear_cadena_extraccion(llm).invoke({"texto": texto})
    return ResultadoExtraccion(factura=factura, avisos=validar_factura(factura))
