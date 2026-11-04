"""Pipeline por lotes: extracción en paralelo -> validación -> informe.

Etapa 1 (paralela): cada factura se extrae de forma independiente, con un
límite de concurrencia para no chocar con los rate limits del proveedor.
Etapa 2 (depende de la 1): se agregan los resultados y se genera un informe
en Markdown y HTML. Un fallo en una factura no tumba el lote: queda
registrado y el resto sigue.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from .extraccion import ResultadoExtraccion, crear_cadena_extraccion, validar_factura


@dataclass
class ResultadoLote:
    correctas: dict[str, ResultadoExtraccion] = field(default_factory=dict)
    con_avisos: dict[str, ResultadoExtraccion] = field(default_factory=dict)
    errores: dict[str, str] = field(default_factory=dict)

    @property
    def total_gastos(self) -> Decimal:
        todas = list(self.correctas.values()) + list(self.con_avisos.values())
        return sum((r.factura.total for r in todas), Decimal("0"))


def procesar_lote(
    archivos: list[Path], llm: BaseChatModel, max_concurrencia: int = 4
) -> ResultadoLote:
    cadena = crear_cadena_extraccion(llm)
    entradas = [{"texto": a.read_text(encoding="utf-8")} for a in archivos]

    # Etapa 1: en paralelo. return_exceptions evita que un error pare el lote.
    salidas = cadena.batch(
        entradas, config={"max_concurrency": max_concurrencia}, return_exceptions=True
    )

    # Etapa 2: clasificar cada resultado.
    lote = ResultadoLote()
    for archivo, salida in zip(archivos, salidas):
        if isinstance(salida, Exception):
            lote.errores[archivo.name] = f"{type(salida).__name__}: {salida}"
            continue
        resultado = ResultadoExtraccion(factura=salida, avisos=validar_factura(salida))
        destino = lote.correctas if resultado.es_valida else lote.con_avisos
        destino[archivo.name] = resultado
    return lote


def informe_markdown(lote: ResultadoLote) -> str:
    lineas = [
        "# Informe de facturas procesadas",
        "",
        f"- Correctas: {len(lote.correctas)}",
        f"- Con avisos: {len(lote.con_avisos)}",
        f"- Errores: {len(lote.errores)}",
        f"- Total gastos extraídos: {lote.total_gastos:.2f} €",
        "",
        "| Archivo | Proveedor | Nº | Fecha | Base | Impuesto | Total | Estado |",
        "| --- | --- | --- | --- | ---: | --- | ---: | --- |",
    ]
    for nombre, r in {**lote.correctas, **lote.con_avisos}.items():
        f = r.factura
        estado = "OK" if r.es_valida else "Revisar"
        lineas.append(
            f"| {nombre} | {f.proveedor} | {f.numero} | {f.fecha.isoformat()} | "
            f"{f.base_imponible:.2f} | {f.impuesto} {f.porcentaje_impuesto} % | "
            f"{f.total:.2f} | {estado} |"
        )

    if lote.con_avisos:
        lineas += ["", "## Avisos", ""]
        for nombre, r in lote.con_avisos.items():
            lineas += [f"- **{nombre}**: {aviso}" for aviso in r.avisos]

    if lote.errores:
        lineas += ["", "## Errores", ""]
        lineas += [f"- **{nombre}**: {error}" for nombre, error in lote.errores.items()]

    return "\n".join(lineas) + "\n"


def informe_html(lote: ResultadoLote) -> str:
    filas = []
    for nombre, r in {**lote.correctas, **lote.con_avisos}.items():
        f = r.factura
        estado = "OK" if r.es_valida else "Revisar"
        celdas = [
            nombre, f.proveedor, f.numero, f.fecha.isoformat(), f"{f.base_imponible:.2f}",
            f"{f.impuesto} {f.porcentaje_impuesto} %", f"{f.total:.2f}", estado,
        ]
        filas.append("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in celdas) + "</tr>")

    avisos = "".join(
        f"<li><strong>{html.escape(n)}</strong>: {html.escape(a)}</li>"
        for n, r in lote.con_avisos.items()
        for a in r.avisos
    )
    errores = "".join(
        f"<li><strong>{html.escape(n)}</strong>: {html.escape(e)}</li>"
        for n, e in lote.errores.items()
    )
    cabecera = "".join(
        f"<th>{c}</th>"
        for c in ["Archivo", "Proveedor", "Nº", "Fecha", "Base", "Impuesto", "Total", "Estado"]
    )
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>Informe de facturas</title>
<style>body{{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:.4rem .6rem;text-align:left}}</style>
</head><body>
<h1>Informe de facturas procesadas</h1>
<p>Correctas: {len(lote.correctas)} · Con avisos: {len(lote.con_avisos)} · Errores: {len(lote.errores)} ·
Total gastos: {lote.total_gastos:.2f} €</p>
<table><thead><tr>{cabecera}</tr></thead><tbody>{"".join(filas)}</tbody></table>
{f"<h2>Avisos</h2><ul>{avisos}</ul>" if avisos else ""}
{f"<h2>Errores</h2><ul>{errores}</ul>" if errores else ""}
</body></html>
"""
