"""Interfaz de línea de comandos.

Ejemplos:
    clinica-ia preguntar "¿Cuánto cuesta la primera visita?"
    clinica-ia extraer data/facturas/factura-001.txt
    clinica-ia lote data/facturas --salida salida/
    clinica-ia agente "Si cancelo una sesión de seguimiento con 10 horas, ¿cuánto pago?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from .llm import crear_llm

DOCS_POR_DEFECTO = Path("data/docs")
FACTURAS_POR_DEFECTO = Path("data/facturas")


def _asistente(llm, docs: Path):
    from .rag import AsistenteRAG, crear_embeddings

    return AsistenteRAG.desde_carpeta(docs, llm, crear_embeddings())


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="clinica-ia", description=__doc__.splitlines()[0])
    parser.add_argument("--proveedor", choices=["openai", "anthropic", "google"])
    parser.add_argument("--modelo")
    sub = parser.add_subparsers(dest="comando", required=True)

    p = sub.add_parser("preguntar", help="Pregunta a la documentación (RAG)")
    p.add_argument("pregunta")
    p.add_argument("--docs", type=Path, default=DOCS_POR_DEFECTO)

    p = sub.add_parser("extraer", help="Extrae y valida una factura")
    p.add_argument("archivo", type=Path)

    p = sub.add_parser("lote", help="Procesa una carpeta de facturas y genera informe")
    p.add_argument("carpeta", type=Path)
    p.add_argument("--salida", type=Path, default=Path("salida"))
    p.add_argument("--concurrencia", type=int, default=4)

    p = sub.add_parser("agente", help="Pregunta al agente con herramientas")
    p.add_argument("pregunta")
    p.add_argument("--docs", type=Path, default=DOCS_POR_DEFECTO)
    p.add_argument("--facturas", type=Path, default=FACTURAS_POR_DEFECTO)

    args = parser.parse_args(argv)
    llm = crear_llm(args.proveedor, args.modelo)

    if args.comando == "preguntar":
        respuesta = _asistente(llm, args.docs).preguntar(args.pregunta)
        print(respuesta.texto)
        print(f"\nFuentes: {', '.join(respuesta.fuentes) or '—'}")

    elif args.comando == "extraer":
        from .extraccion import extraer_factura

        resultado = extraer_factura(args.archivo.read_text(encoding="utf-8"), llm)
        print(resultado.factura.model_dump_json(indent=2))
        for aviso in resultado.avisos:
            print(f"AVISO: {aviso}", file=sys.stderr)

    elif args.comando == "lote":
        from .pipeline import informe_html, informe_markdown, procesar_lote

        archivos = sorted(args.carpeta.glob("*.txt"))
        lote = procesar_lote(archivos, llm, args.concurrencia)
        args.salida.mkdir(parents=True, exist_ok=True)
        (args.salida / "informe.md").write_text(informe_markdown(lote), encoding="utf-8")
        (args.salida / "informe.html").write_text(informe_html(lote), encoding="utf-8")
        print(informe_markdown(lote))
        print(f"Informes guardados en {args.salida}/")

    elif args.comando == "agente":
        from .agente import crear_agente, preguntar_agente

        agente = crear_agente(_asistente(llm, args.docs), llm, args.facturas)
        print(preguntar_agente(agente, args.pregunta))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
