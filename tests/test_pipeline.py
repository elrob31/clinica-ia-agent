from decimal import Decimal

from clinica_ia.pipeline import informe_html, informe_markdown, procesar_lote

from conftest import FACTURAS, json_factura, llm_falso

ARCHIVOS = sorted(FACTURAS.glob("*.txt"))


def test_lote_clasifica_correctas_avisos_y_errores():
    # concurrencia 1 para que el orden de respuestas del modelo falso sea fijo
    llm = llm_falso(
        json_factura(),                      # factura-001: correcta
        json_factura(total="999.99"),        # factura-002: total incoherente
    )
    lote = procesar_lote(ARCHIVOS, llm, max_concurrencia=1)

    assert list(lote.correctas) == ["factura-001.txt"]
    assert list(lote.con_avisos) == ["factura-002.txt"]
    assert lote.total_gastos == Decimal("337.05") + Decimal("999.99")


def test_un_error_no_tumba_el_lote():
    llm = llm_falso(json_factura(), "esto no es JSON")
    lote = procesar_lote(ARCHIVOS, llm, max_concurrencia=1)

    assert len(lote.correctas) == 1
    assert "factura-002.txt" in lote.errores


def test_informes_incluyen_resumen_y_avisos():
    llm = llm_falso(json_factura(), json_factura(total="999.99"))
    lote = procesar_lote(ARCHIVOS, llm, max_concurrencia=1)

    md = informe_markdown(lote)
    assert "Correctas: 1" in md and "## Avisos" in md and "factura-002.txt" in md

    pagina = informe_html(lote)
    assert pagina.startswith("<!doctype html>")
    assert "<table>" in pagina and "Revisar" in pagina
