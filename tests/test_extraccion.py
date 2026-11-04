from datetime import date
from decimal import Decimal

from clinica_ia.extraccion import Factura, extraer_factura, validar_factura

from conftest import FACTURAS, json_factura, llm_falso


def test_extrae_factura_correcta():
    texto = (FACTURAS / "factura-001.txt").read_text(encoding="utf-8")
    resultado = extraer_factura(texto, llm_falso(json_factura()))

    assert resultado.es_valida
    assert resultado.factura.fecha == date(2026, 10, 14)
    assert resultado.factura.total == Decimal("337.05")
    assert len(resultado.factura.lineas) == 3


def test_acepta_json_dentro_de_bloque_de_codigo():
    # Los modelos a menudo envuelven el JSON en ```json ... ```
    resultado = extraer_factura("...", llm_falso(f"```json\n{json_factura()}\n```"))
    assert resultado.factura.numero == "2026-0417"


def test_detecta_total_incorrecto():
    resultado = extraer_factura("...", llm_falso(json_factura(total="340.00")))
    assert not resultado.es_valida
    assert any("total" in a.lower() for a in resultado.avisos)


def test_detecta_cuota_de_impuesto_incorrecta():
    factura = Factura.model_validate_json(json_factura(cuota_impuesto="25.00", total="340.00"))
    avisos = validar_factura(factura)
    assert any("IGIC" in a for a in avisos)


def test_detecta_lineas_que_no_suman_la_base():
    factura = Factura.model_validate_json(json_factura(base_imponible="300.00", cuota_impuesto="21.00", total="321.00"))
    avisos = validar_factura(factura)
    assert any("suma de líneas" in a for a in avisos)


def test_factura_exenta_sin_lineas_es_valida():
    factura = Factura.model_validate_json(
        json_factura(lineas=[], impuesto="EXENTO", porcentaje_impuesto="0", cuota_impuesto="0", total="315.00")
    )
    assert validar_factura(factura) == []
