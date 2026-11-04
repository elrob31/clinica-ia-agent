from langchain_core.documents import Document

from clinica_ia.rag import AsistenteRAG, ajustar_a_presupuesto, cargar_documentos, trocear

from conftest import DOCS, llm_falso


def test_carga_todos_los_documentos_con_su_fuente():
    docs = cargar_documentos(DOCS)
    fuentes = {d.metadata["fuente"] for d in docs}
    assert fuentes == {"horarios-y-servicios.md", "tarifas.md", "politica-cancelacion.md"}


def test_trocear_conserva_la_fuente_en_cada_fragmento():
    fragmentos = trocear(cargar_documentos(DOCS), tamano=200, solape=20)
    assert len(fragmentos) > 3
    assert all("fuente" in f.metadata for f in fragmentos)


def test_presupuesto_de_tokens_corta_por_el_final():
    docs = [Document(page_content="x" * 400, metadata={"fuente": f"{i}.md"}) for i in range(5)]
    # ~100 tokens por fragmento: con 250 caben dos
    seleccion = ajustar_a_presupuesto(docs, max_tokens=250)
    assert [d.metadata["fuente"] for d in seleccion] == ["0.md", "1.md"]


def test_presupuesto_cero_no_devuelve_nada():
    docs = [Document(page_content="hola", metadata={"fuente": "a.md"})]
    assert ajustar_a_presupuesto(docs, max_tokens=0) == []


def test_responde_con_fuentes(embeddings):
    llm = llm_falso("La primera visita cuesta 55 € [tarifas.md]")
    asistente = AsistenteRAG.desde_carpeta(DOCS, llm, embeddings, k=3)

    respuesta = asistente.preguntar("¿Cuánto cuesta la primera visita?")

    assert "55 €" in respuesta.texto
    assert 1 <= len(respuesta.fuentes) <= 3
