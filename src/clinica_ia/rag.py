"""RAG sobre la documentación interna de la clínica.

Flujo: cargar documentos -> trocear -> indexar embeddings -> recuperar los
fragmentos más relevantes -> ajustar al presupuesto de tokens -> responder
citando la fuente.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from langchain.embeddings import init_embeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDINGS_POR_DEFECTO: dict[str, str] = {
    "openai": "text-embedding-3-small",
    "google": "models/text-embedding-004",
}

PROMPT_RAG = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres el asistente de una clínica. Responde SOLO con la información del "
            "contexto. Si el contexto no contiene la respuesta, dilo claramente. "
            "Cita la fuente entre corchetes al final, por ejemplo [tarifas.md].\n\n"
            "Contexto:\n{contexto}",
        ),
        ("human", "{pregunta}"),
    ]
)


def crear_embeddings(proveedor: str | None = None) -> Embeddings:
    """Anthropic no ofrece embeddings, por eso se configuran aparte del LLM."""
    proveedor = (proveedor or os.getenv("EMBEDDINGS_PROVIDER") or "openai").lower()
    if proveedor not in EMBEDDINGS_POR_DEFECTO:
        raise ValueError(f"Embeddings no soportados para '{proveedor}'")
    nombre = "google_genai" if proveedor == "google" else proveedor
    return init_embeddings(EMBEDDINGS_POR_DEFECTO[proveedor], provider=nombre)


def cargar_documentos(carpeta: str | Path) -> list[Document]:
    carpeta = Path(carpeta)
    documentos = [
        Document(page_content=ruta.read_text(encoding="utf-8"), metadata={"fuente": ruta.name})
        for ruta in sorted(carpeta.glob("*.md")) + sorted(carpeta.glob("*.txt"))
    ]
    if not documentos:
        raise FileNotFoundError(f"No hay documentos .md o .txt en {carpeta}")
    return documentos


def trocear(documentos: list[Document], tamano: int = 500, solape: int = 80) -> list[Document]:
    """Fragmentos pequeños con solape: mejor precisión al recuperar y menos tokens."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=tamano,
        chunk_overlap=solape,
        separators=["\n## ", "\n\n", "\n", ". ", " "],
    )
    return splitter.split_documents(documentos)


def ajustar_a_presupuesto(fragmentos: list[Document], max_tokens: int) -> list[Document]:
    """Conserva los fragmentos más relevantes que quepan en ``max_tokens``.

    Los fragmentos llegan ordenados por relevancia, así que cortar por el final
    descarta siempre lo menos útil. El recuento es aproximado (~4 caracteres
    por token), suficiente para controlar costes sin depender de un tokenizador
    concreto de cada proveedor.
    """
    seleccion: list[Document] = []
    usados = 0
    for doc in fragmentos:
        coste = count_tokens_approximately([doc.page_content])
        if usados + coste > max_tokens:
            break
        seleccion.append(doc)
        usados += coste
    return seleccion


def formatear_contexto(fragmentos: list[Document]) -> str:
    return "\n\n---\n\n".join(f"[{d.metadata['fuente']}]\n{d.page_content}" for d in fragmentos)


@dataclass
class Respuesta:
    texto: str
    fuentes: list[str]


class AsistenteRAG:
    def __init__(
        self,
        llm: BaseChatModel,
        embeddings: Embeddings,
        documentos: list[Document],
        k: int = 4,
        max_tokens_contexto: int = 1500,
    ) -> None:
        self.llm = llm
        self.k = k
        self.max_tokens_contexto = max_tokens_contexto
        self.indice = InMemoryVectorStore.from_documents(trocear(documentos), embeddings)
        self.cadena = PROMPT_RAG | llm | StrOutputParser()

    @classmethod
    def desde_carpeta(cls, carpeta: str | Path, llm: BaseChatModel, embeddings: Embeddings, **kw):
        return cls(llm, embeddings, cargar_documentos(carpeta), **kw)

    def recuperar(self, pregunta: str) -> list[Document]:
        candidatos = self.indice.similarity_search(pregunta, k=self.k)
        return ajustar_a_presupuesto(candidatos, self.max_tokens_contexto)

    def preguntar(self, pregunta: str) -> Respuesta:
        fragmentos = self.recuperar(pregunta)
        if not fragmentos:
            return Respuesta("No he encontrado información sobre eso en la documentación.", [])
        texto = self.cadena.invoke(
            {"contexto": formatear_contexto(fragmentos), "pregunta": pregunta}
        )
        fuentes = sorted({d.metadata["fuente"] for d in fragmentos})
        return Respuesta(texto=texto, fuentes=fuentes)
