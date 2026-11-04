# clinica-ia-agent

Agente LLM para la gestión interna de una clínica, construido en Python con LangChain. Funciona con modelos de **OpenAI, Anthropic y Google** cambiando una sola variable de entorno.

Resuelve tres problemas reales de una clínica pequeña:

1. **Extraer datos de facturas** de proveedores (texto libre → JSON validado), incluido el IGIC canario.
2. **Responder preguntas sobre la documentación interna** (horarios, tarifas, políticas) con RAG y citando la fuente.
3. **Un agente con herramientas** que decide cuándo consultar documentos, leer una factura o aplicar una regla de negocio.

## Arquitectura

```
                ┌──────────────┐
  pregunta ───▶ │    Agente    │ ──▶ respuesta citando fuente
                └──────┬───────┘
         ┌─────────────┼──────────────┐
         ▼             ▼              ▼
  buscar_documentacion  leer_factura   calcular_penalizacion
         │             │              (regla determinista)
         ▼             ▼
   RAG (embeddings   Extracción estructurada
   + presupuesto     (Pydantic) + validación
   de tokens)        de importes en Python

  Lote de facturas ─▶ extracción en paralelo ─▶ validación ─▶ informe .md + .html
```

| Módulo | Qué hace |
| --- | --- |
| `llm.py` | Fábrica de modelos multi-proveedor (`init_chat_model`). |
| `extraccion.py` | Texto → `Factura` (Pydantic). Usa salida estructurada nativa si el modelo la soporta y, si no, instrucciones de formato + parser. Valida que líneas, impuesto y total cuadren. |
| `rag.py` | Carga, trocea e indexa documentos. Recupera los fragmentos más relevantes y los recorta a un presupuesto de tokens antes de llamar al modelo. |
| `pipeline.py` | Procesa un lote de facturas en paralelo con límite de concurrencia. Un fallo no detiene el lote. Genera informes en Markdown y HTML. |
| `agente.py` | Agente (`create_agent`) con tres herramientas. Restringe el acceso a archivos a una carpeta. |
| `cli.py` | Línea de comandos: `preguntar`, `extraer`, `lote`, `agente`. |

## Decisiones de diseño

- **El modelo lee, el código calcula.** Los LLM extraen bien, pero las sumas y los porcentajes se comprueban en Python con `Decimal`. Si algo no cuadra, la factura se marca para revisión en lugar de aceptarse sin más.
- **Gestión de contexto.** Fragmentos de ~500 caracteres con solape y un tope de tokens para el contexto: menos coste y menos ruido para el modelo.
- **Robustez en lotes.** `batch(..., return_exceptions=True)` con `max_concurrency` evita los límites de tasa del proveedor y que una factura ilegible tumbe todo el proceso.
- **Reglas de negocio fuera del prompt.** La penalización por cancelación es una herramienta determinista: el agente la llama en vez de "calcularla".
- **Seguridad básica del agente.** La herramienta `leer_factura` no puede leer archivos fuera de su carpeta, aunque el modelo lo intente.
- **Tests sin coste.** Los tests sustituyen el LLM y los embeddings por versiones falsas deterministas: prueban la lógica propia sin claves ni llamadas a la API.

## Instalación

```bash
git clone https://github.com/<tu-usuario>/clinica-ia-agent.git
cd clinica-ia-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[todos,dev]"
cp .env.example .env   # y rellena la clave del proveedor que uses
```

Anthropic no ofrece embeddings, así que el RAG usa OpenAI o Google para indexar (`EMBEDDINGS_PROVIDER`), aunque el modelo de chat sea Claude.

## Uso

```bash
# RAG sobre la documentación
clinica-ia preguntar "¿Cuánto cuesta la primera visita?"

# Extraer una factura
clinica-ia --proveedor anthropic extraer data/facturas/factura-001.txt

# Procesar una carpeta y generar informe Markdown + HTML en salida/
clinica-ia lote data/facturas

# Agente con herramientas
clinica-ia agente "Si cancelo una sesión de seguimiento con 10 horas de antelación, ¿cuánto pago?"
```

## Tests

```bash
pytest
```

21 tests cubren extracción y validación, troceo y presupuesto de tokens, el pipeline por lotes (incluido el manejo de errores), la política de cancelación, la restricción de archivos y el ciclo completo del agente con llamada a herramienta.

## Datos

Los documentos y facturas de `data/` son ficticios, creados para la demo.

## Próximos pasos

- Leer facturas en PDF (extracción de texto u OCR antes del LLM).
- Persistir el índice vectorial (FAISS o pgvector) en lugar de memoria.
- Evaluación automática de respuestas RAG con un conjunto de preguntas de referencia.
