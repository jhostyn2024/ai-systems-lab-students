"""Capacidades del asistente expresadas como tools.

El RAG del Lab 02 se convierte aquí en una tool más (buscar_documentos). Solo hace la
RECUPERACIÓN: devuelve los fragmentos numerados con su fuente, y el LLM que la pidió
redacta la respuesta. Para el agente no hay diferencia entre buscar en documentos,
consultar un servidor MCP o ejecutar una función local: todas son `Tool`.

Examen final (R5): la tool se crea POR SESIÓN con los niveles permitidos para el rol.
Los niveles quedan fijos dentro de la función (closure): NO son un parámetro de la tool,
así que el modelo no los ve ni los puede cambiar. El esquema solo tiene "consulta".

Uso (sin LLM ni API key; requiere el índice de ingest.py):
    uv run python labs/05-examen-final/capabilities.py publico "¿Cuándo es el simulacro del segundo parcial?"
    uv run python labs/05-examen-final/capabilities.py estudiante "¿Cuándo es el simulacro del segundo parcial?"
"""

import asyncio
import sys
from typing import Any

from rag import SOLO_PUBLICO, CourseRAG
from tools import Tool


def rag_tool(rag: CourseRAG, niveles: set[str] = SOLO_PUBLICO) -> Tool:
    # TODO 1 (Lab 04) resuelto. Examen final: `niveles` lo decide el software según el rol de la
    # sesión; por defecto solo "publico" (mínimo privilegio).
    # No se llama a rag.answer porque la tool solo RECUPERA: quien redacta la respuesta es el
    # LLM del agente, que pidió la tool. Si llamáramos a answer, habría una segunda llamada al LLM
    # (más tokens y más latencia) y el agente recibiría una respuesta ya redactada en vez de los
    # fragmentos para citar y combinar con otras tools.

    async def run(args: dict[str, Any]) -> str:
        consulta = str(args.get("consulta", "")).strip()
        if not consulta:
            raise ValueError("falta el argumento 'consulta'")
        contexto, _fuentes = rag.contexto(consulta, niveles)  # fragmentos permitidos + aviso
        return contexto

    return Tool(
        name="buscar_documentos",
        origin="rag",
        read_only=True,
        description=(
            "Busca información en los documentos del curso (programa, docente, horarios, "
            "evaluación, fechas de parciales, políticas, anuncios, preguntas frecuentes y, "
            "según el rol del usuario, guías y monitorías) y devuelve fragmentos numerados "
            "con su fuente."
        ),
        parameters={
            "type": "object",
            "properties": {
                "consulta": {
                    "type": "string",
                    "description": "Pregunta autocontenida para buscar en los documentos, p. ej. 'fecha del primer parcial'.",
                }
            },
            "required": ["consulta"],
        },
        run=run,
    )


# Solo para la demo. En el sistema real esta tabla vive en acceso.py (Paso 4) y el rol sale
# de la sesión creada por el login facial, nunca de lo que se escribe.
NIVELES_DEMO = {
    "publico": {"publico"},
    "estudiante": {"publico", "estudiante"},
    "docente": {"publico", "estudiante", "docente"},
}


async def _demo(rol: str, question: str) -> None:
    from config import load_rag_settings
    from embeddings import Embedder
    from vector_store import VectorStore

    settings = load_rag_settings()
    store = VectorStore.load(settings.index_dir)
    # llm=None: la tool solo recupera, no llama al LLM (rag.answer() sí lo haría).
    rag = CourseRAG(None, Embedder(store.embedding_model), store, settings.top_k, settings.min_score)
    tool = rag_tool(rag, NIVELES_DEMO[rol])
    print(f"[rol {rol}] esquema que ve el modelo: {list(tool.parameters['properties'])}")
    print(f"[rol {rol}] {tool.name}({{'consulta': {question!r}}}) →\n")
    print(await tool.run({"consulta": question}))


if __name__ == "__main__":
    args = sys.argv[1:]
    rol = args.pop(0) if args and args[0] in NIVELES_DEMO else "publico"
    asyncio.run(_demo(rol, " ".join(args) or "¿Cuándo es el simulacro del segundo parcial?"))