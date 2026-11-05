"""Prueba del filtro del RAG por rol (sin LLM ni API key).

Uso:
    uv run python labs/05-examen-final/probar_filtro.py "¿Cuándo es el simulacro del segundo parcial?"
"""

import sys

from config import load_rag_settings
from embeddings import Embedder
from vector_store import VectorStore

NIVELES_POR_ROL = {
    "publico": {"publico"},
    "estudiante": {"publico", "estudiante"},
    "docente": {"publico", "estudiante", "docente"},
}

pregunta = " ".join(sys.argv[1:]) or "¿Cuándo es el simulacro del segundo parcial?"
settings = load_rag_settings()
store = VectorStore.load(settings.index_dir)
vector = Embedder(store.embedding_model).embed_query(pregunta)

print(f"Pregunta: {pregunta}\n")
for rol, niveles in NIVELES_POR_ROL.items():
    print(f"--- rol {rol} (niveles permitidos: {sorted(niveles)})")
    for r in store.search(vector, settings.top_k, niveles):
        print(f"   {r.score:.2f}  [{r.chunk.nivel:10}]  {r.chunk.id}")
    print()