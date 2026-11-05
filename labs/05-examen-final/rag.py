"""RAG: recuperar fragmentos relevantes y usarlos como contexto para el LLM.

Pipeline de consulta (online), en cada pregunta:
    pregunta → embedding → búsqueda en el vector store → contexto → prompt → LLM → respuesta

CourseRAG no sabe nada de la consola: recibe una pregunta (y el historial) y devuelve
la respuesta junto con sus fuentes. Por eso otro programa puede reutilizarlo como una
capacidad más.

Examen final (R5):
  - retrieve() recibe los NIVELES permitidos para el rol y el vector store filtra antes del
    top-k. Por defecto solo "publico" (mínimo privilegio: si alguien olvida pasar los niveles,
    no se filtra nada restringido).
  - nivel_restringido() detecta si entre los mejores resultados SIN filtro hay alguno de un
    nivel no permitido. Solo se usa el NIVEL para avisar ("requiere login"); el TEXTO de esos
    fragmentos nunca se devuelve ni llega al LLM.
"""

from dataclasses import dataclass

from embeddings import Embedder
from llm_client import LLMClient, LLMResponse, Message
from prompts import build_rag_messages
from vector_store import SearchResult, VectorStore

NO_CONTEXT = "(No se encontraron fragmentos relevantes en los documentos del curso.)"
SOLO_PUBLICO = frozenset({"publico"})
ORDEN_NIVELES = ["publico", "estudiante", "docente"]  # para decir el nivel MÍNIMO que haría falta


@dataclass
class RAGAnswer:
    text: str
    sources: list[SearchResult]  # fragmentos recuperados y enviados al LLM
    messages: list[Message]  # lo que realmente recibió el LLM
    response: LLMResponse


def format_context(results: list[SearchResult]) -> str:
    """Numera los fragmentos para que el LLM pueda citarlos: "[1] (evaluacion.md) texto"."""
    if not results:
        return NO_CONTEXT
    return "\n\n".join(
        f"[{i}] ({result.chunk.source}) {result.chunk.text}" for i, result in enumerate(results, 1)
    )


def aviso_restringido(nivel: str, autenticado: bool) -> str:
    """Texto que recibe el LLM cuando hay información relevante en un nivel no permitido."""
    if autenticado:
        accion = "indícale que su rol NO tiene acceso a esa información"
    else:
        accion = "indícale que debe iniciar sesión con /login para consultarla"
    return (
        f"[Aviso del sistema] Hay información relacionada en documentos RESTRINGIDOS (nivel {nivel}) "
        f"que no puedes ver. Si los fragmentos anteriores responden la pregunta, responde con ellos. "
        f"Si no, {accion}. No adivines ni inventes ese contenido."
    )


class CourseRAG:
    def __init__(
        self,
        llm: LLMClient,
        embedder: Embedder,
        store: VectorStore,
        top_k: int,
        min_score: float,
    ):
        self.llm = llm
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(self, question: str, niveles: set[str] = SOLO_PUBLICO) -> list[SearchResult]:
        """Solo fragmentos de los niveles permitidos y por encima del umbral."""
        query_vector = self.embedder.embed_query(question)
        results = self.store.search(query_vector, self.top_k, niveles)
        return [result for result in results if result.score >= self.min_score]

    def nivel_restringido(self, question: str, niveles: set[str]) -> str | None:
        """Nivel mínimo restringido que aparece entre los mejores resultados, o None.

        Busca SIN filtro solo para mirar los niveles dentro del programa: de esos resultados
        no se devuelve ningún texto.
        """
        query_vector = self.embedder.embed_query(question)
        bloqueados = [
            r.chunk.nivel
            for r in self.store.search(query_vector, self.top_k)
            if r.chunk.nivel not in niveles and r.score >= self.min_score
        ]
        return min(bloqueados, key=ORDEN_NIVELES.index) if bloqueados else None

    def contexto(self, question: str, niveles: set[str] = SOLO_PUBLICO) -> tuple[str, list[SearchResult]]:
        """Contexto para el LLM: fragmentos permitidos + aviso si hay información restringida."""
        sources = self.retrieve(question, niveles)
        texto = format_context(sources)
        nivel = self.nivel_restringido(question, niveles)
        if nivel:
            texto += "\n\n" + aviso_restringido(nivel, autenticado=set(niveles) != SOLO_PUBLICO)
        return texto, sources

    def answer(
        self, question: str, history: list[Message] | None = None, niveles: set[str] = SOLO_PUBLICO
    ) -> RAGAnswer:
        context, sources = self.contexto(question, niveles)
        messages = build_rag_messages(history or [], question, context)
        response = self.llm.chat(messages)
        return RAGAnswer(text=response.text, sources=sources, messages=messages, response=response)