"""Orquestador: compone LLM, RAG, tools locales y MCP en un solo asistente.

Dos estrategias para decidir qué capacidad usar:

  modo "router" (flujo fijo, decidido por el software después de clasificar):
      pregunta → router → "directa"   → LLM sin tools               (como el Lab 01)
                        → "documentos" → CourseRAG.answer           (como el Lab 02)
                        → "acciones"   → agente con tools y MCP     (como el Lab 03, con ciclo)

  modo "agente" (flujo dinámico, decidido por el modelo en cada paso):
      pregunta → agente con TODAS las capacidades como tools (incluida buscar_documentos)

El orquestador no sabe nada de la consola ni de Telegram: recibe una pregunta y devuelve la
respuesta con su traza (ruta, pasos, fuentes, llamadas al LLM).

Examen final: answer() recibe la SESIÓN y la función de confirmación del canal. En cada
pregunta arma un agente con:
  1. solo las tools del rol (acceso.TOOLS_POR_ROL): el modelo ni siquiera ve las demás;
  2. buscar_documentos filtrada por los niveles del rol;
  3. el código del estudiante inyectado en las tools "mis_*" (mcp_client.inyectar_identidad);
  4. authorize = acceso.tool_permitida: se vuelve a verificar el rol al ejecutar.
"""

from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Literal

from acceso import TOOLS_CON_IDENTIDAD, Sesion, niveles_permitidos, tool_permitida
from agent import MAX_STEPS, Agent, AgentAnswer, AgentStep, ConfirmFn
from capabilities import rag_tool
from config import load_rag_settings, load_settings
from embeddings import Embedder
from llm_client import LLMClient, LLMResponse, Message
from mcp_client import inyectar_identidad, load_mcp_tools, open_client, read_resource_text
from prompts import build_agent_system_prompt, build_direct_messages
from rag import CourseRAG
from router import RouterResult, route
from tools import LOCAL_TOOLS, Tool
from vector_store import VectorStore

Mode = Literal["router", "agente"]


@dataclass
class OrchestratorAnswer:
    text: str
    mode: Mode
    routing: RouterResult | None = None  # solo en modo router
    steps: list[AgentStep] = field(default_factory=list)  # traza del agente, si se usó
    sources: list[str] = field(default_factory=list)  # fuentes calculadas por el programa (ruta documentos)
    responses: list[LLMResponse] = field(default_factory=list)  # TODAS las llamadas al LLM, en orden
    stopped_by_limit: bool = False
    tools_offered: list[str] = field(default_factory=list)  # examen final: tools que se le dieron al modelo

    def tools_used(self) -> list[str]:
        return [run.call.name for step in self.steps for run in step.tool_runs]

    def tool_runs(self):
        return [run for step in self.steps for run in step.tool_runs]


class Orchestrator:
    def __init__(
        self,
        llm: LLMClient,
        rag: CourseRAG,
        action_tools: list[Tool],
        max_steps: int = MAX_STEPS,
    ):
        self.llm = llm
        self.rag = rag
        self.action_tools = action_tools  # TODAS (locales + MCP); se filtran por sesión en cada pregunta
        self.max_steps = max_steps

    # --- Examen final: control de acceso al armar el agente ---------------------------------

    def tools_for(self, sesion: Sesion, include_rag: bool = True) -> list[Tool]:
        """Solo las tools que el rol puede usar; las "mis_*" con el código de la sesión."""
        tools: list[Tool] = [rag_tool(self.rag, niveles_permitidos(sesion))] if include_rag else []
        for tool in self.action_tools:
            if not tool_permitida(sesion, tool.name):
                continue  # el modelo nunca la verá
            if tool.name in TOOLS_CON_IDENTIDAD:
                if not sesion.codigo:
                    continue
                tool = inyectar_identidad(tool, sesion.codigo)
            tools.append(tool)
        return tools

    def agent_for(self, sesion: Sesion, confirm: ConfirmFn, include_rag: bool = True) -> Agent:
        return Agent(
            self.llm,
            self.tools_for(sesion, include_rag),
            confirm,
            self.max_steps,
            authorize=lambda nombre: tool_permitida(sesion, nombre),  # segunda verificación
            system_prompt=build_agent_system_prompt(sesion.rol, sesion.nombre),
        )

    async def answer(
        self, question: str, history: list[Message], sesion: Sesion, confirm: ConfirmFn, mode: Mode = "agente"
    ) -> OrchestratorAnswer:
        if mode == "agente":
            agent = self.agent_for(sesion, confirm)
            answer = self._from_agent(await agent.run(question, history), mode)
            answer.tools_offered = list(agent.tools)
            return answer

        routing = route(self.llm, question)
        ruta = routing.decision.ruta
        # TODO 5 (Lab 04): ejecuta el camino de la ruta elegida. Examen final: con la sesión.
        if ruta == "directa":
            response = self.llm.chat(build_direct_messages(history, question))
            answer = OrchestratorAnswer(text=response.text, mode=mode, responses=[response])
        elif ruta == "documentos":
            rag_answer = self.rag.answer(question, history, niveles_permitidos(sesion))
            sources = sorted(list(set(result.chunk.source for result in rag_answer.sources)))
            answer = OrchestratorAnswer(
                text=rag_answer.text,
                mode=mode,
                sources=sources,
                responses=[rag_answer.response]
            )
        elif ruta == "acciones":
            agent = self.agent_for(sesion, confirm, include_rag=False)
            answer = self._from_agent(await agent.run(question, history), mode)
            answer.tools_offered = list(agent.tools)
        else:
            # Fallback defensivo por si viene otra ruta
            response = self.llm.chat(build_direct_messages(history, question))
            answer = OrchestratorAnswer(text=response.text, mode=mode, responses=[response])

        answer.routing = routing
        answer.responses.insert(0, routing.response)
        return answer

    @staticmethod
    def _from_agent(result: AgentAnswer, mode: Mode) -> OrchestratorAnswer:
        return OrchestratorAnswer(
            result.text,
            mode,
            steps=result.steps,
            responses=[step.response for step in result.steps],
            stopped_by_limit=result.stopped_by_limit,
        )


def calendario_tool(mcp_client: Any) -> Tool:
    """El resource academico://calendario expuesto como tool pública de solo lectura (R6)."""

    async def run(_arguments: dict[str, Any]) -> str:
        return await read_resource_text(mcp_client, "academico://calendario")

    return Tool(
        name="consultar_calendario",
        description="Devuelve las fechas clave del curso (parciales y proyecto final) del sistema académico.",
        parameters={"type": "object", "properties": {}, "required": []},
        run=run,
        read_only=True,
        origin="servicios-academicos",
    )


async def create_orchestrator(stack: AsyncExitStack) -> Orchestrator:
    """Arma el sistema completo: configuración, LLM, índice, RAG, tools locales y servidor MCP.

    Lanza ValueError (configuración), FileNotFoundError (falta el índice) o un error de
    conexión con el servidor MCP. El servidor queda abierto hasta que se cierre `stack`.
    """
    settings = load_settings()
    rag_settings = load_rag_settings()
    store = VectorStore.load(rag_settings.index_dir)

    llm = LLMClient(settings)
    rag = CourseRAG(
        llm=llm,
        # La pregunta se convierte con el mismo modelo que indexó los documentos.
        embedder=Embedder(store.embedding_model),
        store=store,
        top_k=rag_settings.top_k,
        min_score=rag_settings.min_score,
    )
    mcp_client = await open_client(stack)
    action_tools = [*LOCAL_TOOLS, calendario_tool(mcp_client), *await load_mcp_tools(mcp_client)]
    return Orchestrator(llm, rag, action_tools)