"""Agente: un CICLO en el que el LLM decide el siguiente paso.

En el Lab 03 había una sola ronda: pedir → ejecutar → responder. Aquí se repite
mientras el modelo pida tools, así puede usar el resultado de una tool para decidir
la siguiente (consultar notas → calcular la nota final):

    repetir hasta max_steps:
        LLM(mensajes + definiciones) → ¿tool_calls?
            no → esa es la respuesta final
            sí → ejecutar cada tool, agregar resultados a los mensajes y volver a empezar
    si se agotan los pasos → una última llamada sin tools para responder con lo que hay

El límite existe porque el ciclo lo controla el modelo: sin él, un error o una mala
decisión podría repetirse indefinidamente (y cada vuelta cuesta tokens).

Examen final (R6, defensa en profundidad): el agente recibe `authorize(nombre_tool)`.
Al modelo solo se le entregan las tools de su rol, pero si aun así pide otra (inventando
el nombre, por historial o por un prompt injection), execute() la rechaza con estado
"denegada" ANTES de buscarla o ejecutarla.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from llm_client import LLMClient, LLMResponse, Message, ToolCall
from prompts import build_agent_messages
from tools import Tool

MAX_STEPS = 5
MAX_TOOL_CALLS_PER_STEP = 4  # examen final: corta modelos que piden la misma tool decenas de veces

# Quien usa el agente (la interfaz) decide cómo pedir autorización al usuario.
ConfirmFn = Callable[[Tool, dict[str, Any]], Awaitable[bool]]
# Examen final: ¿el rol de la sesión puede ejecutar esta tool? (acceso.tool_permitida)
AuthorizeFn = Callable[[str], bool]

LIMIT_MESSAGE = (
    "[Aviso del sistema] Se alcanzó el límite de pasos: ya no puedes usar herramientas. "
    "Responde ahora al estudiante con la información que ya tienes y di qué quedó pendiente."
)

REJECTED = "El usuario NO autorizó esta acción, así que no se ejecutó."
SKIPPED = "No se ejecutó: llamada repetida o demasiadas llamadas en un mismo paso."
DENIED = "Acceso denegado: el rol del usuario no tiene permiso para esta herramienta. No se ejecutó."


@dataclass
class ToolRun:
    call: ToolCall  # lo que pidió el modelo
    status: Literal["ok", "error", "rechazada", "denegada"]
    result: str  # lo que recibe el modelo como resultado


@dataclass
class AgentStep:
    response: LLMResponse  # la decisión del modelo en este paso
    tool_runs: list[ToolRun] = field(default_factory=list)  # vacío = respuesta final


@dataclass
class AgentAnswer:
    text: str
    steps: list[AgentStep]  # la traza: qué decidió el modelo en cada vuelta
    messages: list[Message]  # lo que recibió el LLM en la última llamada
    stopped_by_limit: bool  # True si se agotaron los pasos


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        tools: list[Tool],
        confirm: ConfirmFn,
        max_steps: int = MAX_STEPS,
        authorize: AuthorizeFn | None = None,  # examen final
        system_prompt: str | None = None,  # examen final: prompt con los datos de la sesión
    ):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        if len(self.tools) != len(tools):
            raise ValueError("Hay dos tools con el mismo nombre.")
        self.confirm = confirm
        self.max_steps = max_steps
        # Si nadie define una política, se deniega todo (mínimo privilegio).
        self.authorize = authorize or (lambda _nombre: False)
        self.system_prompt = system_prompt

    async def run(self, question: str, history: list[Message] | None = None) -> AgentAnswer:
        messages = build_agent_messages(history or [], question, self.system_prompt)
        definitions = [tool.definition() for tool in self.tools.values()]
        steps: list[AgentStep] = []

        for _ in range(self.max_steps):
            # TODO 2 (Lab 04): un paso del ciclo.
            # (Corregido: self.llm.chat no lleva await)
            response = self.llm.chat(messages, tools=definitions)

            if not response.tool_calls:
                steps.append(AgentStep(response))
                return AgentAnswer(response.text, steps, messages, stopped_by_limit=False)

            messages.append(response.as_message())
            step = AgentStep(response)

            vistas: set[tuple[str, str]] = set()
            for n, call in enumerate(response.tool_calls):
                # Examen final: llamadas repetidas o de más en el mismo paso no se ejecutan.
                clave = (call.name, call.arguments_json or "")
                if n >= MAX_TOOL_CALLS_PER_STEP or clave in vistas:
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": SKIPPED})
                    continue
                vistas.add(clave)
                tool_run = await self.execute(call)
                step.tool_runs.append(tool_run)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": tool_run.result
                })

            steps.append(step)

        # TODO 3 (Lab 04): se agotaron los pasos y el modelo sigue pidiendo tools. Fuerza una respuesta final.
        messages.append({"role": "user", "content": LIMIT_MESSAGE})
        # (Corregido: self.llm.chat no lleva await)
        final_response = self.llm.chat(messages, tools=definitions, allow_tools=False)
        steps.append(AgentStep(final_response))
        return AgentAnswer(final_response.text, steps, messages, stopped_by_limit=True)

    async def execute(self, call: ToolCall) -> ToolRun:
        """Ejecuta una tool pedida por el modelo (igual que en el Lab 03), si el rol lo permite."""
        # Examen final: segunda verificación del rol, ANTES de buscar o ejecutar la tool.
        if not self.authorize(call.name):
            return ToolRun(call, "denegada", DENIED)

        tool = self.tools.get(call.name)
        if tool is None:
            return ToolRun(call, "error", f"Error: no existe la tool {call.name!r}.")
        try:
            # Examen final: algunos modelos mandan "null" para tools sin parámetros.
            arguments = {} if (call.arguments_json or "").strip() in {"", "null", "{}"} else call.arguments()
        except ValueError as exc:
            return ToolRun(call, "error", f"Error: {exc}")

        # Generar no es ejecutar: una acción con efectos solo se ejecuta si el usuario la autoriza.
        if not tool.read_only and not await self.confirm(tool, arguments):
            return ToolRun(call, "rechazada", REJECTED)

        try:
            result = await tool.run(arguments)
        except Exception as exc:  # argumentos inválidos, regla de negocio, servidor caído...
            return ToolRun(call, "error", f"Error: {exc}")
        return ToolRun(call, "ok", result)