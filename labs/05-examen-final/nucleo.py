"""Núcleo del asistente: responder(pregunta, historial, sesion), SIN Telegram (R1).

Lo usan bot.py (Telegram) y pruebas_seguridad.py (suite de ataques). Responsabilidades:
  - llamar al orquestador con la sesión (el control de acceso ocurre ahí y en acceso.py);
  - convertir errores del LLM o del servidor MCP en un mensaje claro (el detalle va al log);
  - avisar de los accesos denegados y las tools restringidas ejecutadas (para la auditoría).

También trae una consola para probar el núcleo sin Telegram, con la traza de cada respuesta:
    uv run python labs/05-examen-final/nucleo.py                  # sin sesión (rol publico)
    uv run python labs/05-examen-final/nucleo.py integrante_1     # sesión SIMULADA del estudiante
    uv run python labs/05-examen-final/nucleo.py integrante_2     # sesión SIMULADA del docente
La sesión simulada existe solo para pruebas locales: en el bot, la sesión la crea el login facial.
"""

import asyncio
import json
import logging
import sys
from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from acceso import SESION_PUBLICA, TOOLS_PUBLICAS, Sesion, usuario_por_id
from agent import ConfirmFn
from llm_client import LLMError, Message
from orchestrator import Mode, Orchestrator, OrchestratorAnswer, create_orchestrator
from tools import Tool

log = logging.getLogger("asistente")

MENSAJE_ERROR_LLM = "⚠️ El modelo de lenguaje no respondió en este momento. Intenta de nuevo en unos segundos."
MENSAJE_ERROR = "⚠️ Ocurrió un error interno procesando tu mensaje. Intenta de nuevo."

# Quien usa el núcleo decide cómo auditar: (evento, detalle) → None. Se conecta en el Paso 7.
AuditarFn = Callable[[str, dict[str, Any]], None]


async def denegar_siempre(_tool: Tool, _arguments: dict[str, Any]) -> bool:
    """Confirmación por defecto: si nadie puede confirmar, la acción NO se ejecuta."""
    return False


@dataclass
class Respuesta:
    texto: str
    answer: OrchestratorAnswer | None = None
    error: str | None = None
    tools_ejecutadas: list[str] = field(default_factory=list)
    tools_denegadas: list[str] = field(default_factory=list)


class Nucleo:
    def __init__(self, orchestrator: Orchestrator, mode: Mode = "agente", auditar: AuditarFn | None = None):
        self.orchestrator = orchestrator
        self.mode = mode
        self.auditar = auditar or (lambda _evento, _detalle: None)

    async def responder(
        self,
        pregunta: str,
        historial: list[Message],
        sesion: Sesion,
        confirm: ConfirmFn = denegar_siempre,
    ) -> Respuesta:
        try:
            answer = await self.orchestrator.answer(pregunta, historial, sesion, confirm, self.mode)
        except LLMError as exc:
            log.error("Error del LLM: %s", exc)
            return Respuesta(MENSAJE_ERROR_LLM, error=str(exc))
        except Exception as exc:  # p. ej. el servidor MCP se cayó
            log.exception("Error procesando la pregunta")
            return Respuesta(MENSAJE_ERROR, error=repr(exc))

        respuesta = Respuesta(answer.text or "(sin respuesta)", answer)
        usuario = sesion.usuario.id if sesion.usuario else None
        for run in answer.tool_runs():
            if run.status == "denegada":
                respuesta.tools_denegadas.append(run.call.name)
                self.auditar("acceso_denegado", {"tool": run.call.name, "rol": sesion.rol, "usuario": usuario})
            elif run.status == "ok":
                respuesta.tools_ejecutadas.append(run.call.name)
                if run.call.name not in TOOLS_PUBLICAS:
                    self.auditar("tool_restringida_ejecutada", {"tool": run.call.name, "rol": sesion.rol, "usuario": usuario})
        return respuesta


def describir_traza(answer: OrchestratorAnswer) -> str:
    """La traza (como --debug del Lab 04): qué tools se ofrecieron y cuáles pidió el modelo."""
    lineas = [f"tools ofrecidas al modelo: {', '.join(answer.tools_offered) or '(ninguna)'}"]
    if answer.routing:
        lineas.append(f"router → {answer.routing.decision.ruta}: {answer.routing.decision.motivo}")
    for i, step in enumerate(answer.steps, start=1):
        if not step.tool_runs:
            lineas.append(f"paso {i}: responde")
        for run in step.tool_runs:
            try:
                args = json.dumps(run.call.arguments(), ensure_ascii=False)
            except ValueError:
                args = run.call.arguments_json
            resultado = run.result.replace("\n", " ")[:110]
            lineas.append(f"paso {i}: {run.call.name}({args}) → {run.status}: {resultado}")
    if answer.sources:
        lineas.append(f"fuentes: {', '.join(answer.sources)}")
    return "\n".join(lineas)


async def crear_nucleo(stack: AsyncExitStack, mode: Mode = "agente", auditar: AuditarFn | None = None) -> Nucleo:
    return Nucleo(await create_orchestrator(stack), mode, auditar)


# --- Consola de prueba (sin Telegram) ---------------------------------------------------------


async def _preguntar(texto: str) -> str:
    return (await asyncio.to_thread(input, texto)).strip()


async def _confirmar_en_consola(tool: Tool, arguments: dict[str, Any]) -> bool:
    print(f"\n[confirmación] El asistente quiere ejecutar: {tool.name}({json.dumps(arguments, ensure_ascii=False)})")
    return (await _preguntar("¿La autorizas? (s/n): ")).lower() in {"s", "si", "sí"}


async def _consola(user_id: str | None) -> None:
    sesion = SESION_PUBLICA
    if user_id:
        usuario = usuario_por_id(user_id)
        if usuario is None:
            sys.exit(f"No existe {user_id!r} en usuarios.json")
        sesion = Sesion(rol=usuario.rol, usuario=usuario)  # SIMULADA: solo para pruebas locales
    print(f"Sesión: rol={sesion.rol} nombre={sesion.nombre} codigo={sesion.codigo}")
    print("Escribe tu pregunta (o 'salir'). Se muestra la traza de cada respuesta.\n")

    async with AsyncExitStack() as stack:
        nucleo = await crear_nucleo(stack, auditar=lambda e, d: print(f"[auditoría] {e} {d}"))
        historial: list[Message] = []
        while (pregunta := await _preguntar("Tú: ")).lower() not in {"salir", "exit"}:
            if not pregunta:
                continue
            respuesta = await nucleo.responder(pregunta, historial, sesion, _confirmar_en_consola)
            print(f"\nAsistente: {respuesta.texto}")
            if respuesta.answer:
                print("\n--- Traza ---\n" + describir_traza(respuesta.answer))
            print()
            historial += [{"role": "user", "content": pregunta}, {"role": "assistant", "content": respuesta.texto}]


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(_consola(sys.argv[1] if len(sys.argv) > 1 else None))