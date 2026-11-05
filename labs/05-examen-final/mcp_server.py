"""Servidor MCP "Servicios académicos" (ficticio).

Expone, con el protocolo estándar MCP, datos y acciones del sistema académico:
  - tools de ESTUDIANTE: consultar_mis_notas, consultar_mis_solicitudes, solicitar_supletorio
  - tools de DOCENTE:    consultar_notas, consultar_solicitudes, registrar_nota
  - resources (la APLICACIÓN decide leerlos): academico://calendario

Examen final (R6): el servidor NO decide quién puede llamar qué (no conoce la sesión). Eso lo
decide el asistente (acceso.py): entrega al modelo solo las tools del rol y las vuelve a
verificar al ejecutarlas. Las tools "mis_*" y solicitar_supletorio reciben el código en el
protocolo, pero el asistente lo OCULTA del esquema que ve el modelo y lo llena con el código
de la sesión (mcp_client.inyectar_identidad).

Normalmente no se ejecuta a mano: mcp_client.py lo lanza como subproceso (transporte stdio).
Para exponerlo por HTTP:
    uv run python labs/05-examen-final/mcp_server.py --http
"""

import json
import sys
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

DATA_FILE = Path(__file__).resolve().parent / "data" / "academico.json"
DATA = json.loads(DATA_FILE.read_text(encoding="utf-8"))
ESTUDIANTES: dict[str, Any] = DATA["estudiantes"]

# Solicitudes creadas en esta ejecución (en memoria: se pierden al cerrar el servidor).
SOLICITUDES: list[dict[str, Any]] = []

CodigoEstudiante = Annotated[str, Field(description="Código del estudiante, p. ej. 2026001")]
# En las tools del estudiante el asistente oculta este parámetro y lo llena desde la sesión.
CodigoSesion = Annotated[str, Field(description="Código del estudiante autenticado (lo pone el asistente)")]

# IMPORTANTE: con el transporte stdio, stdout es el canal del protocolo. Un print() normal
# corrompería los mensajes; para depurar usa print(..., file=sys.stderr).
mcp = MCPServer(
    "servicios-academicos",
    instructions="Servicios académicos (ficticios) del curso de Inteligencia Artificial.",
    log_level="WARNING",
)


def buscar_estudiante(codigo: str) -> dict[str, Any]:
    estudiante = ESTUDIANTES.get(codigo.strip())
    if estudiante is None:
        raise ToolError(f"No existe un estudiante con código {codigo!r}.")
    return estudiante


def _notas(codigo: str, con_observaciones: bool) -> dict[str, Any]:
    estudiante = buscar_estudiante(codigo)
    notas = estudiante["notas"]
    laboratorios = notas["laboratorios"]
    resultado = {
        "codigo": codigo,
        "nombre": estudiante["nombre"],
        "notas": notas,
        "promedio_laboratorios": round(sum(laboratorios) / len(laboratorios), 2) if laboratorios else None,
    }
    if con_observaciones:  # solo la versión del docente las incluye
        resultado["observaciones_docente"] = estudiante.get("observaciones_docente")
    return resultado


# --- Rol estudiante: el código NO lo elige el modelo ---------------------------------------


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def consultar_mis_notas(codigo: CodigoSesion) -> dict[str, Any]:
    """Consulta las notas del estudiante que inició sesión (sin observaciones del docente).

    Una nota null significa que aún no está registrada. Incluye el promedio de los laboratorios.
    """
    return _notas(codigo, con_observaciones=False)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def consultar_mis_solicitudes(codigo: CodigoSesion) -> list[dict[str, Any]]:
    """Lista las solicitudes de supletorio del estudiante que inició sesión."""
    buscar_estudiante(codigo)
    return [s for s in SOLICITUDES if s["codigo"] == codigo]


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False))
def solicitar_supletorio(
    codigo: CodigoSesion,
    parcial: Annotated[Literal[1, 2], Field(description="Número del parcial: 1 o 2")],
    causa: Annotated[
        Literal["incapacidad médica", "calamidad doméstica", "representación institucional"],
        Field(description="Causa justificada de la inasistencia"),
    ],
) -> dict[str, Any]:
    """Registra una solicitud de supletorio del estudiante que inició sesión.

    Es una ACCIÓN: crea un registro en el sistema académico.
    """
    estudiante = buscar_estudiante(codigo)
    if any(s["codigo"] == codigo and s["parcial"] == parcial for s in SOLICITUDES):
        raise ToolError(f"Ya existe una solicitud de supletorio del parcial {parcial} para {codigo}.")
    solicitud = {
        "radicado": f"SUP-2026-{len(SOLICITUDES) + 1:03d}",
        "codigo": codigo,
        "nombre": estudiante["nombre"],
        "parcial": parcial,
        "causa": causa,
        "estado": "pendiente: adjuntar el soporte por Teams dentro de los tres días hábiles siguientes al parcial",
    }
    SOLICITUDES.append(solicitud)
    return solicitud


# --- Rol docente: puede consultar a cualquier estudiante -----------------------------------


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def consultar_notas(codigo: CodigoEstudiante) -> dict[str, Any]:
    """[Docente] Consulta las notas de cualquier estudiante, incluidas las observaciones del docente.

    Una nota null significa que aún no está registrada.
    """
    return _notas(codigo, con_observaciones=True)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def consultar_solicitudes(codigo: CodigoEstudiante) -> list[dict[str, Any]]:
    """[Docente] Lista las solicitudes de supletorio de cualquier estudiante."""
    buscar_estudiante(codigo)
    return [s for s in SOLICITUDES if s["codigo"] == codigo]


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True))
def registrar_nota(
    codigo: CodigoEstudiante,
    evaluacion: Annotated[Literal["parcial_1", "parcial_2", "proyecto"], Field(description="Evaluación a registrar")],
    valor: Annotated[float, Field(ge=0.0, le=5.0, description="Nota de 0.0 a 5.0")],
) -> dict[str, Any]:
    """[Docente] Registra (o reemplaza) la nota de una evaluación de un estudiante.

    Es una ACCIÓN de escritura: el asistente pide confirmación antes de ejecutarla.
    """
    estudiante = buscar_estudiante(codigo)
    anterior = estudiante["notas"][evaluacion]
    estudiante["notas"][evaluacion] = round(valor, 1)  # en memoria: no modifica academico.json
    return {
        "codigo": codigo,
        "nombre": estudiante["nombre"],
        "evaluacion": evaluacion,
        "nota_anterior": anterior,
        "nota_nueva": estudiante["notas"][evaluacion],
    }


@mcp.resource("academico://calendario", description="Fechas clave del curso (parciales y proyecto final)")
def calendario() -> str:
    return "\n".join(f"{item['fecha']}: {item['evento']}" for item in DATA["calendario"])


if __name__ == "__main__":
    if "--http" in sys.argv:
        mcp.run("streamable-http")  # http://127.0.0.1:8000/mcp
    else:
        mcp.run()  # stdio: lo lanza el cliente