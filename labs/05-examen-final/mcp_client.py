"""Cliente MCP: conecta el asistente con el servidor "Servicios académicos".

Responsabilidades:
  1. Lanzar el servidor como subproceso y hablarle por stdin/stdout (transporte stdio).
  2. Descubrir sus tools (list_tools) y CONVERTIRLAS en el mismo `Tool` que las tools locales.
  3. Leer resources cuando la aplicación lo decide.
  4. (Examen final) inyectar_identidad: el código del estudiante lo pone el software.

Uso (sin LLM ni API key):
    uv run python labs/05-examen-final/mcp_client.py
"""

import asyncio
import json
import sys
from contextlib import AsyncExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters
from mcp.types import CallToolResult
from mcp.types import Tool as MCPTool

from tools import Tool, ToolExecutionError

SERVER_SCRIPT = Path(__file__).resolve().parent / "mcp_server.py"


def server_parameters() -> StdioServerParameters:
    return StdioServerParameters(command=sys.executable, args=[str(SERVER_SCRIPT)])


async def open_client(stack: AsyncExitStack) -> Client:
    """Lanza el servidor y se conecta. La conexión se cierra cuando se cierra `stack`."""
    return await stack.enter_async_context(Client(server_parameters()))


def result_to_text(result: CallToolResult) -> str:
    text = "\n".join(block.text for block in result.content if block.type == "text")
    if result.is_error:
        raise ToolExecutionError(text)
    return text


def to_tool(client: Client, mcp_tool: MCPTool) -> Tool:
    """Adaptador: definición MCP → Tool (la misma clase que usan las tools locales)."""

    async def run(arguments: dict[str, Any]) -> str:
        result = await client.call_tool(mcp_tool.name, arguments)
        return result_to_text(result)

    annotations = mcp_tool.annotations
    return Tool(
        name=mcp_tool.name,
        description=mcp_tool.description or "",
        parameters=mcp_tool.input_schema,
        run=run,
        read_only=bool(annotations and annotations.read_only_hint),  # sin anotación: se asume acción
        origin=client.server_info.name if client.server_info else "mcp",
    )


def inyectar_identidad(tool: Tool, codigo: str) -> Tool:
    """Examen final (R6): el software pone el código del estudiante; el modelo no puede elegirlo.

    1. El esquema que ve el LLM ya NO tiene la propiedad "codigo".
    2. Al ejecutar se descarta cualquier "codigo" que el modelo haya enviado y se usa el de la
       sesión (que viene de usuarios.json tras el login facial, nunca del chat).
    """
    properties = {k: v for k, v in tool.parameters.get("properties", {}).items() if k != "codigo"}
    required = [r for r in tool.parameters.get("required", []) if r != "codigo"]
    parameters = {**tool.parameters, "properties": properties, "required": required}
    run_original = tool.run

    async def run(arguments: dict[str, Any]) -> str:
        argumentos = {k: v for k, v in arguments.items() if k != "codigo"}  # lo que diga el modelo, se ignora
        return await run_original({**argumentos, "codigo": codigo})

    return replace(tool, parameters=parameters, run=run)


async def load_mcp_tools(client: Client) -> list[Tool]:
    listed = await client.list_tools()
    return [to_tool(client, mcp_tool) for mcp_tool in listed.tools]


async def read_resource_text(client: Client, uri: str) -> str:
    result = await client.read_resource(uri)
    return "\n".join(content.text for content in result.contents if hasattr(content, "text"))


async def _demo() -> None:
    from acceso import usuario_por_id

    async with Client(server_parameters()) as client:
        info = client.server_info
        print(f"Conectado a {info.name if info else '?'} (protocolo MCP {client.protocol_version})\n")

        tools = await load_mcp_tools(client)
        print("Tools descubiertas:")
        for tool in tools:
            kind = "solo lectura" if tool.read_only else "ACCIÓN"
            print(f"  - {tool.name} [{kind}] parámetros: {list(tool.parameters.get('properties', {}))}")
        by_name = {tool.name: tool for tool in tools}

        # Inyección de identidad con el integrante 1 (estudiante) de usuarios.json
        estudiante = usuario_por_id("integrante_1")
        mis_notas = inyectar_identidad(by_name["consultar_mis_notas"], estudiante.codigo_estudiante)
        print(f"\nSesión simulada: {estudiante.nombre} (código {estudiante.codigo_estudiante})")
        print("Esquema que VE EL MODELO para consultar_mis_notas:", json.dumps(mis_notas.parameters, ensure_ascii=False))

        print("\n1) El modelo la llama normal, sin argumentos:")
        print(await mis_notas.run({}))

        print("\n2) ATAQUE: el modelo intenta pasar codigo='2026001' (Ana Torres):")
        print(await mis_notas.run({"codigo": "2026001"}))


if __name__ == "__main__":
    asyncio.run(_demo())