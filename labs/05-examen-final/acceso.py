"""Control de acceso (AUTORIZACIÓN): qué puede ver y hacer cada rol.

Código determinista, sin LLM. Aquí vive la respuesta a "¿quién puede ver qué?".
El prompt NUNCA es el control de acceso.

  - Sesion: identidad de quien pregunta. La crea SOLO el login facial a partir de
    usuarios.json. Nunca se construye con lo que el usuario escribe en el chat.
  - NIVELES_POR_ROL: qué niveles de documentos ve cada rol en el RAG (R5).
  - TOOLS_POR_ROL: qué tools recibe el modelo y cuáles se pueden ejecutar (R6).
  - TOOLS_CON_IDENTIDAD: tools a las que el SOFTWARE les pone el código del estudiante
    desde la sesión (el modelo no puede elegir de quién son los datos).

Uso (sin LLM ni API key):
    uv run python labs/05-examen-final/acceso.py
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from config import DATA_DIR

USUARIOS_FILE = DATA_DIR / "usuarios.json"

Rol = Literal["publico", "estudiante", "docente"]

# --- Política de acceso ---------------------------------------------------------------------

# RAG: niveles de documentos que ve cada rol.
NIVELES_POR_ROL: dict[str, set[str]] = {
    "publico": {"publico"},
    "estudiante": {"publico", "estudiante"},
    "docente": {"publico", "estudiante", "docente"},
}

# Tools que cualquiera puede usar (incluso sin iniciar sesión).
TOOLS_PUBLICAS = {"buscar_documentos", "consultar_calendario", "fecha_actual", "dias_hasta", "calcular_nota_final"}

# Tools por rol: lo que NO está aquí, el modelo ni lo ve ni lo puede ejecutar.
TOOLS_POR_ROL: dict[str, set[str]] = {
    "publico": TOOLS_PUBLICAS,
    "estudiante": TOOLS_PUBLICAS | {"consultar_mis_notas", "consultar_mis_solicitudes", "solicitar_supletorio"},
    "docente": TOOLS_PUBLICAS | {"consultar_notas", "consultar_solicitudes", "registrar_nota"},
}

# Tools cuyo "codigo" lo pone el software desde la sesión (el modelo no lo ve).
TOOLS_CON_IDENTIDAD = {"consultar_mis_notas", "consultar_mis_solicitudes", "solicitar_supletorio"}


# --- Usuarios y sesión ----------------------------------------------------------------------


@dataclass(frozen=True)
class Usuario:
    id: str
    nombre: str
    rol: Rol
    codigo_estudiante: str | None
    telegram_user_id: int | None


@dataclass(frozen=True)
class Sesion:
    """Identidad autenticada. Sin sesión, el rol es "publico"."""

    rol: Rol
    usuario: Usuario | None = None
    expira: datetime | None = None

    @property
    def nombre(self) -> str | None:
        return self.usuario.nombre if self.usuario else None

    @property
    def codigo(self) -> str | None:
        return self.usuario.codigo_estudiante if self.usuario else None


SESION_PUBLICA = Sesion(rol="publico")


def niveles_permitidos(sesion: Sesion) -> set[str]:
    return NIVELES_POR_ROL[sesion.rol]


def tool_permitida(sesion: Sesion, nombre_tool: str) -> bool:
    """Se pregunta DOS veces: al armar la lista de tools y otra vez al ejecutar cada una."""
    return nombre_tool in TOOLS_POR_ROL[sesion.rol]


def cargar_usuarios() -> list[Usuario]:
    data = json.loads(USUARIOS_FILE.read_text(encoding="utf-8"))
    return [
        Usuario(
            id=u["id"],
            nombre=u["nombre"],
            rol=u["rol"],
            codigo_estudiante=u.get("codigo_estudiante"),
            telegram_user_id=u.get("telegram_user_id"),
        )
        for u in data["usuarios"]
    ]


def usuario_por_id(user_id: str) -> Usuario | None:
    return next((u for u in cargar_usuarios() if u.id == user_id), None)


def usuario_por_telegram(telegram_user_id: int) -> Usuario | None:
    """Factor de posesión: la cuenta de Telegram dice CONTRA QUIÉN se compara el rostro (1:1)."""
    return next((u for u in cargar_usuarios() if u.telegram_user_id == telegram_user_id), None)


if __name__ == "__main__":
    print("Usuarios (de usuarios.json):")
    for usuario in cargar_usuarios():
        print(f"  {usuario.id}: {usuario.nombre} | rol={usuario.rol} | codigo={usuario.codigo_estudiante}")
    print("\nPolítica de acceso:")
    for rol in ("publico", "estudiante", "docente"):
        print(f"\n  [{rol}]")
        print(f"    documentos: {sorted(NIVELES_POR_ROL[rol])}")
        print(f"    tools:      {sorted(TOOLS_POR_ROL[rol])}")