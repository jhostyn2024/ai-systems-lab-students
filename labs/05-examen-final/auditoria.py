"""Log de auditoría en JSONL (una línea JSON por evento).

Guarda: fecha, telegram_user_id, evento, motivo/detalle y similitud facial.
NUNCA guarda imágenes, videos ni embeddings.

Eventos usados:
  login_iniciado, login_aceptado, login_rechazado (con motivo), login_bloqueado,
  logout, sesion_expirada, acceso_denegado (tool no permitida), tool_restringida_ejecutada,
  accion_confirmada, accion_cancelada, error
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

AUDIT_FILE = Path(__file__).resolve().parent / "auditoria.jsonl"  # no se versiona (.gitignore)


class Auditoria:
    def __init__(self, path: Path = AUDIT_FILE):
        self.path = path

    def registrar(self, telegram_user_id: int | None, evento: str, **detalle: Any) -> dict[str, Any]:
        entrada = {
            "fecha": datetime.now().isoformat(timespec="seconds"),
            "telegram_user_id": telegram_user_id,
            "evento": evento,
            **{k: v for k, v in detalle.items() if v is not None},
        }
        if "similitud" in entrada:
            entrada["similitud"] = round(float(entrada["similitud"]), 4)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entrada, ensure_ascii=False) + "\n")
        return entrada


class AuditoriaEnMemoria(Auditoria):
    """Para pruebas (pruebas_seguridad.py): no escribe en disco."""

    def __init__(self):
        self.entradas: list[dict[str, Any]] = []

    def registrar(self, telegram_user_id, evento, **detalle):
        entrada = {"telegram_user_id": telegram_user_id, "evento": evento, **detalle}
        self.entradas.append(entrada)
        return entrada