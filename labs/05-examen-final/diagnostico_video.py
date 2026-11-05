"""Diagnóstico de una video-nota: muestra el giro frame a frame para calibrar la prueba de vida.

Sirve para dos cosas:
  1. Saber si en sus video-notas la izquierda/derecha salen invertidas (cámara en espejo):
     graben una video-nota girando SOLO a su izquierda. Si "giro relativo" sube (+) todo está
     bien; si baja (−), pongan LIVENESS_INVERT=true en el .env.
  2. Ver cuánto giran de verdad para ajustar LIVENESS_YAW_THRESHOLD (por defecto 0.25).

Para conseguir el archivo: en Telegram Desktop, clic derecho sobre la video-nota → "Guardar como".

Uso:
    uv run python labs/05-examen-final/diagnostico_video.py video.mp4 [izquierda|derecha|ambos] [usuario_id]
"""

import sys
from pathlib import Path

import numpy as np

from config import load_identity_settings
from rostro import Rostros, cargar_enrolado
from vida import giro, leer_frames, verificar_video


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return
    path = Path(sys.argv[1])
    settings = load_identity_settings()
    rostros = Rostros()
    frames = leer_frames(path)
    print(f"{len(frames)} frames analizados (LIVENESS_INVERT={settings.liveness_invert})\n")

    signo = -1 if settings.liveness_invert else 1
    giros = []
    for i, frame in enumerate(frames):
        caras = rostros.detectar(frame)
        if len(caras) != 1:
            print(f"frame {i:3d}: {len(caras)} rostros")
            continue
        g = signo * giro(caras[0])
        giros.append(g)
        inicial = float(np.median(giros[:3]))
        rel = g - inicial
        barra = ("+" * int(rel * 20)) if rel > 0 else ("-" * int(-rel * 20))
        print(f"frame {i:3d}: giro relativo {rel:+.3f}  ancho_rostro={caras[0][2]:.0f}px  {barra}")

    if len(sys.argv) >= 4:
        tipo, usuario_id = sys.argv[2], sys.argv[3]
        enrolados = cargar_enrolado(usuario_id)
        if enrolados is None:
            print(f"\n{usuario_id} no está enrolado")
            return
        resultado = verificar_video(
            path, tipo, rostros, enrolados, settings.face_threshold, settings.yaw_threshold, settings.liveness_invert
        )
        print(f"\nResultado para el reto {tipo!r}: {resultado}")


if __name__ == "__main__":
    main()