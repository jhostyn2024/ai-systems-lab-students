"""Prueba de vida: verifica la video-nota del login contra el reto aleatorio.

Por frames del video (≈10 por segundo):
  i.   en cada frame analizado hay EXACTAMENTE un rostro, y es el mismo a lo largo del video;
  ii.  el movimiento pedido ocurre, medido con los landmarks de YuNet:
           giro = (x_nariz − x_medio_ojos) / distancia_entre_ojos
       Mirando al frente el giro ≈ 0. Al girar, la nariz se desplaza hacia un lado.
       Se compara contra la posición inicial (mediana de los primeros frames);
  iii. el frame MÁS FRONTAL (menor |giro|) supera el umbral contra el usuario enrolado (1:1).

Por qué giros y no "acercar el rostro": una foto mostrada en otra pantalla se puede acercar a la
cámara y el rostro crece igual que uno real. Para imitar un giro, en cambio, hay que inclinar
la foto con mucha perspectiva: en nuestra prueba (foto plana inclinada ±45°) el giro llegó a
0.33 en un solo sentido y nunca cumplió el reto "ambos". Por geometría, una cabeza real que
gira θ grados da un giro ≈ 0.4·tan(θ): 45° → 0.40, 60° → 0.69. Umbral por defecto: 0.35
(LIVENESS_YAW_THRESHOLD), por encima de lo que logró la foto inclinada; se calibra con
diagnostico_video.py usando video-notas reales.
Limitación conocida: no detiene una máscara 3D ni un deepfake en tiempo real que gire la cabeza.

El resultado incluye un motivo detallado para la AUDITORÍA; al usuario no se le dice cuál falló.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rostro import Rostros, similitud, similitud_con_enrolado

FRAMES_POR_SEGUNDO = 10  # cuántos frames por segundo se analizan
MAX_FRAMES = 150
MIN_FRAMES_CON_ROSTRO = 8
MIN_PROPORCION_CON_ROSTRO = 0.7  # al menos 70 % de los frames analizados con un rostro
MISMO_ROSTRO_MINIMO = 0.30  # coseno mínimo entre frames frontales del mismo video


@dataclass
class ResultadoVida:
    aceptado: bool
    motivo: str  # "ok" o la verificación que falló (solo para la auditoría)
    similitud: float | None = None
    detalle: dict[str, Any] = field(default_factory=dict)


def giro(cara: np.ndarray) -> float:
    ojo_der_x, ojo_izq_x, nariz_x = cara[4], cara[6], cara[8]
    distancia_ojos = max(abs(ojo_izq_x - ojo_der_x), 1.0)
    return float((nariz_x - (ojo_der_x + ojo_izq_x) / 2) / distancia_ojos)


def leer_frames(path: Path) -> list[np.ndarray]:
    video = cv2.VideoCapture(str(path))
    fps = video.get(cv2.CAP_PROP_FPS) or 30
    paso = max(1, round(fps / FRAMES_POR_SEGUNDO))
    frames, i = [], 0
    while len(frames) < MAX_FRAMES:
        ok, frame = video.read()
        if not ok:
            break
        if i % paso == 0:
            frames.append(frame)
        i += 1
    video.release()
    return frames


def movimiento_cumplido(tipo: str, deltas: list[float], umbral: float) -> bool:
    """deltas = giro de cada frame menos el giro inicial. Positivo = hacia la izquierda del usuario."""
    izquierda = [i for i, d in enumerate(deltas) if d > umbral]
    derecha = [i for i, d in enumerate(deltas) if d < -umbral]
    if tipo == "izquierda":
        return bool(izquierda)
    if tipo == "derecha":
        return bool(derecha)
    if tipo == "ambos":  # primero izquierda, después derecha
        return bool(izquierda) and any(i > izquierda[0] for i in derecha)
    raise ValueError(f"reto desconocido: {tipo}")


def verificar_video(
    path: Path,
    tipo_reto: str,
    rostros: Rostros,
    enrolados: np.ndarray,
    umbral_rostro: float,
    umbral_giro: float = 0.35,
    invertir: bool = False,
) -> ResultadoVida:
    frames = leer_frames(path)
    if not frames:
        return ResultadoVida(False, "video_ilegible")

    validos: list[tuple[np.ndarray, np.ndarray]] = []  # (frame, cara)
    sin_rostro = varios = 0
    for frame in frames:
        caras = rostros.detectar(frame)
        if len(caras) > 1:
            varios += 1
        elif len(caras) == 0:
            sin_rostro += 1
        else:
            validos.append((frame, caras[0]))
    detalle: dict[str, Any] = {"frames": len(frames), "con_rostro": len(validos), "varios": varios}

    # i. exactamente un rostro
    if varios:
        return ResultadoVida(False, "varios_rostros", detalle=detalle)
    if len(validos) < MIN_FRAMES_CON_ROSTRO or len(validos) / len(frames) < MIN_PROPORCION_CON_ROSTRO:
        return ResultadoVida(False, "rostro_insuficiente", detalle=detalle)

    # ii. movimiento pedido
    signo = -1.0 if invertir else 1.0
    giros = [signo * giro(cara) for _, cara in validos]
    inicial = float(np.median(giros[:3]))
    deltas = [g - inicial for g in giros]
    detalle["giro_min"], detalle["giro_max"] = round(min(deltas), 3), round(max(deltas), 3)
    if not movimiento_cumplido(tipo_reto, deltas, umbral_giro):
        return ResultadoVida(False, "movimiento_no_detectado", detalle=detalle)

    # i (segunda parte). el mismo rostro: los frames casi frontales se parecen entre sí
    frontales = sorted(range(len(validos)), key=lambda i: abs(giros[i]))
    mas_frontal = frontales[0]
    ref = rostros.embedding(*validos[mas_frontal])
    candidatos = [i for i in range(len(validos)) if abs(deltas[i]) < umbral_giro / 2]
    for i in candidatos[:: max(1, len(candidatos) // 5)]:  # hasta ~5 frames repartidos en el video
        if similitud(ref, rostros.embedding(*validos[i])) < MISMO_ROSTRO_MINIMO:
            return ResultadoVida(False, "rostro_cambia_en_el_video", detalle=detalle)

    # iii. el frame más frontal contra el usuario enrolado
    sim = similitud_con_enrolado(ref, enrolados)
    if sim < umbral_rostro:
        return ResultadoVida(False, "rostro_no_coincide", similitud=sim, detalle=detalle)
    return ResultadoVida(True, "ok", similitud=sim, detalle=detalle)