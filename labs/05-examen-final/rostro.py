"""Reconocimiento facial con OpenCV (modelos ONNX, CPU). Es el "faces.py" del diagrama del PDF.

  - Detección: YuNet (cv2.FaceDetectorYN) → caja del rostro + 5 landmarks por cara:
        [x, y, w, h, ojo_der_x, ojo_der_y, ojo_izq_x, ojo_izq_y, nariz_x, nariz_y,
         boca_der_x, boca_der_y, boca_izq_x, boca_izq_y, confianza]
  - Reconocimiento: SFace (cv2.FaceRecognizerSF) → alignCrop + feature → embedding de 128 números.

Es la misma idea del RAG del Lab 02: texto → vector y rostro → vector, comparados con coseno.
  - "consulta": el rostro del login.
  - "índice": los 3-5 embeddings enrolados del usuario DUEÑO de la cuenta de Telegram (1:1, no 1:N).
  - umbral: la similitud mínima para aceptar (FACE_MATCH_THRESHOLD, justificado con FAR/FRR).

Uso (prueba con una o dos imágenes):
    uv run python labs/05-examen-final/rostro.py foto1.jpg foto2.jpg
"""

import sys
from pathlib import Path

import cv2
import numpy as np

LAB_DIR = Path(__file__).resolve().parent
MODELS_DIR = LAB_DIR / "modelos"  # no se versiona (.gitignore)
FACES_DIR = LAB_DIR / "rostros"  # embeddings de los enrolados; no se versiona (.gitignore)
YUNET = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
CONFIANZA_MINIMA = 0.8  # score mínimo de YuNet para aceptar una detección
LADO_MAXIMO = 1280  # las fotos del celular se reducen a este tamaño para detectar más rápido


class RostroError(Exception):
    """La imagen no sirve: no hay rostro o hay más de uno."""

    def __init__(self, motivo: str):
        super().__init__(motivo)
        self.motivo = motivo  # "sin_rostro" | "varios_rostros"


def reducir(imagen: np.ndarray) -> np.ndarray:
    alto, ancho = imagen.shape[:2]
    escala = LADO_MAXIMO / max(alto, ancho)
    if escala >= 1:
        return imagen
    return cv2.resize(imagen, (int(ancho * escala), int(alto * escala)))


class Rostros:
    def __init__(self):
        for path in (YUNET, SFACE):
            if not path.exists():
                raise FileNotFoundError(f"Falta el modelo {path}. Descárgalo de opencv/opencv_zoo (Paso 1.5).")
        self.detector = cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), CONFIANZA_MINIMA, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(SFACE), "")

    def detectar(self, imagen: np.ndarray) -> np.ndarray:
        """Todas las caras de la imagen: matriz (n, 15). n = 0 si no hay rostros."""
        alto, ancho = imagen.shape[:2]
        self.detector.setInputSize((ancho, alto))
        _, caras = self.detector.detect(imagen)
        return np.empty((0, 15), dtype=np.float32) if caras is None else caras

    def unico(self, imagen: np.ndarray) -> np.ndarray:
        """La única cara de la imagen. Rechaza si no hay rostro o si hay más de uno (R2)."""
        caras = self.detectar(imagen)
        if len(caras) == 0:
            raise RostroError("sin_rostro")
        if len(caras) > 1:
            raise RostroError("varios_rostros")
        return caras[0]

    def embedding(self, imagen: np.ndarray, cara: np.ndarray) -> np.ndarray:
        """Alinea el rostro con los landmarks y devuelve su vector normalizado (128,)."""
        alineado = self.recognizer.alignCrop(imagen, cara)
        vector = self.recognizer.feature(alineado).flatten().astype(np.float32)
        return vector / max(float(np.linalg.norm(vector)), 1e-12)

    def embedding_de_archivo(self, path: Path) -> np.ndarray:
        imagen = cv2.imread(str(path))
        if imagen is None:
            raise ValueError(f"No se pudo leer la imagen {path} (¿formato .heic?)")
        imagen = reducir(imagen)
        return self.embedding(imagen, self.unico(imagen))


def similitud(a: np.ndarray, b: np.ndarray) -> float:
    """Coseno entre dos embeddings normalizados: 1 = idénticos, 0 = nada que ver."""
    return float(np.dot(a, b))


def similitud_con_enrolado(vector: np.ndarray, enrolados: np.ndarray) -> float:
    """Verificación 1:1: la MEJOR similitud contra los embeddings del dueño de la cuenta."""
    return float(np.max(enrolados @ vector))


# --- Embeddings guardados en rostros/ (solo vectores, nunca fotos) ------------------------


def ruta_enrolado(user_id: str) -> Path:
    return FACES_DIR / f"{user_id}.npy"


def cargar_enrolado(user_id: str) -> np.ndarray | None:
    path = ruta_enrolado(user_id)
    return np.load(path) if path.exists() else None


def guardar_enrolado(user_id: str, embeddings: np.ndarray) -> Path:
    FACES_DIR.mkdir(parents=True, exist_ok=True)
    path = ruta_enrolado(user_id)
    np.save(path, embeddings.astype(np.float32))
    return path


if __name__ == "__main__":
    rostros = Rostros()
    vectores = []
    for archivo in sys.argv[1:]:
        imagen = reducir(cv2.imread(archivo))
        caras = rostros.detectar(imagen)
        print(f"{archivo}: {len(caras)} rostro(s)")
        if len(caras) == 1:
            vectores.append(rostros.embedding(imagen, caras[0]))
    if len(vectores) == 2:
        print(f"similitud coseno = {similitud(*vectores):.3f}")