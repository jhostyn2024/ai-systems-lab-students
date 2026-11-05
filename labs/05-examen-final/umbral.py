"""Tabla FAR / FRR para justificar FACE_MATCH_THRESHOLD con datos (R2).

Fotos (esta carpeta NO se sube al repositorio, está en .gitignore):
    pruebas_umbral/
        integrante_1/   fotos de una persona
        integrante_2/   fotos de otra persona
        (opcional: más carpetas con fotos de un dataset público como LFW)

  Pares GENUINOS:   dos fotos de la MISMA persona  → deberían ACEPTARSE
  Pares IMPOSTORES: dos fotos de personas DISTINTAS → deberían RECHAZARSE

  FAR (False Accept Rate)  = impostores con similitud ≥ umbral / total de impostores
  FRR (False Reject Rate)  = genuinos con similitud < umbral  / total de genuinos

Uso:
    uv run python labs/05-examen-final/umbral.py
    uv run python labs/05-examen-final/umbral.py 0.30 0.40 0.50     # umbrales a evaluar
"""

import itertools
import sys
from pathlib import Path

from rostro import LAB_DIR, Rostros, RostroError, similitud

CARPETA = LAB_DIR / "pruebas_umbral"
UMBRALES = [0.20, 0.30, 0.363, 0.40, 0.45, 0.50, 0.60]
EXTENSIONES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    umbrales = [float(a) for a in sys.argv[1:]] or UMBRALES
    rostros = Rostros()

    vectores = []  # (persona, archivo, embedding)
    for persona in sorted(p for p in CARPETA.iterdir() if p.is_dir()):
        for foto in sorted(f for f in persona.iterdir() if f.suffix.lower() in EXTENSIONES):
            try:
                vectores.append((persona.name, foto.name, rostros.embedding_de_archivo(foto)))
            except (RostroError, ValueError) as exc:
                print(f"  (se omite {persona.name}/{foto.name}: {exc})")

    genuinos, impostores = [], []
    for (p1, f1, v1), (p2, f2, v2) in itertools.combinations(vectores, 2):
        (genuinos if p1 == p2 else impostores).append((similitud(v1, v2), f"{p1}/{f1} vs {p2}/{f2}"))

    personas = len({p for p, _, _ in vectores})
    print(f"\n{len(vectores)} fotos de {personas} personas → {len(genuinos)} pares genuinos y {len(impostores)} impostores")
    if len(genuinos) < 10 or len(impostores) < 20:
        print("¡Atención! El PDF pide al menos 10 pares genuinos y 20 impostores.")
    if not genuinos or not impostores:
        return

    sg = [s for s, _ in genuinos]
    si = [s for s, _ in impostores]
    print(f"\nSimilitud GENUINOS:   mín={min(sg):.3f}  promedio={sum(sg) / len(sg):.3f}  máx={max(sg):.3f}")
    print(f"Similitud IMPOSTORES: mín={min(si):.3f}  promedio={sum(si) / len(si):.3f}  máx={max(si):.3f}")
    peor_genuino = min(genuinos)
    peor_impostor = max(impostores)
    print(f"  genuino más bajo:   {peor_genuino[0]:.3f}  ({peor_genuino[1]})")
    print(f"  impostor más alto:  {peor_impostor[0]:.3f}  ({peor_impostor[1]})")

    print("\n| Umbral | FAR | FRR | Impostores aceptados | Genuinos rechazados |")
    print("|---|---|---|---|---|")
    for u in umbrales:
        fa = sum(s >= u for s in si)
        fr = sum(s < u for s in sg)
        print(f"| {u:.3f} | {fa / len(si):.1%} | {fr / len(sg):.1%} | {fa}/{len(si)} | {fr}/{len(sg)} |")


if __name__ == "__main__":
    main()