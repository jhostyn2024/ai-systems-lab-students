"""Enrolamiento LOCAL de un usuario (R2). NO se puede hacer desde Telegram.

Recibe el id del usuario (de usuarios.json), su telegram_user_id y de 3 a 5 fotos.
  - Guarda SOLO los embeddings en rostros/<id>.npy (las fotos no se copian ni se guardan).
  - Escribe el telegram_user_id en usuarios.json: esa cuenta queda asociada a ese rostro.

¿Por qué local y no desde el chat? Si cualquiera pudiera enrolarse por Telegram, alguien con
el celular (o la cuenta) de otra persona registraría SU PROPIO rostro en esa cuenta y entraría
con el rol de la víctima. El enrolamiento es el momento de mayor confianza: lo hace, en persona,
alguien que verifica la identidad (como cuando la universidad toma la foto del carné).

Uso:
    uv run python labs/05-examen-final/enroll.py integrante_1 123456789 foto1.jpg foto2.jpg foto3.jpg
    uv run python labs/05-examen-final/enroll.py --borrar integrante_1      # borra sus embeddings
"""

import json
import sys
from pathlib import Path

import numpy as np

from acceso import USUARIOS_FILE, usuario_por_id
from rostro import Rostros, RostroError, guardar_enrolado, ruta_enrolado, similitud


def guardar_telegram_id(user_id: str, telegram_user_id: int) -> None:
    data = json.loads(USUARIOS_FILE.read_text(encoding="utf-8"))
    for u in data["usuarios"]:
        if u["id"] != user_id and u.get("telegram_user_id") == telegram_user_id:
            sys.exit(f"La cuenta de Telegram {telegram_user_id} ya pertenece a {u['id']}: una cuenta = un usuario.")
    for u in data["usuarios"]:
        if u["id"] == user_id:
            u["telegram_user_id"] = telegram_user_id
    USUARIOS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def enrolar(user_id: str, telegram_user_id: int, fotos: list[Path]) -> None:
    usuario = usuario_por_id(user_id)
    if usuario is None:
        sys.exit(f"No existe el usuario {user_id!r} en data/usuarios.json")
    if not 3 <= len(fotos) <= 5:
        sys.exit("Se necesitan entre 3 y 5 fotos.")

    rostros = Rostros()
    vectores = []
    for foto in fotos:
        try:
            vectores.append(rostros.embedding_de_archivo(foto))
            print(f"  ✓ {foto.name}")
        except (RostroError, ValueError) as exc:
            sys.exit(f"  ✗ {foto.name}: {exc} (cada foto debe tener exactamente un rostro)")

    sims = [similitud(a, b) for i, a in enumerate(vectores) for b in vectores[i + 1 :]]
    print(f"Similitud entre las fotos enroladas: mín={min(sims):.3f}  máx={max(sims):.3f}")
    matriz = np.stack(vectores)
    guardar_telegram_id(user_id, telegram_user_id)
    path = guardar_enrolado(user_id, matriz)
    print(f"\nEnrolado: {usuario.nombre} (rol {usuario.rol}) con telegram_user_id={telegram_user_id}")
    print(f"Embeddings guardados en {path} → forma {matriz.shape} (solo números; las fotos NO se guardaron)")


def borrar(user_id: str) -> None:
    path = ruta_enrolado(user_id)
    if path.exists():
        path.unlink()
        print(f"Borrado {path}")
    else:
        print(f"No había embeddings para {user_id}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == "--borrar":
        borrar(args[1])
    elif len(args) >= 3:
        enrolar(args[0], int(args[1]), [Path(a) for a in args[2:]])
    else:
        print(__doc__)