"""Sesiones, retos de prueba de vida, intentos fallidos y bloqueo (AUTENTICACIÓN).

Lógica pura en memoria: no conoce Telegram ni OpenCV, así que se prueba sola.

Flujo:
    /login → iniciar_login()   → crea un Reto aleatorio con su hora de emisión
    video-nota → tomar_reto()  → lo entrega UNA sola vez (lo borra) y valida que no haya vencido
    verificación OK → crear_sesion()   (rol, nombre y código salen de usuarios.json)
    verificación falla → registrar_fallo() → al llegar a LOGIN_MAX_ATTEMPTS, bloqueo temporal
    cada mensaje → sesion_de() → si la sesión venció, se borra y se vuelve a rol "publico"

Uso (sin Telegram):
    uv run python labs/05-examen-final/sesiones.py
"""

import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from acceso import SESION_PUBLICA, Sesion, Usuario, usuario_por_telegram
from auditoria import Auditoria
from config import IdentitySettings

# Retos de prueba de vida. Todos se verifican con los landmarks de YuNet (vida.py).
RETOS = {
    "izquierda": "Empieza mirando al frente, gira BIEN la cabeza hacia TU IZQUIERDA y vuelve al frente.",
    "derecha": "Empieza mirando al frente, gira BIEN la cabeza hacia TU DERECHA y vuelve al frente.",
    "ambos": "Empieza mirando al frente, gira BIEN la cabeza primero hacia TU IZQUIERDA y después hacia TU DERECHA.",
}


@dataclass(frozen=True)
class Reto:
    id: str  # identificador aleatorio (útil en la auditoría)
    tipo: str  # clave de RETOS
    emitido: datetime
    usuario_id: str

    @property
    def instruccion(self) -> str:
        return RETOS[self.tipo]


class LoginError(Exception):
    """Motivo por el que no se puede iniciar o completar el login (se audita)."""

    def __init__(self, motivo: str, mensaje: str):
        super().__init__(mensaje)
        self.motivo = motivo
        self.mensaje = mensaje


class GestorSesiones:
    def __init__(
        self,
        settings: IdentitySettings,
        auditoria: Auditoria,
        buscar_usuario: Callable[[int], Usuario | None] = usuario_por_telegram,
        reloj: Callable[[], datetime] = datetime.now,
    ):
        self.settings = settings
        self.auditoria = auditoria
        self.buscar_usuario = buscar_usuario
        self.ahora = reloj
        self._sesiones: dict[int, Sesion] = {}
        self._retos: dict[int, Reto] = {}
        self._fallos: dict[int, int] = {}
        self._bloqueado_hasta: dict[int, datetime] = {}
        self._expiradas: set[int] = set()  # para avisar "tu sesión expiró" una vez

    # --- Sesión ---------------------------------------------------------------------------

    def sesion_de(self, telegram_user_id: int) -> Sesion:
        sesion = self._sesiones.get(telegram_user_id)
        if sesion is None:
            return SESION_PUBLICA
        if sesion.expira and self.ahora() >= sesion.expira:
            del self._sesiones[telegram_user_id]
            self._expiradas.add(telegram_user_id)
            self.auditoria.registrar(telegram_user_id, "sesion_expirada", usuario=sesion.usuario.id)
            return SESION_PUBLICA
        return sesion

    def acaba_de_expirar(self, telegram_user_id: int) -> bool:
        """True una sola vez después de que la sesión expiró."""
        if telegram_user_id in self._expiradas:
            self._expiradas.discard(telegram_user_id)
            return True
        return False

    def crear_sesion(self, telegram_user_id: int, usuario: Usuario, similitud: float) -> Sesion:
        sesion = Sesion(
            rol=usuario.rol,
            usuario=usuario,
            expira=self.ahora() + timedelta(minutes=self.settings.session_minutes),
        )
        self._sesiones[telegram_user_id] = sesion
        self._fallos.pop(telegram_user_id, None)
        self.auditoria.registrar(
            telegram_user_id, "login_aceptado", usuario=usuario.id, rol=usuario.rol, similitud=similitud
        )
        return sesion

    def logout(self, telegram_user_id: int) -> bool:
        sesion = self._sesiones.pop(telegram_user_id, None)
        if sesion:
            self.auditoria.registrar(telegram_user_id, "logout", usuario=sesion.usuario.id)
        return sesion is not None

    # --- Login con reto -------------------------------------------------------------------

    def bloqueado(self, telegram_user_id: int) -> datetime | None:
        hasta = self._bloqueado_hasta.get(telegram_user_id)
        if hasta and self.ahora() < hasta:
            return hasta
        if hasta:  # el bloqueo terminó: se reinicia el contador
            del self._bloqueado_hasta[telegram_user_id]
            self._fallos.pop(telegram_user_id, None)
        return None

    def iniciar_login(self, telegram_user_id: int) -> Reto:
        if hasta := self.bloqueado(telegram_user_id):
            self.auditoria.registrar(telegram_user_id, "login_rechazado", motivo="cuenta_bloqueada")
            raise LoginError("cuenta_bloqueada", f"Demasiados intentos fallidos. Intenta después de las {hasta:%H:%M}.")
        usuario = self.buscar_usuario(telegram_user_id)
        if usuario is None:
            self.auditoria.registrar(telegram_user_id, "login_rechazado", motivo="cuenta_no_enrolada")
            raise LoginError(
                "cuenta_no_enrolada",
                "Esta cuenta de Telegram no está registrada. El enrolamiento se hace localmente con enroll.py.",
            )
        reto = Reto(
            id=secrets.token_hex(4),
            tipo=secrets.choice(list(RETOS)),  # aleatorio: no se puede grabar el video antes
            emitido=self.ahora(),
            usuario_id=usuario.id,
        )
        self._retos[telegram_user_id] = reto  # un nuevo /login reemplaza el reto anterior
        self.auditoria.registrar(telegram_user_id, "login_iniciado", reto=reto.tipo, reto_id=reto.id)
        return reto

    def tiene_reto(self, telegram_user_id: int) -> bool:
        return telegram_user_id in self._retos

    def tomar_reto(self, telegram_user_id: int) -> Reto:
        """Entrega el reto pendiente y lo BORRA: un reto sirve una sola vez."""
        reto = self._retos.pop(telegram_user_id, None)
        if reto is None:
            raise LoginError("sin_reto", "No hay un reto activo (o ya se usó). Escribe /login para recibir uno nuevo.")
        edad = (self.ahora() - reto.emitido).total_seconds()
        if edad > self.settings.liveness_timeout_seconds:
            raise LoginError("reto_vencido", "El reto venció.")
        return reto

    def registrar_fallo(self, telegram_user_id: int, motivo: str, similitud: float | None = None, **detalle) -> bool:
        """Cuenta un intento fallido. Devuelve True si la cuenta quedó bloqueada."""
        self._fallos[telegram_user_id] = self._fallos.get(telegram_user_id, 0) + 1
        intentos = self._fallos[telegram_user_id]
        self.auditoria.registrar(
            telegram_user_id, "login_rechazado", motivo=motivo, similitud=similitud, intento=intentos, **detalle
        )
        if intentos >= self.settings.login_max_attempts:
            hasta = self.ahora() + timedelta(minutes=self.settings.lock_minutes)
            self._bloqueado_hasta[telegram_user_id] = hasta
            self.auditoria.registrar(telegram_user_id, "login_bloqueado", hasta=hasta.isoformat(timespec="seconds"))
            return True
        return False


if __name__ == "__main__":
    from auditoria import AuditoriaEnMemoria
    from config import load_identity_settings

    reloj = [datetime(2026, 11, 20, 10, 0, 0)]
    usuario = Usuario("integrante_1", "Integrante Uno", "estudiante", "2026999", 111)
    auditoria = AuditoriaEnMemoria()
    gestor = GestorSesiones(load_identity_settings(), auditoria, lambda tid: usuario if tid == 111 else None, lambda: reloj[0])

    reto = gestor.iniciar_login(111)
    print("reto:", reto.tipo, "→", reto.instruccion)
    gestor.tomar_reto(111)
    try:
        gestor.tomar_reto(111)
    except LoginError as exc:
        print("segundo uso del reto →", exc.motivo)
    gestor.iniciar_login(111)
    reloj[0] += timedelta(seconds=gestor.settings.liveness_timeout_seconds + 1)
    try:
        gestor.tomar_reto(111)
    except LoginError as exc:
        print("reto tardío →", exc.motivo)
    for _ in range(gestor.settings.login_max_attempts):
        bloqueada = gestor.registrar_fallo(111, "rostro_no_coincide", 0.12)
    print("bloqueada tras fallos:", bloqueada)
    try:
        gestor.iniciar_login(111)
    except LoginError as exc:
        print("login durante bloqueo →", exc.motivo)
    reloj[0] += timedelta(minutes=gestor.settings.lock_minutes + 1)
    gestor.crear_sesion(111, usuario, 0.71)
    print("sesión:", gestor.sesion_de(111).rol)
    reloj[0] += timedelta(minutes=gestor.settings.session_minutes + 1)
    print("tras expirar:", gestor.sesion_de(111).rol)
    print("\nauditoría:")
    for entrada in auditoria.entradas:
        print(" ", entrada)