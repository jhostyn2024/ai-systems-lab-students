"""Adaptador de Telegram (R1): SOLO traduce entre Telegram y el núcleo. No tiene lógica de negocio.

  texto        → nucleo.responder(pregunta, historial, sesion, confirmar_con_botones)
  /login       → sesiones.iniciar_login() → reto aleatorio con tiempo límite
  video-nota   → vida.verificar_video() → sesiones.crear_sesion() o registrar_fallo()
  botones Sí/No → la ConfirmFn del Lab 04, implementada en el canal en vez de la consola

Quién puede ver qué lo deciden acceso.py y el orquestador; quién es quién, sesiones.py y vida.py.
Este archivo no decide nada de eso.

Usa long polling: el bot pregunta a Telegram si hay mensajes nuevos, así que funciona desde un
portátil sin URL pública ni HTTPS.

Uso:
    uv run python labs/05-examen-final/bot.py
"""

import asyncio
import logging
import tempfile
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from acceso import TOOLS_POR_ROL, TOOLS_PUBLICAS, usuario_por_id
from auditoria import Auditoria
from config import load_identity_settings
from llm_client import Message
from nucleo import Nucleo, crear_nucleo
from prompts import AUTORES
from rostro import Rostros, cargar_enrolado
from sesiones import GestorSesiones, LoginError
from tools import Tool
from vida import verificar_video

log = logging.getLogger("bot")

MAX_HISTORIAL = 12  # mensajes (usuario + asistente) que se guardan por chat
MAX_TEXTO = 4000  # Telegram corta en 4096 caracteres
TIEMPO_CONFIRMACION = 60  # segundos para responder Sí/No
RESTRINGIDAS = set().union(*TOOLS_POR_ROL.values()) - TOOLS_PUBLICAS  # su ejecución se audita
RECHAZO_LOGIN = "❌ No se pudo verificar tu identidad. Escribe /login para intentarlo de nuevo."

AYUDA = (
    "Comandos:\n"
    "/start – presentación\n"
    "/login – iniciar sesión con reconocimiento facial y prueba de vida\n"
    "/logout – cerrar sesión\n"
    "/quiensoy – tu telegram_user_id y el estado de tu sesión\n"
    "/ayuda – este mensaje\n\n"
    "Sin sesión puedes preguntar información pública del curso. Con sesión, según tu rol, "
    "puedes consultar información restringida y tus datos académicos."
)


class Bot:
    """Estado del canal: historial por chat y confirmaciones pendientes. Nada de reglas de acceso."""

    def __init__(self, nucleo: Nucleo, sesiones: GestorSesiones, auditoria: Auditoria, rostros: Rostros, settings):
        self.nucleo = nucleo
        self.sesiones = sesiones
        self.auditoria = auditoria
        self.rostros = rostros
        self.settings = settings
        self.historiales: dict[int, list[Message]] = {}
        self.pendientes: dict[str, tuple[int, asyncio.Future]] = {}  # id de confirmación → (usuario, futuro)
        self._contador = 0

    # --- Comandos ------------------------------------------------------------------------

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        sesion = self.sesiones.sesion_de(update.effective_user.id)
        saludo = f"Hola {sesion.nombre}. " if sesion.nombre else "Hola. "
        await update.message.reply_text(
            f"{saludo}Soy el Asistente del Curso de IA desarrollado por {AUTORES}.\n\n"
            "Puedo responder preguntas del curso (fechas, evaluación, políticas…). Para información "
            "restringida o tus datos académicos, inicia sesión con /login.\n\n" + AYUDA
        )

    async def ayuda(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(AYUDA)

    async def quiensoy(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        tid = update.effective_user.id
        sesion = self.sesiones.sesion_de(tid)
        if sesion.usuario:
            estado = f"Sesión activa: {sesion.nombre} · rol {sesion.rol} · expira a las {sesion.expira:%H:%M:%S}"
        else:
            estado = "Sin sesión (rol público)."
        await update.message.reply_text(f"Tu telegram_user_id: {tid}\n{estado}")

    async def login(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        tid = update.effective_user.id
        try:
            reto = self.sesiones.iniciar_login(tid)
        except LoginError as exc:
            await update.message.reply_text(f"❌ {exc.mensaje}")
            return
        await update.message.reply_text(
            "🔐 Inicio de sesión con prueba de vida.\n\n"
            f"Reto: {reto.instruccion}\n\n"
            f"Responde con una VIDEO-NOTA (el video circular: mantén presionado el ícono de la cámara "
            f"junto al campo de texto) en los próximos {self.settings.liveness_timeout_seconds} segundos. "
            "Una foto no sirve."
        )

    async def logout(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        tid = update.effective_user.id
        cerrada = self.sesiones.logout(tid)
        self.historiales.pop(update.effective_chat.id, None)  # el historial puede tener datos restringidos
        await update.message.reply_text("👋 Sesión cerrada." if cerrada else "No tenías una sesión activa.")

    # --- Login: video-nota ---------------------------------------------------------------

    async def video_nota(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        msg = update.message
        tid = update.effective_user.id
        if not self.sesiones.tiene_reto(tid):
            self.auditoria.registrar(tid, "login_rechazado", motivo="sin_reto")
            await msg.reply_text("Primero escribe /login para recibir un reto.")
            return
        try:
            reto = self.sesiones.tomar_reto(tid)  # un reto sirve una sola vez: aquí se consume
        except LoginError as exc:
            await self._fallo(update, exc.motivo)
            return
        if msg.forward_origin is not None:  # video reenviado (grabado antes o de otra persona)
            await self._fallo(update, "mensaje_reenviado")
            return

        usuario = usuario_por_id(reto.usuario_id)
        enrolados = cargar_enrolado(reto.usuario_id) if usuario else None
        if usuario is None or enrolados is None:
            await self._fallo(update, "usuario_sin_embeddings")
            return

        await msg.reply_text("⏳ Verificando…")
        with tempfile.TemporaryDirectory() as carpeta:  # el video se borra al terminar
            path = Path(carpeta) / "video_nota.mp4"
            archivo = await context.bot.get_file(msg.video_note.file_id)
            await archivo.download_to_drive(path)
            resultado = await asyncio.to_thread(
                verificar_video,
                path,
                reto.tipo,
                self.rostros,
                enrolados,
                self.settings.face_threshold,
                self.settings.yaw_threshold,
                self.settings.liveness_invert,
            )

        if not resultado.aceptado:
            await self._fallo(update, resultado.motivo, resultado.similitud, reto=reto.tipo, **resultado.detalle)
            return
        sesion = self.sesiones.crear_sesion(tid, usuario, resultado.similitud)
        self.historiales.pop(update.effective_chat.id, None)  # nueva identidad, historial nuevo
        await msg.reply_text(
            f"✅ Hola, {sesion.nombre} ({sesion.rol}). Sesión iniciada por "
            f"{self.settings.session_minutes} minutos. ¿En qué te ayudo?"
        )

    async def _fallo(self, update: Update, motivo: str, similitud: float | None = None, **detalle) -> None:
        """Cuenta el intento fallido. Al usuario NO se le dice qué verificación falló (R3)."""
        bloqueada = self.sesiones.registrar_fallo(update.effective_user.id, motivo, similitud, **detalle)
        texto = RECHAZO_LOGIN
        if bloqueada:
            texto = (
                "❌ No se pudo verificar tu identidad. Demasiados intentos fallidos: "
                f"intenta de nuevo en {self.settings.lock_minutes} minutos."
            )
        await update.message.reply_text(texto)

    async def foto_o_video(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if self.sesiones.tiene_reto(update.effective_user.id):
            await update.message.reply_text(
                "Una foto o un video normal no sirven para iniciar sesión: responde con una VIDEO-NOTA (video circular)."
            )
        else:
            await update.message.reply_text("Solo proceso texto. Para iniciar sesión usa /login.")

    # --- Preguntas en texto --------------------------------------------------------------

    async def texto(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        tid = update.effective_user.id
        chat_id = update.effective_chat.id
        sesion = self.sesiones.sesion_de(tid)  # sale de sesiones.py, nunca del texto del chat
        if self.sesiones.acaba_de_expirar(tid):
            self.historiales.pop(chat_id, None)
            await update.message.reply_text("⌛ Tu sesión expiró. Para información restringida vuelve a usar /login.")

        pregunta = update.message.text
        historial = self.historiales.setdefault(chat_id, [])
        await context.bot.send_chat_action(chat_id, "typing")

        async def confirmar(tool: Tool, argumentos: dict[str, Any]) -> bool:
            return await self.confirmar_con_botones(update, tool, argumentos)

        respuesta = await self.nucleo.responder(pregunta, historial, sesion, confirmar)
        usuario = sesion.usuario.id if sesion.usuario else None
        for tool in respuesta.tools_denegadas:
            self.auditoria.registrar(tid, "acceso_denegado", tool=tool, rol=sesion.rol, usuario=usuario)
        for tool in respuesta.tools_ejecutadas:
            if tool in RESTRINGIDAS:
                self.auditoria.registrar(tid, "tool_restringida_ejecutada", tool=tool, rol=sesion.rol, usuario=usuario)
        if respuesta.error:
            self.auditoria.registrar(tid, "error", detalle=respuesta.error[:300])

        await update.message.reply_text(respuesta.texto[:MAX_TEXTO])
        if not respuesta.error:
            historial += [{"role": "user", "content": pregunta}, {"role": "assistant", "content": respuesta.texto}]
            del historial[:-MAX_HISTORIAL]

    # --- Confirmación con botones (ConfirmFn del Lab 04) ---------------------------------

    async def confirmar_con_botones(self, update: Update, tool: Tool, argumentos: dict[str, Any]) -> bool:
        self._contador += 1
        clave = str(self._contador)
        futuro = asyncio.get_running_loop().create_future()
        self.pendientes[clave] = (update.effective_user.id, futuro)
        detalle = "\n".join(f"• {k}: {v}" for k, v in argumentos.items())
        teclado = InlineKeyboardMarkup(
            [[InlineKeyboardButton("✅ Sí", callback_data=f"si:{clave}"), InlineKeyboardButton("❌ No", callback_data=f"no:{clave}")]]
        )
        await update.message.reply_text(
            f"⚠️ El asistente quiere ejecutar una ACCIÓN:\n{tool.name}\n{detalle}\n\n¿La autorizas?",
            reply_markup=teclado,
        )
        try:
            autorizada = await asyncio.wait_for(futuro, TIEMPO_CONFIRMACION)
        except asyncio.TimeoutError:
            autorizada = False
        finally:
            self.pendientes.pop(clave, None)
        tid = update.effective_user.id
        self.auditoria.registrar(tid, "accion_confirmada" if autorizada else "accion_cancelada", tool=tool.name)
        return autorizada

    async def boton(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        decision, _, clave = (query.data or "").partition(":")
        pendiente = self.pendientes.get(clave)
        if pendiente is None:
            await query.answer("Esta confirmación ya no está activa.")
            return
        dueno, futuro = pendiente
        if query.from_user.id != dueno:  # solo quien pidió la acción puede confirmarla
            await query.answer("Solo quien pidió la acción puede confirmarla.", show_alert=True)
            return
        await query.answer()
        await query.edit_message_reply_markup(None)
        await query.message.reply_text("✅ Autorizada." if decision == "si" else "❌ Cancelada.")
        if not futuro.done():
            futuro.set_result(decision == "si")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Ningún error tumba el bot: queda en el log y el usuario recibe un mensaje claro (R1)."""
    log.error("Error procesando un mensaje", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text("⚠️ Ocurrió un error interno. Intenta de nuevo.")


async def main() -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO, force=True)
    print("Iniciando el asistente (LLM, RAG, servidor MCP y modelos de rostro)...")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # no imprimir cada consulta a Telegram
    settings = load_identity_settings(require_token=True)
    auditoria = Auditoria()

    async with AsyncExitStack() as stack:
        nucleo = await crear_nucleo(stack)  # LLM + RAG + servidor MCP
        bot = Bot(nucleo, GestorSesiones(settings, auditoria), auditoria, Rostros(), settings)

        # concurrent_updates: mientras una pregunta espera el botón Sí/No, el bot sigue recibiendo mensajes.
        app = Application.builder().token(settings.telegram_token).concurrent_updates(True).build()
        app.add_handler(CommandHandler("start", bot.start))
        app.add_handler(CommandHandler("ayuda", bot.ayuda))
        app.add_handler(CommandHandler("quiensoy", bot.quiensoy))
        app.add_handler(CommandHandler("login", bot.login))
        app.add_handler(CommandHandler("logout", bot.logout))
        app.add_handler(MessageHandler(filters.VIDEO_NOTE, bot.video_nota))
        app.add_handler(MessageHandler(filters.PHOTO | filters.VIDEO, bot.foto_o_video))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.texto))
        app.add_handler(CallbackQueryHandler(bot.boton))
        app.add_error_handler(error_handler)

        async with app:
            await app.start()
            await app.updater.start_polling()
            log.info("Bot en marcha (long polling). Ctrl+C para detenerlo.")
            try:
                await asyncio.Event().wait()  # hasta Ctrl+C
            finally:
                await app.updater.stop()
                await app.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot detenido.")
