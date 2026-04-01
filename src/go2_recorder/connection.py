"""
WebRTC connection management for the Unitree Go2.

Handles AP/STA mode selection, connection with retry logic, and the asyncio
event loop that runs in a background thread.
"""

import asyncio
import logging
import sys
import time
import threading

from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)

# Suppress noisy library logs
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
logging.getLogger("root").setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)

GO2_AP_IP  = "192.168.12.1"
GO2_LAN_IP = "192.168.123.161"

MAX_RETRIES = 3
RETRY_DELAY = 5


def _patch_error_handler():
    """Monkey-patch the library's error handler to avoid corrupting our status line."""
    try:
        import unitree_webrtc_connect.msgs.error_handler as _eh
        _orig = _eh.handle_error

        def _safe(message):
            try:
                _orig(message)
            except Exception:
                data = message.get("data", "")
                sys.stdout.write(f"\n  [robot error] {data}\n")
                sys.stdout.flush()

        _eh.handle_error = _safe
    except Exception:
        pass


def resolve_ip(mode: str, ip: str | None) -> str:
    if mode == "ap":
        return GO2_AP_IP
    if mode == "sta":
        if not ip:
            raise ValueError("--mode sta requires --ip <GO2_IP>")
        return ip
    if mode == "lan":
        return GO2_LAN_IP
    return GO2_AP_IP


def _make_conn(mode: str, ip: str) -> UnitreeWebRTCConnection:
    if mode == "ap":
        return UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)
    return UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=ip)


async def _async_connect(conn: UnitreeWebRTCConnection):
    await asyncio.wait_for(conn.connect(), timeout=15)
    if not conn.isConnected:
        raise ConnectionError("WebRTC connected but data channel did not open")


class Go2Connection:
    """Manages a WebRTC connection to the Go2 with retry logic and a background event loop."""

    def __init__(self, mode: str, ip: str | None):
        _patch_error_handler()
        self.mode = mode
        self.ip = resolve_ip(mode, ip)
        self.conn: UnitreeWebRTCConnection | None = None
        self.loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._loop_thread: threading.Thread | None = None

    def connect(self) -> UnitreeWebRTCConnection:
        """Connect with retries. Returns the live connection. Starts the event loop thread."""
        for attempt in range(1, MAX_RETRIES + 1):
            self.conn = _make_conn(self.mode, self.ip)
            try:
                print(f"  Connecting to Go2 (attempt {attempt}/{MAX_RETRIES}) ...")
                self.loop.run_until_complete(_async_connect(self.conn))
                print("  WebRTC connected\n")
                break
            except (asyncio.TimeoutError, ConnectionError, Exception) as exc:
                if attempt < MAX_RETRIES:
                    print(f"  Connection attempt {attempt} failed: {exc}")
                    print(f"  Waiting {RETRY_DELAY}s before retry ...")
                    time.sleep(RETRY_DELAY)
                    self.loop.close()
                    self.loop = asyncio.new_event_loop()
                else:
                    raise ConnectionError(
                        f"WebRTC connection failed after {MAX_RETRIES} attempts: {exc}"
                    ) from exc

        self._loop_thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._loop_thread.start()
        return self.conn

    def run_coroutine(self, coro, timeout: float = 2.0):
        """Schedule a coroutine on the background loop and wait for the result."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout=timeout)

    def disconnect(self):
        if self.conn:
            try:
                asyncio.run_coroutine_threadsafe(
                    self.conn.disconnect(), self.loop
                ).result(timeout=5)
            except Exception:
                pass
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
