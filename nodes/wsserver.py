"""A tiny dependency-free WebSocket broadcast server (RFC 6455, text frames).

The replay stream publisher only ever pushes text frames to subscribers, so this
implements just enough of the protocol to accept clients and fan out unmasked
server->client text frames. Incoming client frames are drained and ignored except
for close; no third-party dependency is required, matching the rest of this
package's stdlib-only runtime.
"""
from __future__ import annotations

import base64
import hashlib
import socket
import struct
import threading

_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _accept_key(key: str) -> str:
    digest = hashlib.sha1((key + _GUID).encode("latin1")).digest()
    return base64.b64encode(digest).decode("ascii")


def _encode_text(payload: bytes) -> bytes:
    header = bytearray([0x81])  # FIN + text opcode, unmasked (server -> client)
    length = len(payload)
    if length < 126:
        header.append(length)
    elif length < 65536:
        header.append(126)
        header += struct.pack(">H", length)
    else:
        header.append(127)
        header += struct.pack(">Q", length)
    return bytes(header) + payload


class WsBroadcastServer:
    """Accept WebSocket clients on host:port and broadcast text to all of them."""

    def __init__(self, host: str, port: int, initial_text: str = ""):
        self.host = host or "127.0.0.1"
        self.port = int(port)
        self.initial_text = str(initial_text or "")
        self._srv: socket.socket | None = None
        self._clients: set[socket.socket] = set()
        self._lock = threading.RLock()
        self._accept_thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        # Reflect the actually-bound port so callers can pass 0 for an ephemeral one.
        self.port = srv.getsockname()[1]
        srv.listen(16)
        srv.settimeout(0.5)
        self._srv = srv
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True,
                                                name=f"blacknode-ws-accept-{self.port}")
        self._accept_thread.start()

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._srv.accept()  # type: ignore[union-attr]
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                self._handshake(conn)
            except Exception:  # noqa: BLE001 - a bad client must not stop the server
                try:
                    conn.close()
                except OSError:
                    pass
                continue
            with self._lock:
                self._clients.add(conn)
            if self.initial_text:
                try:
                    conn.sendall(_encode_text(self.initial_text.encode("utf-8")))
                except OSError:
                    with self._lock:
                        self._clients.discard(conn)
                    try:
                        conn.close()
                    except OSError:
                        pass

    def _handshake(self, conn: socket.socket) -> None:
        conn.settimeout(5.0)
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = conn.recv(1024)
            if not chunk:
                raise ConnectionError("client closed during handshake")
            data += chunk
            if len(data) > 65536:
                raise ValueError("handshake request too large")
        key = ""
        for line in data.decode("latin1").split("\r\n"):
            if line.lower().startswith("sec-websocket-key:"):
                key = line.split(":", 1)[1].strip()
        if not key:
            raise ValueError("missing Sec-WebSocket-Key header")
        response = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {_accept_key(key)}\r\n\r\n"
        )
        conn.sendall(response.encode("latin1"))
        conn.settimeout(None)

    def broadcast(self, text: str) -> int:
        frame = _encode_text(text.encode("utf-8"))
        with self._lock:
            clients = list(self._clients)
        dead: list[socket.socket] = []
        for conn in clients:
            try:
                conn.sendall(frame)
            except OSError:
                dead.append(conn)
        if dead:
            with self._lock:
                for conn in dead:
                    self._clients.discard(conn)
            for conn in dead:
                try:
                    conn.close()
                except OSError:
                    pass
        return len(clients) - len(dead)

    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    def stop(self) -> None:
        self._stop.set()
        if self._srv is not None:
            try:
                self._srv.close()
            except OSError:
                pass
        with self._lock:
            clients = list(self._clients)
            self._clients.clear()
        for conn in clients:
            try:
                conn.close()
            except OSError:
                pass
