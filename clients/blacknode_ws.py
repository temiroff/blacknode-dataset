"""Minimal dependency-free WebSocket client for Blacknode replay streams.

Only what a subscriber needs: connect, then receive JSON frames. It has no
third-party dependencies so it runs unchanged inside mayapy, an Isaac Lab
Python environment, or a ROS 2 node without installing anything.

    from blacknode_ws import connect
    stream = connect("ws://127.0.0.1:8765")
    while True:
        frame = stream.recv_json()
        if frame is None:
            break
        print(frame["frame_index"], frame["positions"])
"""
from __future__ import annotations

import base64
import json
import os
import socket
import ssl
import struct
from urllib.parse import urlparse


def connect(url: str, timeout: float = 10.0) -> "ReplayStream":
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    sock = socket.create_connection((host, port), timeout=timeout)
    if parsed.scheme == "wss":
        sock = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    request = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n"
    )
    sock.sendall(request.encode("latin1"))
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(1024)
        if not chunk:
            raise ConnectionError("server closed during handshake")
        data += chunk
    status_line = data.split(b"\r\n", 1)[0]
    if b" 101 " not in status_line:
        raise ConnectionError(f"server refused upgrade: {status_line.decode('latin1', 'replace')}")
    sock.settimeout(None)
    return ReplayStream(sock, data.split(b"\r\n\r\n", 1)[1])


class ReplayStream:
    def __init__(self, sock: socket.socket, buffered: bytes = b""):
        self._sock = sock
        self._buf = buffered

    def _read(self, count: int) -> bytes:
        while len(self._buf) < count:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ConnectionError("stream closed")
            self._buf += chunk
        out, self._buf = self._buf[:count], self._buf[count:]
        return out

    def recv(self) -> str | None:
        """Return the next text message, or None when the stream closes."""
        while True:
            b0, b1 = self._read(2)
            opcode = b0 & 0x0F
            masked = b1 & 0x80
            length = b1 & 0x7F
            if length == 126:
                length = struct.unpack(">H", self._read(2))[0]
            elif length == 127:
                length = struct.unpack(">Q", self._read(8))[0]
            mask = self._read(4) if masked else b""
            payload = self._read(length)
            if masked:
                payload = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
            if opcode == 0x8:  # close
                return None
            if opcode == 0x9:  # ping -> pong
                self._send(0xA, payload)
                continue
            if opcode == 0x1:  # text
                return payload.decode("utf-8")
            # 0x0 continuation / 0x2 binary / 0xA pong -> ignore

    def recv_json(self):
        text = self.recv()
        return None if text is None else json.loads(text)

    def _send(self, opcode: int, payload: bytes) -> None:
        header = bytearray([0x80 | opcode])
        length = len(payload)
        mask = os.urandom(4)
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.append(0x80 | 126)
            header += struct.pack(">H", length)
        else:
            header.append(0x80 | 127)
            header += struct.pack(">Q", length)
        header += mask
        masked = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
        self._sock.sendall(bytes(header) + masked)

    def close(self) -> None:
        try:
            self._send(0x8, b"")
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass
