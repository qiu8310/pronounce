from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _json_error(error: str, *, status: int = 400, extra: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    payload: dict[str, Any] = {"ok": False, "error": error}
    if extra:
        payload.update(extra)
    return status, payload


class RepeatHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        super().log_message(fmt, *args)

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any] | None:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0 or length > 1_000_000:
            return None
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] == "/health":
            self._send(200, {"ok": True, "engine": "phoneme", "tts": "kokoro"})
            return
        self._send(*_json_error("not found", status=404))

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        data = self._read_json()
        if data is None:
            self._send(*_json_error("invalid json"))
            return
        if path == "/score":
            if not data.get("ref_wav"):
                self._send(
                    *_json_error(
                        "ref_wav is required",
                        extra={"engine": "phoneme"},
                    )
                )
                return
            self._send(*_json_error("not implemented", extra={"engine": "phoneme"}))
            return
        if path in ("/tts", "/phonemes"):
            self._send(*_json_error("not implemented", extra={"command": path.lstrip("/")}))
            return
        self._send(*_json_error("not found", status=404))


def make_server(host: str, port: int, load: bool = True) -> ThreadingHTTPServer:
    if host not in ALLOWED_HOSTS:
        raise ValueError(f"serve must bind loopback, got {host!r}")
    if load:
        from pronounce.serve.engines import warmup

        warmup()
    return ThreadingHTTPServer((host, port), RepeatHandler)


def serve(host: str = "127.0.0.1", port: int = 8787, load: bool = True) -> None:
    httpd = make_server(host, port, load=load)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
