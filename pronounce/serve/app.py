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

        if path == "/score" and not data.get("ref_wav"):
            self._send(
                *_json_error(
                    "ref_wav is required",
                    extra={"engine": "phoneme"},
                )
            )
            return

        if path not in ("/tts", "/phonemes", "/score"):
            self._send(*_json_error("not found", status=404))
            return

        try:
            from pronounce.serve import engines

            if path == "/tts":
                text = data.get("text")
                out = data.get("out")
                if not text or not out:
                    self._send(
                        *_json_error(
                            "text and out are required",
                            extra={"command": "tts"},
                        )
                    )
                    return
                result = engines.tts_to_file(
                    text=str(text),
                    out=str(out),
                    voice=str(data.get("voice") or "af_heart"),
                    lang=str(data.get("lang") or "en-us"),
                )
                self._send(200, result)
                return

            if path == "/phonemes":
                text = data.get("text")
                if not text:
                    self._send(
                        *_json_error(
                            "text is required",
                            extra={"command": "phonemes"},
                        )
                    )
                    return
                result = engines.dictionary_ipa(
                    text=str(text),
                    lang=str(data.get("lang") or "en-us"),
                )
                self._send(200, result)
                return

            # /score
            text = data.get("text")
            user_wav = data.get("user_wav")
            ref_wav = data.get("ref_wav")
            if not text or not user_wav or not ref_wav:
                self._send(
                    *_json_error(
                        "text, user_wav, and ref_wav are required",
                        extra={"engine": "phoneme"},
                    )
                )
                return
            result = engines.score_phoneme(
                text=str(text),
                user_wav=str(user_wav),
                ref_wav=str(ref_wav),
                lang=str(data.get("lang") or "en-us"),
                device=str(data.get("device") or "cpu"),
            )
            self._send(200, result)
        except (FileNotFoundError, ValueError) as e:
            extra: dict[str, Any] = (
                {"engine": "phoneme"}
                if path == "/score"
                else {"command": path.lstrip("/")}
            )
            self._send(*_json_error(str(e), status=400, extra=extra))
        except Exception as e:
            extra = (
                {"engine": "phoneme"}
                if path == "/score"
                else {"command": path.lstrip("/")}
            )
            self._send(*_json_error(str(e), status=500, extra=extra))


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
