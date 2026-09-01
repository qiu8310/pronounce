import io
import json
import threading
import unittest
from http.client import HTTPConnection

from pronounce.cli import main
from pronounce.serve.app import make_server


class TestServeHttp(unittest.TestCase):
    def setUp(self):
        self.httpd = make_server("127.0.0.1", 0, load=False)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def _request(self, method, path, body=None):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=2)
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json"} if payload else {}
        conn.request(method, path, body=payload, headers=headers)
        res = conn.getresponse()
        raw = res.read().decode("utf-8")
        conn.close()
        return res.status, json.loads(raw)

    def test_health(self):
        status, data = self._request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(
            data, {"ok": True, "engine": "phoneme", "tts": "kokoro"}
        )

    def test_score_requires_ref_wav(self):
        status, data = self._request(
            "POST",
            "/score",
            {"text": "hi", "user_wav": "/tmp/u.wav"},
        )
        self.assertEqual(status, 400)
        self.assertFalse(data["ok"])
        self.assertEqual(data["engine"], "phoneme")
        self.assertIn("ref_wav", data["error"])

    def test_unknown_path(self):
        status, data = self._request("GET", "/nope")
        self.assertEqual(status, 404)
        self.assertFalse(data["ok"])

    def test_cli_rejects_non_loopback(self):
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["serve", "--host", "0.0.0.0", "--no-load"])
        self.assertEqual(code, 1)
        data = json.loads(buf.getvalue())
        self.assertFalse(data["ok"])
        self.assertEqual(data["command"], "serve")


class TestServeBind(unittest.TestCase):
    def test_rejects_non_loopback_host(self):
        from pronounce.serve.app import make_server

        with self.assertRaises(ValueError):
            make_server("0.0.0.0", 8787, load=False)
