"""Serve the Lambda handler locally (offline: no AWS calls) for UI development.

    python scripts/serve_local.py [--port 8080]

Requests are translated into Lambda Function URL events, so this exercises the
exact code that runs in AWS. Without EVIDENCE_BUCKET/ALERT_TOPIC_ARN set, actions
are recorded locally and the fixed safety policy makes the decisions.
"""
from __future__ import annotations

import argparse
import base64
import pathlib
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import lambda_function  # noqa: E402


class Handler(BaseHTTPRequestHandler):
    def _dispatch(self, method: str) -> None:
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else b""
        event = {"rawPath": self.path.split("?", 1)[0], "requestContext": {"http": {"method": method}},
                 "body": base64.b64encode(body).decode() if body else None, "isBase64Encoded": bool(body)}
        resp = lambda_function.handler(event)
        payload = resp["body"].encode("utf-8")
        self.send_response(resp["statusCode"])
        for k, v in resp["headers"].items():
            self.send_header(k, v)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    print(f"EdgeSentry local: http://localhost:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
