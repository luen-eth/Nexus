#!/usr/bin/env python3
"""Offline CI child fixture: accepts Nexus CLI flags, checks environment auth."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--model", default="")
args, _ = parser.parse_known_args()
key = os.environ.get("NEXUS_API_KEY", "")
state = {"loaded": bool(args.model), "path": args.model}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def response(self, code, data):
        payload = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def authorized(self):
        if self.path != "/health" and key and self.headers.get("Authorization") != "Bearer " + key:
            self.response(401, {"error": {"message": "Unauthorized"}})
            return False
        return True

    def do_GET(self):
        if not self.authorized():
            return
        if self.path == "/health":
            self.response(200, {"status": "ok" if state["loaded"] else "degraded", "model_loaded": state["loaded"], "model": "ci-model"})
        elif self.path == "/metrics":
            self.response(200, {"runtime": {"active_sequences": 0, "queued_requests": 0}, "engine": {"host_weight_payload_bytes": 1048576}})
        elif self.path == "/v1/model":
            self.response(200, {"loaded": state["loaded"], "path": state["path"]})
        elif self.path == "/v1/models":
            self.response(200, {"data": [{"id": "ci-model"}] if state["loaded"] else []})
        else:
            self.response(404, {})

    def do_POST(self):
        if not self.authorized():
            return
        data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        if self.path.endswith("/load"):
            state.update(loaded=True, path=data["model"])
        elif self.path.endswith("/reload"):
            state["loaded"] = bool(state["path"])
        elif self.path.endswith("/unload"):
            state["loaded"] = False
        else:
            self.response(404, {})
            return
        self.response(200, {"status": "ok"})


print("Fake Nexus ready", flush=True)
ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
