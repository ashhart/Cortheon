"""Local JSON chat endpoint shared by the Python/Rust differential tests."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread


class Handler(BaseHTTPRequestHandler):
    def respond(self, value):
        data = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        assert self.path == "/v1/models"
        self.respond({"data": [self.server.model]})

    def do_POST(self):
        assert self.path == "/v1/chat/completions"
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert not body.pop("stream", False)
        self.server.requests.append(body)
        self.respond({"choices": [{"message": {"role": "assistant", "content": self.server.reply(body)},
                                   "finish_reason": "stop"}],
                      "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}})

    def log_message(self, *args):
        pass


@contextmanager
def model_fixture(model):
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        server.model = model
        server.requests = []
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
            thread.join()
