import http.server
import threading
import queue
import json
import os
from PyQt6 import QtCore

class QtCommunicator(QtCore.QObject):
    action_signal = QtCore.pyqtSignal(dict)
    
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.action_signal.connect(self.dispatch_action)
        
    def handle_action(self, data):
        # Called from HTTP thread to safely execute on the main thread via Qt Signal
        self.action_signal.emit(data)
        
    def dispatch_action(self, data):
        # Delegate all web action execution to the viewer's registered action handlers
        if hasattr(self.viewer, "handle_web_action"):
            self.viewer.handle_web_action(data)

class ThreadSafeHTTPServer(http.server.ThreadingHTTPServer):
    def __init__(self, server_address, RequestHandlerClass, viewer):
        super().__init__(server_address, RequestHandlerClass)
        self.viewer = viewer
        self.event_queues = []
        self.queues_lock = threading.Lock()
        self.static_routes = {}  # prefix -> local_dir (registered by dynamic backends)

    def handle_error(self, request, client_address):
        # Suppress traceback print for socket/connection abortions when browser tabs close
        import sys
        exc_type, exc_value, _ = sys.exc_info()
        if exc_type in (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            return
        super().handle_error(request, client_address)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # src/web_ui/

class WebServerHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silences console log spam
        pass

    def do_GET(self):
        if self.path == "/api/events":
            self.handle_sse()
            return

        # Strip query parameters (e.g. ?v=1)
        clean_path = self.path.split('?')[0]
        if clean_path == "/" or clean_path == "":
            clean_path = "/index.html"

        # Check dynamically registered static routes first (e.g. for agent files)
        for route_prefix, local_dir in self.server.static_routes.items():
            if clean_path.startswith(route_prefix):
                rel = clean_path[len(route_prefix):]
                normalized = os.path.normpath(rel)
                if normalized.startswith("..") or os.path.isabs(normalized):
                    self.send_error(403, "Forbidden")
                    return
                filepath = os.path.normpath(os.path.join(local_dir, normalized))
                if not filepath.startswith(os.path.normpath(local_dir)):
                    self.send_error(403, "Forbidden")
                    return
                if not os.path.isfile(filepath):
                    self.send_error(404, "File Not Found")
                    return
                ext = os.path.splitext(filepath)[1].lower()
                self.serve_file(filepath, {".json": "application/json", ".md": "text/plain"}.get(ext, "application/octet-stream"))
                return

        # Fallback to serving public files inside BASE_DIR (src/web_ui)
        safe_rel_path = clean_path.lstrip("/")
        normalized = os.path.normpath(safe_rel_path)
        if normalized.startswith("..") or os.path.isabs(normalized):
            self.send_error(403, "Forbidden")
            return

        filepath = os.path.normpath(os.path.join(BASE_DIR, normalized))
        if not filepath.startswith(os.path.normpath(BASE_DIR)):
            self.send_error(403, "Forbidden")
            return

        if not os.path.isfile(filepath):
            self.send_error(404, "File Not Found")
            return

        # Determine MIME type dynamically
        ext = os.path.splitext(filepath)[1].lower()
        mime_types = {
            ".html": "text/html",
            ".css": "text/css",
            ".js": "application/javascript",
            ".json": "application/json",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml",
            ".gif": "image/gif",
            ".ico": "image/x-icon"
        }
        content_type = mime_types.get(ext, "application/octet-stream")
        self.serve_file(filepath, content_type)

    def serve_file(self, filepath, content_type):
        if not os.path.exists(filepath):
            self.send_error(404, f"File {filepath} Not Found")
            return
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with open(filepath, "rb") as f:
            self.wfile.write(f.read())

    def handle_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        
        q = queue.Queue()
        with self.server.queues_lock:
            self.server.event_queues.append(q)
        
        try:
            # Send initial configuration
            initial_data = self.server.viewer.get_initial_web_state()
            self.wfile.write(f"data: {json.dumps({'type': 'init', 'data': initial_data})}\n\n".encode('utf-8'))
            self.wfile.flush()
            
            while True:
                try:
                    event = q.get(timeout=1.0)
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode('utf-8'))
                    self.wfile.flush()
                except queue.Empty:
                    # Send a keep-alive ping to prevent timeouts
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
        except Exception:
            pass
        finally:
            with self.server.queues_lock:
                if q in self.server.event_queues:
                    self.server.event_queues.remove(q)

    def do_POST(self):
        if self.path == "/api/action":
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode('utf-8'))
                self.server.viewer.communicator.handle_action(data)
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok"}).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode('utf-8'))

def start_server(viewer):
    server = ThreadSafeHTTPServer(("localhost", 8000), WebServerHandler, viewer)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server
