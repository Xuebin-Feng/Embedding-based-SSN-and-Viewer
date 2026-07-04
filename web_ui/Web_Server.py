import http.server
import threading
import queue
import json
import os
import numpy as np
from PyQt6 import QtCore, QtGui

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
        action = data.get("action")
        if action == "select":
            indices = data.get("indices", [])
            self.viewer.selected_indices = list(indices)
            self.viewer.update_selection_visual()
            self.viewer.canvas.update()
        elif action == "edit_cell":
            row = data.get("row")
            col = data.get("column")
            value = data.get("value")
            
            meta_entry = self.viewer.metadata.get(col)
            if meta_entry:
                prop_type = meta_entry["type"]
                if prop_type == "number":
                    try:
                        if str(value).strip() == "":
                            parsed_val = np.nan
                        else:
                            parsed_val = float(value)
                    except ValueError:
                        return
                else:
                    parsed_val = str(value)
                
                meta_entry["values"][row] = parsed_val
                # Trigger a redraw of nodes/labels in case they are colored by this metadata
                self.viewer.update_nodes()
                self.viewer.canvas.update()
                
                # Auto-save layout cache on edit
                try:
                    import SSN_Utils as utils
                    cache_path, _ = utils.get_cache_filename()
                    import h5py
                    if os.path.exists(cache_path):
                        with h5py.File(cache_path, "a") as hf:
                            if "metadata" in hf:
                                meta_group = hf["metadata"]
                                if col in meta_group:
                                    del meta_group[col]
                                ds = meta_group.create_dataset(col, data=meta_entry["values"], compression="gzip")
                                ds.attrs["type"] = prop_type
                        print(f"Metadata cell [{row}, {col}] auto-saved to cache.")
                except Exception as e:
                    print(f"Warning: Failed to auto-save metadata edit: {e}")
                    
        elif action == "agent_query":
            query = data.get("query")
            # Run Agent LLM execution!
            import commands.agent as agent_cmd
            agent_cmd.run_web_agent_query(self.viewer, query)
            
        elif action == "set_backend":
            backend_idx = data.get("index")
            import commands.agent as agent_cmd
            if backend_idx == 0:
                agent_cmd.deactivate_agent(self.viewer, quiet=True)
            elif backend_idx == 1:
                agent_cmd.activate_agent(self.viewer, force_backend="api", quiet=True)
            elif backend_idx == 2:
                agent_cmd.activate_agent(self.viewer, force_backend="local", quiet=True)
            
            # Broadcast backend state updated
            self.viewer.broadcast_event({
                "type": "backend_state",
                "llm_loaded": getattr(self.viewer, 'llm_loaded', False),
                "llm_backend": getattr(self.viewer, 'llm_backend', None),
                "llm_model_name": getattr(self.viewer, 'llm_model_name', "Unknown")
            })
        elif action == "clear_history":
            if hasattr(self.viewer, 'llm_history'):
                self.viewer.llm_history = []
            if not hasattr(self.viewer, '_cacheable_attrs'):
                self.viewer._cacheable_attrs = set()
            self.viewer._cacheable_attrs.add("llm_history")

class ThreadSafeHTTPServer(http.server.ThreadingHTTPServer):
    def __init__(self, server_address, RequestHandlerClass, viewer):
        super().__init__(server_address, RequestHandlerClass)
        self.viewer = viewer
        self.event_queues = []
        self.queues_lock = threading.Lock()

    def handle_error(self, request, client_address):
        # Suppress traceback print for socket/connection abortions when browser tabs close
        import sys
        exc_type, exc_value, _ = sys.exc_info()
        if exc_type in (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            return
        super().handle_error(request, client_address)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

        # Prevent directory traversal
        safe_rel_path = clean_path.lstrip("/")
        normalized = os.path.normpath(safe_rel_path)
        if normalized.startswith("..") or os.path.isabs(normalized):
            self.send_error(403, "Forbidden")
            return

        filepath = os.path.normpath(os.path.join(BASE_DIR, normalized))
        # Ensure the resolved file path is physically inside BASE_DIR
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
