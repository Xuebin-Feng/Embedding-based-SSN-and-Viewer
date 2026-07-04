import os
import sys
import json
import urllib.request
import urllib.error
import gc
import Command_Engine
from PyQt6 import QtCore, QtGui, QtWidgets
import re
from vispy import app

def print_help(viewer):
    msg = (
        "Usage: assist [subcommand] OR assist <natural language query>\n\n"
        "Subcommands:\n"
        "  api                      - Activates the hosted API server (e.g. Gemini)\n"
        "  local                    - Activates local backend (probes LM Studio/Ollama, falls back to GGUF)\n"
        "  deactivate / unload / off - Unloads the LLM and frees memory\n"
        "  status                   - Shows current backend and loading status\n"
        "  clear / reset            - Clears current context window history\n\n"
        "Examples:\n"
        "  assist api\n"
        "  assist local\n"
        "  assist color cluster 3 red\n"
        "  assist zoom to 600\n"
        "  assist deactivate"
    )
    Command_Engine.print_help(viewer, msg)

def detect_running_servers():
    """Probes standard local ports for running OpenAI-compatible API servers."""
    probes = [
        {"name": "Ollama", "url": "http://localhost:11434/v1/chat/completions", "test_url": "http://localhost:11434/v1/models", "default_model": "qwen2.5:1.5b"},
        {"name": "LM Studio", "url": "http://localhost:1234/v1/chat/completions", "test_url": "http://localhost:1234/v1/models", "default_model": None},
        {"name": "Llama.cpp", "url": "http://localhost:8080/v1/chat/completions", "test_url": "http://localhost:8080/v1/models", "default_model": None}
    ]
    
    for probe in probes:
        try:
            req = urllib.request.Request(probe["test_url"], method="GET")
            with urllib.request.urlopen(req, timeout=0.5) as response:
                if response.status == 200:
                    model_name = probe["default_model"]
                    try:
                        models_data = json.loads(response.read().decode('utf-8'))
                        if "data" in models_data and len(models_data["data"]) > 0:
                            model_name = models_data["data"][0]["id"]
                    except:
                        pass
                    return {
                        "name": probe["name"],
                        "url": probe["url"],
                        "model": model_name or "default"
                    }
        except Exception:
            continue
            
    try:
        req = urllib.request.Request("http://localhost:11434", method="GET")
        with urllib.request.urlopen(req, timeout=0.5) as response:
            if response.status == 200:
                return {
                    "name": "Ollama",
                    "url": "http://localhost:11434/v1/chat/completions",
                    "model": "qwen2.5:1.5b"
                }
    except Exception:
        pass
        
    return None

def find_local_gguf():
    """Searches commands/assist_resource/ for the first GGUF file."""
    resource_dir = os.path.join("commands", "assist_resource")
    if not os.path.isdir(resource_dir):
        return None
    for file in os.listdir(resource_dir):
        if file.lower().endswith(".gguf"):
            return os.path.join(resource_dir, file)
    return None

def load_gguf_model(gguf_path):
    """Loads GGUF model via llama-cpp-python dynamically."""
    try:
        from llama_cpp import Llama
        return Llama(model_path=gguf_path, n_ctx=2048, verbose=False)
    except ImportError:
        raise ImportError(
            "llama-cpp-python is not installed. "
            "Please install it using 'pip install llama-cpp-python' to run local GGUF models."
        )

def load_config():
    """Loads configuration from commands/assist_resource/assist_config.json and api_secrets.json."""
    config_path = os.path.join("commands", "assist_resource", "assist_config.json")
    secrets_path = os.path.join("commands", "assist_resource", "api_secrets.json")
    defaults = {
        "api_url": None,
        "model_name": None,
        "prefer_backend": "auto",
        "gguf_model_path": None,
        "temperature": 0.0,
        "api_key": None
    }
    
    config = {**defaults}
    
    # 1. Load public settings
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config.update(json.load(f))
        except Exception as e:
            print(f"Warning: Failed to parse assist_config.json: {e}")
            
    # 2. Load private API secrets (ignored by git)
    if os.path.exists(secrets_path):
        try:
            with open(secrets_path, "r", encoding="utf-8") as f:
                secrets = json.load(f)
                if "api_key" in secrets:
                    config["api_key"] = secrets["api_key"]
        except Exception as e:
            print(f"Warning: Failed to parse api_secrets.json: {e}")
            
    return config

def activate_assistant(viewer, force_backend=None, quiet=False):
    """Initializes LLM backend based on config preferences, force override, and availability."""
    config = load_config()
    
    api_url = config.get("api_url")
    model_name = config.get("model_name")
    
    if force_backend:
        prefer_backend = force_backend.lower()
    else:
        prefer_backend = config.get("prefer_backend", "auto").lower()
        
    temperature = config.get("temperature", 0.0)
    api_key = config.get("api_key")
    
    backend_details = None
    
    deactivate_assistant(viewer, quiet=True)
    
    if prefer_backend == "api":
        if not api_url:
            Command_Engine.print_help(viewer, "Error: API url is not configured in assist_config.json.")
            return False
        backend_details = {
            "name": "Hosted API",
            "url": api_url,
            "model": model_name or "default"
        }
        
    elif prefer_backend == "local":
        if not quiet:
            print("Assist: Probing local API servers (Ollama, LM Studio, Llama.cpp)...")
        backend_details = detect_running_servers()
        
        if not backend_details:
            gguf_path = config.get("gguf_model_path")
            if not gguf_path:
                gguf_path = find_local_gguf()
                
            if gguf_path and os.path.exists(gguf_path):
                if not quiet:
                    print(f"Assist: Found local GGUF model: {os.path.basename(gguf_path)}. Loading...")
                if hasattr(app, 'process_events'):
                    app.process_events()
                try:
                    model = load_gguf_model(gguf_path)
                    viewer.llm_backend = "gguf"
                    viewer.llm_model = model
                    viewer.llm_model_name = os.path.basename(gguf_path)
                    viewer.llm_loaded = True
                    viewer.llm_temperature = temperature
                    
                    msg = f"LLM Assist Activated: Local GGUF ({os.path.basename(gguf_path)})"
                    if not quiet:
                        Command_Engine.print_help(viewer, msg)
                    return True
                except Exception as e:
                    print(f"Assist: Failed to load local GGUF model: {e}")
                    Command_Engine.print_help(viewer, f"Error: GGUF model loading failed: {e}")
                    return False
            else:
                Command_Engine.print_help(viewer, "Error: No local running server (LM Studio/Ollama) or local GGUF model found.")
                return False
                
    else: # auto / server / gguf
        if prefer_backend in ["auto", "server"] or api_url:
            if api_url:
                backend_details = {
                    "name": "Custom Server",
                    "url": api_url,
                    "model": model_name or "default"
                }
            else:
                if not quiet:
                    print("Assist: Probing local API servers (Ollama, LM Studio, Llama.cpp)...")
                backend_details = detect_running_servers()
                
        if not backend_details and prefer_backend in ["auto", "gguf"]:
            gguf_path = config.get("gguf_model_path")
            if not gguf_path:
                gguf_path = find_local_gguf()
                
            if gguf_path and os.path.exists(gguf_path):
                if not quiet:
                    print(f"Assist: Found local GGUF model: {os.path.basename(gguf_path)}. Loading...")
                if hasattr(app, 'process_events'):
                    app.process_events()
                try:
                    model = load_gguf_model(gguf_path)
                    viewer.llm_backend = "gguf"
                    viewer.llm_model = model
                    viewer.llm_model_name = os.path.basename(gguf_path)
                    viewer.llm_loaded = True
                    viewer.llm_temperature = temperature
                    
                    msg = f"LLM Assist Activated: Local GGUF ({os.path.basename(gguf_path)})"
                    if not quiet:
                        Command_Engine.print_help(viewer, msg)
                    return True
                except Exception as e:
                    print(f"Assist: Failed to load local GGUF model: {e}")
                    if prefer_backend == "gguf":
                        Command_Engine.print_help(viewer, f"Error: GGUF model loading failed: {e}")
                        return False
                        
    if backend_details:
        viewer.llm_backend = "server"
        viewer.llm_url = backend_details["url"]
        viewer.llm_model_name = backend_details["model"]
        viewer.llm_loaded = True
        viewer.llm_temperature = temperature
        viewer.llm_api_key = api_key
        
        msg = f"LLM Assist Activated: Server ({backend_details['name']}) -> {backend_details['model']}"
        if not quiet:
            Command_Engine.print_help(viewer, msg)
        return True
        
    err_msg = (
        "Error: Could not activate assist.\n"
        "No local running LLM server (Ollama, LM Studio, Llama.cpp) was detected, "
        "and no GGUF file was found in 'commands/assist_resource/'.\n"
        "Please ensure a local server is running or place a GGUF model in the resource folder."
    )
    Command_Engine.print_help(viewer, err_msg)
    return False

def deactivate_assistant(viewer, quiet=False):
    """Deactivates LLM assist and unloads model components from memory."""
    if hasattr(viewer, 'llm_history'):
        viewer.llm_history = []
        
    if not getattr(viewer, 'llm_loaded', False):
        if not quiet:
            Command_Engine.print_help(viewer, "Assist is already inactive.")
        return
        
    backend = getattr(viewer, 'llm_backend', None)
    model_name = getattr(viewer, 'llm_model_name', "Unknown")
    
    if hasattr(viewer, 'llm_model'):
        del viewer.llm_model
    if hasattr(viewer, 'llm_backend'):
        del viewer.llm_backend
    if hasattr(viewer, 'llm_url'):
        del viewer.llm_url
    if hasattr(viewer, 'llm_model_name'):
        del viewer.llm_model_name
    if hasattr(viewer, 'llm_temperature'):
        del viewer.llm_temperature
    if hasattr(viewer, 'llm_api_key'):
        del viewer.llm_api_key
        
    viewer.llm_loaded = False
    
    gc.collect()
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
        
    msg = f"LLM Assist Deactivated. Unloaded: {model_name} ({backend})"
    if not quiet:
        Command_Engine.print_help(viewer, msg)

def print_status(viewer):
    if not getattr(viewer, 'llm_loaded', False):
        Command_Engine.print_help(viewer, "Assist Status: Deactivated (no LLM loaded)")
        return
        
    backend = getattr(viewer, 'llm_backend', None)
    model = getattr(viewer, 'llm_model_name', "Unknown")
    
    if backend == "server":
        url = getattr(viewer, 'llm_url', "")
        msg = f"Assist Status: Active\n  Backend: Server API ({url})\n  Model: {model}"
    else:
        msg = f"Assist Status: Active\n  Backend: Local GGUF\n  Model: {model}"
        
    Command_Engine.print_help(viewer, msg)

def call_api(url, model, system_prompt, user_query, history=None, temperature=0.0, api_key=None):
    """Sends OpenAI-compatible chat request to the LLM server API."""
    if history is None:
        history = []
        
    # Robustly ensure correct chat completion endpoint is appended
    if not url.endswith("/chat/completions"):
        if url.endswith("/"):
            url += "chat/completions"
        else:
            url += "/chat/completions"
            
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt}
        ] + history + [
            {"role": "user", "content": user_query}
        ],
        "temperature": temperature
    }
    
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req, timeout=15.0) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            if "choices" in res_data and len(res_data["choices"]) > 0:
                return res_data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8")
            print(f"\n[Assist Error Detail]\nHTTP Status Code: {e.code}\nServer Response: {body}\n")
        except Exception:
            pass
        raise e
            
    return None

def call_gguf(llm, system_prompt, user_query, history=None, temperature=0.0):
    """Runs query locally on GGUF model using ChatML prompt structure."""
    if history is None:
        history = []
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    for msg in history:
        prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"
    
    res = llm(prompt, max_tokens=150, stop=["<|im_end|>", "\n\n", "Input:"], temp=temperature)
    if "choices" in res and len(res["choices"]) > 0:
        return res["choices"][0]["text"].strip()
class AgentWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(str, str) # Emits: (response, error_message)
    
    def __init__(self, backend, url, model_name, system_prompt, query, history, temperature, api_key, gguf_model):
        super().__init__()
        self.backend = backend
        self.url = url
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.query = query
        self.history = history
        self.temperature = temperature
        self.api_key = api_key
        self.gguf_model = gguf_model
        
    def run(self):
        try:
            translated_output = None
            if self.backend == "server":
                translated_output = call_api(self.url, self.model_name, self.system_prompt, self.query, self.history, self.temperature, self.api_key)
            elif self.backend == "gguf":
                translated_output = call_gguf(self.gguf_model, self.system_prompt, self.query, self.history, self.temperature)
            
            if not translated_output:
                self.finished.emit("", "No response received from LLM.")
            else:
                self.finished.emit(translated_output, "")
        except Exception as e:
            self.finished.emit("", str(e))

class AgentPanel(QtWidgets.QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.worker = None
        self.init_ui()
        self.update_backend_ui_from_state()

    def init_ui(self):
        # Set light theme stylesheet matching the viewer's layout
        self.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                color: #1f2328;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 9.5pt;
            }
            QComboBox {
                background-color: #f6f8fa;
                border: 1px solid #d0d7de;
                border-radius: 4px;
                padding: 4px 8px;
                color: #1f2328;
                min-width: 130px;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                border: 1px solid #d0d7de;
                selection-background-color: #0969da;
                selection-color: #ffffff;
                color: #1f2328;
            }
            QLineEdit {
                background-color: #ffffff;
                border: 1px solid #d0d7de;
                border-radius: 6px;
                padding: 8px 12px;
                color: #1f2328;
            }
            QLineEdit:focus {
                border: 1px solid #0969da;
            }
            QPushButton {
                background-color: #f6f8fa;
                border: 1px solid #d0d7de;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
                color: #1f2328;
            }
            QPushButton:hover {
                background-color: #f3f4f6;
            }
            QPushButton#sendBtn {
                background-color: #0969da;
                border: none;
                color: #ffffff;
            }
            QPushButton#sendBtn:hover {
                background-color: #1a7fec;
            }
            QPushButton#sendBtn:disabled {
                background-color: #8cc2f7;
                color: #ffffff;
            }
            QTextBrowser {
                background-color: #ffffff;
                border: 1px solid #d0d7de;
                border-radius: 8px;
                padding: 8px;
            }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Top Control Bar
        top_bar = QtWidgets.QHBoxLayout()
        
        backend_label = QtWidgets.QLabel("Model:")
        backend_label.setStyleSheet("font-weight: bold; color: #1f2328;")
        top_bar.addWidget(backend_label)

        self.backend_combo = QtWidgets.QComboBox()
        self.backend_combo.addItems(["Deactivated", "API Server", "Local LLM"])
        self.backend_combo.currentIndexChanged.connect(self.on_backend_changed)
        top_bar.addWidget(self.backend_combo)

        top_bar.addStretch()

        self.clear_btn = QtWidgets.QPushButton("Clear")
        self.clear_btn.setToolTip("Clear chat history and context memory")
        self.clear_btn.clicked.connect(self.clear_history)
        top_bar.addWidget(self.clear_btn)

        layout.addLayout(top_bar)

        # Status indicator line
        status_layout = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Status: Deactivated")
        self.status_label.setStyleSheet("color: #57606a; font-size: 8.5pt;")
        status_layout.addWidget(self.status_label)
        layout.addLayout(status_layout)

        # Chat history display area
        self.chat_display = QtWidgets.QTextBrowser()
        self.chat_display.setOpenExternalLinks(True)
        layout.addWidget(self.chat_display)

        # Welcome message
        self.append_system_msg("Welcome to Vispy SSN Agent! Select a model backend above to begin.")

        # Bottom Input Area
        input_layout = QtWidgets.QHBoxLayout()
        self.input_field = QtWidgets.QLineEdit()
        self.input_field.setPlaceholderText("Ask Agent a question or command...")
        self.input_field.returnPressed.connect(self.send_query)
        input_layout.addWidget(self.input_field)

        self.send_btn = QtWidgets.QPushButton("Send")
        self.send_btn.setObjectName("sendBtn")
        self.send_btn.clicked.connect(self.send_query)
        input_layout.addWidget(self.send_btn)

        layout.addLayout(input_layout)

    def update_backend_ui_from_state(self):
        loaded = getattr(self.viewer, 'llm_loaded', False)
        if not loaded:
            self.backend_combo.setCurrentIndex(0)
            self.status_label.setText("Status: Deactivated")
            self.input_field.setEnabled(False)
            self.send_btn.setEnabled(False)
        else:
            backend = getattr(self.viewer, 'llm_backend', None)
            model_name = getattr(self.viewer, 'llm_model_name', "Unknown")
            if backend == "server":
                self.backend_combo.setCurrentIndex(1)
                self.status_label.setText(f"Status: Active Server ({model_name})")
            elif backend == "gguf":
                self.backend_combo.setCurrentIndex(2)
                self.status_label.setText(f"Status: Active Local GGUF ({model_name})")
            self.input_field.setEnabled(True)
            self.send_btn.setEnabled(True)

    def on_backend_changed(self, index):
        if index == 0:
            deactivate_assistant(self.viewer, quiet=True)
        elif index == 1:
            activate_assistant(self.viewer, force_backend="api", quiet=True)
        elif index == 2:
            activate_assistant(self.viewer, force_backend="local", quiet=True)
        self.update_backend_ui_from_state()

    def clear_history(self):
        self.chat_display.clear()
        if hasattr(self.viewer, 'llm_history'):
            self.viewer.llm_history = []
        self.append_system_msg("Chat history and context memory cleared.")

    def append_system_msg(self, msg):
        html = f"""
        <table width="100%">
            <tr>
                <td align="center">
                    <table bgcolor="#f6f8fa" style="border: 1px solid #d0d7de; border-radius: 4px;">
                        <tr>
                            <td style="color: #57606a; font-size: 8.5pt; padding: 3px 8px; font-family: 'Segoe UI', Arial, sans-serif;">
                                {msg}
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
        """
        self.chat_display.append(html)

    def append_user_msg(self, msg):
        msg_escaped = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        html = f"""
        <table width="100%" style="margin: 4px 0;">
            <tr>
                <td align="right">
                    <table bgcolor="#0969da" style="border-radius: 10px;">
                        <tr>
                            <td style="color: #ffffff; padding: 8px 12px; font-family: 'Segoe UI', Arial, sans-serif; font-size: 9.5pt; text-align: left;">
                                {msg_escaped}
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
        """
        self.chat_display.append(html)

    def append_agent_msg(self, explanation, commands):
        exp_escaped = explanation.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        cmd_html = ""
        if commands:
            cmd_escaped = commands.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            cmd_html = f"""
            <table width="100%" bgcolor="#ffffff" style="border: 1px solid #d0d7de; border-radius: 5px; margin-top: 6px;">
                <tr>
                    <td style="padding: 6px 10px; font-family: Consolas, 'Courier New', monospace; font-size: 9pt; color: #0969da; line-height: 1.3;">
                        <b>Executed Command(s):</b><br>{cmd_escaped}
                    </td>
                </tr>
            </table>
            """
        
        html = f"""
        <table width="100%" style="margin: 4px 0;">
            <tr>
                <td align="left">
                    <table bgcolor="#f6f8fa" style="border: 1px solid #d0d7de; border-radius: 10px;">
                        <tr>
                            <td style="color: #1f2328; padding: 8px 12px; font-family: 'Segoe UI', Arial, sans-serif; font-size: 9.5pt; text-align: left;">
                                {exp_escaped}
                                {cmd_html}
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
        """
        self.chat_display.append(html)

    def append_error_msg(self, err_msg):
        err_escaped = err_msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        html = f"""
        <table width="100%">
            <tr>
                <td align="center">
                    <table bgcolor="#ffebe9" style="border: 1px solid #ffc1c0; border-radius: 8px;">
                        <tr>
                            <td style="color: #cf222e; padding: 6px 12px; font-family: 'Segoe UI', Arial, sans-serif; font-size: 9.5pt; text-align: left;">
                                <b>Error:</b> {err_escaped}
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
        """
        self.chat_display.append(html)

    def send_query(self):
        query = self.input_field.text().strip()
        if not query:
            return
        
        if not getattr(self.viewer, 'llm_loaded', False):
            self.append_error_msg("LLM is not loaded. Please select a model backend first.")
            return

        prompt_path = os.path.join("commands", "assist_resource", "system_prompt.md")
        if not os.path.exists(prompt_path):
            self.append_error_msg(f"System prompt file missing at {prompt_path}")
            return
        
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                system_prompt = f.read()
        except Exception as e:
            self.append_error_msg(f"Could not read prompt file: {e}")
            return

        # Disable controls during thinking phase
        self.input_field.setEnabled(False)
        self.send_btn.setEnabled(False)
        self.backend_combo.setEnabled(False)
        self.clear_btn.setEnabled(False)

        model_name = getattr(self.viewer, 'llm_model_name', "LLM")
        self.status_label.setText(f"Status: Thinking ({model_name})...")
        if hasattr(self.viewer, 'console_text'):
            self.viewer.console_text.text = "Agent: Translating query..."
            if hasattr(self.viewer, 'update_console_background'):
                self.viewer.update_console_background()

        self.append_user_msg(query)
        self.input_field.clear()

        # Prepare background arguments
        backend = self.viewer.llm_backend
        url = getattr(self.viewer, 'llm_url', None)
        model_name = self.viewer.llm_model_name
        temperature = getattr(self.viewer, 'llm_temperature', 0.0)
        api_key = getattr(self.viewer, 'llm_api_key', None)
        gguf_model = getattr(self.viewer, 'llm_model', None)
        
        if not hasattr(self.viewer, 'llm_history'):
            self.viewer.llm_history = []
        history = list(self.viewer.llm_history)

        # Launch background thread
        self.worker = AgentWorker(
            backend, url, model_name, system_prompt, query, history,
            temperature, api_key, gguf_model
        )
        self.worker.finished.connect(lambda res, err: self.on_worker_finished(query, res, err))
        self.worker.start()

    def on_worker_finished(self, query, translated_output, error_msg):
        # Re-enable controls
        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.backend_combo.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self.input_field.setFocus()
        
        self.update_backend_ui_from_state()

        if error_msg:
            self.append_error_msg(error_msg)
            if hasattr(self.viewer, 'console_text'):
                self.viewer.console_text.text = f"Agent Error: {error_msg}"
                if hasattr(self.viewer, 'update_console_background'):
                    self.viewer.update_console_background()
            return

        # Parse translated commands and comments
        cmd_lines = []
        explanation_lines = []
        in_code_block = False
        
        for line in translated_output.split("\n"):
            line_strip = line.strip()
            if not line_strip:
                continue
            if line_strip.startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                cmd_lines.append(line_strip)
            else:
                lower_line = line_strip.lower()
                if lower_line.startswith("explanation:") or lower_line.startswith("comment:") or line_strip.startswith("#"):
                    explanation = line_strip
                    if lower_line.startswith("explanation:"):
                        explanation = line_strip[12:].strip()
                    elif lower_line.startswith("comment:"):
                        explanation = line_strip[8:].strip()
                    elif line_strip.startswith("#"):
                        explanation = line_strip[1:].strip()
                    if explanation:
                        explanation_lines.append(explanation)
                elif not lower_line.startswith("input:") and not lower_line.startswith("output:"):
                    clean_line = line_strip.strip("`'")
                    if clean_line:
                        cmd_lines.append(clean_line)

        explanation_str = "\n".join(explanation_lines) if explanation_lines else "Command executed successfully."
        commands_str = "\n".join(cmd_lines)

        # Update viewer LLM history
        if not hasattr(self.viewer, 'llm_history'):
            self.viewer.llm_history = []
        
        self.viewer.llm_history.append({"role": "user", "content": query})
        if commands_str:
            self.viewer.llm_history.append({"role": "assistant", "content": commands_str})
        else:
            self.viewer.llm_history.append({"role": "assistant", "content": f"Explanation: {explanation_str}"})
        self.viewer.llm_history = self.viewer.llm_history[-6:]

        # Append response to UI
        self.append_agent_msg(explanation_str, commands_str)

        # Run commands on main thread
        if cmd_lines:
            for cmd in cmd_lines:
                if hasattr(self.viewer, 'console_text'):
                    self.viewer.console_text.text = f"Running Agent: {cmd}"
                    if hasattr(self.viewer, 'update_console_background'):
                        self.viewer.update_console_background()
                self.viewer.process_command(cmd)
        else:
            if hasattr(self.viewer, 'console_text'):
                self.viewer.console_text.text = "Agent finished processing"
                if hasattr(self.viewer, 'update_console_background'):
                    self.viewer.update_console_background()

def inject_agent_panel(viewer, show_sidebar=True):
    if not hasattr(viewer, 'tab_widget'):
        return None

    tab_idx = -1
    for idx in range(viewer.tab_widget.count()):
        if viewer.tab_widget.tabText(idx) == "Agent":
            tab_idx = idx
            break

    if tab_idx == -1:
        panel = AgentPanel(viewer)
        viewer.tab_widget.addTab(panel, "Agent")
        tab_idx = viewer.tab_widget.count() - 1
        viewer.agent_panel = panel
    else:
        panel = viewer.tab_widget.widget(tab_idx)
        if hasattr(panel, 'update_backend_ui_from_state'):
            panel.update_backend_ui_from_state()

    viewer.tab_widget.setCurrentIndex(tab_idx)
    if show_sidebar:
        viewer.set_sidebar_visible(True)
    return panel

def run(viewer, args):
    # Running 'agent' from CLI launches and focuses the Agent tab in the side panel
    inject_agent_panel(viewer, show_sidebar=True)
    if hasattr(viewer, 'console_text'):
        viewer.console_text.text = "Agent Panel Activated"
        if hasattr(viewer, 'update_console_background'):
            viewer.update_console_background()
