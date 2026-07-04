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
        "Usage: agent [subcommand] OR agent <natural language query>\n\n"
        "Subcommands:\n"
        "  api                      - Activates the hosted API server (e.g. Gemini)\n"
        "  local                    - Activates local backend (probes LM Studio/Ollama, falls back to GGUF)\n"
        "  deactivate / unload / off - Unloads the LLM and frees memory\n"
        "  status                   - Shows current backend and loading status\n"
        "  clear / reset            - Clears current context window history\n\n"
        "Examples:\n"
        "  agent api\n"
        "  agent local\n"
        "  agent color cluster 3 red\n"
        "  agent zoom to 600\n"
        "  agent deactivate"
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
    """Searches web_ui/agent_resource/ for the first GGUF file."""
    resource_dir = os.path.join("web_ui", "agent_resource")
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
    """Loads configuration from web_ui/agent_resource/agent_config.json and api_secrets.json."""
    config_path = os.path.join("web_ui", "agent_resource", "agent_config.json")
    secrets_path = os.path.join("web_ui", "agent_resource", "api_secrets.json")
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
            print(f"Warning: Failed to parse agent_config.json: {e}")
            
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

def activate_agent(viewer, force_backend=None, quiet=False):
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
    
    deactivate_agent(viewer, quiet=True)
    
    if prefer_backend == "api":
        if not api_url:
            Command_Engine.print_help(viewer, "Error: API url is not configured in agent_config.json.")
            return False
        backend_details = {
            "name": "Hosted API",
            "url": api_url,
            "model": model_name or "default"
        }
        
    elif prefer_backend == "local":
        if not quiet:
            print("Agent: Probing local API servers (Ollama, LM Studio, Llama.cpp)...")
        backend_details = detect_running_servers()
        
        if not backend_details:
            gguf_path = config.get("gguf_model_path")
            if not gguf_path:
                gguf_path = find_local_gguf()
                
            if gguf_path and os.path.exists(gguf_path):
                if not quiet:
                    print(f"Agent: Found local GGUF model: {os.path.basename(gguf_path)}. Loading...")
                if hasattr(app, 'process_events'):
                    app.process_events()
                try:
                    model = load_gguf_model(gguf_path)
                    viewer.llm_backend = "gguf"
                    viewer.llm_model = model
                    viewer.llm_model_name = os.path.basename(gguf_path)
                    viewer.llm_loaded = True
                    viewer.llm_temperature = temperature
                    
                    msg = f"LLM Agent Activated: Local GGUF ({os.path.basename(gguf_path)})"
                    if not quiet:
                        Command_Engine.print_help(viewer, msg)
                    return True
                except Exception as e:
                    print(f"Agent: Failed to load local GGUF model: {e}")
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
                    print("Agent: Probing local API servers (Ollama, LM Studio, Llama.cpp)...")
                backend_details = detect_running_servers()
                
        if not backend_details and prefer_backend in ["auto", "gguf"]:
            gguf_path = config.get("gguf_model_path")
            if not gguf_path:
                gguf_path = find_local_gguf()
                
            if gguf_path and os.path.exists(gguf_path):
                if not quiet:
                    print(f"Agent: Found local GGUF model: {os.path.basename(gguf_path)}. Loading...")
                if hasattr(app, 'process_events'):
                    app.process_events()
                try:
                    model = load_gguf_model(gguf_path)
                    viewer.llm_backend = "gguf"
                    viewer.llm_model = model
                    viewer.llm_model_name = os.path.basename(gguf_path)
                    viewer.llm_loaded = True
                    viewer.llm_temperature = temperature
                    
                    msg = f"LLM Agent Activated: Local GGUF ({os.path.basename(gguf_path)})"
                    if not quiet:
                        Command_Engine.print_help(viewer, msg)
                    return True
                except Exception as e:
                    print(f"Agent: Failed to load local GGUF model: {e}")
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
        
        msg = f"LLM Agent Activated: Server ({backend_details['name']}) -> {backend_details['model']}"
        if not quiet:
            Command_Engine.print_help(viewer, msg)
        return True
        
    err_msg = (
        "Error: Could not activate agent.\n"
        "No local running LLM server (Ollama, LM Studio, Llama.cpp) was detected, "
        "and no GGUF file was found in 'web_ui/agent_resource/'.\n"
        "Please ensure a local server is running or place a GGUF model in the resource folder."
    )
    Command_Engine.print_help(viewer, err_msg)
    return False

def deactivate_agent(viewer, quiet=False):
    """Deactivates LLM agent and unloads model components from memory."""
    if hasattr(viewer, 'llm_history'):
        viewer.llm_history = []
        
    if not getattr(viewer, 'llm_loaded', False):
        if not quiet:
            Command_Engine.print_help(viewer, "Agent is already inactive.")
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
        
    msg = f"LLM Agent Deactivated. Unloaded: {model_name} ({backend})"
    if not quiet:
        Command_Engine.print_help(viewer, msg)

def print_status(viewer):
    if not getattr(viewer, 'llm_loaded', False):
        Command_Engine.print_help(viewer, "Agent Status: Deactivated (no LLM loaded)")
        return
        
    backend = getattr(viewer, 'llm_backend', None)
    model = getattr(viewer, 'llm_model_name', "Unknown")
    
    if backend == "server":
        url = getattr(viewer, 'llm_url', "")
        msg = f"Agent Status: Active\n  Backend: Server API ({url})\n  Model: {model}"
    else:
        msg = f"Agent Status: Active\n  Backend: Local GGUF\n  Model: {model}"
        
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
                content = res_data["choices"][0]["message"]["content"].strip()
                usage = res_data.get("usage", {})
                tokens = {
                    "prompt": usage.get("prompt_tokens", 0),
                    "completion": usage.get("completion_tokens", 0),
                    "total": usage.get("total_tokens", 0)
                }
                return {"content": content, "tokens": tokens}
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8")
            print(f"\n[Agent Error Detail]\nHTTP Status Code: {e.code}\nServer Response: {body}\n")
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
        content = res["choices"][0]["text"].strip()
        usage = res.get("usage", {})
        tokens = {
            "prompt": usage.get("prompt_tokens", 0),
            "completion": usage.get("completion_tokens", 0),
            "total": usage.get("total_tokens", 0)
        }
        return {"content": content, "tokens": tokens}
    return None

class AgentWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(str, str, str) # Emits: (response, tokens_json, error_message)
    
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
            res_dict = None
            if self.backend == "server":
                res_dict = call_api(self.url, self.model_name, self.system_prompt, self.query, self.history, self.temperature, self.api_key)
            elif self.backend == "gguf":
                res_dict = call_gguf(self.gguf_model, self.system_prompt, self.query, self.history, self.temperature)
            
            if not res_dict or not res_dict.get("content"):
                self.finished.emit("", "", "No response received from LLM.")
            else:
                tokens_json = json.dumps(res_dict.get("tokens", {}))
                self.finished.emit(res_dict["content"], tokens_json, "")
        except Exception as e:
            self.finished.emit("", "", str(e))

def get_viewer_session_context(viewer):
    """
    Extracts high-level information about the active visualizer state (metadata keys, 
    number of nodes, groups, clusters, reference sequence) to guide the LLM agent.
    """
    import numpy as np
    lines = []
    lines.append("\n--- ACTIVE SSN VIEWER STATE ---")
    
    # 1. Node count
    n_nodes = getattr(viewer, 'n_nodes', 0)
    lines.append(f"Number of Nodes: {n_nodes}")
    
    # 2. Reference sequence
    ref_seq = getattr(viewer, 'resolved_ref_full', None)
    if ref_seq:
        lines.append(f"Active Reference Sequence: {ref_seq}")
        
    # 3. Selection
    selected_indices = getattr(viewer, 'selected_indices', [])
    if selected_indices:
        lines.append(f"Current Selection: {len(selected_indices)} nodes selected (accessible via $sele$)")
    else:
        lines.append("Current Selection: No nodes currently selected")
        
    # 4. Available Metadata keys
    metadata = getattr(viewer, 'metadata', {})
    if metadata:
        lines.append("Available Metadata Properties:")
        for key, prop_data in metadata.items():
            prop_type = prop_data.get("type", "text")
            values = prop_data.get("values")
            val_desc = ""
            if values is not None and len(values) > 0:
                if prop_type == "number":
                    arr = np.asarray(values)
                    valid_vals = arr[~np.isnan(arr)]
                    if len(valid_vals) > 0:
                        val_desc = f" (min: {float(min(valid_vals)):.1f}, max: {float(max(valid_vals)):.1f})"
                else:
                    unique_vals = list(set([str(v) for v in values if v is not None and str(v).strip() != ""]))
                    if len(unique_vals) > 0:
                        sample = ", ".join(unique_vals[:5])
                        val_desc = f" (e.g., {sample})"
            lines.append(f"  - {key}: type={prop_type}{val_desc}")
            
    # 5. Topology Clusters
    cluster_labels = getattr(viewer, 'cluster_labels', None)
    if cluster_labels is not None:
        unique_clusters = np.unique(cluster_labels)
        valid_clusters = [c for c in unique_clusters if str(c).lower() not in ["noise", "none", "-1", "nan"]]
        if valid_clusters:
            lines.append(f"Available Clusters: {', '.join(map(str, valid_clusters[:15]))}" + 
                         ("..." if len(valid_clusters) > 15 else ""))
            
    # 6. Custom Groups
    group_labels = getattr(viewer, 'group_labels', None)
    if group_labels:
        unique_groups = set()
        for g_set in group_labels:
            if isinstance(g_set, set):
                unique_groups.update(g_set)
        if unique_groups:
            lines.append(f"Defined Custom Groups: {', '.join(sorted(unique_groups))}")
            
    lines.append("---------------------------------\n")
    return "\n".join(lines)

def run_web_agent_query(viewer, query):
    if not getattr(viewer, 'llm_loaded', False):
        viewer.broadcast_event({"type": "agent_error", "error": "LLM is not loaded. Please select a model backend first."})
        return

    prompt_path = os.path.join("web_ui", "agent_resource", "system_prompt.md")
    if not os.path.exists(prompt_path):
        viewer.broadcast_event({"type": "agent_error", "error": f"System prompt file missing at {prompt_path}"})
        return
    
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except Exception as e:
        viewer.broadcast_event({"type": "agent_error", "error": f"Could not read prompt file: {e}"})
        return

    # Inject the visualizer state context dynamically into the LLM system prompt
    system_prompt += get_viewer_session_context(viewer)

    # Broadcast thinking status to browser
    model_name = getattr(viewer, 'llm_model_name', "LLM")
    viewer.broadcast_event({"type": "agent_thinking", "model_name": model_name})

    # Prepare background arguments
    backend = viewer.llm_backend
    url = getattr(viewer, 'llm_url', None)
    model_name = viewer.llm_model_name
    temperature = getattr(viewer, 'llm_temperature', 0.0)
    api_key = getattr(viewer, 'llm_api_key', None)
    gguf_model = getattr(viewer, 'llm_model', None)
    
    if not hasattr(viewer, 'llm_history'):
        viewer.llm_history = []
    if not hasattr(viewer, '_cacheable_attrs'):
        viewer._cacheable_attrs = set()
    viewer._cacheable_attrs.add("llm_history")
    
    # Filter history to only include 'role' and 'content' for API request
    history = [{"role": msg["role"], "content": msg["content"]} for msg in viewer.llm_history]

    # Launch background thread
    viewer._web_agent_worker = AgentWorker(
        backend, url, model_name, system_prompt, query, history,
        temperature, api_key, gguf_model
    )
    viewer._web_agent_worker.finished.connect(
        lambda res, tokens, err: on_web_worker_finished(viewer, query, res, tokens, err)
    )
    viewer._web_agent_worker.start()

class RefinementWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(str, str, str) # Emits: (refined_explanation, tokens_json, error_message)
    
    def __init__(self, backend, url, model_name, user_query, commands, terminal_output, temperature, api_key, gguf_model):
        super().__init__()
        self.backend = backend
        self.url = url
        self.model_name = model_name
        self.user_query = user_query
        self.commands = commands
        self.terminal_output = terminal_output
        self.temperature = temperature
        self.api_key = api_key
        self.gguf_model = gguf_model
        
    def run(self):
        try:
            refinement_prompt = (
                "You are a helpful biological assistant.\n"
                "You have just executed the following visualizer commands on behalf of the user:\n"
                f"```\n{self.commands}\n```\n\n"
                "The commands printed the following output in the visualizer terminal:\n"
                f"```\n{self.terminal_output}\n```\n\n"
                f"The user's original request was: \"{self.user_query}\"\n\n"
                "Write a clear, friendly, and concise response explaining the results and answering their question directly. "
                "Do NOT output any commands or repeat 'Explanation:'. Just write the direct answer."
            )
            
            res_dict = None
            if self.backend == "server":
                res_dict = call_api(self.url, self.model_name, "You are a helpful biological assistant.", refinement_prompt, history=None, temperature=self.temperature, api_key=self.api_key)
            elif self.backend == "gguf":
                res_dict = call_gguf(self.gguf_model, "You are a helpful biological assistant.", refinement_prompt, history=None, temperature=self.temperature)
            
            if not res_dict or not res_dict.get("content"):
                self.finished.emit("", "", "No response from refinement.")
            else:
                tokens_json = json.dumps(res_dict.get("tokens", {}))
                self.finished.emit(res_dict["content"], tokens_json, "")
        except Exception as e:
            self.finished.emit("", "", str(e))

def save_and_broadcast_agent_response(viewer, query, explanation, commands, terminal_output, tokens_json):
    if not hasattr(viewer, 'llm_history'):
        viewer.llm_history = []
    if not hasattr(viewer, '_cacheable_attrs'):
        viewer._cacheable_attrs = set()
    viewer._cacheable_attrs.add("llm_history")
    
    tokens = json.loads(tokens_json) if tokens_json else None
    
    # Store complete log including commands/output in history content to maintain full context
    content_payload = explanation
    if commands:
        content_payload += f"\n\nCommands executed:\n```\n{commands}\n```"
    if terminal_output:
        content_payload += f"\n\nTerminal output:\n```\n{terminal_output}\n```"
        
    viewer.llm_history.append({"role": "user", "content": query})
    viewer.llm_history.append({
        "role": "assistant",
        "content": content_payload,
        "explanation": explanation,
        "commands": commands,
        "terminal_output": terminal_output,
        "tokens": tokens
    })
    viewer.llm_history = viewer.llm_history[-10:]
    
    viewer.broadcast_event({
        "type": "agent_response",
        "query": query,
        "explanation": explanation,
        "commands": commands,
        "terminal_output": terminal_output,
        "tokens": tokens,
        "llm_history": viewer.llm_history
    })

def start_refinement_worker(viewer, query, original_explanation, commands, terminal_output, initial_tokens_json):
    model_name = getattr(viewer, 'llm_model_name', "LLM")
    viewer.broadcast_event({"type": "agent_thinking", "model_name": f"{model_name} (Analyzing results)"})
    
    backend = viewer.llm_backend
    url = getattr(viewer, 'llm_url', None)
    temperature = getattr(viewer, 'llm_temperature', 0.0)
    api_key = getattr(viewer, 'llm_api_key', None)
    gguf_model = getattr(viewer, 'llm_model', None)
    
    viewer._refinement_worker = RefinementWorker(
        backend, url, model_name, query, commands, terminal_output, 
        temperature, api_key, gguf_model
    )
    
    def on_refinement_finished(refined_explanation, refinement_tokens_json, err):
        final_explanation = refined_explanation if not err and refined_explanation else original_explanation
        
        combined_tokens_json = initial_tokens_json
        if refinement_tokens_json:
            try:
                t1 = json.loads(initial_tokens_json) if initial_tokens_json else {}
                t2 = json.loads(refinement_tokens_json)
                combined = {
                    "prompt": t1.get("prompt", 0) + t2.get("prompt", 0),
                    "completion": t1.get("completion", 0) + t2.get("completion", 0),
                    "total": t1.get("total", 0) + t2.get("total", 0)
                }
                combined_tokens_json = json.dumps(combined)
            except Exception:
                pass
                
        save_and_broadcast_agent_response(viewer, query, final_explanation, commands, terminal_output, combined_tokens_json)
        
    viewer._refinement_worker.finished.connect(on_refinement_finished)
    viewer._refinement_worker.start()

def on_web_worker_finished(viewer, query, translated_output, tokens_json, error_msg):
    if error_msg:
        viewer.broadcast_event({"type": "agent_error", "error": error_msg})
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

    # Run commands on main thread while capturing standard output
    import io
    import contextlib
    
    captured_outputs = []
    if cmd_lines:
        for cmd in cmd_lines:
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                try:
                    viewer.process_command(cmd)
                except Exception as e:
                    print(f"Error executing command '{cmd}': {e}")
            output = f.getvalue().strip()
            if output:
                captured_outputs.append(output)
                
    terminal_output = "\n".join(captured_outputs) if captured_outputs else ""

    if not terminal_output:
        # Save and broadcast immediately
        save_and_broadcast_agent_response(viewer, query, explanation_str, commands_str, "", tokens_json)
    else:
        # Perform secondary Turn refinement utilizing captured outputs
        start_refinement_worker(viewer, query, explanation_str, commands_str, terminal_output, tokens_json)

def run(viewer, args):
    # Monkeypatch get_initial_web_state to inject llm_history dynamically
    if not hasattr(viewer, '_get_web_state_patched'):
        original_get_web_state = viewer.get_initial_web_state
        
        def patched_get_web_state():
            state = original_get_web_state()
            if isinstance(state, dict):
                state = dict(state)
                state["llm_history"] = getattr(viewer, 'llm_history', [])
            return state
            
        viewer.get_initial_web_state = patched_get_web_state
        viewer._get_web_state_patched = True

    # Dynamically register the Agent button in the side panel
    if hasattr(viewer, 'add_sidebar_button'):
        viewer.add_sidebar_button(
            name="agentBtn",
            label="🤖 Agent",
            callback=viewer.open_agent_ui,
            tooltip="Open AI Agent Chat Console in browser"
        )
        if not hasattr(viewer, 'sidebar_buttons_to_persist'):
            viewer.sidebar_buttons_to_persist = []
        if "agent" not in viewer.sidebar_buttons_to_persist:
            viewer.sidebar_buttons_to_persist.append("agent")

    # If --register-only is specified, return early without opening browser
    if args and args[0] == '--register-only':
        return

    import webbrowser
    webbrowser.open("http://localhost:8000/agent.html")
    if hasattr(viewer, 'console_text'):
        viewer.console_text.text = "Agent UI opened at http://localhost:8000/agent.html"
        if hasattr(viewer, 'update_console_background'):
            viewer.update_console_background()
