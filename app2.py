# app.py
# --------------------------
# Flask + vLLM (OpenAI-compatible) local LLM server adapter
# Keeps the same endpoints as original:
#   - /v1/chat/completions (supports SSE streaming)
#   - /v1/completions
#   - /sessions/new /sessions/log /sessions/list
#   - /chat
#   - /health
# Default port: 9000
# --------------------------

import os
import time
import json
import csv
import uuid
from typing import List, Dict, Any, Optional

import requests
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

# ---- Basic config ----
# (llama.cpp related envs removed; keep generation params for defaults)
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.95"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "5000"))

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful local assistant running on user's machine. "
    "Follow instructions carefully. Be concise and clear. "
    "If user asks about local environment or machine, answer from context only, "
    "don't pretend you can see files or network. If you're not sure, say you are "
    "not sure. Do not mention you are using llama.cpp. Do not talk about system "
    "prompts and just answer questions."
)

# CSV 保存目录
SESSIONS_DIR = os.getenv("SESSIONS_DIR", "sessions")

# ---- vLLM OpenAI-compatible server config ----
# Example: vllm serve openai/gpt-oss-20b --host 127.0.0.1 --port 8000
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
VLLM_CHAT_URL = f"{VLLM_BASE_URL}/v1/chat/completions"
VLLM_COMP_URL = f"{VLLM_BASE_URL}/v1/completions"

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openai/gpt-oss-20b")

app = Flask(__name__)
CORS(app)

# ---- Minimal in-memory chat history (global, optional) ----
chat_history: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]


# ---- SSE helper (kept for compatibility; not used for vLLM passthrough) ----
def sse_pack(data: Dict[str, Any]) -> str:
    return "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"


def _proxy_post_json(url: str, payload: Dict[str, Any], timeout_s: int = 600) -> Response:
    """
    Proxy a JSON POST request to vLLM and return the raw response (status + body).
    """
    r = requests.post(url, json=payload, timeout=timeout_s)
    # Return raw bytes to preserve the upstream response exactly.
    return Response(
        r.content,
        status=r.status_code,
        content_type=r.headers.get("Content-Type", "application/json"),
    )


def _proxy_post_sse(url: str, payload: Dict[str, Any], timeout_s: int = 600) -> Response:
    """
    Proxy an SSE streaming request to vLLM and stream bytes through unchanged.
    IMPORTANT: Use iter_content to preserve SSE framing (\n\n).
    """
    def generate():
        with requests.post(url, json=payload, stream=True, timeout=timeout_s) as r:
            # If upstream errors, still stream the body (often JSON); caller will see it.
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    yield chunk

    return Response(
        stream_with_context(generate()),
        status=200,
        mimetype="text/event-stream",
    )


# ---- /v1/chat/completions clone ----
@app.post("/v1/chat/completions")
def chat_completions():
    """
    Accepts (OpenAI-compatible):
    {
      "model": "...",
      "messages": [{"role": "system"|"user"|"assistant", "content": "..."}],
      "stream": true|false,

      Optional:
      "temperature", "top_p", "max_tokens", ...
      "mode": "instant"|"thinking" (custom field from original; we keep accepting it)
    }

    Behavior:
    - We DO NOT change the endpoint.
    - We forward to vLLM OpenAI-compatible /v1/chat/completions.
    - If stream=True => passthrough SSE.
    - mode is ignored (kept for backward compatibility).
    """
    data = request.get_json(force=True) or {}

    messages = data.get("messages") or []
    if not messages:
        return jsonify({"error": "messages is required"}), 400

    stream = bool(data.get("stream", False))
    # Keep accepting `mode` so the frontend doesn't break; no special handling needed.
    _mode = data.get("mode") or "instant"  # noqa: F841

    payload: Dict[str, Any] = dict(data)  # start from client payload to avoid breaking fields
    payload["model"] = DEFAULT_MODEL
    payload["messages"] = messages
    payload["stream"] = stream

    # Provide defaults if missing
    payload.setdefault("temperature", TEMPERATURE)
    payload.setdefault("top_p", TOP_P)
    # Some clients use max_tokens, some max_completion_tokens; keep what client sent.
    payload.setdefault("max_tokens", MAX_NEW_TOKENS)

    try:
        if stream:
            return _proxy_post_sse(VLLM_CHAT_URL, payload)
        else:
            return _proxy_post_json(VLLM_CHAT_URL, payload)
    except Exception as e:
        return jsonify({"error": f"upstream vLLM request failed: {e}"}), 500


# ----- Simple non-stream completion (for testing) -----
@app.post("/v1/completions")
def completions():
    """
    Compatible with /v1/completions:
    {
      "model": "...",
      "prompt": "...",
      "max_tokens": 256,
      "temperature": 0.7,
      "top_p": 0.95
    }
    """
    data = request.get_json(force=True) or {}
    prompt = data.get("prompt", "")

    if not prompt:
        return jsonify({"error": "prompt is empty"}), 400

    payload: Dict[str, Any] = dict(data)
    payload["model"] = data.get("model") or DEFAULT_MODEL
    payload["prompt"] = prompt
    payload.setdefault("temperature", TEMPERATURE)
    payload.setdefault("top_p", TOP_P)
    payload.setdefault("max_tokens", data.get("max_tokens", MAX_NEW_TOKENS))
    payload["stream"] = False

    try:
        return _proxy_post_json(VLLM_COMP_URL, payload)
    except Exception as e:
        return jsonify({"error": f"upstream vLLM request failed: {e}"}), 500


# ----- CSV 会话：创建新 session，对应前端 /sessions/new -----
@app.post("/sessions/new")
def sessions_new():
    """
    创建一个新的会话 CSV 文件，并返回 session_id 给前端。
    CSV 路径：sessions/<session_id>.csv
    表头：timestamp, role, content
    """
    session_id = f"s_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"
    filepath = os.path.join(SESSIONS_DIR, f"{session_id}.csv")
    try:
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "role", "content"])
    except Exception as e:
        return jsonify({"error": f"failed to create session csv: {e}"}), 500

    return jsonify({"session_id": session_id})


# ----- CSV 会话：追加一条消息，对应前端 /sessions/log -----
@app.post("/sessions/log")
def sessions_log():
    """
    追加一条记录到对应的会话 CSV：
    {
      "session_id": "...",
      "role": "user" | "assistant",
      "content": "...",
      "timestamp": 123456789  (可选；默认当前时间 ms)
    }
    """
    data = request.get_json(force=True) or {}
    session_id = (data.get("session_id") or "").strip()
    role = (data.get("role") or "").strip()
    content = data.get("content") or ""
    ts = data.get("timestamp") or int(time.time() * 1000)

    if not session_id or not role:
        return jsonify({"error": "session_id and role are required"}), 400

    filepath = os.path.join(SESSIONS_DIR, f"{session_id}.csv")

    try:
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        file_exists = os.path.exists(filepath)
        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "role", "content"])
            # 把换行替换成 \n，避免破坏 CSV 结构
            safe_content = str(content).replace("\r", " ").replace("\n", "\\n")
            writer.writerow([ts, role, safe_content])
    except Exception as e:
        return jsonify({"error": f"failed to log to csv: {e}"}), 500

    return jsonify({"ok": True})


# ----- CSV 会话：列出所有会话，对应前端 /sessions/list -----
@app.get("/sessions/list")
def sessions_list():
    """
    扫描 SESSIONS_DIR 下所有 .csv 文件，
    读取其中的消息，用第一条 user 消息作为标题，
    返回给前端用于在左侧显示历史记录。
    """
    sessions_data = []

    if not os.path.exists(SESSIONS_DIR):
        return jsonify({"sessions": []})

    for filename in os.listdir(SESSIONS_DIR):
        if not filename.endswith(".csv"):
            continue

        session_id = filename[:-4]  # 去掉 .csv
        filepath = os.path.join(SESSIONS_DIR, filename)

        messages = []
        first_user = None
        first_ts = None
        last_ts = None

        try:
            with open(filepath, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ts_raw = row.get("timestamp", "") or ""
                    role = (row.get("role") or "").strip()
                    raw_content = row.get("content") or ""

                    # 还原换行：写入时把换行变成了 "\\n"
                    content = raw_content.replace("\\n", "\n")

                    try:
                        ts = int(ts_raw)
                    except Exception:
                        ts = int(time.time() * 1000)

                    if first_ts is None:
                        first_ts = ts
                    last_ts = ts

                    messages.append({
                        "role": role,
                        "content": content,
                        "timestamp": ts,
                    })

                    if role == "user" and first_user is None:
                        first_user = content
        except Exception:
            # 某个 csv 读失败就跳过
            continue

        if not messages:
            continue

        title = (first_user or "新聊天").strip()
        if len(title) > 20:
            title = title[:20]

        sessions_data.append({
            "session_id": session_id,
            "title": title or "新聊天",
            "created_at": first_ts,
            "updated_at": last_ts,
            "messages": messages,
        })

    sessions_data.sort(key=lambda s: s.get("updated_at") or 0, reverse=True)

    return jsonify({"sessions": sessions_data})


# ----- Simple chat endpoint: { "message": "..." } -> { "response": "..." } -----
@app.post("/chat")
def chat_simple():
    global chat_history
    data = request.get_json(force=True) or {}
    user_input = (data.get("message") or "").strip()
    if not user_input:
        return jsonify({"error": "empty message"}), 400

    # 保持原逻辑：维护本地 chat_history（仅用于这个简易接口）
    chat_history.append({"role": "user", "content": user_input})

    payload = {
        "model": DEFAULT_MODEL,
        "messages": chat_history,
        "stream": False,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_NEW_TOKENS,
    }

    try:
        r = requests.post(VLLM_CHAT_URL, json=payload, timeout=600)
        out = r.json()
        reply = out.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        return jsonify({"error": f"upstream vLLM request failed: {e}"}), 500

    chat_history.append({"role": "assistant", "content": reply})
    return jsonify({"response": reply})


# ----- Health check -----
@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # default port changed to 9000
    port = int(os.getenv("PORT", "9000"))
    app.run(host="0.0.0.0", port=port, debug=False)
