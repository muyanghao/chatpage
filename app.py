# app.py
# --------------------------
# Flask + llama.cpp (llama-cpp-python) local LLM server
# With true streaming (SSE) compatible with OpenAI /v1/chat/completions
# --------------------------

import os
import time
import json
import csv
import uuid
from typing import List, Dict, Any

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# ---- Basic config ----
MODEL_PATH = os.getenv("MODEL_PATH", "./model.gguf")
N_CTX = int(os.getenv("N_CTX", "4096"))
N_THREADS = int(os.getenv("N_THREADS", "8"))
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "0"))

# generation params
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.95"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "1024"))

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
SESSIONS_DIR   = os.getenv("SESSIONS_DIR", "sessions")

# ---- Import llama.cpp bindings ----
try:
    from llama_cpp import Llama
except Exception as e:
    raise SystemExit(
        "llama-cpp-python is required. Install with:\n"
        "  pip install llama-cpp-python\n"
        f"Import error: {e}"
    )

app = Flask(__name__)
CORS(app)

# ---- Initialize model once ----
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=N_CTX,
    n_gpu_layers=N_GPU_LAYERS,
    n_threads=N_THREADS,
    verbose=False,
)

# ---- Minimal in-memory chat history (global, optional) ----
chat_history: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

# ---- Prompt builder ----
def build_prompt_from_messages(messages: List[Dict[str, str]]) -> str:
    """
    Very simple prompt builder:
    - start with system prompt
    - then user / assistant turns
    - final line: "Assistant:"
    """
    lines = []

    system = None
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
            break
    if not system:
        system = SYSTEM_PROMPT

    lines.append(f"System: {system}")

    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")

    lines.append("Assistant:")
    return "\n".join(lines)

# ---- Non-stream generation ----
def local_generate(prompt: str) -> str:
    out = llm(
        prompt=prompt,
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        stop=["User:", "Assistant:", "<|eot_id|>"],
    )
    text = out.get("choices", [{}])[0].get("text", "")
    return text.strip()

# ---- SSE helper ----
def sse_pack(data: Dict[str, Any]) -> str:
    return "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"

# ---- /v1/chat/completions clone ----
@app.post("/v1/chat/completions")
def chat_completions():
    """
    Accepts:
    {
      "model": "...",
      "messages": [{"role": "system"|"user"|"assistant", "content": "..."}],
      "stream": true|false,
      "mode": "instant"|"thinking" (自定义，前端用来切换)
    }
    """
    data = request.get_json(force=True) or {}

    messages = data.get("messages") or []
    stream = bool(data.get("stream", False))
    mode = data.get("mode") or "instant"

    if not messages:
        return jsonify({"error": "messages is required"}), 400

    # 把 system prompt 合并一下，防止有多个 system
    # 简单做法：取第一条 system（如果有），否则用默认
    final_messages: List[Dict[str, str]] = []
    system_found = False
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "system" and not system_found:
            final_messages.append({"role": "system", "content": content})
            system_found = True
        elif role in ("user", "assistant"):
            final_messages.append({"role": role, "content": content})
    if not system_found:
        final_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    prompt = build_prompt_from_messages(final_messages)

    # thinking 模式：不走 stream，直接一次性返回最终结果
    if mode == "thinking":
        try:
            out = llm(
                prompt=prompt,
                max_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                stop=["User:", "Assistant:", "<|eot_id|>"],
            )
            text = out.get("choices", [{}])[0].get("text", "")
            text = text.strip()
        except Exception as e:
            return jsonify({"error": f"generation failed: {e}"}), 500

        resp = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "local-llm",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }
        return jsonify(resp)

    # instant 模式：流式输出
    if not stream:
        # 前端如果没开 stream 也走非流式
        try:
            text = local_generate(prompt)
        except Exception as e:
            return jsonify({"error": f"generation failed: {e}"}), 500

        resp = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "local-llm",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }
        return jsonify(resp)

    # ----- stream = True 的情况：返回 SSE -----
    def generate():
        yield sse_pack({
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "local-llm",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ]
        })

        try:
            for chunk in llm(
                prompt=prompt,
                max_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                stop=["User:", "Assistant:", "<|eot_id|>"],
                stream=True,
            ):
                piece = chunk.get("choices", [{}])[0].get("text", "")
                if not piece:
                    continue

                yield sse_pack({
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "local-llm",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": piece},
                            "finish_reason": None,
                        }
                    ]
                })
        except Exception as e:
            yield sse_pack({
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "local-llm",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"\n[Error] {e}"},
                        "finish_reason": "error",
                    }
                ]
            })

        yield "data: [DONE]\n\n"

    return Response(generate(), mimetype="text/event-stream")

# ----- Simple non-stream completion (for testing) -----
@app.post("/v1/completions")
def completions():
    """
    Just to be somewhat compatible with /v1/completions:
    {
      "prompt": "...",
      "max_tokens": 256,
      "temperature": 0.7,
      "top_p": 0.95
    }
    """
    data = request.get_json(force=True) or {}
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", MAX_NEW_TOKENS)
    temperature = data.get("temperature", TEMPERATURE)
    top_p = data.get("top_p", TOP_P)
    stop = data.get("stop")

    if not prompt:
        return jsonify({"error": "prompt is empty"}), 400

    out = llm(
        prompt=prompt,
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        stop=stop or ["User:", "Assistant:", "<|eot_id|>"],
    )
    text = out.get("choices", [{}])[0].get("text", "")

    resp = {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": "local-llm",
        "choices": [
            {
                "index": 0,
                "text": text,
                "finish_reason": "stop",
            }
        ],
        "usage": out.get("usage", {}),
    }
    return jsonify(resp)

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

    chat_history.append({"role": "user", "content": user_input})
    prompt = build_prompt_from_messages(chat_history)

    try:
        reply = local_generate(prompt)
    except Exception as e:
        return jsonify({"error": f"generation failed: {e}"}), 500

    chat_history.append({"role": "assistant", "content": reply})
    return jsonify({"response": reply})

# ----- Health check -----
@app.get("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
