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

# ---- Config via env ----
MODEL_PATH     = os.getenv("MODEL_PATH", "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
N_CTX          = int(os.getenv("N_CTX", "4096"))
N_GPU_LAYERS   = int(os.getenv("N_GPU_LAYERS", "0"))     # 0 = CPU
N_THREADS      = int(os.getenv("N_THREADS", str(os.cpu_count() or 4)))
TEMPERATURE    = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P          = float(os.getenv("TOP_P", "0.95"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "300"))
SYSTEM_PROMPT  = os.getenv("SYSTEM_PROMPT", "You are a assistant and just answer questions.")

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

if not os.path.exists(MODEL_PATH):
    raise SystemExit(f"Model not found: {MODEL_PATH}")

os.makedirs(SESSIONS_DIR, exist_ok=True)

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
    Convert OpenAI-style messages to a single instruction-style prompt that
    works well for Instruct models with llama.cpp completion API.
    """
    sys = SYSTEM_PROMPT
    lines = []
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role == "system" and content:
            sys = content
        elif role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
    # The model will complete the next assistant turn.
    return f"{sys}\n\n" + "\n".join(lines) + "\nAssistant:"

# ---- Non-stream generation ----
def local_generate(prompt: str, *,
                   max_tokens: int = MAX_NEW_TOKENS,
                   temperature: float = TEMPERATURE,
                   top_p: float = TOP_P,
                   stop: List[str] | None = None) -> str:
    out = llm(
        prompt=prompt,
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        stop=stop or ["User:", "Assistant:", "<|eot_id|>"],
    )
    if isinstance(out, dict):
        if "choices" in out and out["choices"]:
            return out["choices"][0].get("text", "")
        return out.get("text", "")
    return str(out)

# ---- THINKING 模式：最多返工 5 轮自我反思推理 ----
def run_thinking_mode(
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: List[str] | None = None,
) -> str:
    """
    多轮自反思：
    - 最多 5 轮（round 1~5）
    - 每一轮模型输出 JSON:
      {
        "answer": "...",
        "critique": "...",
        "quality_score": 1-10,
        "should_stop": true/false
      }
    - 记录评分最高的 answer，最后返回。
    """
    stop = stop or ["User:", "Assistant:", "<|eot_id|>"]
    max_rounds = 5

    conversation_prompt = build_prompt_from_messages(messages)

    best_answer = ""
    best_score = -1
    last_answer = ""
    last_critique = ""

    for i in range(max_rounds):
        round_idx = i + 1

        if i == 0:
            # 第一轮：根据对话生成答案 + 自评
            round_prompt = (
                conversation_prompt
                + f"\n\nYou are now in THINKING mode (round {round_idx} of {max_rounds}).\n"
                  "1. Draft your best possible answer to the user's last request.\n"
                  "2. Analyze your own answer for correctness, clarity, and safety.\n"
                  "3. Give a quality score from 1 to 10 (higher is better).\n"
                  "4. Decide whether further refinement is necessary.\n"
                  "Return ONLY a JSON object with fields:\n"
                  "  - \"answer\": string, your answer for this round\n"
                  "  - \"critique\": string, your self-critique\n"
                  "  - \"quality_score\": number between 1 and 10\n"
                  "  - \"should_stop\": boolean, true if further refinement is not needed\n"
                  "Do NOT include any extra commentary outside the JSON."
            )
        else:
            # 后续轮数：在上一轮答案+自评基础上继续改进
            round_prompt = (
                conversation_prompt
                + f"\n\nYou are refining your previous answer (round {round_idx} of {max_rounds}).\n"
                  "Here was your previous answer:\n"
                + last_answer
                + "\n\nYour previous self-critique was:\n"
                + (last_critique or "(no critique)")
                + "\n\nNow produce a NEW JSON object with improved answer and updated critique.\n"
                  "The JSON MUST have fields:\n"
                  "  - \"answer\": improved answer for this round\n"
                  "  - \"critique\": updated self-critique\n"
                  "  - \"quality_score\": number between 1 and 10\n"
                  "  - \"should_stop\": boolean, true if this answer is good enough\n"
                  "Be honest in the critique. Do NOT mention this multi-round process to the user."
            )

        raw = local_generate(
            round_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        ).strip()

        answer = ""
        critique = ""
        score = 0
        should_stop = False

        try:
            parsed = json.loads(raw)
            answer = str(parsed.get("answer", "")).strip()
            critique = str(parsed.get("critique", "")).strip()
            try:
                score = float(parsed.get("quality_score", 5))
            except Exception:
                score = 5.0
            should_stop = bool(parsed.get("should_stop", False))
        except Exception:
            # JSON 解析失败就当成一个普通答案
            answer = raw
            critique = ""
            score = 5.0
            should_stop = True  # 避免一直循环输出垃圾

        if not answer:
            answer = "I could not produce a valid answer in this round."

        # 更新最佳答案
        if score > best_score:
            best_score = score
            best_answer = answer

        # 保存本轮答案和自评，供下一轮使用
        last_answer = answer
        last_critique = critique

        # 如果模型认为可以停了，就提前结束
        if should_stop:
            break

    final_answer = best_answer or last_answer or "I could not produce a valid answer."
    return final_answer

# ---- Streaming (SSE) helpers ----
def sse_pack(obj: dict) -> str:
    # OpenAI-like SSE frame: "data: {json}\n\n"
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"

def sse_done() -> str:
    return "data: [DONE]\n\n"

def stream_generate_sse(messages, max_tokens, temperature, top_p, stop):
    """
    Use llama_cpp stream=True to yield incremental tokens.
    Emits OpenAI-style SSE deltas: {"choices":[{"delta":{"content":"..."}}]}
    """
    prompt = build_prompt_from_messages(messages)

    for chunk in llm(
        prompt=prompt,
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        stop=stop or ["User:", "Assistant:", "<|eot_id|>"],
        stream=True,
    ):
        piece = chunk.get("choices", [{}])[0].get("text", "")
        if piece:
            yield sse_pack({"choices": [{"delta": {"content": piece}}]})
    yield sse_done()

# ---- Flask app ----
app = Flask(__name__)
CORS(app)  # allow all origins (tighten if needed)

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_path": MODEL_PATH,
        "n_ctx": N_CTX,
        "n_gpu_layers": N_GPU_LAYERS,
        "n_threads": N_THREADS,
        "defaults": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS
        }
    })

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

# ----- Reset conversation -----
@app.post("/reset")
def reset():
    global chat_history
    chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    return jsonify({"message": "reseted"})

# ----- OpenAI-style endpoint (supports stream=true) -----
@app.post("/v1/chat/completions")
def chat_completions():
    """
    JSON (subset of OpenAI):
    {
      "model": "local-llm",
      "messages": [{"role":"system|user|assistant","content":"..."}],
      "max_tokens": 256,
      "temperature": 0.7,
      "top_p": 0.95,
      "stop": ["..."],
      "stream": true|false,
      "mode": "instant" | "thinking"
    }
    """
    data: Dict[str, Any] = request.get_json(force=True) or {}
    messages = data.get("messages") or []
    if not isinstance(messages, list) or not messages:
        return jsonify({"error": "messages must be a non-empty list"}), 400

    max_tokens = int(data.get("max_tokens", MAX_NEW_TOKENS))
    temperature = float(data.get("temperature", TEMPERATURE))
    top_p = float(data.get("top_p", TOP_P))
    stop = data.get("stop") or ["User:", "Assistant:", "<|eot_id|>"]
    stream = bool(data.get("stream", False))
    mode = data.get("mode", "instant")

    # ---------- instant：保持原有流式/非流式逻辑 ----------
    if mode == "instant" and stream:
        # SSE streaming response
        def gen():
            yield from stream_generate_sse(messages, max_tokens, temperature, top_p, stop)

        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable buffering if behind nginx
        }
        return Response(gen(), mimetype="text/event-stream", headers=headers)

    # ---------- 非流式：包括 instant(非流式) + thinking ----------
    t0 = time.time()
    try:
        if mode == "thinking":
            # THINKING 模式：多轮自反思（最多 5 轮）
            content = run_thinking_mode(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )
        else:
            # 普通 instant + 非流式
            prompt = build_prompt_from_messages(messages)
            content = local_generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )
    except Exception as e:
        return jsonify({"error": f"generation failed: {e}"}), 500

    latency_ms = int((time.time() - t0) * 1000)

    return jsonify({
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": data.get("model", "local-llm"),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "latency_ms": latency_ms
        }
    })

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))
    # Flask dev server is fine for LAN; for production streaming consider:
    # gunicorn --workers 1 --threads 8 --timeout 0 -k gthread app:app
    app.run(host=host, port=port, threaded=True)
