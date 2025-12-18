# app.py
# ------------------------------------------------------------
# Flask + vLLM (OpenAI-compatible) local LLM server adapter
#
# Requirements implemented:
# - Let model decide if web browsing is needed (instead of keyword heuristics)
# - If needed: run web search, inject results, then generate final answer
# - Works for stream & non-stream
# - Stream mode sends immediate OpenAI-style "data:" chunk to avoid frontend abort/timeout
# - Robust sessions CSV listing (read all csv even if some rows are broken)
# ------------------------------------------------------------

import os
import time
import json
import csv
import uuid
import re
from typing import List, Dict, Any, Optional, Tuple

import requests
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

# ---- Basic config ----
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.95"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "5000"))

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful local assistant running on user's machine. "
    "Be concise and clear. If you're not sure, say you are not sure."
)

SESSIONS_DIR = os.getenv("SESSIONS_DIR", "sessions")

# ---- vLLM OpenAI-compatible server config ----
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
VLLM_CHAT_URL = f"{VLLM_BASE_URL}/v1/chat/completions"
VLLM_COMP_URL = f"{VLLM_BASE_URL}/v1/completions"
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openai/gpt-oss-20b")

# Model judge config (fast, short)
JUDGE_MAX_TOKENS = int(os.getenv("JUDGE_MAX_TOKENS", "128"))
JUDGE_TEMPERATURE = float(os.getenv("JUDGE_TEMPERATURE", "0.0"))

app = Flask(__name__)
CORS(app)

# (optional) simple /chat history endpoint
chat_history: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]


# ============================================================
# SSE helpers (OpenAI-compatible "data:" chunks)
# ============================================================
def sse_data(obj: Dict[str, Any]) -> bytes:
    return ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8")


def openai_stream_delta(text: str) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl_{uuid.uuid4().hex[:10]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": DEFAULT_MODEL,
        "choices": [{"index": 0, "delta": {"content": text}}],
    }


def openai_stream_keepalive() -> Dict[str, Any]:
    # empty delta to "wake up" frontend SSE parser immediately
    return openai_stream_delta("")


# ============================================================
# Proxy helpers
# ============================================================
def _proxy_post_json(url: str, payload: Dict[str, Any], timeout_s: int = 600) -> Response:
    r = requests.post(url, json=payload, timeout=timeout_s)
    return Response(
        r.content,
        status=r.status_code,
        content_type=r.headers.get("Content-Type", "application/json"),
    )


def _proxy_post_sse(url: str, payload: Dict[str, Any], timeout_s: int = 600) -> Response:
    def generate():
        with requests.post(url, json=payload, stream=True, timeout=timeout_s) as r:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    yield chunk

    return Response(stream_with_context(generate()), status=200, mimetype="text/event-stream")


# ============================================================
# Web search (minimal no-key): DuckDuckGo HTML
# ============================================================
def _ddg_html_search(query: str, max_results: int = 6, timeout_s: int = 20) -> Tuple[List[Dict[str, str]], str]:
    url = "https://duckduckgo.com/html/"
    headers = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/x-www-form-urlencoded"}

    try:
        r = requests.post(url, headers=headers, data={"q": query}, timeout=timeout_s)
        r.raise_for_status()
        html = r.text
    except Exception as e:
        return [], f"ddg_search_failed: {e}"

    links = re.findall(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html)
    snippets = re.findall(
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>|<div[^>]+class="result__snippet"[^>]*>(.*?)</div>',
        html
    )

    def _strip_tags(s: str) -> str:
        s = re.sub(r"<.*?>", "", s)
        s = s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'")
        return re.sub(r"\s+", " ", s).strip()

    results: List[Dict[str, str]] = []
    for i, (href, title_html) in enumerate(links[:max_results]):
        title = _strip_tags(title_html)
        snippet = ""
        if i < len(snippets):
            snippet = _strip_tags(snippets[i][0] or snippets[i][1] or "")
        results.append({"title": title, "url": href, "snippet": snippet})

    return results, ""


def web_search_context(query: str, max_results: int = 6) -> Tuple[str, List[str]]:
    results, err = _ddg_html_search(query, max_results=max_results)
    urls = [r.get("url", "") for r in results if r.get("url")]

    if not results:
        ctx = "Web search returned no results."
        if err:
            ctx = "Web search failed.\n" + err
        return ctx, urls

    lines = []
    for i, r in enumerate(results[:max_results], start=1):
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        snip = (r.get("snippet") or "").strip()
        lines.append(f"{i}. {title}\n   {url}\n   {snip}")

    ctx = "Web search results (use as references, do not fabricate):\n" + "\n".join(lines)
    return ctx, urls


# ============================================================
# Model-based judge: decide if browsing is needed
# ============================================================
JUDGE_SYSTEM = (
    "You are a routing classifier for a local assistant.\n"
    "Decide whether the user's request requires web browsing.\n"
    "Browse if the user asks for: real-time data (current time/date in a location), "
    "weather, breaking/news, prices/stock/crypto rates, live schedules, "
    "recent policy changes, or any factual info you're not confident about.\n"
    "Do NOT browse for: general knowledge, math, coding, writing, translation, or opinions.\n"
    "Return STRICT JSON only, with keys:\n"
    "  need_web: boolean\n"
    "  query: string (if need_web=true, provide an effective web search query; else empty)\n"
    "  reason: string (short)\n"
    "No markdown. No extra text."
)

def _extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try find first {...}
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def model_decide_need_web(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Ask vLLM to decide browsing need.
    Returns dict: {need_web: bool, query: str, reason: str}
    """
    # Build judge messages: system + last user turn(s)
    # Keep it small to be fast.
    last_user = ""
    for m in reversed(messages):
        if (m.get("role") == "user") and (m.get("content") is not None):
            last_user = str(m.get("content"))
            break

    judge_payload = {
        "model": DEFAULT_MODEL,
        "stream": False,
        "temperature": JUDGE_TEMPERATURE,
        "top_p": 1.0,
        "max_tokens": JUDGE_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": last_user},
        ],
    }

    try:
        r = requests.post(VLLM_CHAT_URL, json=judge_payload, timeout=60)
        r.raise_for_status()
        out = r.json()
        txt = out.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    except Exception as e:
        # Fail-safe: if judge fails, default to no browse
        return {"need_web": False, "query": "", "reason": f"judge_failed: {e}"}

    obj = _extract_json_obj(txt) or {}
    need_web = bool(obj.get("need_web", False))
    query = str(obj.get("query", "") or "")
    reason = str(obj.get("reason", "") or "")

    # Safety: if need_web but no query, fall back to last user text
    if need_web and not query.strip():
        query = last_user.strip()

    return {"need_web": need_web, "query": query.strip(), "reason": reason.strip()}


# ============================================================
# /v1/chat/completions
# ============================================================
@app.post("/v1/chat/completions")
def chat_completions():
    data = request.get_json(force=True) or {}
    messages = data.get("messages") or []
    if not messages:
        return jsonify({"error": "messages is required"}), 400

    stream = bool(data.get("stream", False))

    # base payload (we will inject web results if needed)
    payload: Dict[str, Any] = dict(data)
    payload["model"] = DEFAULT_MODEL
    payload["messages"] = messages
    payload["stream"] = stream
    payload.setdefault("temperature", TEMPERATURE)
    payload.setdefault("top_p", TOP_P)
    payload.setdefault("max_tokens", MAX_NEW_TOKENS)

    # 1) Let model decide
    decision = model_decide_need_web(messages)
    need_web = bool(decision.get("need_web", False))
    web_query = (decision.get("query") or "").strip()

    # 2) If stream and need web: keepalive immediately + browse + inject + stream vLLM
    if stream and need_web:
        def generate():
            # Prevent frontend timeout/abort: send a standard data chunk immediately
            yield sse_data(openai_stream_keepalive())

            # OPTIONAL: send markers for UI (front-end can hide/show URL panel)
            # If you don't want these markers shown, handle them in frontend and don't append to text.
            yield sse_data(openai_stream_delta(f"[[WEB_SEARCH_START]]{web_query}"))

            ctx, urls = web_search_context(web_query, max_results=6)

            yield sse_data(openai_stream_delta("[[WEB_SEARCH_URLS]]" + json.dumps(urls, ensure_ascii=False)))
            yield sse_data(openai_stream_delta("[[WEB_SEARCH_END]]"))

            new_messages = [{
                "role": "system",
                "content": (
                    "The following information was retrieved from the web. "
                    "Use it as reference if relevant.\n\n" + ctx
                )
            }] + messages

            vllm_payload = dict(payload)
            vllm_payload["messages"] = new_messages
            vllm_payload["stream"] = True

            with requests.post(VLLM_CHAT_URL, json=vllm_payload, stream=True, timeout=600) as r:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        yield chunk

        return Response(stream_with_context(generate()), status=200, mimetype="text/event-stream")

    # 3) If non-stream and need web: browse then inject then normal json
    if (not stream) and need_web:
        ctx, _urls = web_search_context(web_query, max_results=6)
        payload["messages"] = [{
            "role": "system",
            "content": (
                "The following information was retrieved from the web. "
                "Use it as reference if relevant.\n\n" + ctx
            )
        }] + messages

    # 4) Otherwise passthrough
    try:
        if stream:
            return _proxy_post_sse(VLLM_CHAT_URL, payload)
        return _proxy_post_json(VLLM_CHAT_URL, payload)
    except Exception as e:
        return jsonify({"error": f"upstream vLLM request failed: {e}"}), 500


# ============================================================
# /v1/completions
# ============================================================
@app.post("/v1/completions")
def completions():
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


# ============================================================
# Sessions: /sessions/new /sessions/log /sessions/list
# ============================================================
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


# ============================================================
# /chat (optional simple endpoint, kept)
# ============================================================
@app.post("/chat")
def chat_simple():
    global chat_history
    data = request.get_json(force=True) or {}
    user_input = (data.get("message") or "").strip()
    if not user_input:
        return jsonify({"error": "empty message"}), 400

    chat_history.append({"role": "user", "content": user_input})

    payload = {
        "model": DEFAULT_MODEL,
        "messages": chat_history,
        "stream": False,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_NEW_TOKENS,
    }

    # model-based judge for /chat too
    decision = model_decide_need_web(chat_history)
    if decision.get("need_web"):
        ctx, _urls = web_search_context(decision.get("query") or user_input, max_results=6)
        payload["messages"] = [{
            "role": "system",
            "content": (
                "The following information was retrieved from the web. "
                "Use it as reference if relevant.\n\n" + ctx
            )
        }] + payload["messages"]

    try:
        r = requests.post(VLLM_CHAT_URL, json=payload, timeout=600)
        out = r.json()
        reply = out.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        return jsonify({"error": f"upstream vLLM request failed: {e}"}), 500

    chat_history.append({"role": "assistant", "content": reply})
    return jsonify({"response": reply})


@app.get("/health")
def health():
    return jsonify({"status": "ok", "vllm_base": VLLM_BASE_URL, "model": DEFAULT_MODEL})


if __name__ == "__main__":
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    port = int(os.getenv("PORT", "9000"))
    app.run(host="0.0.0.0", port=port, debug=False)
