// ===== 配置你的后端接口 =====
const API_BASE = "http://100.103.147.79:8001";
const STORAGE_KEY = "chat_sessions_v1";

const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const messagesEl = document.getElementById("messages");
const newChatBtn = document.getElementById("new-chat");
const sessionListEl = document.getElementById("session-list");
const exportBtn = document.getElementById("export");
// 模式选择下拉框
const modeSelect = document.getElementById("mode-select");
// thinking 状态
const thinkingStatus = document.getElementById("thinking-status");
const thinkingTimerEl = document.getElementById("thinking-timer");

let thinkingInterval = null;
let thinkingStart = null;

let sessions = [];
let currentSessionId = null;

window.addEventListener("load", init);

function init() {
  loadSessionsFromStorage();

  if (!Array.isArray(sessions) || sessions.length === 0) {
    createNewSession();
  } else {
    currentSessionId = sessions[0].id;
    renderSessionList();
    renderCurrentSessionMessages();
  }

  newChatBtn.addEventListener("click", () => {
    createNewSession();
  });

  exportBtn.addEventListener("click", () => {
    exportSessions();
  });

  sendBtn.addEventListener("click", () => {
    sendMessage();
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // 点击页面空白处关闭所有会话菜单
  document.addEventListener("click", (e) => {
    if (!e.target.closest(".session-menu-wrapper")) {
      hideAllSessionMenus();
    }
  });
}

// ===== 本地存储（浏览器） =====

function loadSessionsFromStorage() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      sessions = JSON.parse(raw) || [];
    } else {
      sessions = [];
    }
  } catch (e) {
    console.error("Failed to load sessions:", e);
    sessions = [];
  }
}

function saveSessionsToStorage() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
  } catch (e) {
    console.error("Failed to save sessions:", e);
  }
}

// ===== 会话管理 =====

async function createNewSession() {
  // 先在前端生成一个本地 id
  const id = "s_" + Date.now();
  const session = {
    id,
    session_id: null, // 后端返回的真正ID
    title: "新聊天",
    messages: [],
    createdAt: Date.now(),
    updatedAt: Date.now()
  };

  // 新会话放在最前面
  sessions.unshift(session);
  currentSessionId = id;
  saveSessionsToStorage();
  renderSessionList();
  renderCurrentSessionMessages();

  // 调用后端创建 CSV 文件
  try {
    const res = await fetch(API_BASE.replace(/\/$/, "") + "/sessions/new", {
      method: "POST"
    });
    const data = await res.json();
    const sid = data.session_id;

    // 把后端 session_id 记到本地会话里
    const s = getSessionById(id);
    if (s) {
      s.session_id = sid;
      saveSessionsToStorage();
    }
  } catch (e) {
    console.error("createNewSession backend error:", e);
    // 即使后端失败，前端仍然可以用，只是没写 CSV
  }
}

function getSessionById(id) {
  return sessions.find((s) => s.id === id) || null;
}

function getCurrentSession() {
  return getSessionById(currentSessionId);
}

function renderSessionList() {
  sessionListEl.innerHTML = "";

  sessions.forEach((session) => {
    const item = document.createElement("div");
    item.className = "session-item";
    if (session.id === currentSessionId) {
      item.classList.add("active");
    }

    // 左侧标题
    const titleDiv = document.createElement("div");
    titleDiv.className = "session-title";
    titleDiv.textContent = session.title || "未命名对话";
    titleDiv.title = session.title;

    titleDiv.addEventListener("click", () => {
      if (currentSessionId === session.id) return;
      currentSessionId = session.id;
      renderSessionList();
      renderCurrentSessionMessages();
    });

    // 右侧三点菜单
    const menuWrapper = document.createElement("div");
    menuWrapper.className = "session-menu-wrapper";

    const menuBtn = document.createElement("button");
    menuBtn.className = "session-menu-btn";
    menuBtn.innerHTML = "⋯";

    const menu = document.createElement("div");
    menu.className = "session-menu hidden";

    const exportItem = document.createElement("div");
    exportItem.className = "session-menu-item";
    exportItem.textContent = "导出";
    exportItem.addEventListener("click", (e) => {
      e.stopPropagation();
      exportSingleSession(session);
      hideAllSessionMenus();
    });

    const renameItem = document.createElement("div");
    renameItem.className = "session-menu-item";
    renameItem.textContent = "重命名";
    renameItem.addEventListener("click", (e) => {
      e.stopPropagation();
      renameSession(session);
      hideAllSessionMenus();
    });

    const deleteItem = document.createElement("div");
    deleteItem.className = "session-menu-item danger";
    deleteItem.textContent = "删除";
    deleteItem.addEventListener("click", (e) => {
      e.stopPropagation();
      deleteSession(session);
      hideAllSessionMenus();
    });

    menu.appendChild(exportItem);
    menu.appendChild(renameItem);
    menu.appendChild(deleteItem);

    menuBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleSessionMenu(menu);
    });

    menuWrapper.appendChild(menuBtn);
    menuWrapper.appendChild(menu);

    item.appendChild(titleDiv);
    item.appendChild(menuWrapper);

    sessionListEl.appendChild(item);
  });
}

function renderCurrentSessionMessages() {
  messagesEl.innerHTML = "";
  const session = getCurrentSession();
  if (!session) return;

  session.messages.forEach((msg) => {
    const roleClass = msg.role === "user" ? "user" : "bot";
    createBubble(msg.content, roleClass);
  });

  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function updateSessionTitle(session) {
  if (!session) return;

  if (session.title === "新聊天") {
    const firstUserMsg = session.messages.find(
      (m) => m.role === "user" && m.content.trim()
    );
    if (firstUserMsg) {
      session.title = firstUserMsg.content.trim().slice(0, 20);
    }
  }
}

function sortSessionsByUpdatedTime() {
  sessions.sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
}

// ===== 侧边栏菜单辅助函数 =====

function hideAllSessionMenus() {
  document.querySelectorAll(".session-menu").forEach((menu) => {
    menu.classList.add("hidden");
  });
}

function toggleSessionMenu(menu) {
  const isHidden = menu.classList.contains("hidden");
  hideAllSessionMenus();
  if (isHidden) {
    menu.classList.remove("hidden");
  }
}

function exportSingleSession(session) {
  const dataStr =
    "data:text/json;charset=utf-8," +
    encodeURIComponent(JSON.stringify(session, null, 2));
  const a = document.createElement("a");
  const dateStr = new Date().toISOString().slice(0, 10);
  const safeTitle = (session.title || "session").replace(/[\\/:*?"<>|]/g, "_");
  a.href = dataStr;
  a.download = `chat_${safeTitle}_${dateStr}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

function renameSession(session) {
  const name = window.prompt("请输入新的会话名称：", session.title || "");
  if (name == null) return;
  const t = name.trim();
  if (!t) return;
  session.title = t;
  session.updatedAt = Date.now();
  sortSessionsByUpdatedTime();
  saveSessionsToStorage();
  renderSessionList();
}

function deleteSession(session) {
  if (!window.confirm("确定要删除这个会话吗？")) return;
  const idx = sessions.findIndex((s) => s.id === session.id);
  if (idx === -1) return;

  sessions.splice(idx, 1);

  if (sessions.length === 0) {
    currentSessionId = null;
    saveSessionsToStorage();
    messagesEl.innerHTML = "";
    createNewSession();
    return;
  }

  if (currentSessionId === session.id) {
    currentSessionId = sessions[0].id;
  }

  saveSessionsToStorage();
  renderSessionList();
  renderCurrentSessionMessages();
}

// ===== UI 气泡 =====

function createBubble(text, role) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.innerText = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

function appendMessage(text, role, options = {}) {
  const div = createBubble(text, role);
  const session = getCurrentSession();
  let messageRef = null;

  if (session) {
    messageRef = {
      role: role === "user" ? "user" : "assistant",
      content: text,
      time: Date.now()
    };
    session.messages.push(messageRef);
    session.updatedAt = Date.now();

    // 用第一条用户消息前 20 个字生成标题
    updateSessionTitle(session);

    if (!options.skipSave) {
      sortSessionsByUpdatedTime();
      saveSessionsToStorage();
      renderSessionList();
    }
  }

  return { div, messageRef };
}

async function logToBackend(role, content, timestamp) {
  const session = getCurrentSession();
  if (!session || !session.session_id) return;

  try {
    await fetch(API_BASE.replace(/\/$/, "") + "/sessions/log", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: session.session_id,
        role,
        content,
        timestamp
      })
    });
  } catch (e) {
    console.error("logToBackend error:", e);
  }
}

async function sendMessage() {
  const text = input.value.trim();
  if (!text) return;

  const now = Date.now();
  const mode = modeSelect ? modeSelect.value || "instant" : "instant";
  const useStream = mode === "instant";

  // 前端先加一条 user 消息
  const { messageRef: userRef } = appendMessage(text, "user");
  input.value = "";

  // 异步写一条 user 日志到 CSV
  logToBackend("user", text, now);

  // 先创建一个空的 bot 气泡，用于填充内容
  const { div: botBubble, messageRef: botRef } = appendMessage("", "bot", {
    skipSave: true
  });

  sendBtn.disabled = true;

  // thinking 模式：显示动画 + 计时
  if (mode === "thinking") {
    startThinkingUI();
  }

  const url = API_BASE.replace(/\/$/, "") + "/v1/chat/completions";

  const session = getCurrentSession();
  const messagesForBackend = (session?.messages || []).map((m) => ({
    role: m.role,
    content: m.content
  }));

  // mode 传给后端；thinking 模式非流式
  const body = {
    model: "local-llm",
    messages: messagesForBackend,
    stream: useStream,
    mode
  };

  let assistantText = "";

  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });

    if (!res.ok) {
      const errMsg = "Fail to connect (HTTP " + res.status + ")";
      botBubble.textContent = errMsg;
      finalizeBotMessageSave(botRef, errMsg);
      logToBackend("assistant", errMsg, Date.now());
      return;
    }

    if (useStream) {
      // ======== instant 模式：流式输出 ========
      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop();

        for (const chunk of parts) {
          const line = chunk.trim();
          if (!line.startsWith("data:")) continue;

          const payload = line.slice(5).trim();
          if (payload === "[DONE]") {
            break;
          }

          try {
            const obj = JSON.parse(payload);
            const delta = obj?.choices?.[0]?.delta?.content || "";
            if (delta) {
              botBubble.textContent += delta;
              assistantText += delta;
              if (botRef) {
                botRef.content += delta;
              }
              messagesEl.scrollTop = messagesEl.scrollHeight;
            }
          } catch (e) {
            // 忽略解析失败
          }
        }
      }

      finalizeBotMessageSave(botRef, assistantText);
      logToBackend("assistant", assistantText, Date.now());
    } else {
      // ======== thinking 模式：不流式，一次性拿最终答案 ========
      const data = await res.json();
      assistantText = data?.choices?.[0]?.message?.content || "";

      if (!assistantText) {
        assistantText = "[Empty response]";
      }

      botBubble.textContent = assistantText;
      if (botRef) {
        botRef.content = assistantText;
      }
      finalizeBotMessageSave(botRef, assistantText);
      logToBackend("assistant", assistantText, Date.now());
    }
  } catch (err) {
    console.error(err);
    const errMsg = "Fail to connect";
    botBubble.textContent = errMsg;
    finalizeBotMessageSave(botRef, errMsg);
    logToBackend("assistant", errMsg, Date.now());
  } finally {
    sendBtn.disabled = false;
    if (mode === "thinking") {
      stopThinkingUI();
    }
  }
}

function finalizeBotMessageSave(messageRef, finalText) {
  const session = getCurrentSession();
  if (messageRef && session) {
    messageRef.content = finalText;
    session.updatedAt = Date.now();
    sortSessionsByUpdatedTime();
    saveSessionsToStorage();
    renderSessionList();
  }
}

// ===== thinking 模式的 UI 辅助函数 =====

function startThinkingUI() {
  if (!thinkingStatus) return;
  thinkingStatus.classList.remove("hidden");
  thinkingStart = Date.now();
  updateThinkingTimer();
  if (thinkingInterval) clearInterval(thinkingInterval);
  thinkingInterval = setInterval(updateThinkingTimer, 100);
}

function updateThinkingTimer() {
  if (!thinkingTimerEl || thinkingStart == null) return;
  const elapsed = (Date.now() - thinkingStart) / 1000;
  thinkingTimerEl.textContent = elapsed.toFixed(1) + "s";
}

function stopThinkingUI() {
  if (thinkingInterval) {
    clearInterval(thinkingInterval);
    thinkingInterval = null;
  }
  thinkingStart = null;
  if (thinkingStatus) {
    thinkingStatus.classList.add("hidden");
  }
}

function exportSessions() {
  const dataStr =
    "data:text/json;charset=utf-8," +
    encodeURIComponent(JSON.stringify(sessions, null, 2));
  const a = document.createElement("a");
  const dateStr = new Date().toISOString().slice(0, 10);
  a.href = dataStr;
  a.download = `chat_sessions_${dateStr}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}
