// ===================== LawAI — chat frontend =====================
const app = document.getElementById("app");
const chat = document.getElementById("chat");
const greeting = document.getElementById("greeting");
const input = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");

let sessionId = null;   // P3 — phiên hội thoại hiện tại (memory ở backend)
let busy = false;

// ---------- Sidebar collapse ----------
document.getElementById("toggleSidebar").addEventListener("click", () => app.classList.toggle("collapsed"));
document.getElementById("toggleSidebarMobile").addEventListener("click", () => app.classList.toggle("collapsed"));

// ---------- New chat ----------
document.getElementById("newChat").addEventListener("click", () => {
  sessionId = null;     // backend sẽ tạo session mới ở lượt kế
  chat.innerHTML = "";
  chat.classList.remove("active");
  greeting.classList.remove("hidden");
});

// ---------- Textarea auto-grow + Enter to send ----------
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 200) + "px";
});
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});
sendBtn.addEventListener("click", send);

// ---------- Helpers ----------
function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}
function renderMarkdown(text) {
  let html = escapeHtml(text);
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\[([^\]]+)\]/g, '<span style="color:#8ab4f8">[$1]</span>');
  return html;
}
function addMessage(role, contentHtml) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;
  const avatar = role === "user"
    ? '<span class="mini-avatar me">B</span> Bạn'
    : '<span class="mini-avatar ai">L</span> LawAI';
  wrap.innerHTML = `<div class="role">${avatar}</div><div class="bubble">${contentHtml}</div>`;
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
  return wrap;
}
function renderSources(sources) {
  if (!sources || !sources.length) return "";
  const cards = sources.map((s) => `
    <div class="source-card">
      <div class="src-head">
        <span>${escapeHtml(s.source)} <em style="color:#9aa0a6">(${escapeHtml(s.type || "")})</em></span>
        <span class="src-score">score: ${s.score}</span>
      </div>
      <div class="src-preview">${escapeHtml(s.preview)}…</div>
    </div>`).join("");
  return `<details class="sources"><summary>📚 Nguồn tham khảo (${sources.length})</summary>${cards}</details>`;
}

// ---------- Send ----------
async function send() {
  const text = input.value.trim();
  if (!text || busy) return;

  busy = true;
  sendBtn.disabled = true;
  greeting.classList.add("hidden");
  chat.classList.add("active");

  addMessage("user", escapeHtml(text));
  input.value = "";
  input.style.height = "auto";

  const botMsg = addMessage("bot", '<div class="typing"><span></span><span></span><span></span></div>');

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: sessionId }),
    });
    const data = await res.json();
    sessionId = data.session_id || sessionId;   // nhớ session để giữ memory
    const answerHtml = renderMarkdown(data.answer || "(không có phản hồi)");
    botMsg.querySelector(".bubble").innerHTML = answerHtml + renderSources(data.sources);
  } catch (err) {
    botMsg.querySelector(".bubble").innerHTML =
      '<span style="color:#d96570">Lỗi kết nối tới máy chủ. Kiểm tra app.py đang chạy.</span>';
  } finally {
    busy = false;
    sendBtn.disabled = false;
    chat.scrollTop = chat.scrollHeight;
  }
}
