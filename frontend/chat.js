/**
 * Repo chat widget. Sessions via POST /api/chat/sessions; messages via
 * POST /api/chat/sessions/{id}/messages streamed as SSE (`data: {...}\n\n`).
 * Event types: progress (stage), status, token, done (answer, citations, abstained,
 * abstention_reason), error (message). Answers are rendered with textContent only.
 * Requires util.js, api.js (ApiError, seg), ai.js (aiAvailable), graph.js (selectGraphNode),
 * app.js (showWorkspaceTab).
 */

const chatEls = {
  widget: document.getElementById("chat-widget"),
  bubble: document.getElementById("chat-bubble"),
  panel: document.getElementById("chat-panel"),
  title: document.getElementById("chat-session-title"),
  messages: document.getElementById("chat-messages"),
  input: document.getElementById("chat-input"),
  send: document.getElementById("chat-send"),
  status: document.getElementById("chat-status"),
  close: document.getElementById("chat-close"),
  newChat: document.getElementById("chat-new-chat"),
  focusChip: document.getElementById("chat-focus"),
  focusLabel: document.getElementById("chat-focus-label"),
  focusClear: document.getElementById("chat-focus-clear"),
  modelBanner: document.getElementById("chat-model-banner"),
};

const chatState = {
  repoId: null,
  sessionId: null,
  streaming: false,
  focus: null, // {node_id?, file_path?, symbol?}
};

const CHAT_HTTP_FALLBACK = {
  400: "This question was blocked or invalid. Rephrase it about the code.",
  402: "Your plan's chat limit has been reached.",
  429: "Too many messages. Wait a moment and try again.",
};

const PROGRESS_TEXT = {
  validation: "Checking your question…",
  retrieval: "Searching the codebase…",
  retrieval_complete: "Found relevant code…",
  context_assembly: "Gathering context…",
  context_assembly_complete: "Reading relevant snippets…",
  generation: "Writing answer…",
};

const ABSTAIN_FALLBACK = "Sherlock could not find enough evidence in this repository to answer.";

function initChat() {
  chatEls.bubble.addEventListener("click", () => {
    if (chatEls.panel.classList.contains("hidden")) openChatPanel();
    else closeChatPanel();
  });
  chatEls.close.addEventListener("click", () => {
    closeChatPanel();
    chatEls.bubble.focus();
  });
  chatEls.newChat.addEventListener("click", startNewChat);
  chatEls.send.addEventListener("click", sendMessage);
  chatEls.focusClear.addEventListener("click", () => setFocus(null));
  chatEls.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      sendMessage();
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape" || chatEls.panel.classList.contains("hidden")) return;
    closeChatPanel();
    chatEls.bubble.focus(); // Esc returns focus to the bubble that opened the chat
  });
  renderEmptyHint();
  renderModelBanner();
}

function renderEmptyHint() {
  chatEls.messages.replaceChildren(
    el("div", { className: "chat-empty" }, el("p", { text: "Ask anything about this repository." })),
  );
}

/** Shows a subtle notice in the chat panel when no AI model is available (no BYOK key, no platform model). */
function renderModelBanner() {
  chatEls.modelBanner.hidden = aiAvailable();
}

/** Called when a repo is opened (repoId) or the workspace closes (null). */
function setChatRepo(repoId) {
  chatState.repoId = repoId;
  chatState.sessionId = null;
  chatState.streaming = false;
  chatState.focus = null;
  chatEls.title.textContent = "New chat";
  chatEls.send.disabled = false;
  chatEls.status.textContent = "";
  renderEmptyHint();
  renderFocus();
  renderModelBanner();
  chatEls.widget.classList.toggle("hidden", !repoId);
  closeChatPanel();
}

function openChatPanel() {
  chatEls.panel.classList.remove("hidden");
  chatEls.bubble.setAttribute("aria-expanded", "true");
  chatEls.input.focus();
}

function closeChatPanel() {
  chatEls.panel.classList.add("hidden");
  chatEls.bubble.setAttribute("aria-expanded", "false");
}

/** Opens the chat focused on a node (from the node panel's "Ask about this"). */
function openChatWithFocus(node) {
  const focus = {};
  if (node.node_id) focus.node_id = node.node_id;
  if (node.file_path) focus.file_path = node.file_path;
  if (node.symbol) focus.symbol = node.symbol;
  setFocus(Object.keys(focus).length ? focus : null);
  openChatPanel();
}

function setFocus(focus) {
  chatState.focus = focus;
  renderFocus();
}

function renderFocus() {
  const focus = chatState.focus;
  chatEls.focusChip.classList.toggle("hidden", !focus);
  if (focus) chatEls.focusLabel.textContent = `Focused: ${focus.symbol || focus.file_path || focus.node_id}`;
}

function startNewChat() {
  chatState.sessionId = null;
  chatEls.title.textContent = "New chat";
  chatEls.status.textContent = "";
  renderEmptyHint();
}

async function ensureChatSession() {
  if (chatState.sessionId) return chatState.sessionId;
  const data = await API.createChatSession(chatState.repoId);
  chatState.sessionId = data.session_id;
  chatEls.title.textContent = data.title || "Repository chat";
  return chatState.sessionId;
}

function setStreaming(on) {
  chatState.streaming = on;
  chatEls.send.disabled = on;
  chatEls.status.textContent = on ? "Sherlock is thinking..." : "";
}

function appendMessage(role, text) {
  chatEls.messages.querySelector(".chat-empty")?.remove();
  const bubble = el("div", { className: "chat-message-bubble" });
  if (text) bubble.textContent = text;
  const message = el("div", { className: `chat-message ${role}` }, bubble);
  chatEls.messages.append(message);
  chatEls.messages.scrollTop = chatEls.messages.scrollHeight;
  return message;
}

async function sendMessage() {
  const question = chatEls.input.value.trim();
  if (!question || chatState.streaming || !chatState.repoId) return;
  chatEls.input.value = "";
  appendMessage("user", question);
  const reply = appendMessage("assistant", "");
  setStreaming(true);
  try {
    const sessionId = await ensureChatSession();
    const body = { question };
    if (chatState.focus) body.focus = chatState.focus;
    const res = await fetch(`/api/chat/sessions/${seg(sessionId)}/messages`, {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw await chatHttpError(res);
    await consumeChatStream(res, reply);
  } catch (err) {
    renderReplyError(reply, err);
  } finally {
    setStreaming(false);
  }
}

/** HTTP failures become ApiError with the server's {"error"} message and parsed body (for `code`). */
async function chatHttpError(res) {
  let data = null;
  try {
    data = await res.json();
  } catch (_err) {
    data = null; // non-JSON body: fall back to the status message
  }
  const message = (data && data.error) || CHAT_HTTP_FALLBACK[res.status] || `Request failed (${res.status}).`;
  return new ApiError(res.status, message, data);
}

function isNoModelError(err) {
  return err instanceof ApiError && err.status === 409 && Boolean(err.body) && err.body.code === "no_model";
}

/** Shows the error in the reply bubble. Only fetch-level failures count as "network" errors. */
function renderReplyError(reply, err) {
  const bubble = reply.querySelector(".chat-message-bubble");
  reply.classList.add("error");
  if (isNoModelError(err)) {
    bubble.replaceChildren(
      el("p", { className: "chat-notice", text: "Chat needs an AI model before it can answer." }),
      el("a", { className: "chat-upgrade", href: "/settings#keys", text: "Add your API key to start chatting" }),
    );
    return;
  }
  const message = err instanceof ApiError ? err.message : "Network error. Check your connection and try again.";
  const nodes = [el("p", { className: "chat-error", role: "alert", text: message })];
  if (err instanceof ApiError && err.status === 402) {
    nodes.push(el("a", { className: "chat-upgrade", href: "/settings?upgrade=pro", text: "Upgrade plan" }));
  }
  bubble.replaceChildren(...nodes);
}

/** Parses one SSE event block (lines of `data: ...`). Returns null when there is no JSON payload. */
function parseSseEvent(block) {
  const payload = block
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).replace(/^ /, ""))
    .join("\n");
  if (!payload) return null;
  try {
    return JSON.parse(payload);
  } catch (err) {
    console.warn("Skipping malformed chat event:", err);
    return null;
  }
}

/** Friendly status text for a progress stage; unknown stages get a generic line. */
function progressText(event) {
  const count = Number(event.num_chunks ?? event.count);
  if (event.stage === "context_assembly_complete" && Number.isFinite(count) && count > 0) {
    return `Reading ${count} relevant snippets…`;
  }
  return PROGRESS_TEXT[event.stage] || "Working on your answer…";
}

/**
 * Applies one SSE event to the reply. Returns the updated answer text and whether the stream finished.
 * An `error` event throws ApiError(0, message): a stream failure, shown with the server's message.
 */
function applyChatEvent(event, reply, answer) {
  switch (event.type) {
    case "progress":
      chatEls.status.textContent = progressText(event);
      return { answer, finished: false };
    case "status":
      chatEls.status.textContent = event.message || event.text || "Sherlock is thinking...";
      return { answer, finished: false };
    case "token": {
      const next = answer + (event.text || "");
      renderAnswer(reply, next);
      return { answer: next, finished: false };
    }
    case "done": {
      const final = event.answer != null ? event.answer : answer;
      chatEls.status.textContent = "";
      renderAnswer(reply, final);
      renderAbstention(reply, event);
      renderCitations(reply, event.citations || []);
      return { answer: final, finished: true };
    }
    case "error":
      throw new ApiError(0, event.message || event.error || "The answer could not be generated.");
    default:
      return { answer, finished: false };
  }
}

async function consumeChatStream(res, reply) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let answer = "";
  let finished = false;

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop();
    for (const block of blocks) {
      const event = parseSseEvent(block);
      if (!event) continue;
      const outcome = applyChatEvent(event, reply, answer);
      answer = outcome.answer;
      finished = finished || outcome.finished;
    }
  }
  if (!finished) throw new ApiError(0, "The connection closed before the answer finished.");
}

/** Muted note for abstained answers (no answer text): the server's reason, or a generic line. */
function renderAbstention(reply, event) {
  if (!event.abstained) return;
  const reason = typeof event.abstention_reason === "string" && event.abstention_reason.trim()
    ? event.abstention_reason
    : ABSTAIN_FALLBACK;
  reply.querySelector(".chat-message-bubble").append(el("p", { className: "chat-abstain", text: reason }));
}

/** Renders answer text safely: plain paragraphs and ``` fenced blocks as <pre><code>. */
function renderAnswer(reply, text) {
  const bubble = reply.querySelector(".chat-message-bubble");
  const parts = text.split(/```[^\n]*\n?/);
  const nodes = parts.flatMap((part, index) => {
    if (index % 2 === 1) {
      return [el("pre", { className: "chat-code" }, el("code", { text: part.replace(/\n$/, "") }))];
    }
    return part
      .split(/\n{2,}/)
      .filter((para) => para.trim())
      .map((para) => el("p", { className: "chat-para", text: para.trim() }));
  });
  bubble.replaceChildren(...nodes);
}

function renderCitations(reply, citations) {
  if (!citations.length) return;
  const row = el(
    "div",
    { className: "citations" },
    citations.map((c) => el("button", {
      className: "citation-chip",
      type: "button",
      text: `${c.source_id ? `${c.source_id} ` : ""}${c.file_path}${c.start_line ? `:${c.start_line}` : ""}`,
      on: { click: () => openCitation(c) },
    })),
  );
  reply.querySelector(".chat-message-bubble").append(row);
}

/** Citation click: jump to the Graph tab and select the cited node (file node by default). */
function openCitation(citation) {
  if (!citation.file_path) return;
  showWorkspaceTab("graph");
  selectGraphNode(citation.node_id || `file:${citation.file_path}`);
}
