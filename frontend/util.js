/**
 * Shared helpers for the workspace UI.
 * Everything that comes from the server is rendered with textContent or esc().
 * No inline handlers; event listeners are attached from JS.
 */

const SEVERITY_ORDER = ["critical", "high", "medium", "low", "info"];

// Shared, mutable view state for the open repository (read by graph/findings/chat/report).
const workspaceState = {
  repoId: null,
  repoName: "",
  analysisId: null,
};

/** Escape a value for interpolation into an HTML string template. */
function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Join a list of server strings for display, escaped. */
function escList(values, fallback = "None") {
  const items = (values || []).map((v) => String(v));
  return esc(items.length ? items.join(", ") : fallback);
}

/**
 * Build a DOM element. Props: className, text (textContent), on (event map),
 * dataset (object), or any attribute. Children may be nodes, strings, arrays, null.
 */
function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "className") node.className = value;
    else if (key === "text") node.textContent = String(value);
    else if (key === "on") {
      for (const [type, handler] of Object.entries(value)) node.addEventListener(type, handler);
    } else if (key === "dataset") Object.assign(node.dataset, value);
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function setStatus(node, message) {
  if (node) node.textContent = message || "";
}

/** Returns the normalized URL only when it parses as https; "" otherwise. Every external link goes through this. */
function safeExternalLink(raw) {
  try {
    const url = new URL(String(raw || ""));
    return url.protocol === "https:" ? url.href : "";
  } catch (_err) {
    return "";
  }
}

const toastNode = document.getElementById("toast");
let toastTimer = null;

/** Shows a transient message. content: string or array of nodes; tone: "ok" | "error". */
function showToast(content, tone = "ok") {
  clearTimeout(toastTimer);
  toastNode.className = `toast toast-${tone}`;
  toastNode.replaceChildren(...(Array.isArray(content) ? content : [content]));
  toastTimer = setTimeout(() => toastNode.classList.add("hidden"), 7000);
}

function severityBadge(severity) {
  const level = SEVERITY_ORDER.includes(severity) ? severity : "info";
  return el("span", { className: `sev-badge sev-${level}`, text: level });
}

/** Stacked severity bar segments (flex-grow proportional to count). Empty when there are no findings. */
function severitySegments(bySeverity) {
  return SEVERITY_ORDER.filter((sev) => Number(bySeverity[sev]) > 0).map((sev) => {
    const seg = el("span", { className: `sev-seg sev-${sev}-seg`, title: `${bySeverity[sev]} ${sev}` });
    seg.style.flexGrow = String(Number(bySeverity[sev]));
    return seg;
  });
}

/** Legend items: colour dot, count, severity name. */
function severityLegendItems(bySeverity) {
  return SEVERITY_ORDER.filter((sev) => Number(bySeverity[sev]) > 0).map((sev) => el(
    "li",
    {},
    el("span", { className: `sev-dot sev-${sev}-seg`, "aria-hidden": "true" }),
    el("span", { className: "sev-count", text: String(bySeverity[sev]) }),
    el("span", { text: sev }),
  ));
}

const FOCUSABLE_SELECTOR = [
  "a[href]", "button:not([disabled])", "input:not([disabled])", "select:not([disabled])",
  "textarea:not([disabled])", '[tabindex]:not([tabindex="-1"])',
].join(", ");

/** Visible, enabled focusable descendants of root, in DOM order (for focus trapping). */
function focusableIn(root) {
  return Array.from(root.querySelectorAll(FOCUSABLE_SELECTOR)).filter((node) => node.getClientRects().length > 0);
}

function healthLabel(health) {
  return { good: "Healthy", warn: "Needs attention", bad: "At risk" }[health] || "Unrated";
}

function pluralize(count, word) {
  return `${count} ${word}${count === 1 ? "" : "s"}`;
}
