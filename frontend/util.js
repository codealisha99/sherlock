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

function severityBadge(severity) {
  const level = SEVERITY_ORDER.includes(severity) ? severity : "info";
  return el("span", { className: `sev-badge sev-${level}`, text: level });
}

function healthLabel(health) {
  return { good: "Healthy", warn: "Needs attention", bad: "At risk" }[health] || "Unrated";
}

function pluralize(count, word) {
  return `${count} ${word}${count === 1 ? "" : "s"}`;
}
