/**
 * Right-side node detail panel (a bottom sheet on narrow screens): metrics, findings with
 * triage verdicts, neighbors, source, "Ask about this", and "Verify with AI" for the node.
 * Requires util.js, api.js, ai.js, finding-view.js, graph.js (selectGraphNode), chat.js (openChatWithFocus).
 */

const nodePanelEls = {
  panel: document.getElementById("node-panel"),
  content: document.getElementById("node-panel-content"),
  close: document.getElementById("btn-close-node"),
};

const SHEET_QUERY = window.matchMedia("(max-width: 900px)");

// Sequence number so a slow response for an earlier click never overwrites a newer one.
let nodeRequestSeq = 0;
let currentNodeId = null;

function initNodePanel() {
  nodePanelEls.close.addEventListener("click", closeNodePanel);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !nodePanelEls.panel.classList.contains("hidden")) closeNodePanel();
  });
}

function closeNodePanel() {
  nodeRequestSeq += 1;
  currentNodeId = null;
  nodePanelEls.panel.classList.add("hidden");
  nodePanelEls.content.replaceChildren();
}

async function openNodePanel(nodeId) {
  const seq = ++nodeRequestSeq;
  currentNodeId = nodeId;
  nodePanelEls.panel.classList.remove("hidden");
  nodePanelEls.content.replaceChildren(el("p", { className: "text-muted", text: "Loading node..." }));
  try {
    const detail = await API.node(workspaceState.repoId, nodeId);
    if (seq !== nodeRequestSeq) return;
    nodePanelEls.content.replaceChildren(renderNodeDetail(detail));
    // On narrow screens the sheet overlays the graph, so move focus into it.
    if (SHEET_QUERY.matches) nodePanelEls.close.focus({ preventScroll: true });
  } catch (err) {
    if (seq !== nodeRequestSeq) return;
    nodePanelEls.content.replaceChildren(
      el("p", { className: "form-error", role: "alert", text: `Could not load this node: ${err.message}` }),
    );
  }
}

/** Re-renders the open node after a triage run so verdict badges stay current. */
function refreshOpenNodePanel() {
  if (currentNodeId && !nodePanelEls.panel.classList.contains("hidden")) openNodePanel(currentNodeId);
}

function humanize(key) {
  const text = String(key).replace(/_/g, " ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function renderNodeDetail(detail) {
  const node = detail.node || {};
  const findings = detail.findings || [];
  const neighbors = detail.neighbors || {};
  const location = node.file_path ? `${node.file_path}${node.line ? `:${node.line}` : ""}` : "";

  return el(
    "div",
    { className: "node-detail" },
    el("h2", { className: "node-title", text: node.label || node.id || "Node" }),
    el(
      "div",
      { className: "node-tags" },
      el("span", { className: "badge badge-subtle", text: node.type || "node" }),
      el("span", { className: `health-tag health-${node.health || "none"}`, text: healthLabel(node.health) }),
      node.language ? el("span", { className: "badge badge-subtle", text: `Language: ${node.language}` }) : null,
    ),
    location ? el("p", { className: "node-path", text: location }) : null,
    renderMetrics(node.metrics || {}),
    el(
      "button",
      {
        className: "btn btn-primary btn-sm",
        type: "button",
        on: {
          click: () => openChatWithFocus({
            node_id: node.id,
            file_path: node.file_path,
            symbol: node.label,
          }),
        },
      },
      "Ask about this",
    ),
    renderNodeFindings(findings),
    renderNeighborList("Depends on (outgoing)", neighbors.outgoing || []),
    renderNeighborList("Used by (incoming)", neighbors.incoming || []),
    renderSource(detail.source, findings),
  );
}

function renderMetrics(metrics) {
  const entries = Object.entries(metrics);
  if (!entries.length) return null;
  return el(
    "dl",
    { className: "metric-grid" },
    entries.flatMap(([key, value]) => [
      el("dt", { text: humanize(key) }),
      el("dd", { text: String(value) }),
    ]),
  );
}

/** "Verify with AI" for this node's findings that have no verdict yet. */
function nodeVerifyControls(findings) {
  const pending = findings.filter((f) => f.id && !f.triage).map((f) => f.id);
  const available = aiAvailable();
  const label = pending.length ? `Verify with AI (${pending.length})` : "Nothing new to verify";
  return el(
    "div",
    { className: "node-verify" },
    el("button", {
      className: "btn btn-secondary btn-sm",
      type: "button",
      text: label,
      disabled: !available || !pending.length,
      title: available ? undefined : AI_UNAVAILABLE_TITLE,
      on: { click: (e) => verifyFindingIds(pending, e.currentTarget) },
    }),
    available ? null : el("p", { className: "small text-muted" }, aiUnavailableNodes()),
  );
}

function renderNodeFindings(findings) {
  const heading = el("h3", { text: `Findings (${findings.length})` });
  if (!findings.length) {
    return el("section", { className: "node-section" }, heading, el("p", { className: "text-muted", text: "No findings for this node." }));
  }
  return el(
    "section",
    { className: "node-section" },
    heading,
    nodeVerifyControls(findings),
    el("ul", { className: "finding-stack" }, findings.map((f) => el(
      "li",
      { className: "finding-item" },
      el(
        "div",
        { className: "finding-head" },
        severityBadge(f.severity),
        el("strong", { text: f.title || f.rule_id || "Finding" }),
        f.cwe ? el("span", { className: "cwe-tag", text: f.cwe }) : null,
        triageBadge(f),
      ),
      f.line ? el("p", { className: "text-muted", text: `Line ${f.line}` }) : null,
      f.description ? el("p", { text: f.description }) : null,
      f.recommendation ? el("p", { className: "finding-fix" }, el("strong", { text: "Fix: " }), f.recommendation) : null,
      dependencyDetails(f),
      triageDetails(f),
    ))),
  );
}

function renderNeighborList(title, items) {
  return el(
    "section",
    { className: "node-section" },
    el("h3", { text: `${title} (${items.length})` }),
    items.length
      ? el("ul", { className: "neighbor-list" }, items.map((item) => el(
        "li",
        {},
        el(
          "button",
          {
            className: "neighbor-btn",
            type: "button",
            on: { click: () => selectGraphNode(item.id) },
          },
          el("span", { text: item.label || item.id }),
          el("span", { className: "neighbor-meta", text: [item.type, item.edge].filter(Boolean).join(" · ") }),
        ),
      )))
      : el("p", { className: "text-muted", text: "None" }),
  );
}

function renderSource(source, findings) {
  const heading = el("h3", { text: "Source" });
  const lines = (source && source.lines) || [];
  if (!lines.length) {
    return el("section", { className: "node-section" }, heading, el("p", { className: "text-muted", text: "Source not available." }));
  }
  const hitLines = new Set(findings.map((f) => f.line).filter(Boolean));
  return el(
    "section",
    { className: "node-section" },
    heading,
    el("p", { className: "text-muted", text: `${source.path} · lines ${source.start_line}-${source.end_line}` }),
    el(
      "div",
      { className: "code-view", role: "region", "aria-label": `Source of ${source.path}`, tabindex: "0" },
      lines.map((line) => el(
        "span",
        { className: hitLines.has(line.line) ? "code-line hit" : "code-line" },
        el("span", { className: "ln", text: String(line.line) }),
        el("span", { className: "code-text", text: line.content }),
      )),
    ),
  );
}
