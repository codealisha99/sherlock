/**
 * Legacy 10-page analysis report (GET /api/analysis/{id}/report), shown only when the
 * repo has an analysis_id. Server strings are escaped with esc(); mermaid sources are
 * set via textContent on the diagram node, never interpolated as HTML.
 * Requires util.js, api.js, app.js (showWorkspaceTab is not needed here).
 */

if (window.mermaid) {
  mermaid.initialize({
    startOnLoad: false,
    theme: "default",
    securityLevel: "strict",
    themeVariables: {
      background: "#fffdf8",
      primaryColor: "#f5f2ea",
      primaryBorderColor: "#15171c",
      primaryTextColor: "#15171c",
      lineColor: "#8a7f6c",
    },
  });
}

const TAB_PAGE_MAP = {
  overview: 1,
  "repository-map": 2,
  architecture: 3,
  routes: 4,
  flows: 5,
  components: 6,
  data: 7,
  gaps: 8,
  learn: 9,
  teacher: 10,
};
const REPORT_PAGE_COUNT = 10;

const reportEls = {
  body: document.getElementById("report-body"),
  title: document.getElementById("report-title"),
  subtitle: document.getElementById("report-subtitle"),
  indicator: document.getElementById("report-indicator"),
  prev: document.getElementById("report-prev"),
  next: document.getElementById("report-next"),
  tabs: document.getElementById("report-tabs"),
  modal: document.getElementById("code-modal"),
  modalPath: document.getElementById("modal-filepath"),
  modalCode: document.getElementById("modal-code-content"),
  modalClose: document.getElementById("btn-close-modal"),
};

const reportState = {
  analysisId: null,
  report: null,
  page: 1,
  pendingMermaid: [],
};

// Element that had focus before the source viewer opened; focus returns there on close.
let modalReturnFocus = null;

function initReport() {
  reportEls.prev.addEventListener("click", () => navigateReport(reportState.page - 1));
  reportEls.next.addEventListener("click", () => navigateReport(reportState.page + 1));
  reportEls.tabs.querySelectorAll("[data-tab]").forEach((tab) => {
    tab.addEventListener("click", () => navigateReport(TAB_PAGE_MAP[tab.dataset.tab]));
  });
  reportEls.body.addEventListener("click", (e) => {
    const link = e.target.closest("[data-file]");
    if (link) openFileViewer(link.dataset.file);
  });
  reportEls.body.addEventListener("keydown", (e) => {
    const link = e.target.closest("[data-file]");
    if (link && (e.key === "Enter" || e.key === " ")) {
      e.preventDefault();
      openFileViewer(link.dataset.file);
    }
  });
  reportEls.modalClose.addEventListener("click", closeFileViewer);
  reportEls.modal.addEventListener("click", (e) => {
    if (e.target === reportEls.modal) closeFileViewer();
  });
  document.addEventListener("keydown", onModalKeydown);
}

/** Called when a different repo is opened. */
function resetReport() {
  reportState.analysisId = null;
  reportState.report = null;
  reportState.page = 1;
  reportEls.body.replaceChildren();
}

/** Loads the report on first view of the Report tab. Cached per analysis id. */
async function ensureReportLoaded() {
  const analysisId = workspaceState.analysisId;
  if (!analysisId) return;
  if (reportState.report && reportState.analysisId === analysisId) {
    renderReportPage(reportState.page);
    return;
  }
  reportState.analysisId = analysisId;
  reportEls.body.replaceChildren(el("p", { className: "text-muted", text: "Loading report..." }));
  try {
    reportState.report = await API.report(analysisId);
    reportState.page = 1;
    renderReportPage(1);
  } catch (err) {
    reportEls.body.replaceChildren(el("p", { className: "form-error", role: "alert", text: `Could not load the report: ${err.message}` }));
  }
}

function navigateReport(page) {
  if (!reportState.report || page < 1 || page > REPORT_PAGE_COUNT) return;
  renderReportPage(page);
}

function renderReportPage(page) {
  reportState.page = page;
  const pages = reportState.report.pages || [];
  const pageData = pages.find((p) => p.page_number === page);
  reportEls.title.textContent = pageData ? pageData.title : `Section ${page}`;
  reportEls.subtitle.textContent = pageData ? pageData.subtitle || "" : "";
  reportEls.indicator.textContent = `Page ${page} of ${REPORT_PAGE_COUNT}`;
  reportEls.prev.disabled = page === 1;
  reportEls.next.disabled = page === REPORT_PAGE_COUNT;
  reportEls.tabs.querySelectorAll("[data-tab]").forEach((tab) => {
    const active = TAB_PAGE_MAP[tab.dataset.tab] === page;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  });

  reportState.pendingMermaid = [];
  const html = pageData ? pageHtml(page, pageData.content || {}) : '<p class="text-muted">This section is not available.</p>';
  reportEls.body.innerHTML = html;
  hydrateMermaid();
}

/** Replaces placeholder slots with .mermaid nodes (textContent) and renders them. */
function hydrateMermaid() {
  reportEls.body.querySelectorAll("[data-mermaid]").forEach((slot) => {
    const source = reportState.pendingMermaid[Number(slot.dataset.mermaid)];
    slot.replaceWith(el("div", { className: "mermaid", text: source }));
  });
  reportState.pendingMermaid = [];
  const nodes = reportEls.body.querySelectorAll(".mermaid");
  if (!nodes.length || !window.mermaid) return;
  mermaid.run({ nodes }).catch((err) => console.warn("Mermaid render error:", err));
}

function mermaidSlot(source) {
  if (!source) return "";
  reportState.pendingMermaid.push(String(source));
  return `<div class="report-diagram" data-mermaid="${reportState.pendingMermaid.length - 1}"></div>`;
}

function fileLink(path) {
  return `<span class="code-link" role="button" tabindex="0" data-file="${esc(path)}">${esc(path)}</span>`;
}

function badgeRow(values, extraClass = "") {
  return (values || []).map((v) => `<span class="badge badge-subtle ${extraClass}">${esc(v)}</span>`).join("");
}

function pageHtml(page, c) {
  const renderers = [null, page1, page2, page3, page4, page5, page6, page7, page8, page9, page10];
  return renderers[page] ? renderers[page](c) : `<pre class="code-block">${esc(JSON.stringify(c, null, 2))}</pre>`;
}

function page1(c) {
  const entryPoints = (c.entry_points || []).map((ep) => `<li>${fileLink(ep)}</li>`).join("");
  return `
    <div class="card">
      <div class="card-title">Core Identity &amp; Purpose</div>
      <p class="prose-lead">${esc(c.summary || "")}</p>
      <div class="badge-row">${badgeRow(c.languages)}${badgeRow(c.frameworks, "badge-framework")}</div>
    </div>
    <div class="card-grid">
      <div class="card"><div class="card-title">Why This Project Exists</div><p>${esc(c.why_it_exists || "Not stated.")}</p></div>
      <div class="card"><div class="card-title">Problem It Solves</div><p>${esc(c.problem_solved || "Not stated.")}</p></div>
    </div>
    <div class="card"><div class="card-title">Important Entry Points</div><ul class="plain-list">${entryPoints}</ul></div>`;
}

function page2(c) {
  const dirs = (c.directories || []).map((d) => `
    <div class="dir-card">
      <strong class="mono">${esc(d.path)}</strong>
      <p>${esc(d.purpose)}</p>
      <div class="dir-key-files">${(d.key_files || []).map((kf) => fileLink(kf)).join("")}</div>
    </div>`).join("");
  return `
    <div class="card"><div class="card-title">Structural Directory Map</div><div class="dir-grid">${dirs}</div></div>
    <div class="card"><div class="card-title">Repository Hierarchy</div><pre class="code-block tall">${esc(c.tree || "No tree representation")}</pre></div>`;
}

function page3(c) {
  const decisions = (c.decisions || []).map((d) => `
    <div class="decision">
      <strong>${esc(d.decision)}</strong>
      <p><span class="text-ok">Rationale:</span> ${esc(d.rationale)}</p>
      <p class="text-muted"><span class="text-warn">Trade-offs:</span> ${esc(d.trade_offs)}</p>
    </div>`).join("");
  return `
    <div class="card"><div class="card-title">High-Level Architecture Topology</div>${mermaidSlot(c.architecture_diagram) || '<p class="text-muted">No diagram provided.</p>'}</div>
    <div class="card"><div class="card-title">Key Architectural Decisions &amp; Rationale</div>${decisions}</div>`;
}

function page4(c) {
  const endpoints = (c.endpoints || []).map((ep) => `
    <div class="endpoint-row"><code>${esc(ep)}</code><span class="badge badge-subtle">Interface</span></div>`).join("");
  return `
    <div class="card"><div class="card-title">Exposed Routes &amp; Interface Endpoints</div>
    <p>${esc(c.summary || "")}</p><div class="endpoint-list">${endpoints}</div></div>`;
}

function page5(c) {
  const flows = (c.flows || []).map((f) => {
    const steps = (f.steps || []).map((s, idx) => `
      <div class="flow-step-item">
        <div class="flow-step-badge">${esc(s.step_number || idx + 1)}</div>
        <div class="flow-step-content">
          <div class="flow-step-layer">${esc(s.layer)}</div>
          <div><strong>${esc(s.component)}</strong> &mdash; <code>${esc(s.function_name)}</code></div>
          <div>${fileLink(s.file || "")}</div>
          <div class="text-muted">${esc(s.description)}</div>
        </div>
      </div>${idx < (f.steps || []).length - 1 ? '<div class="flow-step-arrow">&darr;</div>' : ""}`).join("");
    return `
      <div class="card inset">
        <h3>${esc(f.name)}</h3>
        <p><strong>Trigger:</strong> <code>${esc(f.trigger)}</code></p>
        <p>${esc(f.description)}</p>
        ${mermaidSlot(f.mermaid_diagram)}
        <div class="flow-steps-container">${steps}</div>
      </div>`;
  }).join("");
  return `<div class="card"><div class="card-title">End-to-End Execution Workflows</div>${flows}</div>`;
}

function page6(c) {
  const components = (c.components || []).map((comp) => `
    <div class="card pastel-lavender">
      <strong>${esc(comp.component)}</strong>
      <div class="card-file">${fileLink(comp.file || "")}</div>
      <p><strong>Responsibility:</strong> ${esc(comp.responsibility)}</p>
      <p class="text-muted">Inputs: ${escList(comp.inputs)}</p>
      <p class="text-muted">Outputs: ${escList(comp.outputs)}</p>
      <p class="text-muted">Side effects: ${escList(comp.side_effects)}</p>
      <p class="text-muted">Assumptions: ${escList(comp.assumptions)}</p>
      <p class="text-muted">Failure modes: ${escList(comp.failure_modes)}</p>
    </div>`).join("");
  return `
    <div class="card"><div class="card-title">Component Explorer</div>
    <p class="prose-lead">What core modules actually do, as opposed to what their names suggest.</p>
    ${mermaidSlot(c.mermaid_graph)}
    <div class="card-grid">${components}</div></div>`;
}

function page7(c) {
  const databases = (c.databases || []).map((db) => `
    <div class="dir-card">
      <strong>${esc(db.name)} (${esc(db.type)})</strong>
      <p>${esc(db.purpose)}</p>
      <p class="text-muted">ORM/Driver: ${esc(db.orm_or_driver)}</p>
      <p class="text-muted">Models/Artifacts: ${escList(db.tables_or_models)}</p>
    </div>`).join("");
  const services = (c.external_services || []).map((s) => `
    <div class="dir-card">
      <strong>${esc(s.name)}</strong>
      <p>${esc(s.purpose)}</p>
      <p class="text-muted">Location: <code>${esc(s.call_location)}</code></p>
      <p><strong class="text-warn">Sent:</strong> ${esc(s.sent_data)}</p>
      <p><strong class="text-ok">Received:</strong> ${esc(s.received_data)}</p>
      <p class="text-danger">Failure risks: ${escList(s.failure_possibilities, "None listed")}</p>
    </div>`).join("");
  return `
    <div class="card"><div class="card-title">Data Movement &amp; External Boundaries</div>
    ${mermaidSlot(c.mermaid_diagram)}
    <h3 class="section-subtitle">Databases &amp; Storage</h3><div class="card-grid">${databases}</div>
    <h3 class="section-subtitle">External Services &amp; AI/ML Integrations</h3><div class="card-grid">${services}</div></div>`;
}

function gapClass(category) {
  const lower = String(category || "").toLowerCase();
  if (lower.includes("confirmed")) return "gap-confirmed";
  if (lower.includes("likely")) return "gap-likely";
  if (lower.includes("architectural")) return "gap-architectural";
  if (lower.includes("missing")) return "gap-missing";
  return "gap-improvement";
}

function page8(c) {
  const gaps = (c.gaps || []).map((g) => `
    <div class="card pastel-blush gap-card ${gapClass(g.category)}">
      <div class="gap-card-head">
        <span class="gap-badge ${gapClass(g.category)}">${esc(g.category)}</span>
        <span class="text-muted">Confidence: <strong>${esc(g.confidence)}</strong></span>
      </div>
      <h3 class="gap-title">${esc(g.gap)}</h3>
      <p><strong>Evidence:</strong> <code>${esc(g.evidence)}</code></p>
      <p><strong class="text-warn">Why it matters:</strong> ${esc(g.why_it_matters)}</p>
      <p><strong class="text-danger">Possible impact:</strong> ${esc(g.possible_impact)}</p>
      <div class="card gap-action"><strong class="text-ok">Recommended contribution:</strong> ${esc(g.possible_contribution)}</div>
    </div>`).join("");
  return `
    <div class="card"><div class="card-title">Codebase Gaps, Technical Debt &amp; Risks</div>
    <p class="prose-lead">${esc(c.summary || "Code health, missing validations, test coverage, and architectural bottlenecks.")}</p>
    <div class="gap-list">${gaps}</div></div>`;
}

function page9(c) {
  const concepts = (c.concepts || []).map((con) => `
    <div class="card inset">
      <h3>${esc(con.concept)}</h3>
      <p>${esc(con.definition)}</p>
      <p><strong class="text-ok">Why relevant here:</strong> ${esc(con.why_relevant_here)}</p>
      <p class="text-muted"><strong class="text-blue">Used in:</strong> <code>${esc(con.where_used_in_project)}</code></p>
    </div>`).join("");
  return `<div class="card"><div class="card-title">Conceptual Primitives to Learn</div><div class="card-grid">${concepts}</div></div>`;
}

function page10(c) {
  const reading = (c.first_files_to_read || []).map((f) => `
    <div class="dir-card">
      <span class="flow-step-badge">${esc(f.order)}</span>
      <div>${fileLink(f.file || "")}<p>${esc(f.why_read_first)}</p>
      ${(f.key_symbols_to_watch || []).length ? `<p class="text-blue"><strong>Key symbols:</strong> ${(f.key_symbols_to_watch || []).map((s) => `<code>${esc(s)}</code>`).join(", ")}</p>` : ""}
      </div>
    </div>`).join("");
  const qa = Object.entries(c.answers_to_25_questions || {}).map(([question, answer], idx) => `
    <div class="card pastel-lavender qa-card">
      <span class="qa-num">${idx + 1}</span>
      <strong>${esc(question)}</strong>
      <p>${esc(answer)}</p>
    </div>`).join("");
  return `
    <div class="card"><div class="card-title">Synthesized Mental Model</div>
      <p class="prose-lead">${esc(c.simple_explanation || "")}</p>
      <div class="card inset"><strong>Interview walkthrough</strong><p>${esc(c.interview_explanation || "")}</p></div>
    </div>
    <div class="card"><div class="card-title">Recommended Reading Order</div><div class="reading-list">${reading}</div></div>
    <div class="card"><div class="card-title">The 25 Core Developer Mental Model Questions</div><div class="qa-list">${qa}</div></div>`;
}

// ---------------------------------------------------------------------------
// Code viewer modal (report file links only; graph uses the node panel)
// ---------------------------------------------------------------------------

/** Escape closes; Tab stays inside the dialog (focus trap). */
function onModalKeydown(e) {
  if (reportEls.modal.classList.contains("hidden")) return;
  if (e.key === "Escape") {
    closeFileViewer();
    return;
  }
  if (e.key !== "Tab") return;
  const items = focusableIn(reportEls.modal);
  if (!items.length) return;
  const first = items[0];
  const last = items[items.length - 1];
  if (e.shiftKey && document.activeElement === first) {
    e.preventDefault();
    last.focus();
  } else if (!e.shiftKey && document.activeElement === last) {
    e.preventDefault();
    first.focus();
  }
}

async function openFileViewer(filepath) {
  const analysisId = reportState.analysisId;
  if (!analysisId) return;
  modalReturnFocus = document.activeElement;
  reportEls.modalPath.textContent = `Loading ${filepath}...`;
  reportEls.modalCode.textContent = "Fetching source...";
  reportEls.modal.classList.remove("hidden");
  reportEls.modalClose.focus();
  try {
    const data = await API.file(analysisId, filepath);
    reportEls.modalPath.textContent = data.path + (data.truncated ? " (truncated)" : "");
    reportEls.modalCode.textContent = data.content;
  } catch (err) {
    reportEls.modalCode.textContent = `Error reading file: ${err.message}`;
  }
}

function closeFileViewer() {
  if (reportEls.modal.classList.contains("hidden")) return;
  reportEls.modal.classList.add("hidden");
  if (modalReturnFocus && modalReturnFocus.isConnected) modalReturnFocus.focus();
  modalReturnFocus = null;
}
