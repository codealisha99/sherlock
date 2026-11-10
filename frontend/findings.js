/**
 * Findings tab: summary cards, severity/category filter chips, sorted list.
 * Clicking a finding switches to the Graph tab and selects its node.
 * Requires util.js, api.js, graph.js (selectGraphNode), app.js (showWorkspaceTab).
 */

const findingsEls = {
  summary: document.getElementById("findings-summary"),
  severity: document.getElementById("filter-severity"),
  category: document.getElementById("filter-category"),
  sort: document.getElementById("findings-sort"),
  list: document.getElementById("findings-list"),
  empty: document.getElementById("findings-empty"),
  status: document.getElementById("findings-status"),
};

const findingsState = {
  findings: [],
  summary: {},
  severities: new Set(), // empty set = no filter
  categories: new Set(),
};

const SORTERS = {
  severity: (a, b) => severityRank(a) - severityRank(b) || compareText(a.file_path, b.file_path),
  file: (a, b) => compareText(a.file_path, b.file_path) || (Number(a.line) || 0) - (Number(b.line) || 0),
  category: (a, b) => compareText(a.category, b.category) || severityRank(a) - severityRank(b),
};

function severityRank(finding) {
  const rank = SEVERITY_ORDER.indexOf(finding.severity);
  return rank === -1 ? SEVERITY_ORDER.length : rank;
}

function compareText(a, b) {
  return String(a || "").localeCompare(String(b || ""));
}

function initFindings() {
  findingsEls.sort.addEventListener("change", renderFindingsList);
}

/** Loads findings for a repo. Returns the summary (or null on failure) for the workspace header. */
async function loadFindings(repoId) {
  setStatus(findingsEls.status, "Loading findings...");
  try {
    const data = await API.findings(repoId);
    findingsState.findings = data.findings || [];
    findingsState.summary = data.summary || {};
    findingsState.severities = new Set();
    findingsState.categories = new Set();
    setStatus(findingsEls.status, "");
    renderFindingsSummary();
    renderFilterChips();
    renderFindingsList();
    return findingsState.summary;
  } catch (err) {
    setStatus(findingsEls.status, `Could not load findings: ${err.message}`);
    findingsEls.list.replaceChildren();
    findingsEls.summary.replaceChildren();
    return null;
  }
}

function statCard(label, value, extraClass = "") {
  return el(
    "div",
    { className: `stat-card ${extraClass}`.trim() },
    el("span", { className: "stat-label", text: label }),
    el("strong", { className: "stat-value", text: String(value) }),
  );
}

function renderFindingsSummary() {
  const summary = findingsState.summary;
  const bySeverity = summary.by_severity || {};
  const byCategory = summary.by_category || {};
  const total = summary.total != null ? summary.total : findingsState.findings.length;
  findingsEls.summary.replaceChildren(
    statCard("Score", summary.score != null ? `${summary.score}/100` : "n/a", "stat-score"),
    statCard("Total", total),
    ...SEVERITY_ORDER.map((sev) => statCard(sev, bySeverity[sev] || 0, `sev-${sev}`)),
    ...Object.entries(byCategory).map(([category, count]) => statCard(category, count)),
  );
}

/** Builds toggle chips for the distinct severities and categories present in the findings. */
function renderFilterChips() {
  const categories = [...new Set(findingsState.findings.map((f) => f.category).filter(Boolean))].sort();
  const present = new Set(findingsState.findings.map((f) => f.severity));
  const severities = SEVERITY_ORDER.filter((sev) => present.has(sev));
  fillChipGroup(findingsEls.severity, severities, "severities");
  fillChipGroup(findingsEls.category, categories, "categories");
}

function fillChipGroup(container, values, key) {
  const chips = values.map((value) => {
    const selected = findingsState[key].has(value);
    return el(
      "button",
      {
        className: "chip",
        type: "button",
        "aria-pressed": String(selected),
        text: value,
        on: { click: () => toggleFilter(key, value) },
      },
    );
  });
  container.replaceChildren(...(chips.length ? chips : [el("span", { className: "text-muted", text: "None" })]));
}

/** Immutable toggle: replaces the set instead of mutating it. */
function toggleFilter(key, value) {
  const next = new Set(findingsState[key]);
  if (next.has(value)) next.delete(value);
  else next.add(value);
  findingsState[key] = next;
  renderFilterChips();
  renderFindingsList();
}

function visibleFindings() {
  const { severities, categories } = findingsState;
  return findingsState.findings.filter((f) => (
    (!severities.size || severities.has(f.severity))
    && (!categories.size || categories.has(f.category))
  ));
}

function renderFindingsList() {
  const sorter = SORTERS[findingsEls.sort.value] || SORTERS.severity;
  const items = [...visibleFindings()].sort(sorter);
  findingsEls.empty.classList.toggle("hidden", items.length > 0);
  findingsEls.list.replaceChildren(...items.map((f) => el(
    "li",
    { className: "finding-li" },
    el(
      "button",
      { className: "finding-row", type: "button", on: { click: () => goToFinding(f) } },
      severityBadge(f.severity),
      el(
        "span",
        { className: "finding-main" },
        el("span", { className: "finding-title", text: f.title || f.rule_id || "Finding" }),
        el("span", { className: "finding-loc", text: `${f.file_path || "unknown file"}${f.line ? `:${f.line}` : ""}` }),
      ),
      el("span", { className: "finding-meta" }, f.category, f.cwe ? ` · ${f.cwe}` : ""),
    ),
  )));
}

/** Jump to the Graph tab and select the finding's node (fall back to its file node). */
function goToFinding(finding) {
  showWorkspaceTab("graph");
  selectGraphNode(finding.node_id || `file:${finding.file_path}`);
}
