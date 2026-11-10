/**
 * Findings tab: summary cards, severity/category/triage filters, sorted list, and AI verification.
 * Clicking a finding's row switches to the Graph tab and selects its node.
 * Requires util.js, api.js, ai.js, finding-view.js, graph.js (selectGraphNode),
 * app.js (showWorkspaceTab, renderWorkspaceHeader).
 */

const findingsEls = {
  summary: document.getElementById("findings-summary"),
  severity: document.getElementById("filter-severity"),
  category: document.getElementById("filter-category"),
  sort: document.getElementById("findings-sort"),
  list: document.getElementById("findings-list"),
  empty: document.getElementById("findings-empty"),
  status: document.getElementById("findings-status"),
  verifyAll: document.getElementById("btn-verify-all"),
  verifyHint: document.getElementById("verify-hint"),
  hideFalsePositives: document.getElementById("filter-hide-fp"),
  hideLowConfidence: document.getElementById("filter-hide-lowconf"),
};

const findingsState = {
  findings: [],
  summary: {},
  severities: new Set(), // empty set = no filter
  categories: new Set(),
  triageFilters: { hideFalsePositives: true, hideLowConfidence: false },
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
  findingsEls.verifyAll.addEventListener("click", (e) => runTriage(undefined, e.currentTarget));
  findingsEls.hideFalsePositives.addEventListener("click", () => toggleTriageFilter("hideFalsePositives"));
  findingsEls.hideLowConfidence.addEventListener("click", () => toggleTriageFilter("hideLowConfidence"));
  renderTriageChips();
  renderVerifyControls();
}

/** Loads findings for a repo (resets severity/category filters). Returns the summary, or null on failure. */
async function loadFindings(repoId) {
  setStatus(findingsEls.status, "Loading findings...");
  findingsState.severities = new Set();
  findingsState.categories = new Set();
  try {
    const data = await API.findings(repoId);
    setStatus(findingsEls.status, "");
    return applyFindings(data);
  } catch (err) {
    setStatus(findingsEls.status, `Could not load findings: ${err.message}`);
    findingsState.findings = [];
    findingsEls.list.replaceChildren();
    findingsEls.summary.replaceChildren();
    return null;
  }
}

/** Re-fetches findings after a triage run. Keeps the active filters; refreshes the header score too. */
async function reloadFindings() {
  const repoId = workspaceState.repoId;
  if (!repoId) return;
  try {
    const data = await API.findings(repoId);
    if (workspaceState.repoId !== repoId) return; // user switched repos meanwhile
    renderWorkspaceHeader(applyFindings(data));
  } catch (err) {
    setStatus(findingsEls.status, `Could not refresh findings: ${err.message}`);
  }
}

function applyFindings(data) {
  findingsState.findings = data.findings || [];
  findingsState.summary = data.summary || {};
  renderFindingsSummary();
  renderFilterChips();
  renderFindingsList();
  return findingsState.summary;
}

function statCard(label, value, extraClass = "") {
  return el(
    "div",
    { className: `stat-card ${extraClass}`.trim() },
    el("span", { className: "stat-label", text: label }),
    el("strong", { className: "stat-value", text: String(value) }),
  );
}

/** Dependency CVE summary; shows the server's error text when the lookup failed. */
function dependencyCard(deps) {
  const card = (value, sub) => el(
    "div",
    { className: "stat-card stat-dep" },
    el("span", { className: "stat-label", text: "Dependencies" }),
    el("strong", { className: "stat-value", text: value }),
    el("span", { className: "stat-sub", text: sub }),
  );
  if (!deps) return card("n/a", "Not scanned");
  if (deps.error) return card("Unavailable", String(deps.error));
  const ecosystems = (deps.ecosystems || []).join(", ") || "no supported manifests";
  return card(`${deps.vulnerable ?? 0} of ${deps.scanned ?? 0}`, `vulnerable · ${ecosystems}`);
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
    dependencyCard(summary.dependencies),
    statCard("Triaged", summary.triaged ?? 0),
    statCard("False positives", summary.false_positives ?? 0),
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

/** Immutable toggle of a triage filter chip (hide false positives / hide low confidence). */
function toggleTriageFilter(key) {
  findingsState.triageFilters = { ...findingsState.triageFilters, [key]: !findingsState.triageFilters[key] };
  renderTriageChips();
  renderFindingsList();
}

function renderTriageChips() {
  const { hideFalsePositives, hideLowConfidence } = findingsState.triageFilters;
  findingsEls.hideFalsePositives.setAttribute("aria-pressed", String(hideFalsePositives));
  findingsEls.hideLowConfidence.setAttribute("aria-pressed", String(hideLowConfidence));
}

/** Top-level "Verify with AI": disabled with a link to Settings when no model is available. */
function renderVerifyControls() {
  const available = aiAvailable();
  findingsEls.verifyAll.disabled = !available;
  findingsEls.verifyAll.title = available ? "" : AI_UNAVAILABLE_TITLE;
  findingsEls.verifyHint.replaceChildren(...(available ? [] : aiUnavailableNodes()));
}

function visibleFindings() {
  const { severities, categories } = findingsState;
  const { hideFalsePositives, hideLowConfidence } = findingsState.triageFilters;
  return findingsState.findings.filter((f) => (
    (!severities.size || severities.has(f.severity))
    && (!categories.size || categories.has(f.category))
    && !(hideFalsePositives && f.triage && f.triage.verdict === "false_positive")
    && !(hideLowConfidence && f.confidence === "low")
  ));
}

function renderFindingsList() {
  const sorter = SORTERS[findingsEls.sort.value] || SORTERS.severity;
  const items = [...visibleFindings()].sort(sorter);
  findingsEls.empty.classList.toggle("hidden", items.length > 0);
  findingsEls.list.replaceChildren(...items.map(findingItem));
}

function locationLabel(finding) {
  const count = Number(finding.count) || 0;
  if (count > 1) return `${count} occurrences`;
  return `${finding.file_path || "unknown file"}${finding.line ? `:${finding.line}` : ""}`;
}

/** Graph target for a finding: its node, else its file node. Empty when repo-level (no node, no file). */
function findingTarget(finding) {
  if (finding.node_id) return finding.node_id;
  return finding.file_path ? `file:${finding.file_path}` : "";
}

/** One finding: a row that jumps to its node (when it has one), then details, triage, and verify. */
function findingItem(finding) {
  const navigable = Boolean(findingTarget(finding));
  const row = el(
    navigable ? "button" : "div",
    {
      className: navigable ? "finding-row" : "finding-row finding-row-static",
      type: navigable ? "button" : undefined,
      on: navigable ? { click: () => goToFinding(finding) } : undefined,
    },
    severityBadge(finding.severity),
    el(
      "span",
      { className: "finding-main" },
      el("span", { className: "finding-title", text: finding.title || finding.rule_id || "Finding" }),
      el("span", { className: "finding-loc", text: locationLabel(finding) }),
    ),
    el(
      "span",
      { className: "finding-side" },
      triageBadge(finding),
      el("span", { className: "finding-meta", text: `${finding.category || ""}${finding.cwe ? ` · ${finding.cwe}` : ""}` }),
    ),
  );
  return el(
    "li",
    { className: "finding-li" },
    row,
    el(
      "div",
      { className: "finding-extra" },
      dependencyDetails(finding),
      occurrencesBlock(finding, (path) => goToFinding({ file_path: path })),
      triageDetails(finding),
      verifyFindingButton(finding),
    ),
  );
}

/** Jump to the Graph tab and select the finding's node (fall back to its file node). No-op without a target. */
function goToFinding(finding) {
  const target = findingTarget(finding);
  if (!target) return;
  showWorkspaceTab("graph");
  selectGraphNode(target);
}
