/**
 * Shared finding renderers for the Findings tab and the node panel: triage verdict badges,
 * triage explanation / suggested fix, dependency (package) details, aggregated occurrences.
 * Server strings go through textContent (el). External links pass safeExternalLink().
 * Requires util.js.
 */

const VERDICT_LABELS = new Map([
  ["confirmed", "Confirmed"],
  ["likely", "Likely"],
  ["false_positive", "False positive"],
  ["uncertain", "Uncertain"],
]);

const OSV_VULN_URL = "https://osv.dev/vulnerability/";

function confidencePercent(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return null;
  return Math.round(Math.min(1, Math.max(0, n)) * 100);
}

/** Verdict badge with confidence, or null when the finding has no usable triage verdict. */
function triageBadge(finding) {
  const verdict = finding.triage && finding.triage.verdict;
  if (!VERDICT_LABELS.has(verdict)) return null;
  const pct = confidencePercent(finding.triage.confidence);
  const label = pct === null ? VERDICT_LABELS.get(verdict) : `${VERDICT_LABELS.get(verdict)} · ${pct}%`;
  return el("span", { className: `verdict verdict-${verdict}`, text: label });
}

/** Expandable explanation and suggested fix (fix is shown in a pre block). */
function triageDetails(finding) {
  const triage = finding.triage;
  if (!triage) return null;
  return el(
    "div",
    { className: "triage-detail" },
    triage.explanation
      ? el("details", {}, el("summary", { text: "Explanation" }), el("p", { text: triage.explanation }))
      : null,
    triage.fix
      ? el("details", {}, el("summary", { text: "Suggested fix" }), el("pre", { className: "code-block triage-fix" }, triage.fix))
      : null,
  );
}

function upgradeText(fixedVersions) {
  const [first, ...rest] = fixedVersions.map(String);
  return `Upgrade to ≥ ${first}${rest.length ? ` (also fixed in ${rest.join(", ")})` : ""}`;
}

/** Advisory ids as links: a reference URL that mentions the id, else the OSV record. Only https renders as a link. */
function advisoryLinks(finding) {
  const ids = Array.isArray(finding.advisory_ids) ? finding.advisory_ids.map(String) : [];
  const refs = Array.isArray(finding.references) ? finding.references.map(String) : [];
  if (!ids.length) return null;
  return el("ul", { className: "advisory-list" }, ids.map((id) => {
    const href = refs.find((ref) => ref.includes(id)) || `${OSV_VULN_URL}${encodeURIComponent(id)}`;
    const safe = safeExternalLink(href);
    const item = safe
      ? el("a", { href: safe, target: "_blank", rel: "noopener noreferrer", text: id })
      : el("span", { text: id });
    return el("li", {}, item);
  }));
}

/** package@version, ecosystem, advisories, and the fixed version for dependency findings. */
function dependencyDetails(finding) {
  if (finding.category !== "dependency" && !finding.package) return null;
  const pkg = `${finding.package || "unknown package"}${finding.version ? `@${finding.version}` : ""}`;
  const fixed = Array.isArray(finding.fixed_versions) ? finding.fixed_versions : [];
  return el(
    "div",
    { className: "dep-detail" },
    el(
      "p",
      { className: "dep-pkg" },
      el("strong", { text: pkg }),
      finding.ecosystem ? el("span", { className: "badge badge-subtle", text: finding.ecosystem }) : null,
    ),
    advisoryLinks(finding),
    fixed.length ? el("p", { className: "dep-fix", text: upgradeText(fixed) }) : null,
  );
}

/** Aggregated findings: count plus an expandable list of file:line occurrences. onOpen(filePath) selects the file node. */
function occurrencesBlock(finding, onOpen) {
  const occurrences = Array.isArray(finding.occurrences) ? finding.occurrences : [];
  const count = Number(finding.count) || occurrences.length;
  if (count <= 1 && !occurrences.length) return null;
  return el(
    "details",
    { className: "occurrences" },
    el("summary", { text: `${count} occurrence${count === 1 ? "" : "s"}` }),
    occurrences.length
      ? el("ul", { className: "occ-list" }, occurrences.map((occ) => {
        const path = String(occ.file_path || "");
        const label = `${path || "unknown file"}${occ.line ? `:${occ.line}` : ""}`;
        return el("li", {}, el("button", {
          className: "neighbor-btn",
          type: "button",
          text: label,
          disabled: !path,
          on: { click: () => onOpen(path) },
        }));
      }))
      : null,
  );
}
