/**
 * AI capabilities (GET /api/capabilities) and finding verification
 * (POST /api/repos/{id}/triage). Verify actions are disabled with a link to
 * Settings when neither the caller's BYOK key nor the platform model is available.
 * Requires util.js (el, showToast, workspaceState), api.js, findings.js (reloadFindings,
 * renderVerifyControls, renderFindingsList), node-panel.js (refreshOpenNodePanel), chat.js.
 */

const AI_UNAVAILABLE_TITLE = "Add your own API key in Settings to verify findings with AI.";

// null = capabilities could not be loaded. Actions stay enabled and the server enforces limits.
const aiState = { capabilities: null };

async function loadCapabilities() {
  try {
    aiState.capabilities = await API.capabilities();
  } catch (_err) {
    // Deliberate fallback: unknown capabilities must not block the UI; a 409 from triage is shown on click.
    aiState.capabilities = null;
  }
}

/** False only when capabilities are known and neither BYOK nor the platform model is available. */
function aiAvailable() {
  const caps = aiState.capabilities;
  if (!caps) return true;
  return Boolean(caps.platform_model || caps.byok);
}

/** Inline hint shown beside disabled AI actions, with a link to the API key section. */
function aiUnavailableNodes() {
  return [
    el("span", { text: `${AI_UNAVAILABLE_TITLE} ` }),
    el("a", { href: "/settings#keys", text: "Open API keys" }),
  ];
}

/** Maps triage failures to user-facing toast content. */
function triageErrorNodes(err) {
  if (!(err instanceof ApiError)) return ["Network error. Check your connection and try again."];
  if (err.status === 402) {
    return [err.message, " ", el("a", { href: "/settings?upgrade=pro", text: "Upgrade plan" })];
  }
  if (err.status === 409) {
    return ["No AI model is available. ", el("a", { href: "/settings#keys", text: "Add your API key" }), " to verify findings."];
  }
  if (err.status === 429) return ["Too many verification requests. Wait a minute and try again."];
  return [err.message];
}

function verdictSummary(result) {
  const findings = result.findings || [];
  const falsePositives = findings.filter((f) => f.triage && f.triage.verdict === "false_positive").length;
  const triaged = result.triaged ?? 0;
  if (triaged === 0) {
    return "Nothing needed verification. AI checks security, secret and dependency findings rated medium or higher that haven't been verified yet.";
  }
  const noun = triaged === 1 ? "finding" : "findings";
  return `Verified ${triaged} ${noun}: ${falsePositives} false positive${falsePositives === 1 ? "" : "s"}.`;
}

function setBusy(button, busy) {
  if (!button) return;
  button.disabled = busy || !aiAvailable();
  button.setAttribute("aria-busy", String(busy));
}

/**
 * Runs triage for the open repo. body undefined = server default selection.
 * On success refreshes findings and the open node panel; errors become a toast.
 */
async function runTriage(body, button) {
  const repoId = workspaceState.repoId;
  if (!repoId) return null;
  const idleLabel = button ? button.textContent : "";
  if (button) button.textContent = "Verifying...";
  setBusy(button, true);
  try {
    const result = await API.triage(repoId, body);
    if (workspaceState.repoId !== repoId) return null; // user navigated away mid-request
    showToast(verdictSummary(result), "ok");
    await reloadFindings();
    refreshOpenNodePanel();
    return result;
  } catch (err) {
    showToast(triageErrorNodes(err), "error");
    return null;
  } finally {
    if (button) button.textContent = idleLabel;
    setBusy(button, false);
  }
}

/** Verifies explicit finding ids (per-finding and per-node actions). */
function verifyFindingIds(findingIds, button) {
  return runTriage({ finding_ids: findingIds }, button);
}

/** Per-finding action. Re-verify when the finding already has a verdict. */
function verifyFindingButton(finding) {
  const available = aiAvailable();
  return el("button", {
    className: "btn btn-secondary btn-sm",
    type: "button",
    text: finding.triage ? "Re-verify" : "Verify with AI",
    disabled: !available || !finding.id,
    title: available ? undefined : AI_UNAVAILABLE_TITLE,
    on: { click: (e) => verifyFindingIds([finding.id], e.currentTarget) },
  });
}

/** Called once capabilities load: refreshes every control that depends on them. */
function refreshAiControls() {
  renderVerifyControls();
  if (findingsState.findings.length) renderFindingsList();
  renderModelBanner();
}
