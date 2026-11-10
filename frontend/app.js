/**
 * Sherlock workspace (/app): auth gate, start screen, job progress, repo workspace shell.
 * Load order (index.html): util.js, api.js, node-panel.js, graph.js, findings.js,
 * chat.js, report.js, app.js. Every module exposes init* functions called from boot().
 */

const POLL_INTERVAL_MS = 1500;
const STAGES = ["cloning", "indexing", "embedding", "scanning", "graphing", "done"];
const STAGE_LABELS = {
  cloning: "Cloning",
  indexing: "Indexing",
  embedding: "Embedding",
  scanning: "Scanning",
  graphing: "Graphing",
  done: "Done",
};

const appEls = {
  globalError: document.getElementById("global-error"),
  authLoading: document.getElementById("auth-loading"),
  userEmail: document.getElementById("user-email"),
  planBadge: document.getElementById("plan-badge"),
  logout: document.getElementById("btn-logout"),
  views: {
    start: document.getElementById("view-start"),
    progress: document.getElementById("view-progress"),
    workspace: document.getElementById("view-workspace"),
  },
  analyzeForm: document.getElementById("analyze-form"),
  urlInput: document.getElementById("repo-url-input"),
  analyzeBtn: document.getElementById("btn-analyze"),
  analyzeError: document.getElementById("analyze-error"),
  repoList: document.getElementById("repo-list"),
  reposEmpty: document.getElementById("repos-empty"),
  jobList: document.getElementById("job-list"),
  jobsEmpty: document.getElementById("jobs-empty"),
  progressUrl: document.getElementById("progress-url"),
  stepper: document.getElementById("stage-stepper"),
  progressTrack: document.getElementById("progress-track"),
  progressFill: document.getElementById("progress-fill"),
  progressStatus: document.getElementById("progress-status"),
  progressError: document.getElementById("progress-error"),
  progressErrorText: document.getElementById("progress-error-text"),
  retryBtn: document.getElementById("btn-retry"),
  backFromProgress: document.getElementById("btn-back-start"),
  repoName: document.getElementById("repo-name"),
  repoUrl: document.getElementById("repo-url"),
  repoScore: document.getElementById("repo-score"),
  severityRow: document.getElementById("severity-counts"),
  backToRepos: document.getElementById("btn-back-repos"),
  wsTabs: Array.from(document.querySelectorAll(".ws-tab")),
  panels: {
    graph: document.getElementById("panel-graph"),
    findings: document.getElementById("panel-findings"),
    report: document.getElementById("panel-report"),
  },
  tabReport: document.getElementById("tab-report"),
};

const progressState = { jobId: null, timer: null, githubUrl: "" };
const listState = { jobs: [] };

document.addEventListener("DOMContentLoaded", boot);

function boot() {
  initGraph();
  initNodePanel();
  initFindings();
  initChat();
  initReport();

  appEls.analyzeForm.addEventListener("submit", onAnalyzeSubmit);
  appEls.retryBtn.addEventListener("click", retryAnalysis);
  appEls.backFromProgress.addEventListener("click", backToStart);
  appEls.backToRepos.addEventListener("click", backToStart);
  appEls.logout.addEventListener("click", logout);
  appEls.wsTabs.forEach((tab) => {
    tab.addEventListener("click", () => showWorkspaceTab(tab.dataset.view));
  });
  checkSession();
}

function showGlobalError(message) {
  appEls.globalError.textContent = message || "";
  appEls.globalError.classList.toggle("hidden", !message);
}

function showView(name) {
  Object.entries(appEls.views).forEach(([key, node]) => node.classList.toggle("hidden", key !== name));
}

// ---------------------------------------------------------------------------
// Auth gate
// ---------------------------------------------------------------------------

function currentRepoParam() {
  return new URLSearchParams(location.search).get("repo") || "";
}

function redirectToLogin() {
  const repo = currentRepoParam();
  const next = "/app" + (repo ? `?repo=${encodeURIComponent(repo)}` : "");
  location.replace(`/login?next=${encodeURIComponent(next)}`);
}

async function checkSession() {
  let session;
  try {
    session = await API.me();
  } catch (err) {
    if (err.status === 401) {
      redirectToLogin();
      return;
    }
    setStatus(appEls.authLoading, `Could not verify your session: ${err.message}`);
    return;
  }
  appEls.authLoading.classList.add("hidden");
  renderUser(session.user || {});
  appEls.urlInput.value = currentRepoParam();
  showView("start");
  loadStartLists();
}

function renderUser(user) {
  appEls.userEmail.textContent = user.email || "";
  const plan = user.plan || "free";
  appEls.planBadge.textContent = `${plan.charAt(0).toUpperCase()}${plan.slice(1)} plan`;
}

async function logout() {
  appEls.logout.disabled = true;
  try {
    await API.logout();
  } catch (err) {
    // A 401 means the session is already gone, which is the state we want.
    if (err.status !== 401) {
      appEls.logout.disabled = false;
      showGlobalError(`Log out failed: ${err.message}`);
      return;
    }
  }
  location.assign("/");
}

// ---------------------------------------------------------------------------
// Start screen: analyze form, repositories, recent jobs
// ---------------------------------------------------------------------------

function friendlyApiError(err) {
  if (!(err instanceof ApiError)) return "Network error. Check your connection and try again.";
  if (err.status === 400) return err.message || "That does not look like a valid GitHub repository URL.";
  if (err.status === 402) return err.message || "You have reached your plan's analysis limit.";
  if (err.status === 429) return err.message || "Too many requests. Wait a moment and try again.";
  return err.message;
}

function renderInlineError(container, err) {
  const nodes = [friendlyApiError(err)];
  if (err instanceof ApiError && err.status === 402) {
    nodes.push(" ", el("a", { href: "/settings?upgrade=pro", text: "Upgrade your plan" }));
  }
  container.replaceChildren(...nodes);
  container.hidden = false;
}

function setAnalyzeBusy(busy) {
  appEls.analyzeBtn.disabled = busy;
  appEls.analyzeBtn.textContent = busy ? "Starting..." : "Analyze";
}

async function onAnalyzeSubmit(event) {
  event.preventDefault();
  const url = appEls.urlInput.value.trim();
  appEls.analyzeError.hidden = true;
  if (!url) {
    renderInlineError(appEls.analyzeError, new ApiError(400, "Enter a GitHub repository URL, for example https://github.com/owner/repo."));
    return;
  }
  setAnalyzeBusy(true);
  try {
    const data = await API.analyzeGithub(url);
    startProgress(data.job.id, url);
  } catch (err) {
    renderInlineError(appEls.analyzeError, err);
  } finally {
    setAnalyzeBusy(false);
  }
}

function repoNameFromUrl(url) {
  return String(url || "").replace(/\.git$/, "").split("/").filter(Boolean).pop() || "Repository";
}

function analysisIdForRepo(repoId) {
  const job = listState.jobs.find((j) => j.repo_id === repoId && j.analysis_id);
  return job ? job.analysis_id : null;
}

async function loadStartLists() {
  const [reposRes, jobsRes] = await Promise.allSettled([API.repos(), API.jobs()]);
  listState.jobs = jobsRes.status === "fulfilled" ? (jobsRes.value.jobs || []) : [];
  renderJobList(jobsRes);
  renderRepoList(reposRes);
}

function setListMessage(list, message) {
  list.replaceChildren(el("li", { className: "form-error", text: message }));
}

function renderRepoList(result) {
  if (result.status !== "fulfilled") {
    setListMessage(appEls.repoList, `Could not load repositories: ${friendlyApiError(result.reason)}`);
    return;
  }
  const repos = result.value.repositories || [];
  appEls.reposEmpty.classList.toggle("hidden", repos.length > 0);
  appEls.repoList.replaceChildren(...repos.map((repo) => {
    const id = repo.id ?? repo.repo_id;
    const name = repo.name ?? repo.repo_name ?? repoNameFromUrl(repo.github_url);
    return el("li", {}, el(
      "button",
      {
        className: "item-row",
        type: "button",
        on: { click: () => openRepo({ id, name, analysisId: analysisIdForRepo(id) }) },
      },
      el("span", { className: "item-title", text: name }),
      el("span", { className: "item-sub", text: repo.github_url || "" }),
    ));
  }));
}

function renderJobList(result) {
  if (result.status !== "fulfilled") {
    setListMessage(appEls.jobList, `Could not load recent jobs: ${friendlyApiError(result.reason)}`);
    return;
  }
  const jobs = listState.jobs;
  appEls.jobsEmpty.classList.toggle("hidden", jobs.length > 0);
  appEls.jobList.replaceChildren(...jobs.map((job) => el("li", {}, el(
    "button",
    {
      className: "item-row",
      type: "button",
      on: { click: () => openJob(job) },
    },
    el("span", { className: "item-title", text: job.result?.repo_name || repoNameFromUrl(job.github_url) }),
    el("span", { className: "item-sub", text: job.github_url || "" }),
    el("span", { className: `status-chip status-${job.status}`, text: job.status }),
  ))));
}

function openJob(job) {
  if (job.status === "done" && job.repo_id) {
    openRepo({
      id: job.repo_id,
      name: job.result?.repo_name || repoNameFromUrl(job.github_url),
      analysisId: job.analysis_id || null,
    });
    return;
  }
  startProgress(job.id, job.github_url || "");
}

// ---------------------------------------------------------------------------
// Job progress: poll GET /api/repos/jobs/{id} every 1.5s
// ---------------------------------------------------------------------------

function stopPolling() {
  if (progressState.timer) clearTimeout(progressState.timer);
  progressState.timer = null;
  progressState.jobId = null;
}

function startProgress(jobId, githubUrl) {
  stopPolling();
  progressState.jobId = jobId;
  progressState.githubUrl = githubUrl || "";
  appEls.progressUrl.textContent = githubUrl || "";
  appEls.progressError.hidden = true;
  appEls.progressTrack.setAttribute("aria-valuenow", "0");
  appEls.progressFill.style.width = "0%";
  setStatus(appEls.progressStatus, "Queued...");
  showView("progress");
  pollJob(jobId);
}

function renderStepper(stage, status) {
  const index = STAGES.indexOf(stage);
  const finished = status === "done";
  appEls.stepper.replaceChildren(...STAGES.map((name, i) => {
    let state = "pending";
    if (finished || i < index) state = "done";
    else if (i === index) state = "active";
    return el(
      "li",
      {
        className: `stage stage-${state}`,
        "aria-current": state === "active" ? "step" : false,
        text: STAGE_LABELS[name],
      },
    );
  }));
}

function renderJobProgress(job) {
  renderStepper(job.stage, job.status);
  const pct = Math.max(0, Math.min(100, Number(job.progress) || 0));
  appEls.progressFill.style.width = `${pct}%`;
  appEls.progressTrack.setAttribute("aria-valuenow", String(pct));
  const label = STAGE_LABELS[job.stage] || job.stage || "Queued";
  setStatus(appEls.progressStatus, `${label} · ${pct}%`);
}

async function pollJob(jobId) {
  if (progressState.jobId !== jobId) return;
  try {
    const data = await API.job(jobId);
    if (progressState.jobId !== jobId) return;
    const job = data.job;
    renderJobProgress(job);
    if (job.status === "done") {
      finishJob(job);
    } else if (job.status === "failed") {
      showJobFailed(job.error || "The analysis failed.");
    } else {
      progressState.timer = setTimeout(() => pollJob(jobId), POLL_INTERVAL_MS);
    }
  } catch (err) {
    if (progressState.jobId !== jobId) return;
    if (err instanceof ApiError && err.status === 404) {
      showJobFailed("This analysis job no longer exists.");
      return;
    }
    setStatus(appEls.progressStatus, `Lost contact with the server. Retrying... (${friendlyApiError(err)})`);
    progressState.timer = setTimeout(() => pollJob(jobId), POLL_INTERVAL_MS);
  }
}

function finishJob(job) {
  stopPolling();
  if (!job.repo_id) {
    showJobFailed("The analysis finished without a repository id.");
    return;
  }
  openRepo({
    id: job.repo_id,
    name: job.result?.repo_name || repoNameFromUrl(job.github_url),
    analysisId: job.analysis_id || null,
  });
}

function showJobFailed(message) {
  stopPolling();
  appEls.progressErrorText.textContent = message;
  appEls.progressError.hidden = false;
  setStatus(appEls.progressStatus, "Failed");
}

async function retryAnalysis() {
  const url = progressState.githubUrl;
  if (!url) {
    backToStart();
    return;
  }
  appEls.retryBtn.disabled = true;
  try {
    const data = await API.analyzeGithub(url);
    startProgress(data.job.id, url);
  } catch (err) {
    appEls.progressErrorText.textContent = friendlyApiError(err);
    appEls.progressError.hidden = false;
  } finally {
    appEls.retryBtn.disabled = false;
  }
}

function backToStart() {
  stopPolling();
  closeRepo();
  showView("start");
  loadStartLists();
}

// ---------------------------------------------------------------------------
// Repo workspace
// ---------------------------------------------------------------------------

function safeGithubLink(url) {
  return /^https:\/\/github\.com\//.test(url || "") ? url : "";
}

async function openRepo({ id, name, analysisId }) {
  stopPolling();
  workspaceState.repoId = id;
  workspaceState.repoName = name || "Repository";
  workspaceState.analysisId = analysisId || null;

  appEls.repoName.textContent = workspaceState.repoName;
  const job = listState.jobs.find((j) => j.repo_id === id);
  const url = safeGithubLink(job && job.github_url);
  appEls.repoUrl.textContent = url || "";
  appEls.repoUrl.href = url || "#";
  appEls.repoUrl.hidden = !url;
  appEls.repoScore.textContent = "Loading...";
  appEls.severityRow.replaceChildren();
  appEls.tabReport.classList.toggle("hidden", !analysisId);

  closeNodePanel();
  resetReport();
  setChatRepo(id);
  showView("workspace");
  showWorkspaceTab("graph");

  const [, summary] = await Promise.all([
    loadGraph(id).catch(() => null), // loadGraph shows its own error in the graph status
    loadFindings(id),
  ]);
  renderWorkspaceHeader(summary);
}

function closeRepo() {
  workspaceState.repoId = null;
  workspaceState.analysisId = null;
  closeNodePanel();
  resetReport();
  setChatRepo(null);
}

function renderWorkspaceHeader(summary) {
  if (!summary) {
    appEls.repoScore.textContent = "Score unavailable";
    return;
  }
  appEls.repoScore.textContent = summary.score != null ? `Score ${summary.score}/100` : "Score unavailable";
  const bySeverity = summary.by_severity || {};
  const counts = SEVERITY_ORDER.filter((sev) => bySeverity[sev]).map((sev) => el(
    "li",
    { className: "severity-count" },
    severityBadge(sev),
    el("span", { text: String(bySeverity[sev]) }),
  ));
  appEls.severityRow.replaceChildren(...(counts.length ? counts : [el("li", { className: "text-muted", text: "No findings" })]));
}

function showWorkspaceTab(name) {
  appEls.wsTabs.forEach((tab) => {
    const active = tab.dataset.view === name;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
  });
  Object.entries(appEls.panels).forEach(([key, panel]) => panel.classList.toggle("hidden", key !== name));
  if (name === "graph") resizeGraph();
  if (name === "report") ensureReportLoaded();
}
