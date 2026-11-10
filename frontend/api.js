/**
 * Thin fetch wrapper for the SaaS API. Same-origin credentials only.
 * Non-2xx responses throw ApiError with the server's {"error": "..."} message.
 */

class ApiError extends Error {
  constructor(status, message, body = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

async function apiRequest(path, options = {}) {
  const init = { credentials: "same-origin", ...options, headers: { ...(options.headers || {}) } };
  if (options.body !== undefined && !(options.body instanceof FormData)) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(options.body);
  }
  const res = await fetch(path, init);
  let data = null;
  try {
    data = await res.json();
  } catch (_err) {
    data = null; // non-JSON body (e.g. proxy error page)
  }
  if (!res.ok) {
    const message = (data && (data.error || data.message)) || `Request failed (${res.status})`;
    throw new ApiError(res.status, message, data);
  }
  return data;
}

function apiGet(path) {
  return apiRequest(path, { method: "GET" });
}

function apiPost(path, body) {
  return apiRequest(path, { method: "POST", body });
}

/** Encode a path segment (repo/job/session ids). */
function seg(value) {
  return encodeURIComponent(String(value));
}

const API = {
  me: () => apiGet("/api/auth/me"),
  logout: () => apiPost("/api/auth/logout", {}),
  analyzeGithub: (url) => apiPost("/api/repos/github", { url }),
  jobs: () => apiGet("/api/repos/jobs"),
  job: (jobId) => apiGet(`/api/repos/jobs/${seg(jobId)}`),
  repos: () => apiGet("/api/repos"),
  graph: (repoId) => apiGet(`/api/repos/${seg(repoId)}/graph`),
  findings: (repoId) => apiGet(`/api/repos/${seg(repoId)}/findings`),
  node: (repoId, nodeId) => apiGet(`/api/repos/${seg(repoId)}/node?id=${encodeURIComponent(nodeId)}`),
  report: (analysisId) => apiGet(`/api/analysis/${seg(analysisId)}/report`),
  file: (analysisId, path) => apiGet(`/api/analysis/${seg(analysisId)}/file?path=${encodeURIComponent(path)}`),
  createChatSession: (repoId) => apiPost("/api/chat/sessions", { repo_id: repoId }),
  capabilities: () => apiGet("/api/capabilities"),
  /** body undefined = server default selection of findings to verify. */
  triage: (repoId, body) => apiPost(`/api/repos/${seg(repoId)}/triage`, body),
};
