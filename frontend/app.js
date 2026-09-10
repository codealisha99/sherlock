/**
 * Sherlock Frontend Application Controller
 * Handles analysis triggering, agent progress polling, 10-page document navigation,
 * and code inspection modal.
 */

let currentAnalysisId = null;
let currentReport = null;
let currentPageNumber = 1;
let pollInterval = null;

// DOM Elements
const btnAnalyze = document.getElementById("btn-analyze");
const inputProjectPath = document.getElementById("project-path-input");
const pipelineProgress = document.getElementById("pipeline-progress");
const activeAgentLabel = document.getElementById("active-agent-label");
const analysisIdBadge = document.getElementById("analysis-id-badge");
const reportContainer = document.getElementById("report-container");
const sidebarProjectName = document.getElementById("sidebar-project-name");
const pageTitle = document.getElementById("page-title");
const pageSubtitle = document.getElementById("page-subtitle");
const pageIndicator = document.getElementById("page-indicator");
const pageBody = document.getElementById("page-body");
const btnPrevPage = document.getElementById("btn-prev-page");
const btnNextPage = document.getElementById("btn-next-page");
const ollamaStatusBadge = document.getElementById("ollama-status-badge");
const ollamaStatusText = document.getElementById("ollama-status-text");

// Modal Elements
const codeModal = document.getElementById("code-modal");
const modalFilepath = document.getElementById("modal-filepath");
const modalCodeContent = document.getElementById("modal-code-content");
const btnCloseModal = document.getElementById("btn-close-modal");

// Agent Chip Mapping
const agentChipMap = {
  "Repository Mapper": document.getElementById("chip-mapper"),
  "Flow Tracer": document.getElementById("chip-flows"),
  "Component Analyst": document.getElementById("chip-components"),
  "Data Analyst": document.getElementById("chip-data"),
  "Gap Analyst": document.getElementById("chip-gaps"),
  "Teacher": document.getElementById("chip-teacher"),
};

// ==============================================================================
// Initialization & Health Check
// ==============================================================================

window.addEventListener("DOMContentLoaded", () => {
  checkOllamaHealth();
  setupEventListeners();
});

async function checkOllamaHealth() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error("Health check failed");
    const data = await res.json();
    const ollama = data.ollama;

    if (ollama.available) {
      ollamaStatusBadge.className = "status-pill status-connected";
      ollamaStatusText.textContent = `Ollama: ${ollama.configured_model}`;
    } else {
      ollamaStatusBadge.className = "status-pill status-offline";
      ollamaStatusText.textContent = "Ollama: Offline (Static Mode Active)";
    }
  } catch (err) {
    ollamaStatusBadge.className = "status-pill status-offline";
    ollamaStatusText.textContent = "Ollama: Offline (Static Mode Active)";
  }
}

function setupEventListeners() {
  btnAnalyze.addEventListener("click", startAnalysis);
  btnPrevPage.addEventListener("click", () => navigateToPage(currentPageNumber - 1));
  btnNextPage.addEventListener("click", () => navigateToPage(currentPageNumber + 1));

  // Sidebar item clicks
  document.querySelectorAll(".nav-item").forEach(item => {
    item.addEventListener("click", () => {
      const pageNum = parseInt(item.getAttribute("data-page"), 10);
      navigateToPage(pageNum);
    });
  });

  // Modal close
  btnCloseModal.addEventListener("click", () => codeModal.classList.add("hidden"));
  codeModal.addEventListener("click", (e) => {
    if (e.target === codeModal) codeModal.classList.add("hidden");
  });
}

// ==============================================================================
// Analysis Trigger & Polling
// ==============================================================================

async function startAnalysis() {
  const projectPath = inputProjectPath.value.trim();
  if (!projectPath) {
    alert("Please enter a valid repository directory path.");
    return;
  }

  btnAnalyze.disabled = true;
  btnAnalyze.innerHTML = '<span class="spinner" style="width:14px;height:14px;display:inline-block;margin:0 6px 0 0;"></span> Analyzing...';
  pipelineProgress.classList.remove("hidden");

  // Reset agent chips
  Object.values(agentChipMap).forEach(chip => {
    chip.className = "agent-chip";
    chip.querySelector(".agent-status-icon").textContent = "⏳";
  });

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_path: projectPath }),
    });

    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.error || "Failed to start analysis");
    }

    const data = await res.json();
    currentAnalysisId = data.analysis_id;
    analysisIdBadge.textContent = currentAnalysisId;
    activeAgentLabel.textContent = "Indexing files...";

    // Start polling status
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(pollProgress, 1000);

  } catch (err) {
    alert("Error: " + err.message);
    btnAnalyze.disabled = false;
    btnAnalyze.innerHTML = '<span class="btn-icon">⚡</span> Analyze Codebase';
  }
}

async function pollProgress() {
  if (!currentAnalysisId) return;

  try {
    const res = await fetch(`/api/analysis/${currentAnalysisId}`);
    if (!res.ok) return;
    const progress = await res.json();

    activeAgentLabel.textContent = progress.status.replace("_", " ").toUpperCase();

    // Update chips
    const statuses = progress.agent_statuses || {};
    for (const [agentName, status] of Object.entries(statuses)) {
      const chip = agentChipMap[agentName];
      if (!chip) continue;

      chip.className = `agent-chip ${status}`;
      const icon = chip.querySelector(".agent-status-icon");
      if (status === "completed") icon.textContent = "✓";
      else if (status === "running") icon.textContent = "⚡";
      else if (status === "failed") icon.textContent = "✕";
      else icon.textContent = "⏳";
    }

    if (progress.status === "completed" || progress.has_report) {
      clearInterval(pollInterval);
      activeAgentLabel.textContent = "DONE";
      btnAnalyze.disabled = false;
      btnAnalyze.innerHTML = '<span class="btn-icon">⚡</span> Analyze Codebase';
      loadFullReport(currentAnalysisId);
    } else if (progress.status === "failed") {
      clearInterval(pollInterval);
      alert("Analysis failed: " + (progress.errors || []).join(", "));
      btnAnalyze.disabled = false;
      btnAnalyze.innerHTML = '<span class="btn-icon">⚡</span> Analyze Codebase';
    }
  } catch (err) {
    console.error("Polling error:", err);
  }
}

// ==============================================================================
// Report Loading & Rendering
// ==============================================================================

async function loadFullReport(analysisId) {
  try {
    const res = await fetch(`/api/analysis/${analysisId}/report`);
    if (!res.ok) throw new Error("Could not load report");
    currentReport = await res.json();

    sidebarProjectName.textContent = currentReport.project_name;
    reportContainer.classList.remove("hidden");
    navigateToPage(1);
  } catch (err) {
    console.error("Load report error:", err);
  }
}

function navigateToPage(pageNum) {
  if (!currentReport || pageNum < 1 || pageNum > 10) return;
  currentPageNumber = pageNum;

  // Update navigation buttons state
  btnPrevPage.disabled = (currentPageNumber === 1);
  btnNextPage.disabled = (currentPageNumber === 10);
  pageIndicator.textContent = `Page ${currentPageNumber} of 10`;

  // Update sidebar active item
  document.querySelectorAll(".nav-item").forEach(item => {
    const num = parseInt(item.getAttribute("data-page"), 10);
    if (num === currentPageNumber) item.classList.add("active");
    else item.classList.remove("active");
  });

  const pageData = currentReport.pages.find(p => p.page_number === currentPageNumber);
  if (!pageData) return;

  pageTitle.textContent = pageData.title;
  pageSubtitle.textContent = pageData.subtitle;

  renderPageContent(currentPageNumber, pageData.content);
}

function renderPageContent(pageNum, content) {
  pageBody.innerHTML = "";

  switch (pageNum) {
    case 1:
      renderPage1(content);
      break;
    case 2:
      renderPage2(content);
      break;
    case 3:
      renderPage3(content);
      break;
    case 4:
      renderPage4(content);
      break;
    case 5:
      renderPage5(content);
      break;
    case 6:
      renderPage6(content);
      break;
    case 7:
      renderPage7(content);
      break;
    case 8:
      renderPage8(content);
      break;
    case 9:
      renderPage9(content);
      break;
    case 10:
      renderPage10(content);
      break;
    default:
      pageBody.innerHTML = `<p>${JSON.stringify(content, null, 2)}</p>`;
  }

  // Trigger Mermaid render if present
  setTimeout(() => {
    try {
      mermaid.run({ querySelector: '.mermaid' });
    } catch (e) {
      console.warn("Mermaid render error:", e);
    }
  }, 100);
}

// ==============================================================================
// Specific Page Renderers
// ==============================================================================

function renderPage1(c) {
  pageBody.innerHTML = `
    <div class="card">
      <div class="card-title">📖 Core Identity & Purpose</div>
      <p style="font-size: 1.05rem; margin-bottom: 14px;">${c.summary || ""}</p>
      <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px;">
        ${(c.languages || []).map(l => `<span class="badge badge-subtle">${l}</span>`).join("")}
        ${(c.frameworks || []).map(f => `<span class="badge badge-subtle" style="color:var(--accent-green);border-color:var(--accent-green);">${f}</span>`).join("")}
      </div>
    </div>

    <div class="card-grid">
      <div class="card">
        <div class="card-title">🎯 Why This Project Exists</div>
        <p>${c.why_it_exists || "Designed to solve concrete engineering domain requirements."}</p>
      </div>
      <div class="card">
        <div class="card-title">💡 Problem It Solves</div>
        <p>${c.problem_solved || "Provides automated, structured codebase comprehension."}</p>
      </div>
    </div>

    <div class="card">
      <div class="card-title">🚀 Important Entry Points</div>
      <ul style="padding-left: 20px;">
        ${(c.entry_points || []).map(ep => `<li style="margin-bottom:6px;"><span class="code-link" onclick="openFileViewer('${ep}')">${ep}</span></li>`).join("")}
      </ul>
    </div>
  `;
}

function renderPage2(c) {
  pageBody.innerHTML = `
    <div class="card">
      <div class="card-title">📂 Structural Directory Map</div>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px; margin-top: 12px;">
        ${(c.directories || []).map(d => `
          <div style="background:var(--bg-tertiary); padding: 12px; border-radius: 6px; border:1px solid var(--border-color);">
            <strong style="color:var(--accent-blue); font-family:var(--font-mono);">${d.path}</strong>
            <p style="font-size:0.85rem; margin-top:4px;">${d.purpose}</p>
            ${d.key_files && d.key_files.length ? `
              <div style="margin-top:6px; display:flex; gap:4px; flex-wrap:wrap;">
                ${d.key_files.map(kf => `<span class="code-link" style="font-size:0.75rem;" onclick="openFileViewer('${kf}')">${kf.split('/').pop()}</span>`).join("")}
              </div>
            ` : ""}
          </div>
        `).join("")}
      </div>
    </div>

    <div class="card">
      <div class="card-title">🌲 Repository Hierarchy</div>
      <pre class="code-block" style="max-height: 400px; overflow-y: auto;">${c.tree || "No tree representation"}</pre>
    </div>
  `;
}

function renderPage3(c) {
  pageBody.innerHTML = `
    <div class="card">
      <div class="card-title">📐 High-Level Architecture Topology</div>
      <div class="mermaid" style="background:var(--bg-tertiary); padding:16px; border-radius:6px; overflow-x:auto;">
        ${c.architecture_diagram || "graph TD; App-->Module;"}
      </div>
    </div>

    <div class="card">
      <div class="card-title">⚖️ Key Architectural Decisions & Rationale</div>
      ${(c.decisions || []).map(d => `
        <div style="margin-bottom: 14px; padding-bottom: 14px; border-bottom: 1px solid var(--border-color);">
          <strong style="color:var(--text-primary); font-size:0.95rem;">${d.decision}</strong>
          <p style="font-size:0.88rem; margin: 4px 0;"><span style="color:var(--accent-green);">Rationale:</span> ${d.rationale}</p>
          <p style="font-size:0.84rem; color:var(--text-muted);"><span style="color:var(--accent-orange);">Trade-offs:</span> ${d.trade_offs}</p>
        </div>
      `).join("")}
    </div>
  `;
}

function renderPage4(c) {
  pageBody.innerHTML = `
    <div class="card">
      <div class="card-title">🔌 Exposed Routes & Interface Endpoints</div>
      <p style="margin-bottom: 16px; color:var(--text-secondary);">${c.summary || ""}</p>
      <div style="display: flex; flex-direction: column; gap: 10px;">
        ${(c.endpoints || []).map(ep => `
          <div style="background:var(--bg-tertiary); padding: 12px 16px; border-radius: 6px; border-left: 3px solid var(--accent-blue); display:flex; justify-content:space-between; align-items:center;">
            <code style="font-size:0.9rem; color:var(--accent-blue); font-family:var(--font-mono);">${ep}</code>
            <span class="badge badge-subtle">Interface</span>
          </div>
        `).join("")}
      </div>
    </div>
  `;
}

function renderPage5(c) {
  pageBody.innerHTML = `
    <div class="card">
      <div class="card-title">⚡ End-to-End Execution Workflows</div>
      <p style="margin-bottom: 20px; color:var(--text-secondary);">Visualizing step-by-step lifecycles from input trigger through service layers to response.</p>
      
      ${(c.flows || []).map(f => `
        <div class="card" style="background:var(--bg-tertiary); margin-bottom: 24px;">
          <h3 style="color:var(--accent-blue); margin-bottom: 6px;">${f.name}</h3>
          <p style="font-size:0.88rem; color:var(--text-secondary); margin-bottom: 14px;"><strong>Trigger:</strong> <code>${f.trigger}</code></p>
          <p style="font-size:0.88rem; margin-bottom: 16px;">${f.description}</p>

          ${f.mermaid_diagram ? `
            <div class="mermaid" style="background:var(--bg-secondary); padding:12px; border-radius:6px; margin-bottom:16px; overflow-x:auto;">
              ${f.mermaid_diagram}
            </div>
          ` : ""}

          <div class="flow-steps-container">
            ${(f.steps || []).map((s, idx) => `
              <div class="flow-step-item">
                <div class="flow-step-badge">${s.step_number || (idx + 1)}</div>
                <div class="flow-step-content">
                  <div class="flow-step-layer">${s.layer}</div>
                  <div style="font-weight:600; margin: 2px 0;">${s.component} &mdash; <code>${s.function_name}</code></div>
                  <div style="font-size:0.82rem; margin: 4px 0;"><span class="code-link" onclick="openFileViewer('${s.file}')">${s.file}</span></div>
                  <div style="font-size:0.85rem; color:var(--text-secondary);">${s.description}</div>
                </div>
              </div>
              ${idx < (f.steps.length - 1) ? '<div class="flow-step-arrow">↓</div>' : ''}
            `).join("")}
          </div>
        </div>
      `).join("")}
    </div>
  `;
}

function renderPage6(c) {
  pageBody.innerHTML = `
    <div class="card">
      <div class="card-title">🧩 Component Explorer</div>
      <p style="margin-bottom: 16px; color:var(--text-secondary);">Deep inspection of what core classes and modules ACTUALLY do (distinguishing naming from concrete behavior).</p>

      ${c.mermaid_graph ? `
        <div class="mermaid" style="background:var(--bg-tertiary); padding:16px; border-radius:6px; margin-bottom:20px; overflow-x:auto;">
          ${c.mermaid_graph}
        </div>
      ` : ""}

      <div class="card-grid">
        ${(c.components || []).map(comp => `
          <div class="card" style="background:var(--bg-tertiary);">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:8px;">
              <strong style="font-size:1.05rem; color:var(--accent-blue); font-family:var(--font-mono);">${comp.component}</strong>
            </div>
            <div style="margin-bottom:10px;">
              <span class="code-link" onclick="openFileViewer('${comp.file}')">${comp.file}</span>
            </div>
            <p style="font-size:0.88rem; margin-bottom:12px;"><strong>Responsibility:</strong> ${comp.responsibility}</p>
            
            <div style="font-size:0.8rem; margin-bottom:6px;">
              <span style="color:var(--text-muted);">Inputs:</span> ${(comp.inputs || []).join(", ") || "None"}
            </div>
            <div style="font-size:0.8rem; margin-bottom:6px;">
              <span style="color:var(--text-muted);">Outputs:</span> ${(comp.outputs || []).join(", ") || "None"}
            </div>
            <div style="font-size:0.8rem; margin-bottom:6px;">
              <span style="color:var(--text-muted);">Side Effects:</span> ${(comp.side_effects || []).join(", ") || "None"}
            </div>
            <div style="font-size:0.8rem; margin-bottom:6px;">
              <span style="color:var(--accent-orange);">Assumptions:</span> ${(comp.assumptions || []).join(", ") || "None"}
            </div>
            <div style="font-size:0.8rem;">
              <span style="color:var(--accent-red);">Failure Modes:</span> ${(comp.failure_modes || []).join(", ") || "None"}
            </div>
          </div>
        `).join("")}
      </div>
    </div>
  `;
}

function renderPage7(c) {
  pageBody.innerHTML = `
    <div class="card">
      <div class="card-title">💾 Data Movement & External Boundaries</div>
      ${c.mermaid_diagram ? `
        <div class="mermaid" style="background:var(--bg-tertiary); padding:16px; border-radius:6px; margin-bottom:20px; overflow-x:auto;">
          ${c.mermaid_diagram}
        </div>
      ` : ""}

      <h3 style="margin: 20px 0 10px; font-size:1rem;">Databases & Storage</h3>
      <div class="card-grid">
        ${(c.databases || []).map(db => `
          <div class="card" style="background:var(--bg-tertiary);">
            <strong style="color:var(--accent-green);">${db.name} (${db.type})</strong>
            <p style="font-size:0.85rem; margin-top:4px;">${db.purpose}</p>
            <div style="font-size:0.8rem; margin-top:8px; color:var(--text-muted);">ORM/Driver: ${db.orm_or_driver}</div>
            <div style="font-size:0.8rem; margin-top:4px;">Models/Artifacts: ${(db.tables_or_models || []).join(", ")}</div>
          </div>
        `).join("")}
      </div>

      <h3 style="margin: 20px 0 10px; font-size:1rem;">External Services & AI/ML Integrations</h3>
      <div class="card-grid">
        ${(c.external_services || []).map(s => `
          <div class="card" style="background:var(--bg-tertiary);">
            <strong style="color:var(--accent-purple);">${s.name}</strong>
            <p style="font-size:0.85rem; margin: 4px 0;">${s.purpose}</p>
            <div style="font-size:0.8rem; color:var(--text-muted);">Location: <code>${s.call_location}</code></div>
            <div style="font-size:0.8rem; margin-top:4px;"><strong style="color:var(--accent-orange);">Sent:</strong> ${s.sent_data}</div>
            <div style="font-size:0.8rem;"><strong style="color:var(--accent-green);">Received:</strong> ${s.received_data}</div>
            <div style="font-size:0.8rem; margin-top:4px; color:var(--accent-red);">Failure Risks: ${(s.failure_possibilities || []).join("; ")}</div>
          </div>
        `).join("")}
      </div>
    </div>
  `;
}

function renderPage8(c) {
  const categoryClass = (cat) => {
    const lower = (cat || "").toLowerCase();
    if (lower.includes("confirmed")) return "gap-confirmed";
    if (lower.includes("likely")) return "gap-likely";
    if (lower.includes("architectural")) return "gap-architectural";
    if (lower.includes("missing")) return "gap-missing";
    return "gap-improvement";
  };

  pageBody.innerHTML = `
    <div class="card">
      <div class="card-title">⚠️ Codebase Gaps, Technical Debt & Risks</div>
      <p style="margin-bottom: 20px; color:var(--text-secondary);">${c.summary || "Evaluated code health, missing validations, test coverage, and architectural bottlenecks."}</p>

      <div style="display: flex; flex-direction: column; gap: 14px;">
        ${(c.gaps || []).map(g => `
          <div class="card" style="background:var(--bg-tertiary); border-left: 4px solid var(--border-color);">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
              <span class="gap-badge ${categoryClass(g.category)}">${g.category}</span>
              <span style="font-size:0.75rem; color:var(--text-muted);">Confidence: <strong>${g.confidence}</strong></span>
            </div>
            <h3 style="font-size:1.05rem; margin-bottom:6px;">${g.gap}</h3>
            <p style="font-size:0.88rem; color:var(--text-secondary); margin-bottom:8px;"><strong>Evidence:</strong> <code>${g.evidence}</code></p>
            <p style="font-size:0.88rem; margin-bottom:8px;"><strong style="color:var(--accent-orange);">Why It Matters:</strong> ${g.why_it_matters}</p>
            <p style="font-size:0.85rem; margin-bottom:8px;"><strong style="color:var(--accent-red);">Possible Impact:</strong> ${g.possible_impact}</p>
            <div style="background:var(--bg-secondary); padding:8px 12px; border-radius:4px; font-size:0.85rem; border:1px solid var(--border-color);">
              <strong style="color:var(--accent-green);">🛠️ Recommended Contribution:</strong> ${g.possible_contribution}
            </div>
          </div>
        `).join("")}
      </div>
    </div>
  `;
}

function renderPage9(c) {
  pageBody.innerHTML = `
    <div class="card">
      <div class="card-title">📚 Conceptual Primitives to Learn</div>
      <p style="margin-bottom: 20px; color:var(--text-secondary);">Engineering concepts required to deeply comprehend and extend this specific codebase.</p>

      <div class="card-grid">
        ${(c.concepts || []).map(con => `
          <div class="card" style="background:var(--bg-tertiary);">
            <h3 style="color:var(--accent-blue); font-size:1.05rem; margin-bottom:6px;">${con.concept}</h3>
            <p style="font-size:0.88rem; margin-bottom:10px;">${con.definition}</p>
            <p style="font-size:0.84rem; color:var(--text-secondary); margin-bottom:6px;"><strong style="color:var(--accent-green);">Why Relevant Here:</strong> ${con.why_relevant_here}</p>
            <p style="font-size:0.8rem; color:var(--text-muted);"><strong style="color:var(--accent-purple);">Used In:</strong> <code>${con.where_used_in_project}</code></p>
          </div>
        `).join("")}
      </div>
    </div>
  `;
}

function renderPage10(c) {
  const qa = c.answers_to_25_questions || {};
  const qaEntries = Object.entries(qa);

  pageBody.innerHTML = `
    <div class="card" style="background: linear-gradient(135deg, var(--bg-secondary) 0%, rgba(188, 140, 255, 0.08) 100%);">
      <div class="card-title" style="color:var(--accent-purple);">🎓 Synthesized Mental Model</div>
      <p style="font-size:1.05rem; margin-bottom:14px; line-height:1.6;">${c.simple_explanation || ""}</p>
      <div style="background:var(--bg-tertiary); padding:14px; border-radius:6px; border:1px solid var(--border-color);">
        <strong style="color:var(--accent-blue);">💼 Interview Walkthrough:</strong>
        <p style="font-size:0.9rem; margin-top:6px; color:var(--text-secondary);">${c.interview_explanation || ""}</p>
      </div>
    </div>

    <div class="card">
      <div class="card-title">📖 Recommended Reading Order for Developers</div>
      <div style="display:flex; flex-direction:column; gap:10px;">
        ${(c.first_files_to_read || []).map(f => `
          <div style="background:var(--bg-tertiary); padding:12px; border-radius:6px; display:flex; align-items:flex-start; gap:12px;">
            <span class="flow-step-badge">${f.order}</span>
            <div style="flex:1;">
              <div><span class="code-link" onclick="openFileViewer('${f.file}')">${f.file}</span></div>
              <p style="font-size:0.85rem; margin-top:4px;">${f.why_read_first}</p>
              ${f.key_symbols_to_watch && f.key_symbols_to_watch.length ? `
                <div style="font-size:0.78rem; color:var(--accent-blue); margin-top:4px;">
                  Key symbols: ${f.key_symbols_to_watch.map(s => `<code>${s}</code>`).join(", ")}
                </div>
              ` : ""}
            </div>
          </div>
        `).join("")}
      </div>
    </div>

    <div class="card">
      <div class="card-title">❓ The 25 Core Developer Mental Model Questions</div>
      <div style="display:flex; flex-direction:column; gap:12px;">
        ${qaEntries.map(([question, answer], idx) => `
          <div style="background:var(--bg-tertiary); padding:14px; border-radius:6px; border-left:3px solid var(--accent-purple);">
            <strong style="color:var(--text-primary); font-size:0.92rem; display:block; margin-bottom:6px;">${question}</strong>
            <p style="font-size:0.88rem; color:var(--text-secondary); line-height:1.5;">${answer}</p>
          </div>
        `).join("")}
      </div>
    </div>
  `;
}

// ==============================================================================
// Interactive Code Inspection Modal
// ==============================================================================

async function openFileViewer(filepath) {
  if (!currentAnalysisId) return;

  modalFilepath.textContent = `Loading ${filepath}...`;
  modalCodeContent.textContent = "Fetching source content...";
  codeModal.classList.remove("hidden");

  try {
    const res = await fetch(`/api/analysis/${currentAnalysisId}/file?path=${encodeURIComponent(filepath)}`);
    if (!res.ok) throw new Error("Could not retrieve file content");
    const data = await res.json();

    modalFilepath.textContent = data.path + (data.truncated ? " (Truncated)" : "");
    modalCodeContent.textContent = data.content;
  } catch (err) {
    modalCodeContent.textContent = `Error reading file: ${err.message}`;
  }
}
