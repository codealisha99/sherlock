/**
 * Cytoscape code graph: build elements from GET /api/repos/{id}/graph,
 * style by health/type, filter, search, layout, and selection.
 * Requires util.js, api.js, node-panel.js (openNodePanel), and the cytoscape CDN script.
 */

const NODE_TYPES = ["file", "class", "function", "method", "external", "other"];
const KNOWN_EDGE_TYPES = ["contains", "imports", "calls"];
const LARGE_GRAPH_THRESHOLD = 400;

/** Node types outside the filter list (e.g. symbols from other languages) are shown as "other", never hidden. */
function normalizeNodeType(type) {
  return NODE_TYPES.includes(type) ? type : "other";
}

const graphEls = {
  canvas: document.getElementById("graph-canvas"),
  status: document.getElementById("graph-status"),
  search: document.getElementById("graph-search"),
  layout: document.getElementById("graph-layout"),
  relayout: document.getElementById("btn-layout"),
  fit: document.getElementById("btn-fit"),
  findingsOnly: document.getElementById("filter-findings-only"),
};

let cyInstance = null;
let graphStats = { total: 0, truncated: false };

const GRAPH_STYLE = [
  {
    selector: "node",
    style: {
      label: "data(label)",
      "font-size": 10,
      "font-family": "Inter, -apple-system, sans-serif",
      color: "#2a2622",
      "text-valign": "bottom",
      "text-margin-y": 4,
      "text-wrap": "ellipsis",
      "text-max-width": 110,
      "min-zoomed-font-size": 8,
      width: "data(size)",
      height: "data(size)",
      "background-color": "#8a8175",
      "border-width": 2,
      "border-color": "#fffdf8",
    },
  },
  { selector: 'node[type = "file"]', style: { shape: "round-rectangle" } },
  { selector: 'node[type = "class"]', style: { shape: "hexagon" } },
  { selector: 'node[type = "function"], node[type = "method"]', style: { shape: "ellipse" } },
  { selector: 'node[type = "external"]', style: { shape: "diamond", "background-color": "#a08fd0" } },
  { selector: 'node[type = "other"]', style: { shape: "rectangle" } },
  { selector: 'node[health = "good"]', style: { "background-color": "#6f8f5e" } },
  { selector: 'node[health = "warn"]', style: { "background-color": "#d9b45b" } },
  { selector: 'node[health = "bad"]', style: { "background-color": "#b3543f" } },
  {
    selector: "edge",
    style: {
      width: 1.5,
      "curve-style": "bezier",
      "line-color": "#c9bfa9",
      "target-arrow-shape": "none",
      opacity: 0.85,
    },
  },
  { selector: 'edge[type = "contains"]', style: { "line-style": "dotted" } },
  {
    selector: 'edge[type = "imports"]',
    style: { "line-color": "#4e7fa6", "target-arrow-color": "#4e7fa6", "target-arrow-shape": "triangle" },
  },
  {
    selector: 'edge[type = "calls"]',
    style: { "line-color": "#cf8260", "target-arrow-color": "#cf8260", "target-arrow-shape": "triangle" },
  },
  { selector: ".match", style: { "border-width": 4, "border-color": "#b98a2f" } },
  { selector: ":selected", style: { "border-width": 4, "border-color": "#2a2622" } },
  { selector: ".filtered", style: { display: "none" } },
];

function initGraph() {
  graphEls.search.addEventListener("input", (e) => searchGraph(e.target.value));
  graphEls.search.addEventListener("keydown", (e) => {
    if (e.key !== "Enter" || !cyInstance) return;
    const first = cyInstance.nodes(":visible").filter((n) => n.hasClass("match")).first();
    if (first.length) selectGraphNode(first.id());
  });
  graphEls.relayout.addEventListener("click", runLayout);
  graphEls.fit.addEventListener("click", fitGraph);
  graphEls.findingsOnly.addEventListener("change", applyGraphFilters);
  document.querySelectorAll('input[name="node-type"], input[name="edge-type"]').forEach((input) => {
    input.addEventListener("change", applyGraphFilters);
  });
}

function sumFindings(findings) {
  return Object.values(findings || {}).reduce((total, n) => total + (Number(n) || 0), 0);
}

function nodeSize(node, degree) {
  const loc = Number(node.metrics && node.metrics.loc) || 0;
  return Math.round(18 + Math.min(40, Math.sqrt(loc) * 2 + degree * 2));
}

/** Convert the contract graph JSON into Cytoscape elements. Drops edges with missing endpoints. */
function buildGraphElements(data) {
  const ids = new Set(data.nodes.map((n) => n.id));
  const edges = data.edges.filter((e) => ids.has(e.source) && ids.has(e.target));
  const degree = new Map();
  for (const e of edges) {
    degree.set(e.source, (degree.get(e.source) || 0) + 1);
    degree.set(e.target, (degree.get(e.target) || 0) + 1);
  }
  const nodeEls = data.nodes.map((n) => ({
    group: "nodes",
    data: {
      id: n.id,
      label: n.label || n.id,
      type: normalizeNodeType(n.type),
      language: n.language || "",
      health: n.health || "none",
      findingsTotal: sumFindings(n.findings),
      size: nodeSize(n, degree.get(n.id) || 0),
    },
  }));
  const edgeEls = edges.map((e, i) => ({
    group: "edges",
    data: { id: `e${i}`, source: e.source, target: e.target, type: e.type },
  }));
  return [...nodeEls, ...edgeEls];
}

function layoutOptions(name, large) {
  const base = { name, animate: !large, fit: true, padding: 40 };
  if (name === "cose") {
    return { ...base, randomize: true, nodeRepulsion: 8000, idealEdgeLength: 90, numIter: large ? 500 : 1000 };
  }
  return base;
}

function runLayout() {
  if (!cyInstance) return;
  const large = cyInstance.nodes().length > LARGE_GRAPH_THRESHOLD;
  cyInstance.layout(layoutOptions(graphEls.layout.value, large)).run();
}

function destroyGraph() {
  if (cyInstance) cyInstance.destroy();
  cyInstance = null;
}

function renderGraph(data) {
  destroyGraph();
  if (!window.cytoscape) {
    setStatus(graphEls.status, "The graph library could not load. Check your connection and reload.");
    return;
  }
  const elements = buildGraphElements(data);
  const large = elements.length > LARGE_GRAPH_THRESHOLD;
  cyInstance = window.cytoscape({
    container: graphEls.canvas,
    elements,
    style: GRAPH_STYLE,
    layout: layoutOptions(graphEls.layout.value, large),
    minZoom: 0.1,
    maxZoom: 3,
  });
  cyInstance.on("tap", "node", (evt) => selectGraphNode(evt.target.id()));
  graphStats = { total: data.nodes.length, truncated: Boolean(data.stats && data.stats.truncated) };
  applyGraphFilters();
}

/** Loads and renders the graph for a repo. Errors are shown in the graph status line and rethrown. */
async function loadGraph(repoId) {
  setStatus(graphEls.status, "Loading graph...");
  try {
    const data = await API.graph(repoId);
    const nodes = data.nodes || [];
    if (!nodes.length) {
      destroyGraph();
      setStatus(graphEls.status, "This repository has no graph nodes yet.");
      return;
    }
    renderGraph({ nodes, edges: data.edges || [], stats: data.stats || {} });
  } catch (err) {
    destroyGraph();
    setStatus(graphEls.status, `Could not load the graph: ${err.message}`);
    throw err;
  }
}

function checkedValues(name) {
  return Array.from(document.querySelectorAll(`input[name="${name}"]:checked`)).map((input) => input.value);
}

/** Hides nodes/edges that fail the type, edge-type, or findings-only filters. */
function applyGraphFilters() {
  if (!cyInstance) return;
  const nodeTypes = new Set(checkedValues("node-type"));
  const edgeTypes = new Set(checkedValues("edge-type"));
  const onlyFindings = graphEls.findingsOnly.checked;
  cyInstance.batch(() => {
    cyInstance.nodes().forEach((n) => {
      const visible = nodeTypes.has(n.data("type")) && (!onlyFindings || n.data("findingsTotal") > 0);
      n.toggleClass("filtered", !visible);
    });
    cyInstance.edges().forEach((e) => {
      const type = e.data("type");
      const visible = (!KNOWN_EDGE_TYPES.includes(type) || edgeTypes.has(type))
        && !e.source().hasClass("filtered")
        && !e.target().hasClass("filtered");
      e.toggleClass("filtered", !visible);
    });
  });
  const shown = cyInstance.nodes().not(".filtered").length;
  const truncated = graphStats.truncated ? " Graph truncated by the server limit." : "";
  setStatus(graphEls.status, `${shown} of ${graphStats.total} nodes shown.${truncated}`);
}

function resetGraphFilters() {
  document.querySelectorAll('input[name="node-type"], input[name="edge-type"]').forEach((input) => {
    input.checked = true;
  });
  graphEls.findingsOnly.checked = false;
  applyGraphFilters();
}

function searchGraph(query) {
  if (!cyInstance) return;
  const q = query.trim().toLowerCase();
  cyInstance.elements().removeClass("match");
  if (!q) return;
  const matches = cyInstance.nodes().filter((n) => {
    if (n.hasClass("filtered")) return false;
    return String(n.data("label")).toLowerCase().includes(q) || n.id().toLowerCase().includes(q);
  });
  matches.addClass("match");
  if (matches.length) {
    cyInstance.animate({ center: { eles: matches.first() } }, { duration: 250 });
  }
  setStatus(graphEls.status, matches.length ? `${matches.length} match${matches.length === 1 ? "" : "es"}.` : "No matching nodes.");
}

/** True for ids like file:<path>, sym:<path>::Name, ext:<module> with a non-empty tail. */
function isSelectableNodeId(nodeId) {
  return typeof nodeId === "string" && /^(file|sym|ext):.+/.test(nodeId);
}

/** Selects a node in the graph (clearing filters that hide it) and opens its detail panel. */
function selectGraphNode(nodeId) {
  if (!isSelectableNodeId(nodeId)) return; // e.g. repo-level findings have no node
  openNodePanel(nodeId);
  if (!cyInstance) return;
  const node = cyInstance.getElementById(nodeId);
  if (node.length === 0) return;
  if (node.hasClass("filtered")) resetGraphFilters();
  cyInstance.elements().unselect();
  node.select();
  cyInstance.animate({ center: { eles: node } }, { duration: 250 });
}

function fitGraph() {
  if (!cyInstance) return;
  cyInstance.fit(cyInstance.elements().not(".filtered"), 40);
}

/** Cytoscape needs a resize after its container was hidden (tab switch). */
function resizeGraph() {
  if (cyInstance) cyInstance.resize();
}
