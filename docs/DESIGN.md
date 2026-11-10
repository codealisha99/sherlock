# Sherlock design system: "Case File"

## The product, in one breath
Paste a GitHub URL → Sherlock clones it, builds a code graph (files, classes, functions,
imports, calls) with embeddings, scans for security, quality, test and dependency (OSV)
issues, maps those findings onto the graph, lets a model verify findings, and answers
questions about any node with file citations. Users: engineers inheriting, auditing or
reviewing an unfamiliar repository. The core job: **see where a codebase is weak, then ask why.**

## Directions considered
| | A. Case File (chosen) | B. Instrument | C. Survey Map |
|---|---|---|---|
| Philosophy | An investigation dossier: evidence, exhibits, annotations | A lab instrument: dark canvas, live signal | Cartography: the repo as terrain |
| Identity | Ink on warm paper, pen-blue accent, serif display, mono margin notes | Graphite, cool white, violet glow | Forest/cream, contour lines, coordinates |
| Layout | Editorial asymmetry, numbered exhibits, ruled ledgers | Full-bleed canvas, floating panels | Map-first, legend-driven |
| Interaction | Things get *marked*: highlight, annotate, cite | Scrubbing, zooming, command palette | Pan, survey, pin |
| Risk | Can feel literary if overdone | Looks like every dark dev tool | Metaphor strains for findings + chat |

**Why A:** the name already promises an investigation; the product's output is literally
evidence (findings with CWE/OSV IDs, file:line citations). Ink-on-paper keeps long
findings readable, and a neutral page lets the only saturated colours be the evidence.

## Rules
1. **Colour means evidence.** Graph nodes are neutral ink; only nodes with findings are coloured
   (amber = needs attention, red = at risk). Brand accent (pen blue) is for interaction only:
   links, primary buttons, focus, selection. Never green-for-brand (the old sage made the whole graph look "healthy").
2. **Three faces, three jobs.** Newsreader = display/headings (italic for emphasis, sparingly).
   Inter = UI and body. IBM Plex Mono = paths, IDs (CWE-327, GHSA-…), numbers, and
   uppercase tracked labels ("eyebrows", table headers).
3. **Rules over shadows.** Surfaces are separated by 1px `--line`. Shadows only for things
   that float (menus, dialogs, chat, sheets).
4. **Radii are small** (4/8/12). No pill buttons.
5. **Every number is real.** No invented stats, customers or testimonials. Illustrations are
   built from a real analysis (pallets/itsdangerous) and labelled as examples.
6. **Motion explains state.** Tokens in `tokens.css` (`--dur-1..4`, `--ease-out`). Press 120ms,
   menus/tabs 200ms, panels/sheets 320ms, reveals 520ms with 45ms stagger. Nothing loops except
   an active-progress indicator. `prefers-reduced-motion` zeroes all durations.
7. **Accessibility is in the tokens.** Text pairs ≥4.5:1, control boundaries ≥3:1
   (`--line-control`), visible `--focus-ring` on every interactive element, 44px touch targets on mobile.

## Tokens
`frontend/tokens.css` is linked before `site.css` / `style.css` on every page. Graph canvas
colours are mirrored as constants in `graph.js` (Cytoscape can't read CSS variables).

## Breakpoints
sm 640 · md 900 · lg 1200. Mobile is designed, not shrunk: the workspace node panel becomes a
bottom sheet, filters collapse into a disclosure, the top bar drops to brand + menu.
