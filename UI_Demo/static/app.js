/* FinReflectKG Frontend Controller */

document.addEventListener("DOMContentLoaded", () => {
  // 1. Tab Navigation
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      document.getElementById(targetId).classList.add("active");
    });
  });

  // 2. Preset Query Chips
  const presetChips = document.querySelectorAll(".chip");
  const queryInput = document.getElementById("query-input");

  presetChips.forEach(chip => {
    chip.addEventListener("click", () => {
      const q = chip.getAttribute("data-query");
      queryInput.value = q;
      executeQuery(q);
    });
  });

  // 3. Tab 1: Execute Unified Query
  const submitBtn = document.getElementById("submit-btn");
  const loadingSpinner = document.getElementById("loading-spinner");
  const resultsPanel = document.getElementById("results-panel");
  const routeBadge = document.getElementById("route-badge");
  const confBadge = document.getElementById("confidence-badge");
  const routeRationale = document.getElementById("route-rationale");
  const evidenceViewer = document.getElementById("evidence-viewer");
  const evidenceCountBadge = document.getElementById("evidence-count-badge");
  const answerViewer = document.getElementById("answer-viewer");
  const latencyBadge = document.getElementById("latency-badge");

  async function executeQuery(queryText) {
    if (!queryText || !queryText.trim()) return;

    // Show loading
    loadingSpinner.classList.remove("hidden");
    resultsPanel.classList.add("hidden");

    try {
      const response = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: queryText.trim() })
      });

      const data = await response.json();
      loadingSpinner.classList.add("hidden");
      resultsPanel.classList.remove("hidden");

      if (data.status === "out_of_scope") {
        routeBadge.textContent = "OUT_OF_SCOPE";
        routeBadge.className = "badge badge-amber";
        confBadge.textContent = "Guardrail Active";
        routeRationale.textContent = data.routing.reasoning;

        evidenceViewer.innerHTML = `<p class="placeholder-text">No Knowledge Graph retrieval performed for out-of-scope query.</p>`;
        evidenceCountBadge.textContent = "0 Sources";
        answerViewer.innerHTML = marked.parse(data.answer);
        latencyBadge.textContent = `Latency: ${data.latency_sec}s`;
        return;
      }

      // Normal Success
      const routing = data.routing;
      const execution = data.execution;

      routeBadge.textContent = routing.route;
      routeBadge.className = "badge badge-route";
      confBadge.textContent = `Confidence: ${(routing.confidence * 100).toFixed(0)}%`;
      routeRationale.textContent = routing.reasoning;

      // Evidence Rendering
      if (routing.route === "GRAPH_MULTIHOP" && execution.evidence_context) {
        evidenceViewer.innerHTML = marked.parse(execution.evidence_context);
        evidenceCountBadge.textContent = `${execution.evidence_count} Facts`;
      } else if (routing.route === "SYMBOLIC_COMPUTE" && execution.calculation_summary) {
        evidenceViewer.innerHTML = `
          <h4>Deterministic Calculation Record</h4>
          <pre class="code-block">${execution.calculation_summary}</pre>
          <p><em>Symbolic arithmetic tool executed directly over verified filing records.</em></p>
        `;
        evidenceCountBadge.textContent = "1 Verified Calculation";
      } else {
        evidenceViewer.innerHTML = `
          <h4>Dense Vector Text Passages</h4>
          <p>Retrieved ${execution.evidence_count || 4} high-relevance chunks from Apple/Microsoft SEC 10-K filings using BGE-small embeddings.</p>
        `;
        evidenceCountBadge.textContent = `${execution.evidence_count || 4} Chunks`;
      }

      // Answer Rendering
      answerViewer.innerHTML = marked.parse(execution.answer);
      latencyBadge.textContent = `Latency: ${execution.latency_sec}s`;

    } catch (err) {
      loadingSpinner.classList.add("hidden");
      resultsPanel.classList.remove("hidden");
      answerViewer.innerHTML = `<p style="color: #f43f5e;">Error processing query: ${err.message}</p>`;
    }
  }

  submitBtn.addEventListener("click", () => {
    executeQuery(queryInput.value);
  });

  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      executeQuery(queryInput.value);
    }
  });

  // 4. Tab 2: Adaptive Routing Ablation
  const runRouteAblationBtn = document.getElementById("run-route-ablation-btn");
  const routeAblationInput = document.getElementById("route-ablation-input");
  const routeAblationResults = document.getElementById("route-ablation-results");

  runRouteAblationBtn.addEventListener("click", async () => {
    const q = routeAblationInput.value.trim();
    if (!q) return;

    runRouteAblationBtn.disabled = true;
    runRouteAblationBtn.textContent = "Comparing...";

    try {
      const resp = await fetch("/api/route_ablation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q })
      });
      const data = await resp.json();

      routeAblationResults.classList.remove("hidden");
      document.getElementById("ablation-rec-route").textContent = data.recommended_route;
      document.getElementById("ablation-vec-time").textContent = `${data.vector_route.latency_sec}s`;
      document.getElementById("ablation-graph-time").textContent = `${data.graph_route.latency_sec}s`;
      
      const savingsEl = document.getElementById("ablation-savings");
      savingsEl.textContent = `${data.latency_savings_pct > 0 ? '-' : ''}${Math.abs(data.latency_savings_pct)}%`;
      document.getElementById("ablation-route-reason").textContent = `Routing Rationale: ${data.reasoning}`;

      document.getElementById("ablation-vec-answer").innerHTML = marked.parse(data.vector_route.answer);
      document.getElementById("ablation-graph-answer").innerHTML = marked.parse(data.graph_route.answer);
    } catch (err) {
      alert("Error running route ablation: " + err.message);
    } finally {
      runRouteAblationBtn.disabled = false;
      runRouteAblationBtn.textContent = "Compare Routes";
    }
  });

  // 5. Tab 3: Evidence Formatting Ablation
  const runFormatAblationBtn = document.getElementById("run-format-ablation-btn");
  const formatAblationInput = document.getElementById("format-ablation-input");
  const formatAblationResults = document.getElementById("format-ablation-results");

  runFormatAblationBtn.addEventListener("click", async () => {
    const q = formatAblationInput.value.trim();
    if (!q) return;

    runFormatAblationBtn.disabled = true;
    runFormatAblationBtn.textContent = "Evaluating...";

    try {
      const resp = await fetch("/api/format_ablation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q })
      });
      const data = await resp.json();

      formatAblationResults.classList.remove("hidden");

      // Raw Baseline
      document.getElementById("raw-evidence-pre").textContent = data.raw_baseline.evidence_context || "No triples found.";
      document.getElementById("raw-answer-div").innerHTML = marked.parse(data.raw_baseline.answer);

      // Structured Proposed
      document.getElementById("struct-evidence-div").innerHTML = marked.parse(data.structured_proposed.evidence_context || "No structured tables.");
      document.getElementById("struct-answer-div").innerHTML = marked.parse(data.structured_proposed.answer);
    } catch (err) {
      alert("Error running formatting ablation: " + err.message);
    } finally {
      runFormatAblationBtn.disabled = false;
      runFormatAblationBtn.textContent = "Run Formatting Ablation";
    }
  });

  // Auto-run initial query on load
  executeQuery(queryInput.value);
});
