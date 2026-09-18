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

      // Graph Connectivity & PPR Badges
      const connBadge = document.getElementById("connectivity-badge");
      const pprBadge = document.getElementById("ppr-badge");
      const entitiesRow = document.getElementById("entities-row");

      const connData = routing.graph_connectivity;
      if (connData && connData.is_connected) {
        connBadge.textContent = `Graph Density: ${connData.matched_edge_count} Edges`;
        connBadge.classList.remove("hidden");
      } else {
        connBadge.classList.add("hidden");
      }

      if (routing.ppr_ranked_triples_count && routing.ppr_ranked_triples_count > 0) {
        pprBadge.textContent = `PPR Subgraph Ranking: Top ${routing.ppr_ranked_triples_count} Triples`;
        pprBadge.classList.remove("hidden");
      } else {
        pprBadge.classList.add("hidden");
      }

      // Render Extracted Entities
      if (routing.extracted_entities && routing.extracted_entities.length > 0) {
        entitiesRow.innerHTML = routing.extracted_entities.map(e => `
          <span class="chip" style="font-size: 11px; padding: 2px 8px; border-color: rgba(56,189,248,0.3);">
            <strong style="color: #38bdf8;">${e.name}</strong> 
            <span class="chip-type" style="background: rgba(168,85,247,0.2); color: #c084fc; margin-left: 4px;">${e.type}</span>
          </span>
        `).join("");
        entitiesRow.classList.remove("hidden");
      } else {
        entitiesRow.innerHTML = "";
        entitiesRow.classList.add("hidden");
      }

      // Evidence Rendering
      if (routing.route === "SIMPLE_VECTOR") {
        // For SIMPLE_VECTOR route: show top 2 retrieved text passage previews (first 3-4 lines each)
        let passageHtml = "";
        const chunks = execution.retrieved_chunks || [];
        const topChunks = chunks.slice(0, 2);
        
        if (topChunks.length > 0) {
          passageHtml = topChunks.map((chunk, idx) => {
            const rawLines = chunk.split('\n').map(l => l.trim()).filter(l => l.length > 0);
            const previewLines = rawLines.slice(0, 4).join('\n');
            const previewText = previewLines.length > 0 ? previewLines : chunk.slice(0, 260);
            return `
              <div style="background: rgba(15,23,42,0.6); border: 1px solid rgba(56,189,248,0.25); border-radius: 6px; padding: 12px; margin-bottom: 12px;">
                <div style="font-weight: 600; font-size: 12px; color: #38bdf8; margin-bottom: 6px; display: flex; align-items: center; justify-content: space-between;">
                  <span>📄 SEC 10-K Excerpt Passage ${idx + 1}</span>
                  <span style="font-size: 10px; background: rgba(56,189,248,0.15); color: #7dd3fc; padding: 2px 6px; border-radius: 3px;">Top Embedding Match</span>
                </div>
                <p style="font-size: 13px; line-height: 1.55; color: #cbd5e1; margin: 0; white-space: pre-line; font-style: italic;">"${previewText}..."</p>
              </div>
            `;
          }).join("");
        } else if (execution.evidence_context) {
          passageHtml = marked.parse(execution.evidence_context);
        } else {
          passageHtml = `<p class="placeholder-text">No passages retrieved.</p>`;
        }

        evidenceViewer.innerHTML = `
          <h4 style="color: #38bdf8; margin-bottom: 10px; font-size: 14px;">📄 Top Retrieved Text Passages</h4>
          ${passageHtml}
        `;
        evidenceCountBadge.textContent = `${topChunks.length || execution.evidence_count || 2} Passages`;

      } else if (routing.route === "GRAPH_MULTIHOP") {
        // For GRAPH_MULTIHOP route: show top 4-5 PPR-ranked graph triples with entity types
        evidenceViewer.innerHTML = `
          <div class="graph-evidence-container">
            <h4 style="color: #c084fc; margin-bottom: 10px; font-size: 14px;">🕸️ Top PPR-Ranked Graph Triples</h4>
            ${marked.parse(execution.evidence_context || "No graph triples available.")}
          </div>
        `;
        evidenceCountBadge.textContent = `${execution.evidence_count || 5} PPR Triples`;

      } else if (routing.route === "HYBRID") {
        // For HYBRID route: split left panel into two sections (Graph Evidence & Vector Passages)
        evidenceViewer.innerHTML = `
          <div class="hybrid-evidence-container">
            ${marked.parse(execution.evidence_context || "No hybrid evidence available.")}
          </div>
        `;
        evidenceCountBadge.textContent = `${execution.evidence_count || 6} Merged Facts`;

      } else if (routing.route === "SYMBOLIC_COMPUTE") {
        // For SYMBOLIC_COMPUTE route: show extracted metric values and the calculation summary
        let metricBox = "";
        const ev = execution.extracted_values;
        if (ev && ev.metric) {
          metricBox = `
            <div style="background: rgba(15,23,42,0.6); border: 1px solid rgba(16,185,129,0.3); border-radius: 6px; padding: 12px; margin-bottom: 12px;">
              <div style="font-weight: 600; font-size: 11px; color: #34d399; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.05em;">
                📊 Extracted SEC Filing Metric Values
              </div>
              <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 13px;">
                <div><span style="color: #94a3b8;">Entity:</span> <strong style="color: #f1f5f9;">${ev.ticker}</strong></div>
                <div><span style="color: #94a3b8;">Metric:</span> <strong style="color: #f1f5f9;">${ev.metric}</strong></div>
                <div><span style="color: #94a3b8;">FY${ev.year_start}:</span> <strong style="color: #38bdf8;">$${ev.value_start ? ev.value_start.toLocaleString() : '--'} M</strong></div>
                <div><span style="color: #94a3b8;">FY${ev.year_end}:</span> <strong style="color: #38bdf8;">$${ev.value_end ? ev.value_end.toLocaleString() : '--'} M</strong></div>
              </div>
            </div>
          `;
        }

        evidenceViewer.innerHTML = `
          <div class="symbolic-evidence-container">
            <h4 style="color: #34d399; margin-bottom: 10px; font-size: 14px;">🧮 Symbolic Computation Record</h4>
            ${metricBox}
            <div style="font-size: 12px; font-weight: 600; color: #94a3b8; margin-bottom: 6px;">Calculation Summary & Verification:</div>
            <pre class="code-block" style="border-color: rgba(16,185,129,0.3); color: #6ee7b7; font-size: 13px; line-height: 1.5; padding: 12px;">${execution.calculation_summary || 'Formula evaluated over graph metric nodes.'}</pre>
            <p style="font-size: 12px; color: #94a3b8; margin-top: 6px;"><em>Symbolic arithmetic tool executed directly over verified SEC filing records without probabilistic rounding.</em></p>
          </div>
        `;
        evidenceCountBadge.textContent = "1 Verified Calculation";

      } else {
        evidenceViewer.innerHTML = marked.parse(execution.evidence_context || "<p class='placeholder-text'>No evidence retrieved.</p>");
        evidenceCountBadge.textContent = `${execution.evidence_count || 0} Sources`;
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
      
      // Graph Connectivity & PPR Status
      const densityEl = document.getElementById("ablation-graph-density");
      if (densityEl) {
        const edgeCount = (data.graph_connectivity && typeof data.graph_connectivity.matched_edge_count === 'number') 
          ? data.graph_connectivity.matched_edge_count.toLocaleString() 
          : "0";
        densityEl.textContent = `${edgeCount} Edges`;
      }

      const pprEl = document.getElementById("ablation-ppr-status");
      if (pprEl) {
        const count = data.ppr_ranked_triples_count || 0;
        pprEl.textContent = count > 0 ? `Active (Top ${count})` : "Bypassed (Vector)";
      }

      const savingsEl = document.getElementById("ablation-savings");
      savingsEl.textContent = `${data.latency_savings_pct > 0 ? '-' : ''}${Math.abs(data.latency_savings_pct)}%`;
      document.getElementById("ablation-route-reason").textContent = `Routing Rationale: ${data.reasoning}`;

      // Tab 2 Entities Row
      const tab2EntitiesRow = document.getElementById("ablation-entities-row");
      if (tab2EntitiesRow) {
        if (data.extracted_entities && data.extracted_entities.length > 0) {
          tab2EntitiesRow.innerHTML = data.extracted_entities.map(e => `
            <span class="chip" style="font-size: 11px; padding: 2px 8px; border-color: rgba(56,189,248,0.3);">
              <strong style="color: #38bdf8;">${e.name}</strong> 
              <span class="chip-type" style="background: rgba(168,85,247,0.2); color: #c084fc; margin-left: 4px;">${e.type}</span>
            </span>
          `).join("");
        } else {
          tab2EntitiesRow.innerHTML = `<span style="font-size: 12px; color: #94a3b8;">No complex graph entities required for simple vector lookup.</span>`;
        }
      }

      const vecBadge = document.getElementById("ablation-vec-badge");
      if (vecBadge) vecBadge.textContent = `${data.vector_route.latency_sec}s (${data.vector_route.evidence_count || 4} chunks)`;
      const graphBadge = document.getElementById("ablation-graph-badge");
      if (graphBadge) graphBadge.textContent = `${data.graph_route.latency_sec}s (${data.graph_route.evidence_count || 12} triples)`;
      const hybridBadge = document.getElementById("ablation-hybrid-badge");
      if (hybridBadge && data.hybrid_route) {
        hybridBadge.textContent = `${data.hybrid_route.latency_sec}s (${data.hybrid_route.evidence_count} merged facts)`;
      }

      document.getElementById("ablation-vec-answer").innerHTML = marked.parse(data.vector_route.answer);
      document.getElementById("ablation-graph-answer").innerHTML = marked.parse(data.graph_route.answer);
      const hybridAnsEl = document.getElementById("ablation-hybrid-answer");
      if (hybridAnsEl && data.hybrid_route) {
        hybridAnsEl.innerHTML = marked.parse(data.hybrid_route.answer);
      }
    } catch (err) {
      alert("Error running route ablation: " + err.message);
    } finally {
      runRouteAblationBtn.disabled = false;
      runRouteAblationBtn.textContent = "Inspect Adaptive Routing";
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
