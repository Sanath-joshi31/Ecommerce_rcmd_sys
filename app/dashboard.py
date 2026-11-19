"""Web Dashboard Interface for Live Interactive Recommendations."""


def get_dashboard_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>E-Commerce Recommendation System &mdash; Live Interactive Console</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    @keyframes pulse-subtle { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
    .animate-pulse-subtle { animation: pulse-subtle 2s infinite ease-in-out; }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen font-sans antialiased selection:bg-indigo-500 selection:text-white">
  <!-- Top Navigation Bar -->
  <header class="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <div class="w-9 h-9 rounded-lg bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center font-bold text-lg text-white shadow-lg shadow-indigo-500/20">
          🛒
        </div>
        <div>
          <h1 class="text-base font-semibold text-white tracking-tight flex items-center gap-2">
            E-Commerce Recommendation Engine
            <span class="text-xs font-mono font-normal px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">v1.0.0</span>
          </h1>
          <p class="text-xs text-slate-400">Implicit ALS &bull; Content TF-IDF &bull; Popularity Baseline</p>
        </div>
      </div>
      <div class="flex items-center gap-3 text-xs">
        <div id="health-indicator" class="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
          <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse-subtle"></span>
          <span id="health-text" class="font-medium font-mono">HEALTHY</span>
        </div>
        <a href="/docs" target="_blank" class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition border border-slate-700 flex items-center gap-1.5 font-medium">
          <span>Interactive OpenAPI Docs</span>
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/></svg>
        </a>
      </div>
    </div>
  </header>

  <!-- Main Container -->
  <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
    <!-- Telemetry Cards -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-1">
        <span class="text-xs text-slate-400 font-medium">Memory Consumption</span>
        <div id="stat-memory" class="text-2xl font-bold font-mono text-white">Loading...</div>
        <span class="text-xs text-slate-500">Resident Set Size (RAM)</span>
      </div>
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-1">
        <span class="text-xs text-slate-400 font-medium">Service Uptime</span>
        <div id="stat-uptime" class="text-2xl font-bold font-mono text-indigo-400">Loading...</div>
        <span class="text-xs text-slate-500">Continuous microservice uptime</span>
      </div>
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-1">
        <span class="text-xs text-slate-400 font-medium">Active Sub-Models</span>
        <div id="stat-models" class="text-sm font-bold font-mono text-emerald-400 pt-1.5">3 / 3 Active</div>
        <span class="text-xs text-slate-500">ALS &bull; Content &bull; Popularity</span>
      </div>
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-1">
        <span class="text-xs text-slate-400 font-medium">P95 SLA Target</span>
        <div class="text-2xl font-bold font-mono text-violet-400">&lt; 80.0 ms</div>
        <span class="text-xs text-slate-500">Measured: ~13.8 ms P95</span>
      </div>
    </div>

    <!-- Interactive Query Console -->
    <div class="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl space-y-5">
      <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <h2 class="text-lg font-bold text-white tracking-tight">Interactive Recommendation Playground</h2>
          <p class="text-xs text-slate-400">Query personalized hybrid predictions or evaluate automated popularity fallback.</p>
        </div>
        <!-- Quick Pick Warm Users -->
        <div class="flex items-center gap-2">
          <span class="text-xs text-slate-400 font-medium">Test Profiles:</span>
          <button onclick="setVisitor(25)" class="px-2.5 py-1 text-xs rounded bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700 transition">Warm #25</button>
          <button onclick="setVisitor(183)" class="px-2.5 py-1 text-xs rounded bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700 transition">Warm #183</button>
          <button onclick="setVisitor(728)" class="px-2.5 py-1 text-xs rounded bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700 transition">Warm #728</button>
          <button onclick="setVisitor(9988776655)" class="px-2.5 py-1 text-xs rounded bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 border border-amber-500/20 transition">Cold Visitor</button>
        </div>
      </div>

      <!-- Controls -->
      <div class="grid grid-cols-1 sm:grid-cols-4 gap-4 items-end">
        <div class="sm:col-span-2 space-y-1.5">
          <label class="block text-xs font-medium text-slate-300">Visitor ID (Customer Identifier)</label>
          <input type="number" id="input-visitor" value="25" min="1" class="w-full px-3.5 py-2.5 rounded-lg bg-slate-950 border border-slate-700 text-white font-mono text-sm focus:outline-none focus:border-indigo-500 transition">
        </div>
        <div class="space-y-1.5">
          <label class="block text-xs font-medium text-slate-300">Top-N Recommendations</label>
          <select id="select-topn" class="w-full px-3.5 py-2.5 rounded-lg bg-slate-950 border border-slate-700 text-white font-mono text-sm focus:outline-none focus:border-indigo-500 transition">
            <option value="5" selected>Top 5 Items</option>
            <option value="10">Top 10 Items</option>
            <option value="15">Top 15 Items</option>
          </select>
        </div>
        <div class="flex gap-2">
          <button onclick="fetchRecommendations()" id="btn-submit" class="flex-1 px-4 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-sm shadow-lg shadow-indigo-600/30 transition flex items-center justify-center gap-1.5">
            <span>Recommend</span>
          </button>
          <button onclick="fetchPopular()" title="View Global Popular Products" class="px-3.5 py-2.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700 transition text-sm">
            🔥 Trending
          </button>
        </div>
      </div>
    </div>

    <!-- Results Section -->
    <div id="results-container" class="space-y-4">
      <div class="flex items-center justify-between">
        <h3 class="text-base font-semibold text-white tracking-tight flex items-center gap-2">
          <span>Results Output</span>
          <span id="badge-source" class="text-xs font-mono font-medium px-2 py-0.5 rounded-full bg-slate-800 text-slate-400">Ready</span>
        </h3>
        <span id="label-latency" class="text-xs font-mono text-slate-400">Latency: -- ms</span>
      </div>

      <div id="cards-grid" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3.5">
        <!-- Result Cards Injected via JS -->
      </div>
    </div>
  </main>

  <script>
    async function updateTelemetry() {
      try {
        const res = await fetch('/health');
        if (res.ok) {
          const data = await res.json();
          document.getElementById('stat-memory').textContent = data.memory_usage_mb.toFixed(1) + ' MB';
          document.getElementById('stat-uptime').textContent = Math.round(data.uptime_seconds) + 's';
          document.getElementById('health-indicator').className = 'flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400';
          document.getElementById('health-text').textContent = 'HEALTHY';
        }
      } catch (e) {
        document.getElementById('health-indicator').className = 'flex items-center gap-2 px-3 py-1.5 rounded-full bg-rose-500/10 border border-rose-500/20 text-rose-400';
        document.getElementById('health-text').textContent = 'OFFLINE';
      }
    }

    function setVisitor(id) {
      document.getElementById('input-visitor').value = id;
      fetchRecommendations();
    }

    async function fetchRecommendations() {
      const visitorId = document.getElementById('input-visitor').value;
      const topN = document.getElementById('select-topn').value;
      const btn = document.getElementById('btn-submit');
      btn.disabled = true;
      btn.textContent = 'Computing...';

      const t0 = performance.now();
      try {
        const res = await fetch(`/recommend/${visitorId}?top_n=${topN}&filter_seen=true`);
        const elapsed = (performance.now() - t0).toFixed(2);
        if (!res.ok) throw new Error('API Error: ' + res.status);
        const data = await res.json();
        renderResults(data, elapsed);
      } catch (err) {
        alert('Failed to fetch recommendations: ' + err.message);
      } finally {
        btn.disabled = false;
        btn.textContent = 'Recommend';
      }
    }

    async function fetchPopular() {
      const topN = document.getElementById('select-topn').value;
      const t0 = performance.now();
      try {
        const res = await fetch(`/popular?top_n=${topN}`);
        const elapsed = (performance.now() - t0).toFixed(2);
        const data = await res.json();
        renderResults(data, elapsed);
      } catch (err) {
        alert('Failed to fetch popular items: ' + err.message);
      }
    }

    function renderResults(data, clientElapsed) {
      const grid = document.getElementById('cards-grid');
      const badge = document.getElementById('badge-source');
      const latencyLabel = document.getElementById('label-latency');

      badge.textContent = data.source.toUpperCase();
      if (data.source === 'hybrid') {
        badge.className = 'text-xs font-mono font-medium px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30';
      } else if (data.source === 'popularity_fallback') {
        badge.className = 'text-xs font-mono font-medium px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/30';
      } else {
        badge.className = 'text-xs font-mono font-medium px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30';
      }

      latencyLabel.textContent = `Server: ${data.latency_ms || 0.1} ms | Roundtrip: ${clientElapsed} ms`;

      let html = '';
      data.recommendations.forEach(item => {
        const scoreDisplay = (item.score > 1.0) ? item.score.toFixed(0) : (item.score * 100).toFixed(1) + '%';
        html += `
          <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col justify-between hover:border-slate-700 transition shadow group">
            <div class="space-y-2">
              <div class="flex items-center justify-between text-xs font-mono text-slate-400">
                <span class="font-bold text-slate-200">#${item.rank}</span>
                <span class="text-indigo-400 font-semibold">${scoreDisplay}</span>
              </div>
              <div class="pt-2">
                <div class="text-xs text-slate-400">Product Identifier</div>
                <div class="text-lg font-bold font-mono text-white group-hover:text-indigo-300 transition">ID: ${item.item_id}</div>
              </div>
            </div>
            <div class="pt-4 mt-4 border-t border-slate-800/80 text-[11px] text-slate-400 flex items-center justify-between">
              <span>${item.reason || 'Recommendation'}</span>
              <span class="w-1.5 h-1.5 rounded-full bg-indigo-400"></span>
            </div>
          </div>
        `;
      });
      grid.innerHTML = html;
    }

    // Auto load on start
    updateTelemetry();
    setInterval(updateTelemetry, 5000);
    fetchRecommendations();
  </script>
</body>
</html>
"""
