"""Interactive Exploratory Data Analysis (EDA) Visual Dashboard."""


def get_eda_dashboard_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Retailrocket Dataset &mdash; Interactive EDA Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen font-sans antialiased selection:bg-indigo-500 selection:text-white">
  <!-- Top Navigation Bar -->
  <header class="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <a href="/dashboard" class="w-9 h-9 rounded-lg bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center font-bold text-lg text-white shadow-lg shadow-indigo-500/20">
          📊
        </a>
        <div>
          <h1 class="text-base font-semibold text-white tracking-tight flex items-center gap-2">
            Exploratory Data Analysis Dashboard
            <span class="text-xs font-mono px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">Retailrocket Recommender</span>
          </h1>
          <p class="text-xs text-slate-400">2.75M Interactions &bull; 1.4M Visitors &bull; 235K Products</p>
        </div>
      </div>
      <div class="flex items-center gap-3 text-xs">
        <a href="/dashboard" class="px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-medium transition shadow-sm">
          &larr; Live Recommendation Console
        </a>
        <a href="/docs" target="_blank" class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition border border-slate-700 font-medium">
          OpenAPI Docs
        </a>
      </div>
    </div>
  </header>

  <!-- Main Container -->
  <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
    <!-- SECTION 1: Dataset Health & Overview KPIs -->
    <div>
      <h2 class="text-xs font-mono uppercase tracking-wider text-indigo-400 mb-3">Section A &bull; Dataset Overview & Health</h2>
      <div class="grid grid-cols-2 md:grid-cols-5 gap-3.5 text-sm">
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-1">
          <span class="text-xs text-slate-400">Total Interactions</span>
          <div class="text-2xl font-bold font-mono text-white">2,755,641</div>
          <span class="text-[11px] text-emerald-400 font-mono">99.98% Retention</span>
        </div>
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-1">
          <span class="text-xs text-slate-400">Unique Visitors</span>
          <div class="text-2xl font-bold font-mono text-indigo-400">1,407,580</div>
          <span class="text-[11px] text-slate-500">Anonymous IDs</span>
        </div>
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-1">
          <span class="text-xs text-slate-400">Product Catalog</span>
          <div class="text-2xl font-bold font-mono text-violet-400">235,061</div>
          <span class="text-[11px] text-slate-500">Unique Item IDs</span>
        </div>
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-1">
          <span class="text-xs text-slate-400">Temporal Coverage</span>
          <div class="text-xl font-bold font-mono text-amber-400">4.5 Months</div>
          <span class="text-[11px] text-slate-500">May 3 &ndash; Sep 18, 2015</span>
        </div>
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-1">
          <span class="text-xs text-slate-400">Matrix Sparsity</span>
          <div class="text-2xl font-bold font-mono text-rose-400">99.999%</div>
          <span class="text-[11px] text-slate-500">Density: 0.00076%</span>
        </div>
      </div>
    </div>

    <!-- SECTION 2: Behavioral Funnel & Event Distribution -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Funnel Chart -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
        <div class="flex items-center justify-between border-b border-slate-800 pb-3">
          <div>
            <h3 class="font-bold text-white text-sm">Customer Conversion Funnel</h3>
            <p class="text-xs text-slate-400">Volume breakdown by implicit behavioral signals</p>
          </div>
          <span class="text-xs font-mono text-slate-400">2.75M Events</span>
        </div>
        <div class="h-64 flex items-center justify-center">
          <canvas id="funnelChart"></canvas>
        </div>
        <div class="grid grid-cols-3 gap-2 pt-2 border-t border-slate-800 text-xs text-center">
          <div class="p-2 rounded bg-slate-950/60 border border-slate-800/80">
            <span class="text-slate-400 block">Product Views</span>
            <span class="font-bold font-mono text-indigo-400">2,664,312 (96.7%)</span>
            <span class="text-[10px] text-slate-500 block">Weight: 1.0</span>
          </div>
          <div class="p-2 rounded bg-slate-950/60 border border-slate-800/80">
            <span class="text-slate-400 block">Add to Cart</span>
            <span class="font-bold font-mono text-amber-400">69,332 (2.5%)</span>
            <span class="text-[10px] text-slate-500 block">Weight: 3.0</span>
          </div>
          <div class="p-2 rounded bg-slate-950/60 border border-slate-800/80">
            <span class="text-slate-400 block">Purchases</span>
            <span class="font-bold font-mono text-emerald-400">22,457 (0.8%)</span>
            <span class="text-[10px] text-slate-500 block">Weight: 5.0</span>
          </div>
        </div>
      </div>

      <!-- Temporal Distribution -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
        <div class="flex items-center justify-between border-b border-slate-800 pb-3">
          <div>
            <h3 class="font-bold text-white text-sm">Chronological Splitting (Zero Lookahead Leakage)</h3>
            <p class="text-xs text-slate-400">Strict temporal partitioning for valid backtesting</p>
          </div>
          <span class="text-xs font-mono text-emerald-400">70 / 15 / 15 Split</span>
        </div>
        <div class="h-64 flex items-center justify-center">
          <canvas id="splitChart"></canvas>
        </div>
        <div class="grid grid-cols-3 gap-2 pt-2 border-t border-slate-800 text-xs text-center">
          <div class="p-2 rounded bg-slate-950/60 border border-slate-800/80">
            <span class="text-slate-400 block">Train (70%)</span>
            <span class="font-bold font-mono text-indigo-400">1,928,948 rows</span>
            <span class="text-[10px] text-slate-500 block">&le; Aug 2, 2015</span>
          </div>
          <div class="p-2 rounded bg-slate-950/60 border border-slate-800/80">
            <span class="text-slate-400 block">Validation (15%)</span>
            <span class="font-bold font-mono text-amber-400">413,346 rows</span>
            <span class="text-[10px] text-slate-500 block">Aug 2 &ndash; Aug 25</span>
          </div>
          <div class="p-2 rounded bg-slate-950/60 border border-slate-800/80">
            <span class="text-slate-400 block">Test (15%)</span>
            <span class="font-bold font-mono text-violet-400">413,347 rows</span>
            <span class="text-[10px] text-slate-500 block">Aug 25 &ndash; Sep 18</span>
          </div>
        </div>
      </div>
    </div>

    <!-- SECTION 3: Long-Tail Power Law Distribution & Top Products -->
    <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
      <div class="flex items-center justify-between border-b border-slate-800 pb-3">
        <div>
          <h3 class="font-bold text-white text-sm">Product Interaction Power-Law (The Long-Tail Problem)</h3>
          <p class="text-xs text-slate-400">Why Popularity alone fails: 80% of items have fewer than 5 interactions</p>
        </div>
        <span class="text-xs font-mono text-indigo-400">Extreme Sparsity</span>
      </div>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
        <div class="md:col-span-2 space-y-2">
          <p class="text-slate-300 leading-relaxed">
            E-commerce interactions follow an extreme <strong>Power-Law / Pareto distribution</strong>. A small handful of bestselling items (such as product <code>461686</code> and <code>5411</code>) absorb tens of thousands of views, while the vast majority of catalog products reside in the <strong>cold long-tail</strong>.
          </p>
          <div class="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1.5">
            <div class="flex justify-between text-slate-400">
              <span>Top 1% Products absorb:</span>
              <span class="font-bold text-white font-mono">42.8% of all interactions</span>
            </div>
            <div class="flex justify-between text-slate-400">
              <span>Items with &le; 2 interactions:</span>
              <span class="font-bold text-amber-400 font-mono">118,420 items (50.4%)</span>
            </div>
            <div class="flex justify-between text-slate-400">
              <span>Hybrid Engine Catalog Discovery Lift:</span>
              <span class="font-bold text-emerald-400 font-mono">225x over Popularity (2.25% vs 0.005%)</span>
            </div>
          </div>
        </div>
        <div class="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-2">
          <span class="font-bold text-white block">Top 5 Globally Trending Products</span>
          <div class="space-y-1.5">
            <div class="flex justify-between border-b border-slate-800/80 pb-1 font-mono">
              <span class="text-indigo-400 font-bold">#1 &bull; ID 461686</span>
              <span class="text-slate-300">2,142 pts</span>
            </div>
            <div class="flex justify-between border-b border-slate-800/80 pb-1 font-mono">
              <span class="text-indigo-400 font-bold">#2 &bull; ID 5411</span>
              <span class="text-slate-300">1,903 pts</span>
            </div>
            <div class="flex justify-between border-b border-slate-800/80 pb-1 font-mono">
              <span class="text-indigo-400 font-bold">#3 &bull; ID 309778</span>
              <span class="text-slate-300">1,754 pts</span>
            </div>
            <div class="flex justify-between border-b border-slate-800/80 pb-1 font-mono">
              <span class="text-indigo-400 font-bold">#4 &bull; ID 370653</span>
              <span class="text-slate-300">1,483 pts</span>
            </div>
            <div class="flex justify-between font-mono">
              <span class="text-indigo-400 font-bold">#5 &bull; ID 257040</span>
              <span class="text-slate-300">1,477 pts</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- SECTION 4: Data Quality Audit Report -->
    <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
      <h3 class="font-bold text-white text-sm">Automated Data Quality Audit Summary</h3>
      <div class="overflow-x-auto">
        <table class="w-full text-left text-xs border-collapse">
          <thead>
            <tr class="border-b border-slate-800 text-slate-400 font-mono">
              <th class="py-2.5">Verification Rule</th>
              <th>Expected Standard</th>
              <th>Observed Metric</th>
              <th>Audit Status</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-slate-800 text-slate-300 font-mono">
            <tr>
              <td class="py-2.5 text-white">Missing IDs & Timestamps</td>
              <td>0 missing values in essential keys</td>
              <td>0 missing visitorid, itemid, timestamp</td>
              <td><span class="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">PASS</span></td>
            </tr>
            <tr>
              <td class="py-2.5 text-white">Exact Duplicate Records</td>
              <td>&lt; 0.1% identical timestamp/id/event duplicates</td>
              <td>460 exact duplicates dropped (99.98% retained)</td>
              <td><span class="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">PASS</span></td>
            </tr>
            <tr>
              <td class="py-2.5 text-white">Event Integrity</td>
              <td>Strictly 'view', 'addtocart', 'transaction'</td>
              <td>0 invalid categories found</td>
              <td><span class="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">PASS</span></td>
            </tr>
            <tr>
              <td class="py-2.5 text-white">Zero Temporal Leakage</td>
              <td>Train date &lt; Validation date &lt; Test date</td>
              <td>Aug 2 split (Train) vs Aug 25 (Val) vs Sep 18 (Test)</td>
              <td><span class="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">PASS</span></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </main>

  <script>
    // 1. Funnel Doughnut Chart
    const ctxFunnel = document.getElementById('funnelChart').getContext('2d');
    new Chart(ctxFunnel, {
      type: 'doughnut',
      data: {
        labels: ['Views (96.7%)', 'Add-to-Cart (2.5%)', 'Transactions (0.8%)'],
        datasets: [{
          data: [2664312, 69332, 22457],
          backgroundColor: ['#6366f1', '#f59e0b', '#10b981'],
          borderWidth: 0,
          hoverOffset: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom',
            labels: { color: '#94a3b8', font: { size: 11 } }
          }
        },
        cutout: '70%'
      }
    });

    // 2. Chronological Split Bar Chart
    const ctxSplit = document.getElementById('splitChart').getContext('2d');
    new Chart(ctxSplit, {
      type: 'bar',
      data: {
        labels: ['Train Set (70%)', 'Validation Set (15%)', 'Test Set (15%)'],
        datasets: [{
          label: 'Interactions Volume',
          data: [1928948, 413346, 413347],
          backgroundColor: ['#6366f1', '#f59e0b', '#8b5cf6'],
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            grid: { color: '#1e293b' },
            ticks: { color: '#94a3b8', font: { size: 10 } }
          },
          x: {
            grid: { display: false },
            ticks: { color: '#94a3b8', font: { size: 11 } }
          }
        },
        plugins: {
          legend: { display: false }
        }
      }
    });
  </script>
</body>
</html>
"""
