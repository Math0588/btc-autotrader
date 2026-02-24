/**
 * BTC Options Surface & Matrix Dashboard
 * ========================================
 * Frontend JavaScript – handles data fetching, Plotly charts,
 * options matrix rendering, and real-time updates.
 */

// ─── State ──────────────────────────────────────────────────────────────────
let STATE = {
    data: null,
    spot: 0,
    selectedExpiry: null,
    autoRefreshInterval: null,
    activeTab: 'surface',
};

const PLOTLY_DARK_LAYOUT = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'Inter, sans-serif', color: '#94a3b8', size: 11 },
    margin: { l: 60, r: 30, t: 40, b: 60, pad: 4 },
    scene: {
        bgcolor: 'rgba(0,0,0,0)',
        xaxis: { gridcolor: 'rgba(148,163,184,0.08)', zerolinecolor: 'rgba(148,163,184,0.1)', color: '#94a3b8' },
        yaxis: { gridcolor: 'rgba(148,163,184,0.08)', zerolinecolor: 'rgba(148,163,184,0.1)', color: '#94a3b8' },
        zaxis: { gridcolor: 'rgba(148,163,184,0.08)', zerolinecolor: 'rgba(148,163,184,0.1)', color: '#94a3b8' },
    },
    xaxis: { gridcolor: 'rgba(148,163,184,0.08)', zerolinecolor: 'rgba(148,163,184,0.1)', color: '#94a3b8' },
    yaxis: { gridcolor: 'rgba(148,163,184,0.08)', zerolinecolor: 'rgba(148,163,184,0.1)', color: '#94a3b8' },
};

const PLOTLY_CONFIG = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
};

// ─── Bootstrap ──────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initRefreshButton();
    initProbabilityCalculator();
    initStrategy();
    fetchData();

    // Auto-refresh every 30s
    STATE.autoRefreshInterval = setInterval(fetchData, 30000);
});

// ─── Data Fetching ──────────────────────────────────────────────────────────
async function fetchData() {
    const refreshBtn = document.getElementById('refresh-btn');
    refreshBtn.classList.add('spinning');

    try {
        const [spotRes, dataRes] = await Promise.all([
            fetch('/api/spot'),
            fetch('/api/options_data'),
        ]);

        const spotJson = await spotRes.json();
        const dataJson = await dataRes.json();

        if (dataJson.error) throw new Error(dataJson.error);

        STATE.spot = spotJson.spot;
        STATE.data = dataJson;

        updateHeader();
        renderActiveTab();
        hideLoading();

    } catch (err) {
        console.error('Fetch error:', err);
        document.getElementById('loading-overlay').querySelector('.loading-text').textContent =
            `Error: ${err.message}. Retrying...`;
    } finally {
        refreshBtn.classList.remove('spinning');
    }
}

function hideLoading() {
    document.getElementById('loading-overlay').classList.add('hidden');
}

function updateHeader() {
    const d = STATE.data;
    document.getElementById('spot-price').textContent = `$${STATE.spot.toLocaleString('en-US', { maximumFractionDigits: 0 })}`;
    document.getElementById('total-options').textContent = d.total_options.toLocaleString();
    document.getElementById('total-expiries').textContent = d.expiries.length;
    const now = new Date();
    document.getElementById('last-update').textContent = now.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    document.getElementById('footer-timestamp').textContent = `${now.toLocaleDateString('fr-FR')} ${now.toLocaleTimeString('fr-FR')}`;
}

// ─── Tab Navigation ─────────────────────────────────────────────────────────
function initTabs() {
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const tabId = tab.dataset.tab;
            document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.getElementById(`content-${tabId}`).classList.add('active');
            STATE.activeTab = tabId;
            renderActiveTab();
        });
    });
}

function initRefreshButton() {
    document.getElementById('refresh-btn').addEventListener('click', fetchData);
}

function renderActiveTab() {
    if (!STATE.data) return;
    switch (STATE.activeTab) {
        case 'surface': renderVolSurface(); break;
        case 'matrix': renderOptionsMatrix(); break;
        case 'smiles': renderVolSmiles(); break;
        case 'heatmap': renderHeatmap(); break;
        case 'probability': /* no re-render needed, user-driven */ break;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// VOLATILITY SURFACE (3D)
// ═══════════════════════════════════════════════════════════════════════════
function renderVolSurface() {
    const surface = STATE.data.surface;
    if (!surface || !surface.x || surface.x.length === 0) {
        document.getElementById('surface-chart').innerHTML =
            '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#64748b;">No surface data available</div>';
        return;
    }

    const colorscale = document.getElementById('surface-colorscale').value;
    const xAxisType = document.getElementById('surface-xaxis').value;

    const xLabel = xAxisType === 'moneyness' ? 'Moneyness (K/S)' : 'Strike ($)';
    const xValues = xAxisType === 'moneyness' ? surface.x : surface.x.map(m => m * STATE.spot);

    // Main surface
    const surfaceTrace = {
        type: 'surface',
        x: xValues,
        y: surface.y,
        z: surface.z,
        colorscale: colorscale,
        opacity: 0.92,
        lighting: {
            ambient: 0.6,
            diffuse: 0.5,
            specular: 0.3,
            roughness: 0.8,
        },
        contours: {
            z: { show: true, usecolormap: true, highlightcolor: '#f7931a', project: { z: true } },
        },
        colorbar: {
            title: { text: 'IV (%)', font: { size: 11, color: '#94a3b8' } },
            tickfont: { size: 10, color: '#94a3b8' },
            bgcolor: 'rgba(0,0,0,0)',
            bordercolor: 'rgba(148,163,184,0.1)',
            len: 0.6,
            thickness: 15,
        },
        hovertemplate:
            `<b>${xLabel}:</b> %{x:.3f}<br>` +
            '<b>DTE:</b> %{y:.0f} days<br>' +
            '<b>IV:</b> %{z:.1f}%<extra></extra>',
    };

    // ATM line on the surface
    const atmX = xAxisType === 'moneyness' ? 1.0 : STATE.spot;

    const layout = {
        ...PLOTLY_DARK_LAYOUT,
        title: {
            text: `BTC Implied Volatility Surface — Spot: $${STATE.spot.toLocaleString()}`,
            font: { size: 15, color: '#f1f5f9', family: 'Inter, sans-serif' },
            x: 0.02,
            xanchor: 'left',
        },
        scene: {
            ...PLOTLY_DARK_LAYOUT.scene,
            xaxis: {
                ...PLOTLY_DARK_LAYOUT.scene.xaxis,
                title: { text: xLabel, font: { size: 11 } },
            },
            yaxis: {
                ...PLOTLY_DARK_LAYOUT.scene.yaxis,
                title: { text: 'Days to Expiry', font: { size: 11 } },
            },
            zaxis: {
                ...PLOTLY_DARK_LAYOUT.scene.zaxis,
                title: { text: 'Implied Volatility (%)', font: { size: 11 } },
            },
            camera: {
                eye: { x: 1.6, y: -1.8, z: 0.7 },
                up: { x: 0, y: 0, z: 1 },
            },
            aspectratio: { x: 1.2, y: 1.2, z: 0.7 },
        },
        margin: { l: 0, r: 0, t: 50, b: 0 },
    };

    Plotly.newPlot('surface-chart', [surfaceTrace], layout, PLOTLY_CONFIG);

    // Attach event listeners for controls
    ['surface-colorscale', 'surface-xaxis', 'surface-type'].forEach(id => {
        const el = document.getElementById(id);
        el.onchange = renderVolSurface;
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// OPTIONS MATRIX
// ═══════════════════════════════════════════════════════════════════════════
function renderOptionsMatrix() {
    const d = STATE.data;
    const container = document.getElementById('options-matrix-table');
    const expirySelect = document.getElementById('matrix-expiry');
    const displaySelect = document.getElementById('matrix-display');
    const allExpiriesToggle = document.getElementById('matrix-all-expiries');

    // Populate expiry dropdown
    const currentVal = expirySelect.value;
    expirySelect.innerHTML = '';
    d.expiries.forEach(exp => {
        const opt = document.createElement('option');
        opt.value = exp;
        opt.textContent = `${exp} (${d.expiry_groups[exp].dte.toFixed(0)}d)`;
        expirySelect.appendChild(opt);
    });
    if (currentVal && d.expiries.includes(currentVal)) {
        expirySelect.value = currentVal;
    }

    const showAll = allExpiriesToggle.checked;
    const displayField = displaySelect.value;
    const expiriesToShow = showAll ? d.expiries : [expirySelect.value || d.expiries[0]];

    let html = '';

    expiriesToShow.forEach(exp => {
        const group = d.expiry_groups[exp];
        if (!group) return;

        const calls = group.calls || [];
        const puts = group.puts || [];

        // Get all unique strikes
        const strikeSet = new Set();
        calls.forEach(c => strikeSet.add(c.strike));
        puts.forEach(p => strikeSet.add(p.strike));
        const strikes = Array.from(strikeSet).sort((a, b) => a - b);

        // Build lookup maps
        const callMap = {};
        calls.forEach(c => callMap[c.strike] = c);
        const putMap = {};
        puts.forEach(p => putMap[p.strike] = p);

        const totalContracts = calls.length + puts.length;

        // Expiry group header
        html += `
        <div class="expiry-group-header" onclick="this.nextElementSibling.classList.toggle('collapsed')">
            <h3>
                📅 ${exp}
                <span class="dte-badge">${group.dte.toFixed(0)} DTE</span>
                <span class="count-badge">${totalContracts} contracts</span>
            </h3>
        </div>`;

        html += '<div class="matrix-table-wrapper">';
        html += '<table class="options-matrix">';

        // Header — Calls columns | Strike | Puts columns
        const callHeaders = getDisplayHeaders(displayField, 'call');
        const putHeaders = getDisplayHeaders(displayField, 'put');

        html += '<thead><tr>';
        callHeaders.forEach(h => {
            html += `<th class="calls-header">${h}</th>`;
        });
        html += '<th class="strike-header">STRIKE</th>';
        putHeaders.forEach(h => {
            html += `<th class="puts-header">${h}</th>`;
        });
        html += '</tr></thead>';

        // Body
        html += '<tbody>';
        strikes.forEach(strike => {
            const call = callMap[strike];
            const put = putMap[strike];
            const isATM = Math.abs(strike - STATE.spot) / STATE.spot < 0.02;
            const isCallITM = strike < STATE.spot;
            const isPutITM = strike > STATE.spot;

            html += '<tr>';

            // Call cells
            html += renderCellGroup(call, displayField, 'call', isCallITM, isATM);

            // Strike cell
            html += `<td class="strike-cell ${isATM ? 'atm' : ''}">$${strike.toLocaleString()}</td>`;

            // Put cells
            html += renderCellGroup(put, displayField, 'put', isPutITM, isATM);

            html += '</tr>';
        });
        html += '</tbody></table></div>';
    });

    container.innerHTML = html;

    // Attach event listeners
    expirySelect.onchange = renderOptionsMatrix;
    displaySelect.onchange = renderOptionsMatrix;
    allExpiriesToggle.onchange = renderOptionsMatrix;
}

function getDisplayHeaders(field, type) {
    const prefix = type === 'call' ? 'C' : 'P';
    switch (field) {
        case 'iv':
            return [`${prefix} IV%`, `${prefix} Bid`, `${prefix} Ask`, `${prefix} Vol`, `${prefix} OI`];
        case 'mark_usd':
            return [`${prefix} Mark$`, `${prefix} Bid$`, `${prefix} Ask$`, `${prefix} Vol`];
        case 'delta':
            return [`${prefix} Δ`, `${prefix} Γ`, `${prefix} ν`, `${prefix} Θ`];
        case 'gamma':
            return [`${prefix} Γ`, `${prefix} IV%`, `${prefix} Vol`];
        case 'vega':
            return [`${prefix} ν`, `${prefix} IV%`, `${prefix} Vol`];
        case 'theta':
            return [`${prefix} Θ`, `${prefix} IV%`, `${prefix} Vol`];
        case 'volume':
            return [`${prefix} Vol`, `${prefix} OI`, `${prefix} IV%`];
        case 'open_interest':
            return [`${prefix} OI`, `${prefix} Vol`, `${prefix} IV%`];
        default:
            return [`${prefix} IV%`, `${prefix} Bid`, `${prefix} Ask`];
    }
}

function renderCellGroup(opt, field, type, isITM, isATM) {
    const cls = type === 'call' ? 'call-cell' : 'put-cell';
    const itmCls = isITM ? ' itm' : '';
    const atmCls = isATM ? ' atm' : '';
    const extraCls = `${cls}${itmCls}${atmCls}`;

    if (!opt) {
        const count = getDisplayHeaders(field, type).length;
        let cells = '';
        for (let i = 0; i < count; i++) cells += `<td class="${extraCls}">—</td>`;
        return cells;
    }

    switch (field) {
        case 'iv':
            return `
                <td class="${extraCls}">${opt.iv.toFixed(1)}%</td>
                <td class="${extraCls}">${fmtUSD(opt.bid_usd)}</td>
                <td class="${extraCls}">${fmtUSD(opt.ask_usd)}</td>
                <td class="${extraCls}">${opt.volume}</td>
                <td class="${extraCls}">${opt.open_interest}</td>`;
        case 'mark_usd':
            return `
                <td class="${extraCls}">${fmtUSD(opt.mark_usd)}</td>
                <td class="${extraCls}">${fmtUSD(opt.bid_usd)}</td>
                <td class="${extraCls}">${fmtUSD(opt.ask_usd)}</td>
                <td class="${extraCls}">${opt.volume}</td>`;
        case 'delta':
            return `
                <td class="${extraCls}">${opt.delta.toFixed(4)}</td>
                <td class="${extraCls}">${opt.gamma.toFixed(8)}</td>
                <td class="${extraCls}">${opt.vega.toFixed(2)}</td>
                <td class="${extraCls}">${opt.theta.toFixed(2)}</td>`;
        case 'gamma':
            return `
                <td class="${extraCls}">${opt.gamma.toFixed(8)}</td>
                <td class="${extraCls}">${opt.iv.toFixed(1)}%</td>
                <td class="${extraCls}">${opt.volume}</td>`;
        case 'vega':
            return `
                <td class="${extraCls}">${opt.vega.toFixed(2)}</td>
                <td class="${extraCls}">${opt.iv.toFixed(1)}%</td>
                <td class="${extraCls}">${opt.volume}</td>`;
        case 'theta':
            return `
                <td class="${extraCls}">${opt.theta.toFixed(2)}</td>
                <td class="${extraCls}">${opt.iv.toFixed(1)}%</td>
                <td class="${extraCls}">${opt.volume}</td>`;
        case 'volume':
            return `
                <td class="${extraCls}">${opt.volume}</td>
                <td class="${extraCls}">${opt.open_interest}</td>
                <td class="${extraCls}">${opt.iv.toFixed(1)}%</td>`;
        case 'open_interest':
            return `
                <td class="${extraCls}">${opt.open_interest}</td>
                <td class="${extraCls}">${opt.volume}</td>
                <td class="${extraCls}">${opt.iv.toFixed(1)}%</td>`;
        default:
            return `
                <td class="${extraCls}">${opt.iv.toFixed(1)}%</td>
                <td class="${extraCls}">${fmtUSD(opt.bid_usd)}</td>
                <td class="${extraCls}">${fmtUSD(opt.ask_usd)}</td>`;
    }
}

function fmtUSD(val) {
    if (val == null || val === 0) return '—';
    if (val >= 1000) return `$${(val / 1000).toFixed(1)}k`;
    if (val >= 1) return `$${val.toFixed(2)}`;
    return `$${val.toFixed(4)}`;
}

// ═══════════════════════════════════════════════════════════════════════════
// VOLATILITY SMILES
// ═══════════════════════════════════════════════════════════════════════════
function renderVolSmiles() {
    const d = STATE.data;
    const checkboxContainer = document.getElementById('smile-expiry-checkboxes');

    // Create checkboxes for each expiry (up to 8 initially checked)
    if (checkboxContainer.children.length === 0) {
        d.expiries.forEach((exp, i) => {
            const label = document.createElement('label');
            label.className = `expiry-check-label ${i < 6 ? 'checked' : ''}`;
            label.innerHTML = `<input type="checkbox" value="${exp}" ${i < 6 ? 'checked' : ''}>${exp}`;
            label.addEventListener('click', () => {
                const cb = label.querySelector('input');
                setTimeout(() => {
                    label.classList.toggle('checked', cb.checked);
                    plotSmiles();
                }, 10);
            });
            checkboxContainer.appendChild(label);
        });
    }

    plotSmiles();
}

function plotSmiles() {
    const d = STATE.data;
    const checked = Array.from(document.querySelectorAll('#smile-expiry-checkboxes input:checked'))
        .map(cb => cb.value);

    if (checked.length === 0) {
        Plotly.purge('smile-chart');
        return;
    }

    // Color palette
    const colors = [
        '#f7931a', '#10b981', '#3b82f6', '#8b5cf6', '#ef4444',
        '#ec4899', '#06b6d4', '#eab308', '#f97316', '#14b8a6',
        '#6366f1', '#d946ef',
    ];

    const traces = [];

    checked.forEach((exp, i) => {
        const group = d.expiry_groups[exp];
        if (!group) return;
        const color = colors[i % colors.length];

        // Calls
        const callData = (group.calls || []).filter(c => c.iv > 0).sort((a, b) => a.strike - b.strike);
        if (callData.length > 0) {
            traces.push({
                x: callData.map(c => c.strike),
                y: callData.map(c => c.iv),
                mode: 'lines+markers',
                name: `${exp} Calls (${group.dte.toFixed(0)}d)`,
                line: { color: color, width: 2.5, shape: 'spline' },
                marker: { size: 4, color: color },
                hovertemplate: `<b>${exp} Call</b><br>Strike: $%{x:,.0f}<br>IV: %{y:.1f}%<extra></extra>`,
            });
        }

        // Puts (dashed)
        const putData = (group.puts || []).filter(p => p.iv > 0).sort((a, b) => a.strike - b.strike);
        if (putData.length > 0) {
            traces.push({
                x: putData.map(p => p.strike),
                y: putData.map(p => p.iv),
                mode: 'lines+markers',
                name: `${exp} Puts (${group.dte.toFixed(0)}d)`,
                line: { color: color, width: 1.5, dash: 'dot', shape: 'spline' },
                marker: { size: 3, color: color, symbol: 'diamond' },
                hovertemplate: `<b>${exp} Put</b><br>Strike: $%{x:,.0f}<br>IV: %{y:.1f}%<extra></extra>`,
            });
        }
    });

    // ATM line
    traces.push({
        x: [STATE.spot, STATE.spot],
        y: [0, 200],
        mode: 'lines',
        name: `ATM ($${STATE.spot.toLocaleString()})`,
        line: { color: '#f7931a', width: 1.5, dash: 'dash' },
        showlegend: true,
        hoverinfo: 'skip',
    });

    const layout = {
        ...PLOTLY_DARK_LAYOUT,
        title: {
            text: 'Volatility Smiles by Expiry',
            font: { size: 15, color: '#f1f5f9' },
            x: 0.02, xanchor: 'left',
        },
        xaxis: {
            ...PLOTLY_DARK_LAYOUT.xaxis,
            title: { text: 'Strike ($)', font: { size: 12 } },
            tickformat: '$,.0f',
        },
        yaxis: {
            ...PLOTLY_DARK_LAYOUT.yaxis,
            title: { text: 'Implied Volatility (%)', font: { size: 12 } },
            rangemode: 'tozero',
        },
        legend: {
            bgcolor: 'rgba(17,24,39,0.8)',
            bordercolor: 'rgba(148,163,184,0.1)',
            font: { size: 10, color: '#94a3b8' },
            x: 1, xanchor: 'right', y: 1,
        },
        hovermode: 'closest',
    };

    Plotly.newPlot('smile-chart', traces, layout, PLOTLY_CONFIG);
}

// ═══════════════════════════════════════════════════════════════════════════
// IV HEATMAP
// ═══════════════════════════════════════════════════════════════════════════
function renderHeatmap() {
    const d = STATE.data;
    const optionType = document.getElementById('heatmap-type').value;
    const colorscale = document.getElementById('heatmap-colorscale').value;

    // Filter data for the selected type
    const filtered = d.options.filter(o => o.type === optionType && o.iv > 0);
    if (filtered.length === 0) {
        document.getElementById('heatmap-chart').innerHTML =
            '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#64748b;">No data</div>';
        return;
    }

    // Build heatmap grid: X = expiry, Y = strike, Z = IV
    const expiries = [...new Set(filtered.map(o => o.expiry))];
    // Sort by DTE
    expiries.sort((a, b) => {
        const aDte = filtered.find(o => o.expiry === a)?.dte || 0;
        const bDte = filtered.find(o => o.expiry === b)?.dte || 0;
        return aDte - bDte;
    });

    const strikes = [...new Set(filtered.map(o => o.strike))].sort((a, b) => a - b);

    // Limit strikes around ATM for readability
    const atmIdx = strikes.findIndex(s => s >= STATE.spot);
    const range = 25;
    const startIdx = Math.max(0, atmIdx - range);
    const endIdx = Math.min(strikes.length, atmIdx + range);
    const displayStrikes = strikes.slice(startIdx, endIdx);

    // Build Z matrix
    const lookup = {};
    filtered.forEach(o => {
        lookup[`${o.expiry}_${o.strike}`] = o.iv;
    });

    const z = displayStrikes.map(strike =>
        expiries.map(exp => lookup[`${exp}_${strike}`] || null)
    );

    const trace = {
        z: z,
        x: expiries,
        y: displayStrikes.map(s => `$${s.toLocaleString()}`),
        type: 'heatmap',
        colorscale: colorscale,
        hovertemplate:
            '<b>Expiry:</b> %{x}<br>' +
            '<b>Strike:</b> %{y}<br>' +
            '<b>IV:</b> %{z:.1f}%<extra></extra>',
        colorbar: {
            title: { text: 'IV (%)', font: { size: 11, color: '#94a3b8' } },
            tickfont: { size: 10, color: '#94a3b8' },
            bgcolor: 'rgba(0,0,0,0)',
            bordercolor: 'rgba(148,163,184,0.1)',
        },
        zsmooth: 'best',
    };

    const layout = {
        ...PLOTLY_DARK_LAYOUT,
        title: {
            text: `IV Heatmap — ${optionType === 'C' ? 'Calls' : 'Puts'}`,
            font: { size: 15, color: '#f1f5f9' },
            x: 0.02, xanchor: 'left',
        },
        xaxis: {
            ...PLOTLY_DARK_LAYOUT.xaxis,
            title: { text: 'Expiry', font: { size: 12 } },
            tickangle: -45,
        },
        yaxis: {
            ...PLOTLY_DARK_LAYOUT.yaxis,
            title: { text: 'Strike', font: { size: 12 } },
            type: 'category',
        },
        margin: { l: 90, r: 30, t: 50, b: 80 },
    };

    Plotly.newPlot('heatmap-chart', [trace], layout, PLOTLY_CONFIG);

    // Attach event listeners
    document.getElementById('heatmap-type').onchange = renderHeatmap;
    document.getElementById('heatmap-colorscale').onchange = renderHeatmap;
}

// ═══════════════════════════════════════════════════════════════════════════
// PROBABILITY CALCULATOR
// ═══════════════════════════════════════════════════════════════════════════

function initProbabilityCalculator() {
    // Set default date to 3 months from now
    const dateInput = document.getElementById('proba-target-date');
    const defaultDate = new Date();
    defaultDate.setMonth(defaultDate.getMonth() + 3);
    dateInput.value = defaultDate.toISOString().split('T')[0];
    dateInput.min = new Date().toISOString().split('T')[0];

    // Scenario toggle buttons
    const aboveBtn = document.getElementById('scenario-above');
    const belowBtn = document.getElementById('scenario-below');

    aboveBtn.addEventListener('click', () => {
        aboveBtn.classList.add('active');
        belowBtn.classList.remove('active');
    });

    belowBtn.addEventListener('click', () => {
        belowBtn.classList.add('active');
        aboveBtn.classList.remove('active');
    });

    // Calculate button
    document.getElementById('proba-calculate-btn').addEventListener('click', calculateProbability);

    // Preset buttons
    document.querySelectorAll('.preset-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const price = parseFloat(btn.dataset.price);
            const scenario = btn.dataset.scenario;

            document.getElementById('proba-target-price').value = price;

            if (scenario === 'above') {
                aboveBtn.classList.add('active');
                belowBtn.classList.remove('active');
            } else {
                belowBtn.classList.add('active');
                aboveBtn.classList.remove('active');
            }

            calculateProbability();
        });
    });

    // Enter key on inputs
    document.getElementById('proba-target-price').addEventListener('keydown', e => {
        if (e.key === 'Enter') calculateProbability();
    });
    document.getElementById('proba-target-date').addEventListener('keydown', e => {
        if (e.key === 'Enter') calculateProbability();
    });
}

async function calculateProbability() {
    const targetPrice = parseFloat(document.getElementById('proba-target-price').value);
    const targetDate = document.getElementById('proba-target-date').value;
    const scenario = document.getElementById('scenario-above').classList.contains('active') ? 'above' : 'below';

    if (!targetPrice || targetPrice <= 0) {
        showProbaError('Veuillez entrer un prix cible valide.');
        return;
    }
    if (!targetDate) {
        showProbaError('Veuillez sélectionner une date cible.');
        return;
    }

    // Show loading
    showProbaState('loading');

    try {
        const resp = await fetch('/api/probability', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                target_price: targetPrice,
                target_date: targetDate,
                scenario: scenario,
            }),
        });

        const data = await resp.json();

        if (data.error) {
            showProbaError(data.error);
            return;
        }

        displayProbabilityResult(data);

    } catch (err) {
        showProbaError(`Erreur réseau: ${err.message}`);
    }
}

function showProbaState(state) {
    const placeholder = document.getElementById('proba-placeholder');
    const loading = document.getElementById('proba-loading');
    const result = document.getElementById('proba-result');
    const error = document.getElementById('proba-error');

    placeholder.classList.add('hidden');
    loading.classList.add('hidden');
    result.classList.add('hidden');
    error.classList.add('hidden');

    switch (state) {
        case 'placeholder': placeholder.classList.remove('hidden'); break;
        case 'loading': loading.classList.remove('hidden'); break;
        case 'result': result.classList.remove('hidden'); break;
        case 'error': error.classList.remove('hidden'); break;
    }
}

function showProbaError(msg) {
    showProbaState('error');
    document.getElementById('proba-error-msg').textContent = msg;
}

function displayProbabilityResult(data) {
    showProbaState('result');

    const isAbove = data.scenario === 'above';
    const scenarioText = isAbove
        ? `P( BTC > $${data.target_price.toLocaleString()} ) au ${data.target_date}`
        : `P( BTC < $${data.target_price.toLocaleString()} ) au ${data.target_date}`;

    // Scenario label
    const label = document.getElementById('proba-scenario-label');
    label.textContent = scenarioText;
    label.style.color = isAbove ? 'var(--green)' : 'var(--red)';

    // Big number
    const bigNum = document.getElementById('proba-big-number');
    const probRN = data.probability_risk_neutral;
    bigNum.textContent = `${probRN.toFixed(1)}%`;
    bigNum.style.color = isAbove ? 'var(--green)' : 'var(--red)';
    bigNum.style.textShadow = isAbove
        ? '0 0 30px rgba(16,185,129,0.4)'
        : '0 0 30px rgba(239,68,68,0.4)';
    // Re-trigger animation
    bigNum.style.animation = 'none';
    bigNum.offsetHeight; // force reflow
    bigNum.style.animation = 'numberPop 0.5s var(--ease-spring)';

    // Real-world probability
    document.getElementById('proba-rw-value').textContent = `${data.probability_real_world.toFixed(1)}%`;

    // IV
    document.getElementById('proba-iv-value').textContent = `${data.interpolated_iv_pct.toFixed(1)}%`;
    document.getElementById('proba-iv-method').textContent = `Méthode: ${data.interpolation_method}`;

    // Moneyness
    document.getElementById('proba-moneyness-value').textContent = data.target_moneyness.toFixed(3);
    document.getElementById('proba-dte-display').textContent = `${data.dte_days.toFixed(0)} jours`;

    // Quadrant probabilities
    document.getElementById('quad-above-rn').textContent = `${data.prob_above_rn}%`;
    document.getElementById('quad-below-rn').textContent = `${data.prob_below_rn}%`;
    document.getElementById('quad-above-rw').textContent = `${data.prob_above_rw}%`;
    document.getElementById('quad-below-rw').textContent = `${data.prob_below_rw}%`;

    // BS Parameters grid
    const paramsGrid = document.getElementById('proba-params-grid');
    const params = [
        { label: 'Spot (S)', value: `$${data.spot.toLocaleString()}` },
        { label: 'Strike (K)', value: `$${data.target_price.toLocaleString()}` },
        { label: 'σ (sigma)', value: `${(data.sigma * 100).toFixed(2)}%` },
        { label: 'T (années)', value: data.T_years.toFixed(4) },
        { label: 'DTE (jours)', value: `${data.dte_days.toFixed(1)}j` },
        { label: 'd₁', value: data.d1.toFixed(4) },
        { label: 'd₂ (RN)', value: data.d2.toFixed(4) },
        { label: 'd₂ (RW)', value: data.d2_real.toFixed(4) },
        { label: 'Taux sans risque', value: `${(data.risk_free_rate * 100).toFixed(1)}%` },
        { label: 'Drift μ', value: `${(data.real_world_drift * 100).toFixed(0)}%` },
        { label: 'Moneyness', value: data.target_moneyness.toFixed(4) },
        { label: 'Interpolation', value: data.interpolation_method },
    ];
    paramsGrid.innerHTML = params.map(p => `
        <div class="param-item">
            <span class="param-label">${p.label}</span>
            <span class="param-value">${p.value}</span>
        </div>
    `).join('');

    // Nearby options table
    const nearbyDiv = document.getElementById('proba-nearby-table');
    if (data.nearby_options && data.nearby_options.length > 0) {
        let html = `<table class="nearby-table">
            <thead><tr>
                <th>Expiry</th><th>Strike</th><th>DTE</th><th>IV%</th><th>Moneyness</th><th>Distance</th>
            </tr></thead><tbody>`;
        data.nearby_options.forEach(n => {
            html += `<tr>
                <td>${n.expiry}</td>
                <td>$${n.strike.toLocaleString()}</td>
                <td>${n.dte.toFixed(0)}j</td>
                <td>${n.iv.toFixed(1)}%</td>
                <td>${n.moneyness.toFixed(3)}</td>
                <td>${n.distance.toFixed(3)}</td>
            </tr>`;
        });
        html += '</tbody></table>';
        nearbyDiv.innerHTML = html;
    } else {
        nearbyDiv.innerHTML = '<p style="color:var(--text-muted);font-size:0.8rem;">Pas d\'options proches disponibles</p>';
    }

    // Probability distribution chart
    renderProbabilityDistChart(data);
}

function renderProbabilityDistChart(data) {
    const S = data.spot;
    const K = data.target_price;
    const sigma = data.sigma;
    const T = data.T_years;
    const r = data.risk_free_rate;
    const isAbove = data.scenario === 'above';

    // Generate log-normal distribution PDF
    const mu_ln = Math.log(S) + (r - 0.5 * sigma * sigma) * T;
    const sigma_ln = sigma * Math.sqrt(T);

    const nPoints = 500;
    const xMin = S * 0.3;
    const xMax = S * 3.0;
    const step = (xMax - xMin) / nPoints;

    const xVals = [];
    const yVals = [];
    const fillColors = [];

    for (let i = 0; i <= nPoints; i++) {
        const x = xMin + i * step;
        if (x <= 0) continue;
        // Log-normal PDF
        const lnx = Math.log(x);
        const pdf = (1 / (x * sigma_ln * Math.sqrt(2 * Math.PI))) *
            Math.exp(-0.5 * Math.pow((lnx - mu_ln) / sigma_ln, 2));
        xVals.push(x);
        yVals.push(pdf);
    }

    // Build shaded area for the probability region
    const xFill = [];
    const yFill = [];
    for (let i = 0; i < xVals.length; i++) {
        const inRegion = isAbove ? xVals[i] >= K : xVals[i] <= K;
        if (inRegion) {
            xFill.push(xVals[i]);
            yFill.push(yVals[i]);
        }
    }

    const traces = [
        // Full distribution
        {
            x: xVals,
            y: yVals,
            mode: 'lines',
            name: 'Distribution Log-Normale',
            line: { color: '#94a3b8', width: 2 },
            hovertemplate: 'Prix: $%{x:,.0f}<br>Densité: %{y:.6f}<extra></extra>',
        },
        // Shaded probability region
        {
            x: xFill,
            y: yFill,
            fill: 'tozeroy',
            mode: 'lines',
            name: `P(${isAbove ? 'Above' : 'Below'}) = ${data.probability_risk_neutral.toFixed(1)}%`,
            line: { color: 'transparent' },
            fillcolor: isAbove ? 'rgba(16,185,129,0.3)' : 'rgba(239,68,68,0.3)',
            hoverinfo: 'skip',
        },
    ];

    // Current spot line
    const maxY = Math.max(...yVals) * 1.1;
    traces.push({
        x: [S, S],
        y: [0, maxY],
        mode: 'lines',
        name: `Spot: $${S.toLocaleString()}`,
        line: { color: '#f7931a', width: 2, dash: 'dash' },
        hoverinfo: 'skip',
    });

    // Target price line
    traces.push({
        x: [K, K],
        y: [0, maxY],
        mode: 'lines',
        name: `Cible: $${K.toLocaleString()}`,
        line: { color: isAbove ? '#10b981' : '#ef4444', width: 2.5 },
        hoverinfo: 'skip',
    });

    const layout = {
        ...PLOTLY_DARK_LAYOUT,
        title: {
            text: `Distribution Log-Normale de BTC à T+${data.dte_days.toFixed(0)}j`,
            font: { size: 13, color: '#f1f5f9' },
            x: 0.02, xanchor: 'left',
        },
        xaxis: {
            ...PLOTLY_DARK_LAYOUT.xaxis,
            title: { text: 'Prix BTC ($)', font: { size: 11 } },
            tickformat: '$,.0f',
        },
        yaxis: {
            ...PLOTLY_DARK_LAYOUT.yaxis,
            title: { text: 'Densité de probabilité', font: { size: 11 } },
            showticklabels: false,
        },
        legend: {
            bgcolor: 'rgba(17,24,39,0.8)',
            bordercolor: 'rgba(148,163,184,0.1)',
            font: { size: 10, color: '#94a3b8' },
            x: 1, xanchor: 'right', y: 1,
        },
        margin: { l: 40, r: 20, t: 40, b: 50 },
        showlegend: true,
    };

    Plotly.newPlot('proba-dist-chart', traces, layout, { ...PLOTLY_CONFIG, displayModeBar: false });
}

// ═══════════════════════════════════════════════════════════════════════════
// STRATEGY: Polymarket ↔ Options Arbitrage
// ═══════════════════════════════════════════════════════════════════════════

function initStrategy() {
    document.getElementById('strategy-scan-btn').addEventListener('click', scanStrategy);
}

async function scanStrategy() {
    // Show loading
    showStrategyState('loading');

    try {
        const resp = await fetch('/api/strategy');
        const data = await resp.json();

        if (data.error) {
            showStrategyError(data.error);
            return;
        }

        displayStrategyResults(data);
    } catch (err) {
        showStrategyError(`Erreur réseau: ${err.message}`);
    }
}

function showStrategyState(state) {
    const placeholder = document.getElementById('strategy-placeholder');
    const loading = document.getElementById('strategy-loading');
    const results = document.getElementById('strategy-results');
    const error = document.getElementById('strategy-error');
    const summary = document.getElementById('strategy-summary');

    placeholder.classList.add('hidden');
    loading.classList.add('hidden');
    results.classList.add('hidden');
    error.classList.add('hidden');
    summary.style.display = 'none';

    switch (state) {
        case 'placeholder': placeholder.classList.remove('hidden'); break;
        case 'loading': loading.classList.remove('hidden'); break;
        case 'results':
            results.classList.remove('hidden');
            summary.style.display = 'grid';
            break;
        case 'error': error.classList.remove('hidden'); break;
    }
}

function showStrategyError(msg) {
    showStrategyState('error');
    document.getElementById('strategy-error-msg').textContent = msg;
}

function displayStrategyResults(data) {
    showStrategyState('results');

    const s = data.summary;
    const opps = data.opportunities || [];
    const trades = data.trades || [];

    // Summary cards
    document.getElementById('strat-total-markets').textContent = s.total_markets_scanned;
    document.getElementById('strat-grade-a').textContent = s.grade_a_count || 0;
    document.getElementById('strat-win-rate').textContent = `${(s.weighted_win_rate || 0).toFixed(0)}%`;
    document.getElementById('strat-total-exposure').textContent = `$${s.total_exposure_usd.toLocaleString()}`;

    const pnlEl = document.getElementById('strat-expected-pnl');
    pnlEl.textContent = `$${s.total_expected_pnl.toFixed(2)}`;
    pnlEl.style.color = s.total_expected_pnl >= 0 ? 'var(--green)' : 'var(--red)';

    // Sort: show trades first (sorted by grade A>B>C then by E[PnL]), then NO_TRADE
    const gradeOrder = { A: 0, B: 1, C: 2 };
    const sorted = [...opps].sort((a, b) => {
        // Trades first
        const aIsTrade = a.direction !== 'NO_TRADE' ? 0 : 1;
        const bIsTrade = b.direction !== 'NO_TRADE' ? 0 : 1;
        if (aIsTrade !== bIsTrade) return aIsTrade - bIsTrade;
        // Then by grade
        const ga = gradeOrder[a.grade] ?? 9;
        const gb = gradeOrder[b.grade] ?? 9;
        if (ga !== gb) return ga - gb;
        // Then by expected PnL
        return (b.expected_pnl_per_dollar || 0) - (a.expected_pnl_per_dollar || 0);
    });

    // Render table
    const tbody = document.getElementById('strategy-tbody');
    let html = '';

    sorted.forEach(o => {
        if (o.direction === 'NO_TRADE') return; // Only show actionable trades

        const gradeClass = o.grade === 'A' ? 'grade-a' : o.grade === 'B' ? 'grade-b' : 'grade-c';
        const gradeEmoji = o.grade === 'A' ? '🟢' : o.grade === 'B' ? '🟡' : '⚪';

        const winColor = o.win_probability > 60 ? '#10b981' : o.win_probability > 45 ? '#f59e0b' : '#ef4444';
        const winBarWidth = Math.min(o.win_probability, 100);

        const edgeClass = o.edge_rn > 0 ? 'edge-positive' : o.edge_rn < 0 ? 'edge-negative' : '';
        const edgeSign = o.edge_rn > 0 ? '+' : '';

        const desc = o.trade_description || (o.direction === 'BUY_NO' ? `BTC < $${o.barrier_price.toLocaleString()}` : `BTC > $${o.barrier_price.toLocaleString()}`);

        html += `<tr class="${gradeClass}-row">
            <td><span class="grade-badge ${gradeClass}">${gradeEmoji} ${o.grade}</span></td>
            <td class="trade-desc-cell" title="${o.title}">
                <div class="trade-desc">${desc}</div>
                <div class="trade-dte-sub">${o.dte_days.toFixed(0)}j — ${o.title.substring(o.title.lastIndexOf(' on ') + 4).replace('?', '')}</div>
            </td>
            <td>$${o.barrier_price.toLocaleString()}</td>
            <td>${o.dte_days.toFixed(0)}j</td>
            <td>
                <div class="win-rate-cell">
                    <span class="win-rate-num" style="color:${winColor}">${o.win_probability.toFixed(0)}%</span>
                    <div class="win-rate-bar">
                        <div class="win-rate-fill" style="width:${winBarWidth}%;background:${winColor}"></div>
                    </div>
                </div>
            </td>
            <td style="color:#10b981;font-weight:600">+${o.profit_if_win_pct.toFixed(0)}%</td>
            <td class="${edgeClass}">${edgeSign}${o.edge_rn.toFixed(1)}%</td>
            <td>${o.entry_price.toFixed(2)}$</td>
            <td>$${o.position_usd.toFixed(0)}</td>
            <td style="color:${o.expected_pnl_per_dollar >= 0 ? 'var(--green)' : 'var(--red)'}">
                ${o.expected_pnl_per_dollar >= 0 ? '+' : ''}${(o.expected_pnl_per_dollar * 100).toFixed(1)}%
            </td>
        </tr>`;
    });

    tbody.innerHTML = html || '<tr><td colspan="10" style="color:var(--text-muted);padding:30px">Aucun marché BTC barrier trouvé sur Polymarket</td></tr>';

    // Render edge chart
    renderStrategyChart(opps);
}

function renderStrategyChart(opps) {
    if (!opps || opps.length === 0) {
        Plotly.purge('strategy-edge-chart');
        return;
    }

    // Bar chart: Edge per market
    const labels = opps.map(o => {
        const barrier = o.barrier_price >= 1000 ? `$${(o.barrier_price / 1000).toFixed(0)}k` : `$${o.barrier_price}`;
        return `${o.scenario === 'above' ? '>' : '<'} ${barrier} (${o.dte_days.toFixed(0)}j)`;
    });

    const edges = opps.map(o => o.edge_rn);
    const colors = edges.map(e => e > 3 ? '#10b981' : e < -3 ? '#ef4444' : '#64748b');

    const pmProbs = opps.map(o => o.pm_implied_prob);
    const modelProbs = opps.map(o => o.model_prob_rn);

    const traces = [
        {
            x: labels,
            y: pmProbs,
            name: 'Polymarket',
            type: 'bar',
            marker: { color: 'rgba(139, 92, 246, 0.7)', line: { width: 1, color: '#8b5cf6' } },
            hovertemplate: '%{x}<br>PM: %{y:.1f}%<extra></extra>',
        },
        {
            x: labels,
            y: modelProbs,
            name: 'Modèle Options',
            type: 'bar',
            marker: { color: 'rgba(247, 147, 26, 0.7)', line: { width: 1, color: '#f7931a' } },
            hovertemplate: '%{x}<br>Model: %{y:.1f}%<extra></extra>',
        },
    ];

    const layout = {
        ...PLOTLY_DARK_LAYOUT,
        title: {
            text: 'Polymarket vs Modèle Options — Probabilités Comparées',
            font: { size: 14, color: '#f1f5f9' },
            x: 0.02, xanchor: 'left',
        },
        barmode: 'group',
        xaxis: {
            ...PLOTLY_DARK_LAYOUT.xaxis,
            tickangle: -45,
            tickfont: { size: 9 },
        },
        yaxis: {
            ...PLOTLY_DARK_LAYOUT.yaxis,
            title: { text: 'Probabilité (%)', font: { size: 11 } },
        },
        legend: {
            bgcolor: 'rgba(17,24,39,0.8)',
            bordercolor: 'rgba(148,163,184,0.1)',
            font: { size: 10, color: '#94a3b8' },
            x: 1, xanchor: 'right', y: 1,
        },
        margin: { l: 50, r: 20, t: 50, b: 100 },
    };

    Plotly.newPlot('strategy-edge-chart', traces, layout, { ...PLOTLY_CONFIG, displayModeBar: false });
}

function truncate(str, maxLen) {
    return str.length > maxLen ? str.substring(0, maxLen) + '...' : str;
}

// ─── Utility ────────────────────────────────────────────────────────────────
function debounce(fn, ms) {
    let timer;
    return (...args) => {
        clearTimeout(timer);
        timer = setTimeout(() => fn(...args), ms);
    };
}
