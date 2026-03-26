const COLORS = {
    accent: '#6c63ff',
    accentBright: '#8b83ff',
    green: '#34d399',
    orange: '#f59e0b',
    red: '#ef4444',
    blue: '#3b82f6',
    text: '#e4e4ef',
    textMuted: '#8888a0',
    textDim: '#5a5a70',
    border: '#2a2a3a',
    bgCard: '#12121a',
    bg: '#0a0a0f',
};

const ALGO_COLORS = {
    'Cleora':           '#a78bfa',
    'ProNE':            '#f59e0b',
    'RandNE':           '#ef4444',
    'NetMF':            '#3b82f6',
    'DeepWalk':         '#f472b6',
};

const DATASETS = ['ego-Facebook', 'Cora', 'CiteSeer', 'PubMed', 'PPI'];
const ALGORITHMS = ['Cleora', 'ProNE', 'RandNE', 'NetMF', 'DeepWalk'];

const SUMMARY_DATA = {
    'Cleora':          [0.989, 0.816, 0.584, 0.802, 0.103],
    'ProNE':           [0.062, 0.159, 0.140, 0.324, 0.042],
    'RandNE':          [0.198, 0.280, 0.250, 0.357, 0.039],
    'NetMF':           [0.979, 0.808, 0.579, null,  null],
    'DeepWalk':        [0.969, 0.806, 0.587, null,  null],
};

const FAILURE_STATUS = {
    'NetMF':           [null, null, null, 'OOM', 'OOM'],
    'DeepWalk':        [null, null, null, 'Timed Out', 'Timed Out'],
};

const SPEED_DATA = {
    algorithms: ['Cleora', 'RandNE', 'ProNE', 'NetMF', 'DeepWalk'],
    facebook:   [0.77,     0.10,     2.78,    44.0,    46.9],
    cora:       [0.27,     0.05,     0.36,    6.09,    58.8],
    citeseer:   [0.25,     0.03,     0.12,    18.1,    24.6],
    pubmed:     [0.65,     0.18,     1.21,    null,    null],
    ppi:        [1.46,     0.77,     3.41,    null,    null],
    roadnet:    [31.500,   null,     null,    null,    null],
};

const SPEED_FAILURE = {
    pubmed:     [null, null, null, 'OOM', 'Timed Out'],
    ppi:        [null, null, null, 'OOM', 'Timed Out'],
    roadnet:    [null, 'OOM', 'OOM', 'OOM', 'OOM'],
};

const MEMORY_DATA = {
    algorithms: ['Cleora', 'RandNE', 'ProNE', 'DeepWalk', 'NetMF'],
    facebook:   [21,       40,       64,      538,        1047],
    cora:       [15,       24,       41,      260,        313],
    citeseer:   [18,       30,       50,      362,        374],
    pubmed:     [98,       176,      293,     null,       null],
    ppi:        [251,      540,      875,     null,       null],
    roadnet:    [4129,     null,     null,    null,       null],
};

const MEMORY_FAILURE = {
    pubmed:     [null, null, null, 'OOM', 'OOM'],
    ppi:        [null, null, null, 'OOM', 'OOM'],
    roadnet:    [null, 'OOM', 'OOM', 'OOM', 'OOM'],
};

const SCATTER_DATA = {
    'ego-Facebook': {
        'Cleora':          { acc: 0.989, time: 0.77 },
        'NetMF':           { acc: 0.979, time: 44.0 },
        'DeepWalk':        { acc: 0.969, time: 46.9 },
        'RandNE':          { acc: 0.198, time: 0.10 },
        'ProNE':           { acc: 0.062, time: 2.78 },
    },
    'Cora': {
        'Cleora':          { acc: 0.816, time: 0.27 },
        'NetMF':           { acc: 0.808, time: 6.09 },
        'DeepWalk':        { acc: 0.806, time: 58.8 },
        'RandNE':          { acc: 0.280, time: 0.05 },
        'ProNE':           { acc: 0.159, time: 0.36 },
    },
    'CiteSeer': {
        'Cleora':          { acc: 0.584, time: 0.25 },
        'NetMF':           { acc: 0.579, time: 18.1 },
        'DeepWalk':        { acc: 0.587, time: 24.6 },
        'RandNE':          { acc: 0.250, time: 0.03 },
        'ProNE':           { acc: 0.140, time: 0.12 },
    },
    'PubMed': {
        'Cleora':          { acc: 0.802, time: 0.65 },
        'RandNE':          { acc: 0.357, time: 0.18 },
        'ProNE':           { acc: 0.324, time: 1.21 },
    },
    'PPI': {
        'Cleora':          { acc: 0.103, time: 1.46 },
        'ProNE':           { acc: 0.042, time: 3.41 },
        'RandNE':          { acc: 0.039, time: 0.77 },
    },
};

function chartDefaults() {
    Chart.defaults.color = COLORS.textMuted;
    Chart.defaults.borderColor = COLORS.border;
    Chart.defaults.font.family = "'Graphik', -apple-system, BlinkMacSystemFont, sans-serif";
    Chart.defaults.font.size = 12;
    Chart.defaults.plugins.tooltip.backgroundColor = COLORS.bgCard;
    Chart.defaults.plugins.tooltip.borderColor = COLORS.border;
    Chart.defaults.plugins.tooltip.borderWidth = 1;
    Chart.defaults.plugins.tooltip.titleColor = COLORS.text;
    Chart.defaults.plugins.tooltip.bodyColor = COLORS.textMuted;
    Chart.defaults.plugins.tooltip.padding = 12;
    Chart.defaults.plugins.tooltip.cornerRadius = 8;
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.legend.labels.pointStyle = 'circle';
    Chart.defaults.plugins.legend.labels.padding = 16;
}

const failureBarPlugin = {
    id: 'failureBars',
    afterDatasetsDraw(chart) {
        const { ctx } = chart;
        const failureMap = chart.options._failureMap;
        if (!failureMap) return;

        const isHorizontal = chart.options.indexAxis === 'y';
        const categoryScale = isHorizontal ? chart.scales.y : chart.scales.x;
        const valueScale = isHorizontal ? chart.scales.x : chart.scales.y;
        const numDatasets = chart.data.datasets.length;

        chart.data.datasets.forEach((ds, dsIndex) => {
            const algoOrKey = ds.label;
            ds.data.forEach((val, dataIndex) => {
                if (val !== null) return;

                let failureText = null;
                for (const [key, statuses] of Object.entries(failureMap)) {
                    if (key === algoOrKey && statuses[dataIndex]) {
                        failureText = statuses[dataIndex];
                    }
                }
                if (!failureText) return;

                const meta = chart.getDatasetMeta(dsIndex);
                const bar = meta.data[dataIndex];
                if (!bar) return;

                const color = '#666680';
                const barX = bar.x;
                const barY = bar.y;

                let refBar = null;
                for (let si = 0; si < numDatasets; si++) {
                    const refMeta = chart.getDatasetMeta(si);
                    const refEl = refMeta.data[dataIndex];
                    if (refEl && chart.data.datasets[si].data[dataIndex] !== null) {
                        refBar = refEl;
                        break;
                    }
                }
                if (!refBar) return;

                const barWidth = refBar.width || 16;

                ctx.save();
                ctx.setLineDash([5, 4]);
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.globalAlpha = 0.6;

                if (isHorizontal) {
                    const xStart = valueScale.getPixelForValue(0);
                    const xEnd = valueScale.right - 20;
                    const halfH = barWidth / 2;
                    ctx.strokeRect(xStart, barY - halfH, xEnd - xStart, barWidth);

                    for (let lx = xStart + 8; lx < xEnd; lx += 12) {
                        ctx.beginPath();
                        ctx.moveTo(lx, barY - halfH);
                        ctx.lineTo(lx + 8, barY + halfH);
                        ctx.stroke();
                    }

                    ctx.setLineDash([]);
                    ctx.globalAlpha = 0.8;
                    ctx.fillStyle = color;
                    ctx.font = "bold 11px 'Graphik', sans-serif";
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(failureText, (xStart + xEnd) / 2, barY);
                } else {
                    const yBottom = valueScale.getPixelForValue(0);
                    const yTop = valueScale.top + 20;
                    const halfW = barWidth / 2;
                    ctx.strokeRect(barX - halfW, yTop, barWidth, yBottom - yTop);

                    for (let ly = yBottom - 8; ly > yTop; ly -= 12) {
                        ctx.beginPath();
                        ctx.moveTo(barX - halfW, ly);
                        ctx.lineTo(barX + halfW, ly - 8);
                        ctx.stroke();
                    }

                    ctx.setLineDash([]);
                    ctx.globalAlpha = 0.8;
                    ctx.fillStyle = color;
                    ctx.font = "bold 11px 'Graphik', sans-serif";
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.save();
                    ctx.translate(barX, (yTop + yBottom) / 2);
                    ctx.rotate(-Math.PI / 2);
                    ctx.fillText(failureText, 0, 0);
                    ctx.restore();
                }

                ctx.restore();
            });
        });
    }
};

function buildAccuracyChart() {
    const ctx = document.getElementById('chart-accuracy').getContext('2d');
    const datasets = ALGORITHMS.map(algo => ({
        label: algo,
        data: SUMMARY_DATA[algo],
        backgroundColor: ALGO_COLORS[algo] + 'cc',
        borderColor: ALGO_COLORS[algo],
        borderWidth: 1,
        borderRadius: 3,
    }));

    new Chart(ctx, {
        type: 'bar',
        data: { labels: DATASETS, datasets },
        plugins: [failureBarPlugin],
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            _failureMap: FAILURE_STATUS,
            plugins: {
                title: { display: true, text: 'Node Classification Accuracy — Real Datasets, Nearest Centroid, 256 dim', color: COLORS.text, font: { size: 14, weight: 500 }, padding: { bottom: 20 } },
                tooltip: {
                    callbacks: {
                        label: tooltipCtx => {
                            const algo = tooltipCtx.dataset.label;
                            const idx = tooltipCtx.dataIndex;
                            const status = FAILURE_STATUS[algo] && FAILURE_STATUS[algo][idx];
                            if (status) return algo + ': ' + status;
                            if (tooltipCtx.raw === null) return algo + ': N/A';
                            return algo + ': ' + tooltipCtx.raw.toFixed(3);
                        }
                    }
                },
            },
            scales: {
                x: { grid: { display: false }, ticks: { color: COLORS.textMuted } },
                y: { min: 0, max: 1.05, ticks: { color: COLORS.textMuted, callback: v => v.toFixed(1) }, grid: { color: COLORS.border + '60' } },
            },
        },
    });
}

function buildSpeedChart() {
    const ctx = document.getElementById('chart-speed').getContext('2d');

    const datasets_labels = ['ego-Facebook', 'Cora', 'CiteSeer', 'PubMed', 'PPI'];
    const algos = SPEED_DATA.algorithms;
    const dsKeys = ['facebook', 'cora', 'citeseer', 'pubmed', 'ppi'];

    const algoDatasets = algos.map(algo => {
        const algoIdx = SPEED_DATA.algorithms.indexOf(algo);
        return {
            label: algo,
            data: dsKeys.map(k => SPEED_DATA[k][algoIdx]),
            backgroundColor: ALGO_COLORS[algo] + 'cc',
            borderColor: ALGO_COLORS[algo],
            borderWidth: 1,
            borderRadius: 3,
        };
    });

    const speedFailureMap = {};
    algos.forEach(algo => {
        const algoIdx = SPEED_DATA.algorithms.indexOf(algo);
        const statuses = dsKeys.map(k => SPEED_FAILURE[k] ? SPEED_FAILURE[k][algoIdx] : null);
        if (statuses.some(s => s)) speedFailureMap[algo] = statuses;
    });

    new Chart(ctx, {
        type: 'bar',
        data: { labels: datasets_labels, datasets: algoDatasets },
        plugins: [failureBarPlugin],
        options: {
            responsive: true,
            maintainAspectRatio: false,
            _failureMap: speedFailureMap,
            plugins: {
                title: { display: true, text: 'Embedding Time (seconds) — Real Datasets, Log Scale', color: COLORS.text, font: { size: 14, weight: 500 }, padding: { bottom: 20 } },
                tooltip: {
                    callbacks: {
                        label: tooltipCtx => {
                            const algo = tooltipCtx.dataset.label;
                            const idx = tooltipCtx.dataIndex;
                            const status = speedFailureMap[algo] && speedFailureMap[algo][idx];
                            if (status) return algo + ': ' + status;
                            return tooltipCtx.raw !== null ? algo + ': ' + tooltipCtx.raw + 's' : '';
                        }
                    }
                },
            },
            scales: {
                x: { grid: { display: false }, ticks: { color: COLORS.textMuted } },
                y: {
                    type: 'logarithmic',
                    grid: { color: COLORS.border + '60' },
                    ticks: { color: COLORS.textMuted, callback: v => v + 's' },
                },
            },
        },
    });
}

function buildMemoryChart() {
    const ctx = document.getElementById('chart-memory').getContext('2d');
    const algos = MEMORY_DATA.algorithms;

    const datasets_labels = ['ego-Facebook', 'Cora', 'CiteSeer', 'PubMed', 'PPI'];
    const dsKeys = ['facebook', 'cora', 'citeseer', 'pubmed', 'ppi'];

    const algoDatasets = algos.map(algo => {
        const algoIdx = MEMORY_DATA.algorithms.indexOf(algo);
        return {
            label: algo,
            data: dsKeys.map(k => MEMORY_DATA[k][algoIdx]),
            backgroundColor: ALGO_COLORS[algo] + 'cc',
            borderColor: ALGO_COLORS[algo],
            borderWidth: 1,
            borderRadius: 3,
        };
    });

    const memFailureMap = {};
    algos.forEach(algo => {
        const algoIdx = MEMORY_DATA.algorithms.indexOf(algo);
        const statuses = dsKeys.map(k => MEMORY_FAILURE[k] ? MEMORY_FAILURE[k][algoIdx] : null);
        if (statuses.some(s => s)) memFailureMap[algo] = statuses;
    });

    new Chart(ctx, {
        type: 'bar',
        data: { labels: datasets_labels, datasets: algoDatasets },
        plugins: [failureBarPlugin],
        options: {
            responsive: true,
            maintainAspectRatio: false,
            _failureMap: memFailureMap,
            plugins: {
                title: { display: true, text: 'Peak Memory Usage (MB) — Real Datasets, Log Scale', color: COLORS.text, font: { size: 14, weight: 500 }, padding: { bottom: 20 } },
                tooltip: {
                    callbacks: {
                        label: tooltipCtx => {
                            const algo = tooltipCtx.dataset.label;
                            const idx = tooltipCtx.dataIndex;
                            const status = memFailureMap[algo] && memFailureMap[algo][idx];
                            if (status) return algo + ': ' + status;
                            return tooltipCtx.raw !== null ? algo + ': ' + tooltipCtx.raw + ' MB' : '';
                        }
                    }
                },
            },
            scales: {
                x: { grid: { display: false }, ticks: { color: COLORS.textMuted } },
                y: {
                    type: 'logarithmic',
                    grid: { color: COLORS.border + '60' },
                    ticks: { color: COLORS.textMuted, callback: v => v >= 1000 ? (v/1000).toFixed(1) + ' GB' : v + ' MB' },
                },
            },
        },
    });
}

function buildScatterChart() {
    const ctx = document.getElementById('chart-scatter').getContext('2d');
    const datasetColors = {
        'ego-Facebook': COLORS.accent,
        'Cora': '#10b981',
        'CiteSeer': COLORS.orange,
        'PubMed': COLORS.blue,
        'PPI': '#f472b6',
    };

    const datasets = Object.entries(SCATTER_DATA).map(([dsName, algos]) => ({
        label: dsName,
        data: Object.entries(algos).map(([algo, d]) => ({
            x: d.time,
            y: d.acc,
            algo: algo,
        })),
        backgroundColor: datasetColors[dsName] + 'cc',
        borderColor: datasetColors[dsName],
        borderWidth: 1,
        pointRadius: 7,
        pointHoverRadius: 10,
    }));

    new Chart(ctx, {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: 'Accuracy vs Embedding Time — Real Datasets, 256 dim', color: COLORS.text, font: { size: 14, weight: 500 }, padding: { bottom: 20 } },
                tooltip: {
                    callbacks: {
                        label: tooltipCtx => {
                            const p = tooltipCtx.raw;
                            return p.algo + ' — Acc: ' + p.y.toFixed(3) + ', Time: ' + p.x + 's';
                        }
                    }
                },
            },
            scales: {
                x: {
                    title: { display: true, text: 'Embedding Time (seconds)', color: COLORS.textMuted },
                    grid: { color: COLORS.border + '60' },
                    ticks: { color: COLORS.textMuted, callback: v => v + 's' },
                },
                y: {
                    min: 0,
                    max: 1.05,
                    title: { display: true, text: 'Accuracy', color: COLORS.textMuted },
                    grid: { color: COLORS.border + '60' },
                    ticks: { color: COLORS.textMuted, callback: v => v.toFixed(1) },
                },
            },
        },
    });
}

function initToggles() {
    document.querySelectorAll('.bench-toggle-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const section = btn.closest('.bench-section');
            const chartView = section.querySelector('.bench-chart-view');
            const tableView = section.querySelector('.bench-table-view');
            const btns = section.querySelectorAll('.bench-toggle-btn');

            btns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            if (btn.dataset.view === 'chart') {
                chartView.style.display = 'block';
                tableView.style.display = 'none';
            } else {
                chartView.style.display = 'none';
                tableView.style.display = 'block';
            }
        });
    });
}

document.addEventListener('DOMContentLoaded', () => {
    initToggles();

    if (typeof Chart === 'undefined') {
        document.querySelectorAll('.bench-section').forEach(section => {
            const chartView = section.querySelector('.bench-chart-view');
            const tableView = section.querySelector('.bench-table-view');
            if (chartView) chartView.style.display = 'none';
            if (tableView) tableView.style.display = 'block';
            const btns = section.querySelectorAll('.bench-toggle-btn');
            btns.forEach(b => {
                b.classList.toggle('active', b.dataset.view === 'table');
            });
        });
        return;
    }

    chartDefaults();
    buildAccuracyChart();
    buildSpeedChart();
    buildMemoryChart();
    buildScatterChart();
});
