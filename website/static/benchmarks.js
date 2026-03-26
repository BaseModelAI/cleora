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
    'HOPE':             '#34d399',
    'GraRep':           '#fb923c',
    'Node2Vec':         '#22d3ee',
};

const DATASETS = ['ego-Facebook', 'Cora', 'CiteSeer', 'PubMed', 'PPI'];
const ALGORITHMS = ['Cleora', 'ProNE', 'RandNE', 'HOPE', 'NetMF', 'GraRep', 'DeepWalk', 'Node2Vec'];

const SUMMARY_DATA = {
    'Cleora':          [0.990, 0.861, 0.824, 0.879, 1.000],
    'ProNE':           [0.075, 0.179, 0.189, 0.339, 0.023],
    'RandNE':          [0.212, 0.247, 0.244, 0.351, 0.073],
    'HOPE':            [0.890, 0.821, 0.740, null,  null],
    'NetMF':           [0.957, 0.839, 0.810, null,  null],
    'GraRep':          [null,  0.809, 0.756, null,  null],
    'DeepWalk':        [0.958, 0.835, 0.806, null,  null],
    'Node2Vec':        [0.958, 0.835, 0.806, null,  null],
};

const FAILURE_STATUS = {
    'HOPE':            [null,         null, null, 'Timed Out', 'Timed Out'],
    'NetMF':           [null,         null, null, 'OOM',       'OOM'],
    'GraRep':          ['Timed Out',  null, null, 'OOM',       'OOM'],
    'DeepWalk':        [null,         null, null, 'Timed Out', 'Timed Out'],
    'Node2Vec':        [null,         null, null, 'Timed Out', 'Timed Out'],
};

const SPEED_DATA = {
    algorithms: ['Cleora', 'RandNE', 'ProNE', 'HOPE', 'NetMF', 'GraRep', 'DeepWalk', 'Node2Vec'],
    facebook:   [1.23,     0.07,     0.26,    31.48,  28.81,   null,     59.21,      67.90],
    cora:       [1.03,     0.03,     0.13,    15.97,  4.23,    16.40,    24.11,      25.79],
    citeseer:   [0.99,     0.02,     0.14,    19.56,  6.58,    27.25,    29.29,      29.59],
    pubmed:     [1.40,     0.22,     0.75,    null,   null,    null,     null,       null],
    ppi:        [1.23,     0.07,     1.45,    null,   null,    null,     null,       null],
    roadnet:    [31.500,   null,     null,    null,   null,    null,     null,       null],
};

const SPEED_FAILURE = {
    facebook:   [null, null, null, null,         null, 'Timed Out', null,         null],
    pubmed:     [null, null, null, 'Timed Out',  'OOM', 'OOM',     'Timed Out',  'Timed Out'],
    ppi:        [null, null, null, 'Timed Out',  'OOM', 'OOM',     'Timed Out',  'Timed Out'],
    roadnet:    [null, 'OOM', 'OOM', 'OOM',      'OOM', 'OOM',    'OOM',        'OOM'],
};

const MEMORY_DATA = {
    algorithms: ['Cleora', 'RandNE', 'ProNE', 'HOPE', 'NetMF', 'GraRep', 'DeepWalk', 'Node2Vec'],
    facebook:   [22,       42,       67,      857,    1098,    null,     572,        572],
    cora:       [14,       24,       40,      330,    332,     322,      227,        227],
    citeseer:   [16,       27,       45,      430,    335,     411,      294,        294],
    pubmed:     [97,       175,      291,     null,   null,    null,     null,       null],
    ppi:        [21,       40,       64,      null,   null,    null,     null,       null],
    roadnet:    [4129,     null,     null,    null,   null,    null,     null,       null],
};

const MEMORY_FAILURE = {
    facebook:   [null, null, null, null,         null, 'Timed Out', null,         null],
    pubmed:     [null, null, null, 'Timed Out',  'OOM', 'OOM',     'Timed Out',  'Timed Out'],
    ppi:        [null, null, null, 'Timed Out',  'OOM', 'OOM',     'Timed Out',  'Timed Out'],
    roadnet:    [null, 'OOM', 'OOM', 'OOM',      'OOM', 'OOM',    'OOM',        'OOM'],
};

const SCATTER_DATA = {
    'ego-Facebook': {
        'Cleora':          { acc: 0.990, time: 1.23 },
        'ProNE':           { acc: 0.075, time: 0.26 },
        'RandNE':          { acc: 0.212, time: 0.07 },
        'HOPE':            { acc: 0.890, time: 31.48 },
        'NetMF':           { acc: 0.957, time: 28.81 },
        'DeepWalk':        { acc: 0.958, time: 59.21 },
        'Node2Vec':        { acc: 0.958, time: 67.90 },
    },
    'Cora': {
        'Cleora':          { acc: 0.861, time: 1.03 },
        'ProNE':           { acc: 0.179, time: 0.13 },
        'RandNE':          { acc: 0.247, time: 0.03 },
        'HOPE':            { acc: 0.821, time: 15.97 },
        'NetMF':           { acc: 0.839, time: 4.23 },
        'GraRep':          { acc: 0.809, time: 16.40 },
        'DeepWalk':        { acc: 0.835, time: 24.11 },
        'Node2Vec':        { acc: 0.835, time: 25.79 },
    },
    'CiteSeer': {
        'Cleora':          { acc: 0.824, time: 0.99 },
        'ProNE':           { acc: 0.189, time: 0.14 },
        'RandNE':          { acc: 0.244, time: 0.02 },
        'HOPE':            { acc: 0.740, time: 19.56 },
        'NetMF':           { acc: 0.810, time: 6.58 },
        'GraRep':          { acc: 0.756, time: 27.25 },
        'DeepWalk':        { acc: 0.806, time: 29.29 },
        'Node2Vec':        { acc: 0.806, time: 29.59 },
    },
    'PubMed': {
        'Cleora':          { acc: 0.879, time: 1.40 },
        'ProNE':           { acc: 0.339, time: 0.75 },
        'RandNE':          { acc: 0.351, time: 0.22 },
    },
    'PPI': {
        'Cleora':          { acc: 1.000, time: 1.23 },
        'ProNE':           { acc: 0.023, time: 1.45 },
        'RandNE':          { acc: 0.073, time: 0.07 },
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
                    const xStart = valueScale.type === 'logarithmic' ? valueScale.left : valueScale.getPixelForValue(0);
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
                    const yBottom = valueScale.type === 'logarithmic' ? valueScale.bottom : valueScale.getPixelForValue(0);
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
