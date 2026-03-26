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
    'Cleora (whiten)':  '#a78bfa',
    'Cleora':           '#6c63ff',
    'ProNE':            '#f59e0b',
    'RandNE':           '#ef4444',
    'NetMF':            '#3b82f6',
    'DeepWalk':         '#f472b6',
};

const DATASETS = ['ego-Facebook', 'PPI-large', 'Flickr', 'ogbn-arxiv', 'Yelp'];
const ALGORITHMS = ['Cleora (whiten)', 'Cleora', 'ProNE', 'RandNE', 'NetMF', 'DeepWalk'];

const SUMMARY_DATA = {
    'Cleora (whiten)': [0.932, 0.985, 0.502, 0.624, null],
    'Cleora':          [0.350, 0.025, 0.157, 0.038, 0.013],
    'ProNE':           [0.019, 0.008, 0.142, 0.026, null],
    'RandNE':          [0.120, 0.014, 0.153, 0.032, null],
    'NetMF':           [0.889, null,  null,  null,  null],
    'DeepWalk':        [0.885, null,  null,  null,  null],
};

const MLP_DATA = {
    'Cleora (whiten)': [0.973, null, null, null, null],
    'Cleora':          [0.379, null, null, null, null],
    'ProNE':           [0.130, null, null, null, null],
    'RandNE':          [0.130, null, null, null, null],
    'NetMF':           [0.639, null, null, null, null],
    'DeepWalk':        [0.620, null, null, null, null],
};

const SPEED_DATA = {
    algorithms: ['Cleora', 'Cleora (whiten)', 'RandNE', 'ProNE', 'NetMF', 'DeepWalk'],
    facebook:   [0.111,    0.430,              0.070,    0.264,   35.229,  50.093],
    ppi_large:  [0.707,    1.702,              1.863,    7.286,   null,    null],
    flickr:     [0.869,    2.218,              2.169,    10.732,  null,    null],
    ogbn_arxiv: [1.290,    3.623,              3.204,    15.725,  null,    null],
    yelp:       [7.076,    null,               null,     null,    null,    null],
    roadnet:    [5.312,    null,               null,     null,    null,    null],
};

const MEMORY_DATA = {
    algorithms: ['Cleora', 'Cleora (whiten)', 'RandNE', 'ProNE', 'DeepWalk', 'NetMF'],
    facebook:   [3.9,      25.2,               39.8,     64.0,    540.8,      1047.4],
    ppi_large:  [55.6,     335.2,              541.0,    875.8,   null,        null],
    flickr:     [87.2,     524.5,              830.4,    1354.9,  null,        null],
    ogbn_arxiv: [165.4,    993.8,              1550.8,   2545.5,  null,        null],
    yelp:       [700.0,    null,               null,     null,    null,        null],
    roadnet:    [1919.1,   null,               null,     null,    null,        null],
};

const SCATTER_DATA = {
    'ego-Facebook': {
        'Cleora (whiten)': { acc: 0.932, time: 0.430 },
        'NetMF':           { acc: 0.889, time: 35.229 },
        'DeepWalk':        { acc: 0.885, time: 50.093 },
        'RandNE':          { acc: 0.120, time: 0.070 },
        'ProNE':           { acc: 0.019, time: 0.264 },
        'Cleora':          { acc: 0.350, time: 0.111 },
    },
    'PPI-large': {
        'Cleora (whiten)': { acc: 0.985, time: 1.702 },
        'Cleora':          { acc: 0.025, time: 0.707 },
        'ProNE':           { acc: 0.008, time: 7.286 },
        'RandNE':          { acc: 0.014, time: 1.863 },
    },
    'Flickr': {
        'Cleora (whiten)': { acc: 0.502, time: 2.218 },
        'Cleora':          { acc: 0.157, time: 0.869 },
        'ProNE':           { acc: 0.142, time: 10.732 },
        'RandNE':          { acc: 0.153, time: 2.169 },
    },
    'ogbn-arxiv': {
        'Cleora (whiten)': { acc: 0.624, time: 3.623 },
        'Cleora':          { acc: 0.038, time: 1.290 },
        'RandNE':          { acc: 0.032, time: 3.204 },
        'ProNE':           { acc: 0.026, time: 15.725 },
    },
};

const CV_DATA = {
    datasets:     ['ego-Facebook', 'PPI-large', 'Flickr', 'ogbn-arxiv'],
    meanAccuracy: [0.931, 0.985, 0.507, 0.620],
    stdAccuracy:  [0.017, 0.001, 0.006, 0.003],
    meanF1:       [0.813, 0.985, 0.507, 0.620],
    stdF1:        [0.025, 0.001, 0.006, 0.003],
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
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                title: { display: true, text: 'Node Classification Accuracy — Nearest Centroid, 256 dim', color: COLORS.text, font: { size: 16, weight: 500 }, padding: { bottom: 20 } },
                tooltip: {
                    callbacks: {
                        label: ctx => {
                            if (ctx.raw === null) return ctx.dataset.label + ': N/A';
                            return ctx.dataset.label + ': ' + ctx.raw.toFixed(3);
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

function buildMLPChart() {
    const ctx = document.getElementById('chart-mlp');
    if (!ctx) return;

    const datasets = ALGORITHMS.map(algo => ({
        label: algo,
        data: [MLP_DATA[algo][0]],
        backgroundColor: ALGO_COLORS[algo] + 'cc',
        borderColor: ALGO_COLORS[algo],
        borderWidth: 1,
        borderRadius: 3,
    }));

    new Chart(ctx.getContext('2d'), {
        type: 'bar',
        data: { labels: ['ego-Facebook'], datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                title: { display: true, text: 'MLP Classifier Accuracy — ego-Facebook, 256 dim', color: COLORS.text, font: { size: 16, weight: 500 }, padding: { bottom: 20 } },
                tooltip: {
                    callbacks: {
                        label: ctx => {
                            if (ctx.raw === null) return ctx.dataset.label + ': N/A';
                            return ctx.dataset.label + ': ' + ctx.raw.toFixed(3);
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
    const algos = SPEED_DATA.algorithms;

    const dsConfigs = [
        { key: 'facebook',   label: 'ego-Facebook (4k)',   color: COLORS.accent },
        { key: 'ppi_large',  label: 'PPI-large (57k)',     color: '#10b981' },
        { key: 'flickr',     label: 'Flickr (89k)',        color: COLORS.orange },
        { key: 'ogbn_arxiv', label: 'ogbn-arxiv (169k)',   color: COLORS.blue },
        { key: 'yelp',       label: 'Yelp (717k)',         color: '#f472b6' },
        { key: 'roadnet',    label: 'roadNet-CA (2M)',     color: COLORS.green },
    ];
    const datasets = dsConfigs.map(d => ({
        label: d.label,
        data: SPEED_DATA[d.key],
        backgroundColor: d.color + 'cc',
        borderColor: d.color,
        borderWidth: 1,
        borderRadius: 3,
    }));

    new Chart(ctx, {
        type: 'bar',
        data: { labels: algos, datasets },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: 'Embedding Time (seconds) — Linear Scale', color: COLORS.text, font: { size: 16, weight: 500 }, padding: { bottom: 20 } },
                tooltip: {
                    callbacks: { label: ctx => ctx.raw !== null ? ctx.dataset.label + ': ' + ctx.raw + 's' : '' }
                },
            },
            scales: {
                x: {
                    grid: { color: COLORS.border + '60' },
                    ticks: { color: COLORS.textMuted, callback: v => v + 's' },
                },
                y: { grid: { display: false }, ticks: { color: COLORS.textMuted } },
            },
        },
    });
}

function buildMemoryChart() {
    const ctx = document.getElementById('chart-memory').getContext('2d');
    const algos = MEMORY_DATA.algorithms;

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: algos,
            datasets: [
                { key: 'facebook',   label: 'ego-Facebook (4k)',   color: COLORS.accent },
                { key: 'ppi_large',  label: 'PPI-large (57k)',     color: '#10b981' },
                { key: 'flickr',     label: 'Flickr (89k)',        color: COLORS.orange },
                { key: 'ogbn_arxiv', label: 'ogbn-arxiv (169k)',   color: COLORS.blue },
                { key: 'yelp',       label: 'Yelp (717k)',         color: '#f472b6' },
                { key: 'roadnet',    label: 'roadNet-CA (2M)',     color: COLORS.green },
            ].map(d => ({
                label: d.label,
                data: MEMORY_DATA[d.key],
                backgroundColor: d.color + 'cc',
                borderColor: d.color,
                borderWidth: 1,
                borderRadius: 3,
            })),
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: 'Peak Memory Usage (MB) — Linear Scale', color: COLORS.text, font: { size: 16, weight: 500 }, padding: { bottom: 20 } },
                tooltip: {
                    callbacks: { label: ctx => ctx.raw !== null ? ctx.dataset.label + ': ' + ctx.raw + ' MB' : '' }
                },
            },
            scales: {
                x: { grid: { display: false }, ticks: { color: COLORS.textMuted } },
                y: {
                    grid: { color: COLORS.border + '60' },
                    ticks: { color: COLORS.textMuted, callback: v => v + ' MB' },
                },
            },
        },
    });
}

function buildScatterChart() {
    const ctx = document.getElementById('chart-scatter').getContext('2d');
    const datasetColors = {
        'ego-Facebook': COLORS.accent,
        'PPI-large': '#10b981',
        'Flickr': COLORS.orange,
        'ogbn-arxiv': COLORS.blue,
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
                title: { display: true, text: 'Accuracy vs Embedding Time — 256 dim', color: COLORS.text, font: { size: 16, weight: 500 }, padding: { bottom: 20 } },
                tooltip: {
                    callbacks: {
                        label: ctx => {
                            const p = ctx.raw;
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

function buildCVChart() {
    const ctx = document.getElementById('chart-cv').getContext('2d');
    const errorBarsAccuracy = CV_DATA.datasets.map((_, i) => ({
        min: CV_DATA.meanAccuracy[i] - CV_DATA.stdAccuracy[i],
        max: CV_DATA.meanAccuracy[i] + CV_DATA.stdAccuracy[i],
    }));
    const errorBarsF1 = CV_DATA.datasets.map((_, i) => ({
        min: CV_DATA.meanF1[i] - CV_DATA.stdF1[i],
        max: CV_DATA.meanF1[i] + CV_DATA.stdF1[i],
    }));

    const errorBarPlugin = {
        id: 'errorBars',
        afterDatasetsDraw(chart) {
            const { ctx: c } = chart;
            chart.data.datasets.forEach((ds, dsIndex) => {
                if (!ds.errorBars) return;
                const meta = chart.getDatasetMeta(dsIndex);
                meta.data.forEach((bar, i) => {
                    const eb = ds.errorBars[i];
                    if (!eb) return;
                    const yScale = chart.scales.y;
                    const yMin = yScale.getPixelForValue(eb.min);
                    const yMax = yScale.getPixelForValue(eb.max);
                    const x = bar.x;
                    c.save();
                    c.strokeStyle = ds.borderColor || COLORS.text;
                    c.lineWidth = 2;
                    c.beginPath();
                    c.moveTo(x, yMin);
                    c.lineTo(x, yMax);
                    c.stroke();
                    c.beginPath();
                    c.moveTo(x - 4, yMin);
                    c.lineTo(x + 4, yMin);
                    c.stroke();
                    c.beginPath();
                    c.moveTo(x - 4, yMax);
                    c.lineTo(x + 4, yMax);
                    c.stroke();
                    c.restore();
                });
            });
        },
    };

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: CV_DATA.datasets,
            datasets: [
                {
                    label: 'Mean Accuracy',
                    data: CV_DATA.meanAccuracy,
                    backgroundColor: COLORS.accent + 'cc',
                    borderColor: COLORS.accent,
                    borderWidth: 1,
                    borderRadius: 3,
                    errorBars: errorBarsAccuracy,
                },
                {
                    label: 'Mean F1',
                    data: CV_DATA.meanF1,
                    backgroundColor: COLORS.green + 'cc',
                    borderColor: COLORS.green,
                    borderWidth: 1,
                    borderRadius: 3,
                    errorBars: errorBarsF1,
                },
            ],
        },
        plugins: [errorBarPlugin],
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: 'Cross-Validation: Cleora (whiten, 16 iter, 256 dim)', color: COLORS.text, font: { size: 16, weight: 500 }, padding: { bottom: 20 } },
                tooltip: {
                    callbacks: {
                        label: ctx => {
                            const i = ctx.dataIndex;
                            const dsIdx = ctx.datasetIndex;
                            if (dsIdx === 0) return 'Accuracy: ' + ctx.raw.toFixed(3) + ' \u00b1 ' + CV_DATA.stdAccuracy[i].toFixed(3);
                            return 'F1: ' + ctx.raw.toFixed(3) + ' \u00b1 ' + CV_DATA.stdF1[i].toFixed(3);
                        }
                    }
                },
            },
            scales: {
                x: { grid: { display: false }, ticks: { color: COLORS.textMuted } },
                y: { min: 0, max: 1.1, grid: { color: COLORS.border + '60' }, ticks: { color: COLORS.textMuted, callback: v => v.toFixed(1) } },
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
    buildMLPChart();
    buildSpeedChart();
    buildMemoryChart();
    buildScatterChart();
    buildCVChart();
});
