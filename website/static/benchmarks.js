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
    'Cleora':     '#6c63ff',
    'Cleora-sym': '#8b83ff',
    'ProNE':      '#f59e0b',
    'RandNE':     '#ef4444',
    'NetMF':      '#3b82f6',
    'DeepWalk':   '#f472b6',
    'Node2Vec':   '#fb923c',
};

const DATASETS = ['ego-Facebook', 'PPI-large', 'Flickr', 'ogbn-arxiv', 'Yelp'];
const ALGORITHMS = ['Cleora', 'Cleora-sym', 'ProNE', 'RandNE', 'NetMF', 'DeepWalk', 'Node2Vec'];

const SUMMARY_DATA = {
    'Cleora':     [0.964, 0.026, 0.158, 0.038, 0.013],
    'Cleora-sym': [0.293, 0.026, 0.158, 0.038, null],
    'ProNE':      [0.021, 0.008, 0.139, 0.026, null],
    'RandNE':     [0.318, 0.011, 0.146, 0.030, null],
    'NetMF':      [0.944, null,  null,  null,  null],
    'DeepWalk':   [0.912, null,  null,  null,  null],
    'Node2Vec':   [0.918, null,  null,  null,  null],
};

const SPEED_DATA = {
    algorithms: ['Cleora', 'Cleora-sym', 'RandNE', 'ProNE', 'NetMF', 'DeepWalk', 'Node2Vec'],
    facebook:   [0.109,    0.103,        0.232,   1.429,    17.920,  32.352,      111.426],
    ppi_large:  [0.330,    0.251,        1.072,   8.338,    null,    null,         null],
    flickr:     [0.475,    0.436,        1.332,   5.569,    null,    null,         null],
    ogbn_arxiv: [0.747,    0.752,        2.046,   8.333,    null,    null,         null],
    yelp:       [3.304,    3.162,        null,    null,     null,    null,         null],
    roadnet:    [4.242,    4.269,        8.968,   57.716,   null,    null,         null],
};

const MEMORY_DATA = {
    algorithms: ['Cleora', 'Cleora-sym', 'RandNE', 'ProNE', 'Node2Vec', 'DeepWalk', 'NetMF'],
    facebook:   [15.78,    15.78,        146.28,   248.98,  600,        600,         1107],
    ppi_large:  [27.81,    27.80,        290.79,   458.38,  null,       null,         null],
    flickr:     [43.58,    43.58,        438.17,   700.79,  null,       null,         null],
    ogbn_arxiv: [82.69,    82.69,        806.62,   1305,    null,       null,         null],
    yelp:       [350,      350,          null,     null,    null,       null,         null],
    roadnet:    [1934,     1934,         8868,     14648,   null,       null,         null],
};

const SCATTER_DATA = {
    'ego-Facebook': {
        'Cleora':     { acc: 0.964, time: 0.740 },
        'NetMF':      { acc: 0.944, time: 17.920 },
        'Node2Vec':   { acc: 0.918, time: 111.426 },
        'DeepWalk':   { acc: 0.912, time: 32.352 },
        'RandNE':     { acc: 0.318, time: 0.232 },
        'Cleora-sym': { acc: 0.293, time: 0.103 },
        'ProNE':      { acc: 0.021, time: 1.429 },
    },
    'ogbn-arxiv': {
        'Cleora':     { acc: 0.038, time: 0.747 },
        'Cleora-sym': { acc: 0.038, time: 0.752 },
        'RandNE':     { acc: 0.030, time: 2.046 },
        'ProNE':      { acc: 0.026, time: 8.333 },
    },
};

const CV_DATA = {
    datasets:     ['ego-Facebook'],
    meanAccuracy: [0.888],
    stdAccuracy:  [0.015],
    meanF1:       [0.757],
    stdF1:        [0.021],
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
                title: { display: true, text: 'Accuracy per Algorithm — ego-Facebook', color: COLORS.text, font: { size: 16, weight: 500 }, padding: { bottom: 20 } },
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
                title: { display: true, text: 'Embedding Time (seconds, log scale)', color: COLORS.text, font: { size: 16, weight: 500 }, padding: { bottom: 20 } },
                tooltip: {
                    callbacks: { label: ctx => ctx.raw !== null ? ctx.dataset.label + ': ' + ctx.raw + 's' : '' }
                },
            },
            scales: {
                x: {
                    type: 'logarithmic',
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
                title: { display: true, text: 'Peak Memory Usage (MB, log scale)', color: COLORS.text, font: { size: 16, weight: 500 }, padding: { bottom: 20 } },
                tooltip: {
                    callbacks: { label: ctx => ctx.raw !== null ? ctx.dataset.label + ': ' + ctx.raw + ' MB' : '' }
                },
            },
            scales: {
                x: { grid: { display: false }, ticks: { color: COLORS.textMuted } },
                y: {
                    type: 'logarithmic',
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
                title: { display: true, text: 'Accuracy vs Embedding Time', color: COLORS.text, font: { size: 16, weight: 500 }, padding: { bottom: 20 } },
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
                    type: 'logarithmic',
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
                title: { display: true, text: 'Cross-Validation: Cleora on ego-Facebook (1024 dim, 4 iter)', color: COLORS.text, font: { size: 16, weight: 500 }, padding: { bottom: 20 } },
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
    buildSpeedChart();
    buildMemoryChart();
    buildScatterChart();
    buildCVChart();
});
