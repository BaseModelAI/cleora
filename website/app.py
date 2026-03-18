from flask import Flask, render_template, redirect, jsonify
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/docs')
def docs():
    return render_template('docs.html')

@app.route('/changelog')
def changelog():
    return render_template('changelog.html')

@app.route('/benchmarks')
def benchmarks():
    return render_template('benchmarks.html')

@app.route('/api')
def api_ref():
    return render_template('api.html')

@app.route('/use-cases')
def use_cases():
    return render_template('use-cases.html')

@app.route('/architecture')
def architecture():
    return render_template('architecture.html')

@app.route('/features')
def features():
    return render_template('functions.html')

@app.route('/functions')
def functions_redirect():
    return redirect('/features', code=301)

@app.route('/sync')
def sync_page():
    return render_template('sync.html')

@app.route('/api/sync/start', methods=['POST'])
def api_sync_start():
    from sync_worker import start_sync
    ok, msg = start_sync()
    if ok:
        return jsonify({"status": "started", "message": msg}), 200
    return jsonify({"status": "error", "message": msg}), 409

@app.route('/api/sync/status')
def api_sync_status():
    from sync_worker import get_status
    return jsonify(get_status())

@app.route('/api/sync/results')
def api_sync_results():
    from sync_worker import get_results
    results = get_results()
    if results is None:
        return jsonify({"status": "no_results", "message": "No benchmark results available yet. Run a sync first."}), 404
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
