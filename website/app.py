from flask import Flask, render_template, redirect
import os

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
