"""Simple Python code execution server with matplotlib support."""
import base64
import io
import os
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from flask import Flask, request, jsonify

# Configure matplotlib for non-GUI backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

TIMEOUT = int(os.environ.get('EXECUTION_TIMEOUT', 30))


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'runtime': 'python'})


def capture_figures() -> list[str]:
    """Capture all open matplotlib figures as base64 PNG images."""
    images = []
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        images.append(img_base64)
        buf.close()
    plt.close('all')
    return images


@app.route('/execute', methods=['POST'])
def execute():
    """Execute Python code and return the output."""
    data = request.get_json()
    if not data or 'code' not in data:
        return jsonify({'error': 'No code provided'}), 400

    code = data['code']
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    result = {
        'stdout': '',
        'stderr': '',
        'return_value': None,
        'error': None,
        'images': [],
    }

    # Clear any existing figures before execution
    plt.close('all')

    try:
        # Create a clean namespace with matplotlib available
        namespace = {
            '__builtins__': __builtins__,
            'plt': plt,
            'matplotlib': matplotlib,
        }

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, namespace)

        result['stdout'] = stdout_capture.getvalue()
        result['stderr'] = stderr_capture.getvalue()

        # Capture any matplotlib figures
        result['images'] = capture_figures()

        # Try to capture last expression value if it exists
        if '_result' in namespace:
            result['return_value'] = str(namespace['_result'])

    except Exception as e:
        result['stdout'] = stdout_capture.getvalue()
        result['stderr'] = stderr_capture.getvalue()
        result['error'] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        plt.close('all')

    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
