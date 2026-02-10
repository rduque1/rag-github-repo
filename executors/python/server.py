"""Simple Python code execution server."""
import io
import os
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from flask import Flask, request, jsonify

app = Flask(__name__)

TIMEOUT = int(os.environ.get('EXECUTION_TIMEOUT', 30))


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'runtime': 'python'})


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
    }

    try:
        # Create a clean namespace for execution
        namespace = {'__builtins__': __builtins__}

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, namespace)

        result['stdout'] = stdout_capture.getvalue()
        result['stderr'] = stderr_capture.getvalue()

        # Try to capture last expression value if it exists
        if '_result' in namespace:
            result['return_value'] = str(namespace['_result'])

    except Exception as e:
        result['stdout'] = stdout_capture.getvalue()
        result['stderr'] = stderr_capture.getvalue()
        result['error'] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
