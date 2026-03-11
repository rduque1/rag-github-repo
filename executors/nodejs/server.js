/**
 * Simple Node.js code execution server.
 */
const express = require('express');
const vm = require('vm');

const app = express();
app.use(express.json());

const TIMEOUT = parseInt(process.env.EXECUTION_TIMEOUT || '30', 10) * 1000;

app.get('/health', (req, res) => {
  res.json({ status: 'healthy', runtime: 'nodejs' });
});

app.post('/execute', (req, res) => {
  const { code } = req.body;

  if (!code) {
    return res.status(400).json({ error: 'No code provided' });
  }

  const result = {
    stdout: '',
    stderr: '',
    return_value: null,
    error: null,
  };

  // Capture console output
  const logs = [];
  const errors = [];

  const sandbox = {
    console: {
      log: (...args) => logs.push(args.map(String).join(' ')),
      error: (...args) => errors.push(args.map(String).join(' ')),
      warn: (...args) => logs.push('[WARN] ' + args.map(String).join(' ')),
      info: (...args) => logs.push('[INFO] ' + args.map(String).join(' ')),
    },
    require: (mod) => {
      // Allow only safe built-in modules
      const allowed = ['path', 'url', 'querystring', 'util', 'crypto', 'fs'];
      if (allowed.includes(mod)) {
        return require(mod);
      }
      throw new Error(`Module '${mod}' is not allowed`);
    },
    setTimeout: undefined,
    setInterval: undefined,
    process: undefined,
    _result: undefined,
  };

  try {
    const context = vm.createContext(sandbox);
    const script = new vm.Script(code);
    const returnValue = script.runInContext(context, { timeout: TIMEOUT });

    result.stdout = logs.join('\n');
    result.stderr = errors.join('\n');

    if (returnValue !== undefined) {
      result.return_value = String(returnValue);
    }
  } catch (err) {
    result.stdout = logs.join('\n');
    result.stderr = errors.join('\n');
    result.error = `${err.name}: ${err.message}\n${err.stack}`;
  }

  res.json(result);
});

const PORT = 3000;
app.listen(PORT, '0.0.0.0', () => {
  console.log(`Node.js executor running on port ${PORT}`);
});
