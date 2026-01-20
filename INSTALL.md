# Installation Guide

## Quick Start

1. **Install the package in development mode** (required):
   ```bash
   pip install -e .
   ```

2. **Verify installation**:
   ```bash
   python -c "from utils.reward import compute_reward; print('Installation successful!')"
   ```

## Why is this needed?

The project is structured as a Python package. Installing it with `pip install -e .` makes all modules available for import without needing to modify `sys.path` or use relative imports.

## Alternative: Manual path setup (not recommended)

If you don't want to install the package, you can add the `src` directory to `PYTHONPATH`:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

However, using `pip install -e .` is the recommended approach as it's cleaner and more maintainable.

## Troubleshooting

### ImportError: No module named 'utils'

**Solution**: Run `pip install -e .` in the project root directory.

### ImportError: No module named 'policy'

**Solution**: Make sure you've installed the package with `pip install -e .`

### Changes not reflected after installation

If you make changes to the code and they're not being used:
- Make sure you used `pip install -e .` (editable install)
- If you used `pip install .` (non-editable), reinstall with `-e` flag
