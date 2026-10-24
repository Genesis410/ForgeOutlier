# Contributing to ForgeOutlier Lite

Thank you for contributing to ForgeOutlier Lite.

## Development Setup
1. Clone the repository.
2. Install test dependencies:
   ```bash
   pip install -e .[dev,eval]
   ```
3. Run tests locally:
   ```bash
   python -m pytest tests/
   ```

## Code Standards
- Code formatting and linting: `ruff check .` and `ruff format .`
- Strict type hints: `mypy src/`
- Every PR must maintain test coverage $\ge 85\%$ across the codebase.
- No network calls permitted in library runtime code.
