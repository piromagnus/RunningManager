# Contributing to Running Manager

Thank you for your interest in contributing to Running Manager! We welcome contributions from everyone. This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and constructive in all interactions.

## Ways to Contribute

### 1. Report Bugs

If you find a bug, please create an issue on GitHub with:
- **Clear title**: Describe the bug briefly
- **Reproduction steps**: How to reproduce the issue
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Screenshots**: If applicable
- **Environment**: Python version, OS, and any relevant configuration

### 2. Suggest Features

Have a great idea? We'd love to hear it! Create a GitHub issue with:
- **Use case**: Why this feature would be useful
- **Description**: How the feature should work
- **Examples**: Usage examples or mockups
- **Priority**: How urgent/important is this feature?

### 3. Improve Documentation

Documentation improvements are always welcome:
- Clarify existing docs
- Add examples
- Fix typos
- Translate to other languages
- Add architecture diagrams

### 4. Submit Code

### Before Starting

1. **Check existing issues**: Look for related open issues or PRs
2. **Discuss major changes**: For significant features, open an issue first to discuss approach
3. **Follow the workflow**: Fork → Branch → Develop → Test → PR

### Development Workflow

1. **Fork the repository** on GitHub

2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/RunningManager.git
   cd RunningManager
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/piromagnus/RunningManager.git
   ```

4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or for bug fixes:
   git checkout -b fix/your-bug-fix
   ```

5. **Install dependencies**:
   ```bash
   uv sync
   ```

6. **Make your changes**

7. **Run tests**:
   ```bash
   pytest
   ```

8. **Run linting**:
   ```bash
   ruff check . && ruff format .
   ```

9. **Commit with clear messages**:
   ```bash
   git commit -m "feat: add feature description"
   # or
   git commit -m "fix: resolve bug description"
   ```

10. **Push to your fork**:
    ```bash
    git push origin feature/your-feature-name
    ```

11. **Create a Pull Request** on GitHub with:
    - Clear title describing the change
    - Description of what changes and why
    - Link to related issues (e.g., `Fixes #123`)
    - Screenshots or demo if UI-related

## Code Style Guide

### Python Style

- **Line Length**: 100 characters (enforced by Ruff)
- **Imports**: Sorted alphabetically, grouped by stdlib/third-party/local
- **Formatting**: Use `ruff format` for consistent formatting
- **Type Hints**: Add type annotations to function signatures
- **Docstrings**: Use Google-style docstrings for public functions

Example:
```python
from typing import Optional
from services.planner_service import PlannerService

def calculate_training_load(
    activities: list[dict],
    window_days: int = 7
) -> Optional[float]:
    """Calculate rolling training load for activities.
    
    Args:
        activities: List of activity dictionaries with metrics
        window_days: Number of days for rolling window (default 7)
    
    Returns:
        Training load value or None if no data
    """
    if not activities:
        return None
    
    # Implementation here
    return load_value
```

### Commit Messages

Follow conventional commits:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation
- `style:` for formatting changes
- `test:` for test additions/modifications
- `refactor:` for code refactoring
- `perf:` for performance improvements

Example:
```
feat: add Garmin activity import

- Implement OAuth2 flow for Garmin Connect
- Parse activity data from Garmin API
- Store timeseries data to CSV

Fixes #42
```

### CSV and Data

- **Decimals**: Always use `.` (period) as decimal separator, never `,`
- **Headers**: Keep column names consistent with `persistence/repositories.py`
- **File Locking**: Always use `CsvStorage` for reading/writing, never raw pandas
- **Migration**: Follow `PlannedSessionsRepo._migrate_headers_if_needed` pattern when adding fields

### Streamlit UI

- **Caching**: Use `@st.cache_data` for expensive computations with TTL
- **Session State**: Use consistent prefixes (e.g., `planner_*`, `analytics_*`)
- **Formatting**: Use `utils/formatting.py` for display; keep raw decimals in storage
- **Styling**: Centralize CSS in `utils/styling.py` where possible

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/services/test_planner_service.py

# Run with coverage
pytest --cov=. --cov-report=html

# Run in verbose mode
pytest -v
```

### Writing Tests

- **Location**: Tests go in `tests/` directory, mirroring source structure
- **Naming**: Use `test_*.py` files and `test_*` functions
- **Coverage**: Aim for >80% coverage
- **Isolation**: Use fixtures from `tests/conftest.py` for common setup
- **Determinism**: Avoid time-dependent tests; use `freezegun` if needed
- **Mocking**: Mock external APIs; use fakes for portalocker

Example:
```python
import pytest
from services.planner_service import PlannerService

def test_estimate_fundamental_distance():
    """Test fundamental distance estimation."""
    service = PlannerService()
    distance = service.estimate_fundamental_distance(
        pace_kmh=10.0,
        duration_minutes=60
    )
    assert distance == pytest.approx(10.0, rel=0.01)
```

## Documentation

When adding/changing features:

1. **Update AGENTS.md** if changing architecture
2. **Add docstrings** to public functions
3. **Update README.md** for new setup steps or features
4. **Consider adding examples** in docs/examples/ for complex features

## PR Review Process

1. **Automated checks**: Tests and linting must pass
2. **Code review**: Maintainers will review for:
   - Code quality and style
   - Test coverage
   - Documentation completeness
   - Architecture alignment
3. **Feedback**: Address review comments
4. **Approval**: PR is merged once approved

## Performance Considerations

When contributing, consider:
- **CSV Loading**: Use caching for frequently accessed data
- **Streamlit Rendering**: Minimize st.write() calls; batch updates
- **Metrics Computation**: Profile expensive operations; consider async
- **API Calls**: Implement rate limiting; cache responses

## Security Considerations

- **Secrets**: Never hardcode API keys or tokens; use environment variables
- **Encryption**: Use `utils/crypto.py` for sensitive data
- **Validation**: Validate user input in pages and services
- **Dependencies**: Be cautious adding new dependencies; prefer using existing ones
- **Logging**: Redact sensitive data using `utils/config.redact()`

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Issues**: Search existing issues before creating new ones
- **Slack/Discord**: If available, join our community chat
- **Mentorship**: First-time contributors welcome! Ask for help if stuck

## Recognition

We recognize all contributors! Those who contribute will be:
- Added to the CONTRIBUTORS.md file
- Mentioned in release notes
- Credited in the GitHub repo

## License

By contributing to Running Manager, you agree that your contributions will be licensed under the GPLv3 license. This ensures the project remains free and open for everyone.

---

Thank you for contributing to Running Manager! We appreciate your effort in making this project better. 🚀
