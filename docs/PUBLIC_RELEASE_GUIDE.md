# Public Release & Open-Source Guide

This document outlines the improvements made to prepare **Running Manager** for public release as an open-source project.

## 📋 Overview of Updates

Running Manager is a sophisticated Streamlit application for managing trail running coaching. This guide explains:
1. What the project does
2. How to set it up
3. How to install and run it
4. How to contribute
5. What open-source practices have been implemented

## What is Running Manager?

**Running Manager** is a comprehensive web application designed for coaches and athletes to:

- **Plan Training**: Create weekly training sessions with multiple session types (fundamental endurance, long runs, intervals, races)
- **Track Activities**: Import activities from Strava or Garmin; manually link planned sessions with logged activities
- **Analyze Performance**: Visualize training load, adherence, and performance metrics over time
- **Manage Templates**: Save reusable session templates for consistent planning
- **Compute Metrics**: Calculate distance-equivalent, TRIMP, and rolling training loads

### Key Features

- **Multi-user Planning**: Single coach managing one or more athletes
- **Strava OAuth Integration**: Secure, encrypted token storage; incremental 14-day sync
- **Advanced Analytics**: Planned vs actual comparison, rolling training windows, performance trends
- **Interval Workouts**: Editor for structured interval sessions with repeatable blocks
- **Data Privacy**: CSV-based local storage; no cloud dependency for core features
- **Locale Support**: Fr-FR formatting for French users; extensible to other languages

## Installation & Setup

### Quick Start (5 minutes)

```bash
# Clone the repository
git clone https://github.com/yourusername/RunningManager.git
cd RunningManager

# Install dependencies
uv sync

# Set up environment
cp .env.example .env
# Edit .env with your configuration

# Run the application
uv run streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Detailed Installation

See [README.md](../README.md) for:
- Full prerequisites and dependency installation
- Environment variable configuration
- Strava API setup and OAuth flow
- Mapbox configuration for enhanced maps
- Troubleshooting guide

### Development Setup

```bash
# Install dev dependencies
uv sync

# Run tests
pytest

# Run linting
ruff check . && ruff format .

# Check test coverage
pytest --cov=.
```

## Architecture Overview

### High-Level Structure

```
Running Manager
├── pages/           # Streamlit multi-page UI
│   ├── Planner.py      → Weekly session planning
│   ├── Dashboard.py     → Training load trends
│   ├── Analytics.py     → Planned vs actual analysis
│   ├── Activities.py    → Activity feed & linking
│   └── Settings.py      → Configuration & integrations
├── services/        # Domain business logic
│   ├── planner_service.py      → Estimation algorithms
│   ├── metrics_service.py       → Metrics computation
│   ├── analytics_service.py     → Analytics queries
│   ├── strava_service.py        → Strava OAuth & sync
│   └── ...
├── persistence/     # CSV-based data layer
│   ├── csv_storage.py      → Pandas + portalocker IO
│   └── repositories.py      → CRUD operations per table
├── utils/           # Helper modules
│   ├── config.py        → Environment & secrets
│   ├── formatting.py     → Locale-specific display
│   ├── crypto.py        → Token encryption
│   └── ...
└── data/            # CSV tables & timeseries
    ├── activities.csv
    ├── planned_sessions.csv
    ├── weekly_metrics.csv
    └── timeseries/      → Per-activity HR, pace, etc.
```

### Data Model

**Core Tables**:
- `activities.csv`: Imported from Strava/Garmin
- `planned_sessions.csv`: Coach-created training plans
- `links.csv`: Manual matches between planned sessions and activities
- `weekly_metrics.csv`: Aggregated performance data
- Derived tables: `daily_metrics.csv`, `activities_metrics.csv`, etc.

**Key Invariants**:
- CSV decimals: Always `.` (period), never `,` (comma)
- File locking: `portalocker` for concurrent access safety
- Encryption: Sensitive tokens encrypted with Fernet
- Locale: Display uses `utils/formatting.py`; storage uses raw decimals

See [AGENTS.md](../AGENTS.md) for comprehensive architecture details.

## Contributing

We welcome contributions! Here's how to get involved:

### For First-Time Contributors

1. **Pick an issue**: Look for [good-first-issue](https://github.com/yourusername/RunningManager/labels/good-first-issue) tags
2. **Fork & branch**: `git checkout -b feature/your-feature`
3. **Develop locally**: `uv sync` and make your changes
4. **Test & lint**: `pytest` and `ruff check .`
5. **Create PR**: Push and open a pull request with clear description

### Contribution Areas

| Area | Examples | Difficulty |
|------|----------|-----------|
| **Bug Fixes** | Fix reported issues; improve error handling | Easy-Medium |
| **Features** | Garmin integration; TCX export; new metrics | Medium-Hard |
| **Documentation** | Improve guides; add examples; fix typos | Easy |
| **Performance** | Optimize CSV loading; speed up charts; cache improvements | Medium-Hard |
| **Testing** | Add tests; improve coverage; integration tests | Medium |
| **UI/UX** | Improve interface; accessibility; internationalization | Medium |

### Code Style

- **Line Length**: 100 characters (Ruff)
- **Formatting**: Black-compatible via Ruff
- **Type Hints**: Required for public APIs
- **Docstrings**: Google-style format
- **Tests**: >80% coverage target

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.

## Open-Source Governance

### License: GPLv3

Running Manager is licensed under the **GNU General Public License v3.0**. This means:

✅ **You can**:
- Use the software freely
- Modify the source code
- Distribute modified versions
- Use for commercial purposes

⚠️ **You must**:
- Include the license in distributions
- Disclose source code modifications
- Apply the same license to derivative works
- Retain copyright notices

📄 See [LICENSE](../LICENSE) for full legal text.

### Code of Conduct

We maintain a welcoming community following the [Contributor Covenant](https://www.contributor-covenant.org/). See [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md) for our standards and enforcement process.

### Community Support

- **Issues**: Report bugs or request features via GitHub Issues
- **Discussions**: Ask questions or discuss ideas in GitHub Discussions
- **Email**: Contact maintainers at [INSERT EMAIL]
- **Documentation**: See `AGENTS.md` for developer guidelines

## Security & Privacy

### Data Security

- **CSV Storage**: Local file-based; no cloud dependency required
- **Token Encryption**: Strava tokens encrypted with Fernet before storage
- **Environment Variables**: Secrets loaded via `.env`; never hardcoded
- **Logging**: Sensitive data redacted using `utils/config.redact()`

### Best Practices

1. Generate unique `ENCRYPTION_KEY` per environment
2. Never commit `.env` files (git-ignored by default)
3. Use HTTPS in production
4. Rotate encryption keys periodically
5. Monitor token usage for suspicious activity

## Performance Optimization

### Implemented

- `@st.cache_data` for expensive DataFrame operations
- CSV file locking prevents data corruption
- Incremental Strava sync (only 14-day window)
- Raw JSON caching for API responses

### Recommendations

- Profile metrics computation for large datasets
- Consider async operations for API calls
- Batch Streamlit UI updates
- Implement pagination for large activity lists

See [docs/production_readiness.md](./production_readiness.md) for deployment insights.

## Testing Strategy

### Coverage

- **Unit Tests**: Services, repositories, utilities
- **Integration Tests**: CSV operations, metrics pipeline
- **UI Tests**: Streamlit page interactions (limited)

### Running Tests

```bash
# Full suite
pytest

# Specific file
pytest tests/services/test_planner_service.py

# With coverage report
pytest --cov=. --cov-report=html
```

### Fakes & Fixtures

- `conftest.py` provides `portalocker` stub for deterministic testing
- `freezegun` for time-dependent tests
- Mock external APIs to avoid network calls

## Deployment Considerations

### Development

```bash
uv run streamlit run app.py
```

Runs on `http://localhost:8501` (default).

### Production

- Consider Docker containerization
- Use persistent volume for `data/` directory
- Set appropriate environment variables
- Enable HTTPS for Strava OAuth
- Monitor logs for errors
- Implement backups for CSV data

See [docs/production_readiness.md](./production_readiness.md) for detailed deployment guide.

## Roadmap & Future Work

### Short-term (Next Release)

- [ ] Garmin Connect integration (currently stubbed)
- [ ] TCX export for interval workouts
- [ ] Improved error handling and logging
- [ ] Performance optimizations

### Medium-term

- [ ] Multi-athlete dashboard
- [ ] Advanced AI-powered training recommendations
- [ ] Integration with additional platforms (Apple Health, etc.)
- [ ] Comprehensive API documentation

### Long-term Vision

- [ ] SaaS offering for coaches managing multiple athletes
- [ ] Mobile-first interface
- [ ] Real-time collaboration features
- [ ] Advanced analytics and machine learning

## Resources

### Documentation

- [README.md](../README.md) — Getting started
- [AGENTS.md](../AGENTS.md) — Architecture & conventions
- [CONTRIBUTING.md](../CONTRIBUTING.md) — How to contribute
- [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md) — Community standards
- [production_readiness.md](./production_readiness.md) — Deployment guide

### External Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Strava API Docs](https://developers.strava.com/docs/)
- [Running Power Research](https://docs.google.com/document/d/1oHKR9IfEqz6t-O2a5y76dBhvG5RM8F53kQGZhsOGqiA/)
- [TRIMP Formula](https://en.wikipedia.org/wiki/Training_impulse)
- [GPLv3 License](https://www.gnu.org/licenses/gpl-3.0.txt)

## Getting Help

### Common Questions

**Q: How do I set up Strava integration?**
A: See the [Environment Setup](../README.md#-environment-setup) section in README.md.

**Q: Can I use this commercially?**
A: Yes, as long as you comply with GPLv3 terms (share source code, use same license).

**Q: How do I report a security issue?**
A: Please email [INSERT SECURITY EMAIL] rather than using public GitHub issues.

**Q: Can I contribute?**
A: Yes! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

### Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/RunningManager/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/RunningManager/discussions)
- **Email**: [maintainer@example.com]

## Summary

Running Manager is ready for public use and community contributions! We've implemented:

✅ Comprehensive README with installation and usage guides
✅ GPLv3 license for legal clarity
✅ Contributing guidelines for smooth collaboration
✅ Code of Conduct for a welcoming community
✅ Architecture documentation in AGENTS.md
✅ Production readiness guide for deployment

We welcome your contributions, feedback, and ideas. Happy coding! 🚀

---

**Last Updated**: October 2024
**Version**: 1.0-alpha (Pre-release)
**Maintainer(s)**: [Your Name/Team]
