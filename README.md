# Running Manager

[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-green.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37+-red.svg)](https://streamlit.io)

**Running Manager** is an open-source Streamlit application for managing trail running coaching. It provides coaches and athletes with tools to plan training sessions, track activities from Strava or Garmin, analyze performance metrics, and manage running goals.

## ✨ Features

- **Weekly Planner**: Create and manage training sessions with flexible session types (fundamental endurance, long runs, intervals, races)
- **Session Templates**: Save and reuse session configurations for consistent planning
- **Strava Integration**: OAuth-based sync with automatic activity import (14-day window, incremental updates)
- **Activity Linking**: Manual matching of planned sessions with logged activities; RPE tracking
- **Performance Analytics**: 
  - Planned vs actual workload comparison (weekly/daily)
  - Training load metrics (acute/chronic TRIMP, distance-equivalent)
  - Multi-metric visualization (time, distance, DistEq, TRIMP)
  - Speed-effort scatter plots
- **Advanced Metrics**:
  - Distance-equivalent (DistEq) calculations accounting for elevation gain
  - Sport-specific equivalence (e.g., bike ride DistEq factors)
  - TRIMP (Training Impulse) using HR reserve weighting
  - Rolling training loads (7-day acute, 28-day chronic)
- **Interval Workouts**: Structured interval step editor with repeatable blocks and recovery phases
- **Garmin Support**: (Stub) Framework for future Garmin Connect integration
- **FR/EN Locale**: Fr-FR formatting for European users; extensible to other languages

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#installation)
- [Environment Setup](#-environment-setup)
- [Usage](#usage)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [License](#license)
- [Support](#support)

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or later
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/piromagnus/RunningManager
cd RunningManager

# Install dependencies
uv sync

rm -rf data/* # Clean the data directory to start fresh
# Or with pip:
# pip install -r requirements.txt

# Run the application
uv run streamlit run app.py
```

The app will open at `http://localhost:8501`.

## 🔧 Environment Setup

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

### Essential Variables

```bash
# Data storage directory
DATA_DIR=./data

# Strava API Integration (optional)
STRAVA_CLIENT_ID=your_client_id
STRAVA_CLIENT_SECRET=your_client_secret
STRAVA_REDIRECT_URI=http://localhost:8501/callback

# Encryption key for token storage (required if using Strava)
ENCRYPTION_KEY=<generate-via-command-below>

# Mapbox for premium basemaps (optional)
MAPBOX_API_KEY=your_mapbox_token
```

#### Generate Encryption Key

```bash
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

#### Strava Setup

1. Go to [Strava API Settings](https://www.strava.com/settings/api)
2. Create an app with:
   - **Application Name**: RunningManager
   - **Category**: Other
   - **Callback Domain**: `localhost` (dev) or your domain (prod)
3. Copy `Client ID` and `Client Secret` to `.env`
4. Set `STRAVA_REDIRECT_URI` (default: `"http://localhost:8501/Settings"`)

#### Mapbox Setup (Optional)

1. Create account at [Mapbox](https://www.mapbox.com)
2. Generate access token
3. Add to `.env` as `MAPBOX_API_KEY`

## Usage

### Starting the App

```bash
uv run streamlit run app.py
```

### Pages

- **Planner**: Create and manage weekly training sessions
- **Activities**: View activity feed, link unlinked sessions, browse activity details
- **Dashboard**: Training load trends and performance analytics
- **Analytics**: Planned vs actual metrics with flexible date ranges and category filters
- **Settings**: Strava integration, metrics recomputation, thresholds and goals configuration
- **Athlete**: Athlete profile management
- **Goals**: Set and track race/performance goals

### Testing

```bash
# Run the full test suite
pytest

# Run with coverage
pytest --cov
```

### Linting

```bash
# Lint code (Ruff, line length 100)
ruff check .

# Format code
ruff format .
```

## Architecture

### High-Level Overview

```
app.py (Streamlit entry)
├── pages/ (UI multi-page app)
│   ├── Planner.py: Session planning
│   ├── Dashboard.py: Training load trends
│   ├── Analytics.py: Planned vs actual analysis
│   ├── Activities.py: Activity feed & linking
│   └── Settings.py: Integrations & config
├── services/ (Domain logic)
│   ├── planner_service.py: Estimation (pace, distance, TRIMP)
│   ├── metrics_service.py: Metrics pipeline
│   ├── analytics_service.py: Analytics computations
│   ├── strava_service.py: OAuth & sync
│   └── ...
├── persistence/ (CSV storage)
│   ├── csv_storage.py: Pandas + portalocker IO
│   └── repositories.py: CRUD repos
└── utils/ (Helpers)
    ├── config.py: Env loading & secrets
    ├── formatting.py: fr-FR locale display
    ├── crypto.py: Token encryption
    └── ...
```

### Data Model

**CSV Tables** (stored in `DATA_DIR`, default `./data/`):
- `activities.csv`: Imported from Strava/Garmin
- `planned_sessions.csv`: Coach-created training sessions
- `links.csv`: Manual matches between sessions and activities
- `activities_metrics.csv`: Computed per-activity metrics
- `planned_metrics.csv`: Computed per-session estimates
- `weekly_metrics.csv`: Weekly aggregates (planned/actual)
- `daily_metrics.csv`: Daily aggregates with rolling windows
- `athletes.csv`, `settings.csv`, `thresholds.csv`, `goals.csv`, `templates.csv`, etc.

**Timeseries**: `data/timeseries/{activityId}.csv` — sampled HR, pace, elevation, etc.

### Key Invariants

- **CSV Decimals**: Always use `.` (period) as decimal separator; never locale-specific commas
- **Locale Display**: Use `utils/formatting.py` for UI display only; storage keeps raw decimals
- **File Locking**: All CSV reads/writes use `portalocker` for safe concurrent access
- **Secrets**: Never hardcode; load via `.env`; encrypt sensitive tokens with Fernet

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Install dev dependencies: `uv sync`
4. Make your changes
5. Run tests: `pytest`
6. Lint: `ruff check . && ruff format .`
7. Commit with clear messages: `git commit -m "feat: add feature X"`
8. Push and create a Pull Request

### Code Style

- **Line Length**: 100 characters (Ruff configured)
- **Imports**: Sorted with Ruff (isort rules)
- **Format**: Black-compatible formatting via Ruff
- **Type Hints**: Use type annotations for public APIs
- **Testing**: Add tests for new behavior; maintain >80% coverage

### Contribution Areas

- **Bug Fixes**: Report issues on GitHub; PRs welcome
- **Features**: Garmin integration, additional analytics, TCX export enhancements
- **Documentation**: Improve guides, add examples, translate to other languages
- **Performance**: Optimize CSV loading, chart rendering, metrics computation
- **Testing**: Expand test coverage, add integration tests
- **UI/UX**: Improve interface clarity, accessibility, internationalization

### Reporting Issues

When reporting bugs, please include:
- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs or error messages

### Pull Request Process

1. Ensure all tests pass: `pytest`
2. Update `AGENTS.md` or `docs/` if adding/changing architecture
3. Add tests for new features or bug fixes
4. Keep commits focused and descriptive
5. Reference related issues (e.g., `Fixes #123`)

## 📄 License

This project is licensed under the **GNU General Public License v3.0** (GPLv3). See [LICENSE](LICENSE) for full details.

**Summary**: You are free to use, modify, and distribute this software, provided that:
- You retain the copyright notice and license
- You provide a copy of the license with any distribution
- Any modifications are also licensed under GPLv3
- You disclose the source code if distributing the software

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Strava API Docs](https://developers.strava.com/docs/)
- [Garmin Connect Documentation](https://developer.garmin.com/)
- [TRIMP Formula References](https://en.wikipedia.org/wiki/Training_impulse)

## 🛟 Support

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/piromagnus/RunningManager/issues)
- **Discussions**: Join conversations on [GitHub Discussions](https://github.com/piromagnus/RunningManager/discussions)
- **Documentation**: See `AGENTS.md` for developer guidelines

## 👥 Contributors

Thanks to all contributors who have helped improve Running Manager!

---

**Running Manager** is built by a trail running enthusiast, for coaches and athletes who want sophisticated training analytics in open source. We welcome your contributions and feedback!
