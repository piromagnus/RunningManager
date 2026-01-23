# Tests

Pytest test suite with fixtures and fakes.

## Files

| File | Purpose |
|------|---------|
| `conftest.py` | Shared fixtures, portalocker/Babel fakes |
| `test_planner_service.py` | PlannerService tests |
| `test_planner_service_distance_eq.py` | DistanceEq calculation tests |
| `test_metrics_service.py` | Metrics pipeline tests |
| `test_metrics_sport_type.py` | Sport type categorization tests |
| `test_analytics_service.py` | Analytics service tests |
| `test_csv_storage.py` | CsvStorage tests |
| `test_repositories.py` | Repository CRUD tests |
| `test_interval_utils.py` | Interval normalization tests |
| `test_intervals_thresholds.py` | Threshold-based interval tests |
| `test_strava_service.py` | Strava OAuth/sync tests |
| `test_linking_service.py` | Activity linking tests |
| `test_activity_feed_service.py` | Feed building tests |
| `test_activity_detail_service.py` | Detail loading tests |
| `test_lap_metrics_service.py` | Lap metrics tests |
| `test_templates_service.py` | Template CRUD tests |
| `test_session_templates_service.py` | Session template tests |
| `test_planner_presenter.py` | Presenter layer tests |
| `test_pacer_service.py` | Race pacing tests |
| `test_formatting.py` | Formatting helper tests |
| `test_config.py` | Config loading tests |
| `test_time_ids.py` | Time/ID utility tests |
| `test_gpx_parser.py` | GPX parsing tests |

## conftest.py

### Fakes
```python
# Portalocker stub (no actual locking in tests)
class _DummyLock:
    def __enter__(self): return self
    def __exit__(self, *_): return False

# Babel stub (simplified formatting)
class _FakeBabelNumbers:
    @staticmethod
    def format_decimal(value, format=None, locale=None):
        # Returns comma-separated string
```

### Key Fixtures
```python
@pytest.fixture
def planner(tmp_path):
    # CsvStorage + ThresholdsRepo with test thresholds
    # Returns PlannerService

@pytest.fixture
def interval_steps_legacy():
    # Legacy repeats format

@pytest.fixture
def interval_steps_loops():
    # Current loops format

@pytest.fixture
def week_sessions(interval_steps_loops):
    # Sample week of sessions
```

## Running Tests

```bash
pytest                    # All tests
pytest tests/test_*.py    # Specific file
pytest -k "test_name"     # By name pattern
pytest -v                 # Verbose output
```

## Test Patterns

### Service Tests
```python
def test_service_method(tmp_path):
    storage = CsvStorage(base_dir=tmp_path)
    service = SomeService(storage)
    result = service.method(...)
    assert result == expected
```

### Repository Tests
```python
def test_repo_crud(tmp_path):
    storage = CsvStorage(base_dir=tmp_path)
    repo = SomeRepo(storage)
    id = repo.create({...})
    item = repo.get(id)
    assert item is not None
```

## Invariants

- **Deterministic**: No network calls, no real time dependencies
- **Isolated**: Use `tmp_path` fixture for file operations
- **Fakes**: Use conftest fakes for portalocker/Babel
- **Coverage**: Focus on domain invariants and edge cases

## Related Files

- `services/`: Services under test
- `persistence/`: Storage under test
- `utils/`: Utilities under test

## Maintaining This File

Update when:
- Adding new test files
- Adding new fixtures
- Changing test patterns
- Adding new fakes
