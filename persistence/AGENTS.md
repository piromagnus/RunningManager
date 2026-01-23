# Persistence Layer

CSV-backed storage with pandas and file locking.

## Files

| File | Purpose |
|------|---------|
| `csv_storage.py` | Thread-safe CSV I/O with portalocker |
| `repositories.py` | Typed repositories per table (headers, CRUD) |

## csv_storage.py

Core I/O abstraction:
- `read_csv(filename)`: Shared lock read
- `write_csv(filename, df)`: Exclusive lock write
- `append_row(filename, row, headers)`: Atomic append
- `upsert(filename, keys, row)`: Update or insert by key columns

Key behaviors:
- Creates parent directories automatically
- Handles empty files with header initialization
- Uses `portalocker.LOCK_SH` (read) and `LOCK_EX` (write)

## repositories.py

Base pattern:
```python
@dataclass
class BaseRepo:
    storage: CsvStorage
    file_name: str
    headers: List[str]
    id_column: str
```

Available repositories:
| Class | File | ID Column |
|-------|------|-----------|
| `ActivitiesRepo` | activities.csv | activityId |
| `PlannedSessionsRepo` | planned_sessions.csv | plannedSessionId |
| `LinksRepo` | links.csv | linkId |
| `ActivitiesMetricsRepo` | activities_metrics.csv | activityId |
| `PlannedMetricsRepo` | planned_metrics.csv | plannedSessionId |
| `DailyMetricsRepo` | daily_metrics.csv | dailyId |
| `WeeklyMetricsRepo` | weekly_metrics.csv | weekStartDate |
| `ThresholdsRepo` | thresholds.csv | thresholdId |
| `GoalsRepo` | goals.csv | goalId |
| `TemplatesRepo` | templates.csv | templateId |
| `SessionTemplatesRepo` | session_templates.csv | templateId |
| `AthletesRepo` | athlete.csv | athleteId |
| `SettingsRepo` | settings.csv | coachId |
| `TokensRepo` | tokens.csv | athleteId |

## Adding New Tables

1. Define headers list in `repositories.py`
2. Create repository class extending `BaseRepo`
3. Implement `_migrate_headers_if_needed()` if adding columns later
4. Update `data/AGENTS.md` with schema

## Column Migrations

Pattern for adding columns:
```python
def _migrate_headers_if_needed(self) -> None:
    path = self._path()
    if not path.exists():
        return
    df = self.storage.read_csv(self.file_name)
    if any(h not in df.columns for h in self.headers):
        df = _ensure_headers(df, self.headers)
        self.storage.write_csv(self.file_name, df)
```

## Invariants

- **Never bypass CsvStorage** for CSV I/O
- **Decimal separator**: Always `.` in storage
- **Headers**: Must match `self.headers` list exactly

## Related Files

- `data/AGENTS.md`: Schema documentation
- `services/metrics_service.py`: Uses repos for metrics pipeline
- `tests/test_csv_storage.py`: Storage tests
- `tests/test_repositories.py`: Repository tests

## Maintaining This File

Update when:
- Adding new repository classes
- Changing header lists
- Modifying locking behavior
- Adding new CsvStorage methods
