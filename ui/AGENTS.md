# UI Components

Complex UI widgets requiring significant state management.

## Files

| File | Purpose |
|------|---------|
| `interval_editor.py` | Interval step editor with loops and actions |

## interval_editor.py

Full-featured interval workout editor.

### Main Function
```python
def render_interval_editor(
    athlete_id: str,
    initial_steps: Optional[Dict],
    planner: PlannerService,
    key_prefix: str = "interval",
) -> Dict[str, Any]:
    # Returns normalized steps JSON
```

### Features
- Warmup/cooldown duration inputs
- Loop management (add/remove loops)
- Actions per loop (run, recovery)
- Target type selection (hr, pace, sensation)
- Threshold label selection
- Ascent/descent per action
- Between-loop recovery

### State Keys
Uses `{key_prefix}_*` pattern:
- `{prefix}_warmup_sec`
- `{prefix}_cooldown_sec`
- `{prefix}_loops`
- `{prefix}_between_loop_recover`

### Steps JSON Schema
```json
{
  "warmupSec": 600,
  "cooldownSec": 300,
  "betweenLoopRecoverSec": 60,
  "preBlocks": [],
  "loops": [
    {
      "repeats": 5,
      "actions": [
        {
          "kind": "run",
          "sec": 180,
          "targetType": "hr",
          "targetLabel": "Threshold 60",
          "ascendM": 0,
          "descendM": 0
        },
        {
          "kind": "recovery",
          "sec": 60,
          "targetType": null,
          "targetLabel": null,
          "ascendM": 0,
          "descendM": 0
        }
      ]
    }
  ],
  "postBlocks": []
}
```

## Related Files

- `pages/Planner.py`: Embeds interval editor
- `pages/SessionCreator.py`: Uses for template creation
- `services/interval_utils.py`: Step normalization
- `services/planner_service.py`: Duration/distance calculations
- `widgets/session_forms.py`: Interval form wrapper

## Maintaining This File

Update when:
- Adding new action types
- Changing steps schema
- Adding new editor features
- Modifying state management
