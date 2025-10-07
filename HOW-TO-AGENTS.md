# HOW-TO-AGENTS

This guide teaches AI coding agents how to work effectively in any repository. It focuses on safety, determinism, clarity, and collaboration. Use it to create or refine an `AGENTS.md` file for a specific project.

## Goals for AI Agents

- Keep changes small, deterministic, and reversible
- Follow the project’s standards and protect invariants
- Prefer readable, maintainable code over cleverness
- Add or update tests when behavior changes
- Explain your plan, then implement it and verify

## Discovery First

Before editing, always perform a quick discovery pass:

- Read top-level docs: `README.md`, `AGENTS.md` (if present), contributing docs, architecture notes
- Examine configuration and project files (e.g., `pyproject`, `package.json`, `requirements.txt`)
- Skim key modules and the test suite to learn invariants, public APIs, and patterns
- Identify security-sensitive areas (secrets, auth, PII) and established practices

Tip: Create a short todo for complex tasks and mark progress after each step.

## Edit Safely and Incrementally

- Make the smallest viable change; avoid sprawling refactors unless requested
- Preserve existing formatting and style; do not reformat unrelated code
- Add imports and dependencies explicitly; keep dependency changes minimal
- Use feature flags or guards when changing behavior that might be risky
- Write or adapt tests in the same change to validate new behavior

## Code Quoting Rules (for Cursor-style agents)

Use two different formats depending on the source of the code you show in messages:

1) Existing code (use code references, include line numbers and file path):

```
12:34:path/to/file.py
# code here from the repository
```

2) New or proposed code (use standard fenced code blocks with a language tag):

```python
def example():
    return 42
```

Rules:
- Never include line numbers inside code content
- Do not indent the opening/closing triple backticks
- Do not mix formats in the same block

## Testing Mindset

- Run existing tests locally if possible; add tests for new logic or bug fixes
- Prefer focused unit tests; keep tests deterministic (no network or time flakiness unless mocked)
- Validate edge cases and error paths, not just the happy path

## Security & Secrets

- Never log raw secrets or tokens; implement secret redaction helpers
- Load secrets from environment or secret stores; never hardcode
- Avoid printing sensitive data in exceptions or test output
- Be careful with PII and access tokens in fixtures and snapshots

## Data & Migration Safety

- Understand storage format invariants (e.g., CSV header order, decimal separator)
- Backfill or migrate data with idempotent scripts; re-run should be safe
- Validate inputs and guard against partial writes; use locks if the codebase does

## Conventions & Style

- Match the project’s language version and linting settings
- Prefer early returns and shallow nesting; avoid broad try/except
- Use explicit, descriptive names (avoid single-letter variables)
- Keep comments concise and purposeful (explain non-obvious rationale)

## Git & PR Flow

- Use feature branches; keep commits logical and scoped
- Follow the repository’s commit message style (e.g., Conventional Commits)
- In PRs, explain the change, risks, and how you tested it

## Task Management (optional but recommended)

- If a task system exists (e.g., Taskmaster), use it to plan, expand, and track work
- Break large efforts into subtasks; mark them in-progress and done as you go

## Template: AGENTS.md (General)

Copy and adapt this structure for any repository.

```markdown
# AGENTS.md

## Project Overview
- Purpose, scope, primary users

## Architecture & Layout
- Brief module map and key directories

## Setup
- Prerequisites and install steps
- Environment variables and secrets management

## Run, Test, Build
- Commands to run the app, test suite, and build or package

## Code Style & Invariants
- Language version, linters/formatters
- Domain invariants that must be preserved

## Security Practices
- How secrets and tokens are managed; redaction rules

## Data & Storage
- Data formats, headers, and migration notes

## Development Workflow
- Branching, commit style, PR guidance
- Task tracking (if applicable)

## Common Pitfalls
- Known sharp edges and how to avoid them
```

## Checklist for Agents

- [ ] I read the top-level docs and tests
- [ ] I identified invariants and security constraints
- [ ] I proposed a small plan with clear steps
- [ ] I implemented with minimal, readable changes
- [ ] I added/updated tests and ran them
- [ ] I documented new behavior or decisions where needed


