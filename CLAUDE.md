run command
```bash
cd c:/code/smtl-code && uv run python -m experiments.run_baseline <model> <n_rows>
```

Example:
```bash
uv run python -m experiments.run_baseline gpt-4o-mini 2
uv run python -m experiments.run_baseline claude-sonnet-4-6 100
```