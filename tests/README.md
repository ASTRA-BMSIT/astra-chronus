# tests/

Per-module gates, run in CI. One test file per module, named after the thing
it gates (`tests/test_residual_gru.py`).

CI runs the same three checks you should run before pushing (CONTRIBUTING,
Step 5):

```bash
ruff check src tests scripts
ruff format --check src tests scripts
bandit -r src -ll -q
pytest tests/ -q
```

Run only the tests related to your change, not the full suite on every
commit. CI skips pytest until the first `test_*.py` lands here.
