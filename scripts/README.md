# scripts/

One entry point per task, named for the task: `train_*.py`, `eval_*.py`, and
the split generator. Every run is launched from a committed config:

```bash
python scripts/train_prediction.py --config configs/prediction/residual_gru_v1.yaml
python scripts/eval_prediction.py  --config configs/prediction/residual_gru_v1.yaml
```

Then log the run in [EXPERIMENT_LOG.md](../docs/EXPERIMENT_LOG.md) in the
same change. Training never lives in notebooks (README section 9).
