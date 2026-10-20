# configs/

One YAML per experiment, and every YAML is committed. Hyperparameters live
here, never hardcoded in a source file; a hardcoded threshold is flagged in
review (CONTRIBUTING, Step 3).

Workstreams get their own subdirectory as they grow, for example
`configs/prediction/residual_gru_v1.yaml`.

Every row in [EXPERIMENT_LOG.md](../docs/EXPERIMENT_LOG.md) names a config
path and a seed. That is what makes a claimed number re-runnable: a result
that cannot be traced back to a committed config and seed is changed to
`TBD`.
