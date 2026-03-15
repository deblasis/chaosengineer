# Workload: Autoresearch Irish Folk Music

## Context
Training a small language model on the Sanderwoods Irishman ABC notation dataset
(Irish folk music in ABC format). Based on the experiment documented by Onchain AI Garage.
The goal is to minimize val_bpb within a fixed 5-minute training budget per experiment.
The dataset is smaller and lower-entropy than text — ABC notation has a limited character
set, rigid syntax, and repeated patterns. Optimal strategies tend toward smaller, faster
models that see the data many times within the budget.

## Experiment Space
- Directional: "batch_size" (currently 524288)
- Directional: "depth" (currently 8)
- Directional: "aspect_ratio" (currently 64)
- Directional: "learning_rate" (currently 0.04)
- Directional: "warmup_fraction" (currently 0.0)
- Directional: "cooldown_fraction" (currently 0.5)
- Directional: "weight_decay" (currently 0.0)
- Directional: "head_dim" (currently 128)
- Enum: "value_embeddings" options: true, false
- Enum: "window_pattern" options: SSSL, SSSS, SSLL

## Execution
- Command: `uv run train.py > run.log 2>&1`
- Time budget per experiment: 5 minutes

## Evaluation
- Type: automatic
- Metric: val_bpb (lower is better)
- Parse: `grep "^val_bpb:" run.log | awk '{print $2}'`
- Secondary metrics: peak_vram_mb, num_steps

## Resources
- Per worker: 1 GPU
- Available: 1

## Budget
- Max experiments: 20
- Max wall time: 2h

## Constraints
- Files workers may modify: train.py, prepare.py
- Simplicity criterion: prefer simpler code, reject complexity for marginal gains
