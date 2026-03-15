# Workload: Autoresearch Climbmix

## Context
Training a small language model on the climbmix-400b dataset (Karpathy's autoresearch).
The goal is to minimize val_bpb within a fixed 5-minute training budget per experiment.
Only train.py may be modified — prepare.py contains the fixed evaluation harness,
data loading, and tokenizer.

## Experiment Space
- Directional: "batch_size" (currently 524288)
- Directional: "depth" (currently 8)
- Directional: "aspect_ratio" (currently 64)
- Directional: "learning_rate" (currently 0.04)
- Directional: "warmup_fraction" (currently 0.0)
- Directional: "cooldown_fraction" (currently 0.5)
- Directional: "weight_decay" (currently 0.0)
- Directional: "head_dim" (currently 128)
- Enum: "activation" options: GeLU, SiLU, ReLU
- Enum: "window_pattern" options: SSSL, SSSS, SSLL
- Diverse: "optimizer_strategy"

## Execution
- Command: `uv run train.py > run.log 2>&1`
- Time budget per experiment: 5 minutes

## Evaluation
- Type: automatic
- Metric: val_bpb (lower is better)
- Parse: `grep "^val_bpb:" run.log | awk '{print $2}'`
- Secondary metrics: peak_vram_mb, mfu_percent, num_steps

## Resources
- Per worker: 1 GPU
- Available: 1

## Budget
- Max experiments: 100
- Max wall time: 8h

## Constraints
- Files workers may modify: train.py
- Do not modify prepare.py
- Do not install new packages
- Simplicity criterion: prefer simpler code, reject complexity for marginal gains
