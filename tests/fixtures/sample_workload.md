# Workload: Neural Network Architecture Search

## Context
Training a small language model on climbmix-400b dataset. The goal is to find
the best hyperparameters and architecture choices to minimize val_bpb within
a fixed 5-minute training budget.

## Experiment Space
- Directional: "learning_rate" (currently 0.04)
- Directional: "depth" (currently 8)
- Enum: "activation" options: GeLU, SiLU, ReLU
- Diverse: "attention_mechanism"
Constraint: depth * 64 must stay under 1024 for memory

## Execution
- Command: `uv run train.py > run.log 2>&1`
- Time budget per experiment: 5 minutes

## Evaluation
- Type: automatic
- Metric: val_bpb (lower is better)
- Parse: `grep "^val_bpb:" run.log`
- Secondary metrics: train_loss, perplexity

## Resources
- Per worker: 1 GPU
- Available: 4

## Budget
- Max API cost: $50
- Max experiments: 100
- Max wall time: 8h

## Constraints
- Files workers may modify: train.py
- Do not modify prepare.py
