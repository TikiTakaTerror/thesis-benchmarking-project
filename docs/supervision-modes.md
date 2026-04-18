# Supervision Modes

R5 makes the supervision configs executable.

## Implemented Modes

### `full`

- labels: enabled
- concepts: enabled
- logic constraints: enabled
- concept supervision fraction: `1.0`

Meaning:
- task labels are used during training
- concept labels are used for every training example
- extra logic-constraint loss remains enabled where the family exposes it

### `label_only`

- labels: enabled
- concepts: disabled
- logic constraints: disabled
- concept supervision fraction: `0.0`

Meaning:
- task labels are used during training
- concept labels are removed from the training batches
- extra logic-constraint loss is disabled where the family exposes it

Current family behavior:
- `pipeline`: concept supervision is removed, but the symbolic layer still defines the label path
- `deepproblog`: concept supervision is removed, but the logic program still defines the label path
- `ltn`: concept supervision is removed and `satisfaction_weight` is forced to `0.0`

### `concept_50`

- labels: enabled
- concepts: enabled
- logic constraints: enabled
- concept supervision fraction: `0.5`

Meaning:
- task labels are used during training
- exactly 50% of training examples keep concept supervision
- the masked subset is deterministic given the run seed

## How Partial Concept Supervision Works

R5 applies a per-example boolean `concept_supervision_mask` to the training batches.

The models then:

- use task-label loss for the full batch
- use concept loss only for the supervised subset
- report:
  - `train_concept_supervised_examples`
  - `train_concept_supervision_fraction`

Evaluation still uses the full concept annotations when they exist, because supervision settings affect training, not the stored dataset labels.

## Verification

Run:

```bash
cd /Users/abdullahsaeed/thesis-benchmarking-project
source .venv/bin/activate
python scripts/check_supervision_modes.py
```

Expected final line:

```text
[OK] Supervision mode check passed.
```
