# Dataset Infrastructure Setup

## Goal

Phase 2 adds the dataset foundation only.

It provides:
- a common dataset adapter interface
- a reusable prepared-manifest dataset format
- an `MNLogicDatasetAdapter`
- a demo dataset generator for verification
- a dataset validation script

It does not provide:
- the real MNLogic download process
- image preprocessing for training
- Kand-Logic integration
- model or training code

Since R2, this repository also includes a converter for raw official rsbench MNLogic output:

- [docs/mnlogic-rsbench-conversion.md](/Users/abdullahsaeed/thesis-benchmarking-project/docs/mnlogic-rsbench-conversion.md)

## Prepared Dataset Contract

The dataset adapter expects this layout:

```text
data/processed/<dataset_name>/
  images/
    ...
  metadata/
    concept_schema.json
    label_schema.json
  splits/
    train.csv
    val.csv
    test.csv
    ood.csv        # optional
```

## Required Split CSV Columns

Every split CSV must contain:
- `sample_id`
- `image_path`
- `label_id`

Concept annotations must be stored as one column per concept with this prefix:
- `concept__<concept_name>`

Any additional columns are treated as metadata.

## Schema File Format

### `concept_schema.json`

```json
{
  "dataset_name": "mnlogic",
  "concepts": [
    {
      "name": "is_red",
      "index": 0,
      "type": "binary",
      "description": "Example concept"
    }
  ]
}
```

### `label_schema.json`

```json
{
  "dataset_name": "mnlogic",
  "labels": [
    {
      "id": 0,
      "name": "negative",
      "description": "Example label"
    }
  ]
}
```

## Exact Verification Flow

Run these commands from the project root:

```bash
cd /Users/abdullahsaeed/thesis-benchmarking-project
source .venv/bin/activate
python scripts/create_mnlogic_demo_dataset.py --output-dir data/processed/mnlogic_demo
python scripts/check_mnlogic_dataset.py --dataset-dir data/processed/mnlogic_demo
```

## Expected Verification Output

The validation command should end with output similar to:

```text
[OK] Dataset root: /Users/abdullahsaeed/thesis-benchmarking-project/data/processed/mnlogic_demo
[OK] Concepts loaded: 3
[OK] Labels loaded: 2
[OK] train split: 4 samples
[OK] val split: 2 samples
[OK] test split: 2 samples
[OK] ood split: 2 samples
[OK] Dataset validation passed.
```

Exact version text and absolute paths may differ slightly, but the split counts should match.

## Real MNLogic Placement Contract

When you later have the real prepared MNLogic dataset, place it here:

- `/Users/abdullahsaeed/thesis-benchmarking-project/data/processed/mnlogic/`

Required files:
- `/Users/abdullahsaeed/thesis-benchmarking-project/data/processed/mnlogic/metadata/concept_schema.json`
- `/Users/abdullahsaeed/thesis-benchmarking-project/data/processed/mnlogic/metadata/label_schema.json`
- `/Users/abdullahsaeed/thesis-benchmarking-project/data/processed/mnlogic/splits/train.csv`
- `/Users/abdullahsaeed/thesis-benchmarking-project/data/processed/mnlogic/splits/val.csv`
- `/Users/abdullahsaeed/thesis-benchmarking-project/data/processed/mnlogic/splits/test.csv`
- optional: `/Users/abdullahsaeed/thesis-benchmarking-project/data/processed/mnlogic/splits/ood.csv`

Then validate it with:

```bash
python scripts/check_mnlogic_dataset.py
```

Without `--dataset-dir`, the script uses the default root from `src/configs/datasets/mnlogic.yaml`.

## Current Limitation

The adapter validates file layout and manifest consistency, but it does not decode images into tensors yet. That is intentional and belongs to a later phase.

There is also a known upstream caveat for the official `xor.yml` generator output:
- default `val`, `test`, and `ood` splits can become single-class
- this is a raw-data generation issue, not a prepared-dataset conversion issue
