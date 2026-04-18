# MNLogic Rsbench Conversion

## Goal

R2 converts raw MNLogic output produced by the official `rsbench-code/rssgen` XOR generator
into this repository's prepared-manifest dataset format.

The result is a real prepared dataset under:

- `/Users/abdullahsaeed/thesis-benchmarking-project/data/processed/mnlogic/`

This phase still does **not** make the managed run system use real MNLogic yet.
That wiring belongs to later phases.

## Raw Input Expected

The converter expects the official generator output layout:

```text
data/raw/mnlogic/rsbench_generator_output/
  train/
    0.png
    0.joblib
    ...
  val/
  test/
  ood/
```

Each `.joblib` file contains:
- `label`
- `meta.concepts`

## Prepared Output Produced

The converter writes:

```text
data/processed/mnlogic/
  images/
    train/
    val/
    test/
    ood/
  metadata/
    concept_schema.json
    label_schema.json
    source_info.json
  splits/
    train.csv
    val.csv
    test.csv
    ood.csv
```

## Conversion Command

Run this from the project root:

```bash
cd /Users/abdullahsaeed/thesis-benchmarking-project
source .venv/bin/activate
python scripts/prepare_mnlogic_from_rsbench.py --overwrite
```

## Validation Command

Then validate the prepared dataset:

```bash
python scripts/check_mnlogic_dataset.py
```

## Important Upstream Caveat

The official raw output generated from `examples_config/xor.yml` has a split issue:

- `val`
- `test`
- `ood`

become single-class with the default upstream proportions.

This was observed directly in the generated raw dataset and comes from the current upstream
generator behavior, not from this repository's conversion script.

That means:
- R2 gives you a real prepared MNLogic dataset
- but it is not yet a thesis-ready balanced evaluation dataset

We should address that before using it for final experiments.

## Current Usefulness

After R2, the project has:
- real raw MNLogic source data
- real prepared MNLogic files in the local adapter format
- a reproducible conversion path

After R3 and R4, the backend will be able to train on this prepared dataset directly.
