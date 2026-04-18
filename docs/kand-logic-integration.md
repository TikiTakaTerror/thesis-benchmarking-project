# Kand-Logic Integration

## Goal

R9 adds real Kand-Logic support end to end:

- raw data generation from the local official `rsbench-code/rssgen` checkout
- conversion into this repository's prepared-manifest format
- real managed runs through the existing runner
- backend API launch
- frontend launch

## Raw Dataset Generation

The project now includes:

- `scripts/generate_kand_logic_from_rsbench.py`

Default raw output location:

- `/Users/abdullahsaeed/thesis-benchmarking-project/data/raw/kand_logic/rsbench_generator_output/`

Run:

```bash
cd /Users/abdullahsaeed/thesis-benchmarking-project
.venv/bin/python scripts/generate_kand_logic_from_rsbench.py --overwrite
```

This uses the dedicated generator environment at:

- `/Users/abdullahsaeed/thesis-benchmarking-project/external/rsbench-code/rssgen/.venv-rssgen/`

## Prepared Dataset Conversion

The project now includes:

- `scripts/prepare_kand_logic_from_rsbench.py`

Default prepared output location:

- `/Users/abdullahsaeed/thesis-benchmarking-project/data/processed/kand_logic/`

Run:

```bash
cd /Users/abdullahsaeed/thesis-benchmarking-project
source .venv/bin/activate
python scripts/prepare_kand_logic_from_rsbench.py --overwrite
```

## Prepared Dataset Layout

The converted dataset uses the same prepared-manifest contract as MNLogic:

```text
data/processed/kand_logic/
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

## Concept Representation

Raw Kand-Logic concepts are categorical object attributes. The current model families require binary concept supervision, so the converter expands each object slot into one-hot binary concepts.

With the official local config:

- `3` figures per image
- `3` object slots per figure
- `3` shapes
- `3` colors

This becomes:

- `3 * 3 * (3 + 3) = 54` binary concepts

Example concept names:

- `fig1_obj1_shape_circle`
- `fig1_obj1_shape_square`
- `fig1_obj1_shape_triangle`
- `fig1_obj1_color_red`
- `fig1_obj1_color_yellow`
- `fig1_obj1_color_blue`

## Logic Representation

The converter also compiles the official Kand symbolic rule into the local rule-tree format already used by:

- the pipeline symbolic executor
- the LTN logic layer
- the DeepProbLog rule compiler

That compiled rule is stored in:

- `data/processed/kand_logic/metadata/source_info.json`

## Validation

Run:

```bash
cd /Users/abdullahsaeed/thesis-benchmarking-project
source .venv/bin/activate
python scripts/check_kand_logic_dataset.py
```

## Real Managed-Run Verification

Run:

```bash
cd /Users/abdullahsaeed/thesis-benchmarking-project
source .venv/bin/activate
python scripts/check_real_kand_logic_run.py
```

This checks:

- runtime config generation for `pipeline`, `ltn`, and `deepproblog`
- one managed Kand-Logic run per family
- backend API launch at `POST /api/v1/runs/launch/kand_logic`
- frontend launch from `/`

## Important Practical Note

DeepProbLog exact inference is much slower on Kand-Logic than on MNLogic because the Kand rule is much larger.

That is why the Kand smoke check uses:

- `limit_per_split=4` for `pipeline` and `ltn`
- `limit_per_split=1` for `deepproblog`

This is an honest runtime limitation of the current exact-inference path, not a hidden failure.
