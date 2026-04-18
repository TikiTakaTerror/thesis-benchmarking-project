#!/usr/bin/env python3
"""Generate raw Kand-Logic data with the local official rsbench generator checkout."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GENERATOR_ROOT = PROJECT_ROOT / "external" / "rsbench-code" / "rssgen"
DEFAULT_PYTHON = DEFAULT_GENERATOR_ROOT / ".venv-rssgen" / "bin" / "python"
DEFAULT_CONFIG = DEFAULT_GENERATOR_ROOT / "examples_config" / "kandinksy.yml"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "kand_logic" / "rsbench_generator_output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generator-root",
        default=str(DEFAULT_GENERATOR_ROOT),
        help="Path to external/rsbench-code/rssgen.",
    )
    parser.add_argument(
        "--python-executable",
        default=str(DEFAULT_PYTHON),
        help="Python executable inside the rsbench generator environment.",
    )
    parser.add_argument(
        "--config-path",
        default=str(DEFAULT_CONFIG),
        help="Kand-Logic generator YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
        help="Raw output directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Generator seed.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Override the YAML sample_size value.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output directory if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generator_root = resolve_path(args.generator_root)
    python_executable = resolve_path(args.python_executable, preserve_symlink=True)
    config_path = resolve_path(args.config_path)
    output_dir = resolve_path(args.output_dir)

    if not generator_root.exists():
        return fail(f"Generator root not found: {generator_root}")
    if not python_executable.exists():
        return fail(
            f"Generator Python executable not found: {python_executable}\n"
            "Expected the dedicated rssgen environment at "
            f"{DEFAULT_PYTHON}"
        )
    if not config_path.exists():
        return fail(f"Generator config not found: {config_path}")

    if output_dir.exists():
        if not args.overwrite:
            return fail(
                f"Output directory already exists: {output_dir}\n"
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(python_executable),
        "-m",
        "rssgen",
        str(config_path),
        "kandinsky",
        str(output_dir),
        "--seed",
        str(int(args.seed)),
    ]
    if args.n_samples is not None:
        command.extend(["--n_samples", str(int(args.n_samples))])

    try:
        subprocess.run(command, cwd=generator_root, check=True)
    except subprocess.CalledProcessError as exc:
        return fail(
            "rsbench Kand-Logic generation failed.\n"
            f"Command: {' '.join(command)}\n"
            f"Exit code: {exc.returncode}"
        )

    split_paths = [output_dir / split_name for split_name in ("train", "val", "test", "ood")]
    missing_splits = [str(path) for path in split_paths if not path.exists()]
    if missing_splits:
        return fail(f"Raw Kand-Logic generation is missing split directories: {missing_splits}")

    print(f"[OK] Generator root: {generator_root}")
    print(f"[OK] Generator config: {config_path}")
    print(f"[OK] Raw Kand-Logic output written: {output_dir}")
    for split_path in split_paths:
        num_png = sum(1 for _ in split_path.glob('*.png'))
        num_joblib = sum(1 for _ in split_path.glob('*.joblib'))
        print(
            f"[OK] {split_path.name} split: png={num_png}, joblib={num_joblib}"
        )
    return 0


def resolve_path(raw_path: str, *, preserve_symlink: bool = False) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path if preserve_symlink else path.resolve()
    combined = PROJECT_ROOT / path
    return combined if preserve_symlink else combined.resolve()


def fail(message: str) -> int:
    print(f"[ERROR] {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
