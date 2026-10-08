"""Verify that built wheels contain the Python API and license."""

import argparse
from pathlib import Path
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+", help="Wheel files or directories containing wheels.")
    args = parser.parse_args()
    wheels = []
    for path in args.paths:
        wheels.extend(path.glob("*.whl") if path.is_dir() else [path])
    if not wheels:
        raise SystemExit("No wheel found.")
    for wheel in wheels:
        with ZipFile(wheel) as archive:
            names = archive.namelist()
            required = [
                "bornsim/__init__.py",
                "bornsim/model.py",
                "bornsim/series.py",
                "bornsim/volume.py",
                "bornsim/green.py",
                "bornsim/ensemble.py",
                "bornsim/_validation.py",
                "bornsim/api.py",
                "bornsim/source.py",
                "bornsim/solver.py",
                "bornsim/results.py",
                "bornsim/_archives.py",
                "bornsim/_result_plotting.py",
                "bornsim/_volume_plotting.py",
                "bornsim/units.py",
                "bornsim/media.py",
                "bornsim/geometry.py",
                "bornsim/angular_data.py",
                "bornsim/rotation.py",
                "bornsim/material.py",
                "bornsim/ensemble_sampling.py",
                "bornsim/grid.py",
                "bornsim/sampling.py",
                "bornsim/_result_validation.py",
            ]
            if not all(name in names for name in required):
                raise SystemExit(f"Missing package sources in {wheel.name}")
            if not any(name.endswith("/licenses/LICENSE") for name in names):
                raise SystemExit(f"Missing MIT license in {wheel.name}")
            if "bornsim/dashboard.py" in names or "bornsim/__main__.py" in names:
                raise SystemExit(f"Dashboard sources leaked into {wheel.name}")
            if any(name.endswith("/entry_points.txt") for name in names):
                raise SystemExit(f"Unexpected console launcher in {wheel.name}")
            if any(name.startswith(("tests/", "docs/", "tools/")) for name in names):
                raise SystemExit(f"Development files leaked into {wheel.name}")
        print(f"Verified {wheel.name}")


if __name__ == "__main__":
    main()
