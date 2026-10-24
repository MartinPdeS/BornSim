# Contributing to BornSim

Use Python 3.11 or newer and an isolated environment:

```sh
python -m venv .venv
make editable PYTHON=.venv/bin/python
make check PYTHON=.venv/bin/python
make docs PYTHON=.venv/bin/python
make package-check PYTHON=.venv/bin/python
make release-check PYTHON=.venv/bin/python
git diff --check
```

On Windows use `.venv/Scripts/python.exe`. The public API lives in `bornsim/`; tests are grouped by analytical, numerical and packaging behavior. Use explicit units and keyword arguments in examples. Numerical tests should include independent references and asymmetric data, rather than merely duplicating implementation formulas.

Install local hooks with `.venv/bin/pre-commit install`. Format changes with `make format`. Documentation sources are in `docs/source`; runnable examples are in `docs/examples`. Scratch research belongs in `development/` and is excluded from package distributions.

## Releases

Versions are declared in project metadata and checked across the package, citation, Zenodo and Conda files. Commit work before tagging. `make tag VERSION=v0.1.0` creates a local release commit and annotated tag. `make release patch`, `minor`, or `major` chooses a version above the highest semantic Git tag, aligns all metadata, commits, tags and pushes the resulting HEAD and that specific tag. These commands are never part of ordinary tests.

CI builds/tests on Python 3.11–3.13 across Linux, macOS and Windows. The tag-triggered PyPI workflow requires the repository's `PYPI_API_TOKEN` secret. Documentation builds are checked by CI; hosting configuration is separate. Conda builds can be verified manually through the Conda workflow; no channel upload is configured. Remote repository creation and publishing credentials must be configured before publication.

## Shared release versioning

Versions are explicit in `[project].version`; builds never derive versions from
Git. Install the pinned release tool in the interpreter used by Make:

```sh
python -m pip install "martinpdes-maintenance==0.2.2"
make release-check PYTHON=python
```

The `release` extra declares the same tool. The scripts in `tools/` delegate to
it, and `[tool.mps-maintenance]` declares the package's release files. Use
`python tools/release_tag.py vX.Y.Z --dry-run` to preview changes from a clean,
full checkout. The tool aligns configured metadata, commits, and creates an
annotated tag; it never pushes. Existing Makefile publishing shortcuts retain
their push behavior. Fetch remote tags first. With no release tags, `next` uses
the explicit project version as its starting point.
