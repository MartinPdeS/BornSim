"""Check observable release behavior in disposable repositories only."""

from pathlib import Path
import json
import re
import shutil
import subprocess
import sys
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[2]


def command(directory, *args):
    return subprocess.run(args, cwd=directory, capture_output=True, text=True, check=False)


@pytest.fixture
def release_repo(tmp_path):
    for name in ("pyproject.toml", "CITATION.cff", ".zenodo.json", "bornsim/_version.py", "conda.recipe/meta.yaml"):
        destination = tmp_path / name

        destination.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(ROOT / name, destination)

    shutil.copytree(ROOT / "tools", tmp_path / "tools", ignore=shutil.ignore_patterns("__pycache__"))

    for args in (
        ("init", "-b", "main"),
        ("config", "user.email", "tests@example.invalid"),
        ("config", "user.name", "BornSim tests"),
        ("add", "."),
        ("commit", "-m", "Initial fixture"),
    ):
        result = command(tmp_path, "git", *args)

        assert result.returncode == 0, result.stderr

    return tmp_path


def test_project_metadata_matches_installed_api():
    import bornsim

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]

    assert project["version"] == bornsim.__version__

    assert project["license"] == "MIT"

    assert not project.get("scripts")

    assert "gui" not in project["optional-dependencies"]

    assert (ROOT / project["readme"]).is_file()


def test_release_aligns_metadata_and_creates_annotated_tag(release_repo):
    project = tomllib.loads((release_repo / "pyproject.toml").read_text())["project"]

    major, minor, _ = map(int, project["version"].split("."))

    minor += 1

    version = f"{major}.{minor}.0"

    tag = f"v{version}"

    result = command(release_repo, sys.executable, "tools/release_tag.py", tag)

    assert result.returncode == 0, result.stderr

    assert command(release_repo, "git", "cat-file", "-t", tag).stdout.strip() == "tag"

    assert command(release_repo, "git", "status", "--porcelain").stdout == ""

    check = command(release_repo, sys.executable, "tools/check_release.py", "--version", tag)

    assert check.returncode == 0, check.stdout + check.stderr

    assert json.loads((release_repo / ".zenodo.json").read_text())["version"] == version

    for kind, expected in [
        ("patch", f"v{major}.{minor}.1"),
        ("minor", f"v{major}.{minor + 1}.0"),
        ("major", f"v{major + 1}.0.0"),
    ]:
        next_version = command(release_repo, sys.executable, "tools/next_release_version.py", kind)

        assert next_version.returncode == 0, next_version.stderr

        assert next_version.stdout.strip() == expected

    repeated = command(release_repo, sys.executable, "tools/release_tag.py", tag)

    assert repeated.returncode != 0

    assert "already exists" in repeated.stderr


@pytest.mark.parametrize("kind", ["patch", "minor", "major"])
def test_first_release_uses_project_version_without_creating_tags(release_repo, kind):
    version = tomllib.loads((release_repo / "pyproject.toml").read_text())["project"]["version"]

    major, minor, patch = map(int, version.split("."))

    expected = {
        "patch": f"v{major}.{minor}.{patch + 1}",
        "minor": f"v{major}.{minor + 1}.0",
        "major": f"v{major + 1}.0.0",
    }

    result = command(release_repo, sys.executable, "tools/next_release_version.py", kind)

    assert result.returncode == 0, result.stderr

    assert result.stdout.strip() == expected[kind]

    assert command(release_repo, "git", "tag", "--list").stdout == ""

    assert command(release_repo, "git", "status", "--porcelain").stdout == ""


def test_release_rejects_dirty_tree_and_invalid_tags(release_repo):
    before = (release_repo / "pyproject.toml").read_text()

    (release_repo / "unrelated.txt").write_text("Uncommitted work\n")

    result = command(release_repo, sys.executable, "tools/release_tag.py", "v0.2.0")

    assert result.returncode != 0

    assert "not clean" in result.stderr

    assert (release_repo / "pyproject.toml").read_text() == before

    result = command(release_repo, sys.executable, "tools/release_tag.py", "invalid")

    assert result.returncode != 0

    assert "vMAJOR.MINOR.PATCH" in result.stderr

    assert command(release_repo, "git", "tag", "--list").stdout == ""


def test_version_mismatch_is_detected(release_repo):
    citation = release_repo / "CITATION.cff"

    citation.write_text(re.sub(r"^version:.*$", 'version: "9.0.0"', citation.read_text(), flags=re.MULTILINE))

    result = command(release_repo, sys.executable, "tools/check_release.py")

    assert result.returncode != 0

    assert "CITATION.cff declares 9.0.0" in result.stderr
