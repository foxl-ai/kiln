"""The version is written once, in kiln/__init__.py, and everything else follows it.

pyproject.toml reads it (setuptools `attr:`), so a built or installed Kiln reports the same
string; CHANGELOG.md's newest release heading must name it, so a release cannot ship notes for
another version. Release rules: release/RELEASING.md (dev repo only).
"""

import importlib.metadata
import os
import re

import pytest

import kiln

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# https://semver.org/spec/v2.0.0.html, the "suggested regular expression" (core and pre-release; no build).
SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
                    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?$")


def test_version_is_semver():
    assert SEMVER.match(kiln.__version__), kiln.__version__


def test_pyproject_takes_the_version_from_the_package():
    tomllib = pytest.importorskip("tomllib")  # Python >= 3.11
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
        meta = tomllib.load(f)
    project = meta["project"]
    assert "version" not in project, "pyproject.toml must not carry its own version; it reads kiln.__version__"
    assert "version" in project["dynamic"]
    assert meta["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "kiln.__version__"}


def test_changelog_names_the_version():
    with open(os.path.join(ROOT, "CHANGELOG.md")) as f:
        text = f.read()
    # Keep a Changelog: "## [Unreleased]" may sit on top; the first dated heading is the newest release.
    released = re.findall(r"^## \[([^\]]+)\] - (\d{4}-\d{2}-\d{2})$", text, re.M)
    assert released, "CHANGELOG.md has no '## [X.Y.Z] - YYYY-MM-DD' heading"
    assert released[0][0] == kiln.__version__, (released[0][0], kiln.__version__)
    assert f"[{kiln.__version__}]: https://github.com/foxl-ai/kiln/releases/tag/v{kiln.__version__}" in text


def test_installed_metadata_matches_when_installed():
    """In the container Kiln is pip-installed: the distribution must report this same version. A
    source checkout run with PYTHONPATH=. has no distribution, which is fine."""
    try:
        installed = importlib.metadata.version("kiln")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("kiln is not pip-installed here (PYTHONPATH checkout)")
    assert installed == kiln.__version__, f"installed distribution {installed} != kiln.__version__ {kiln.__version__}"
