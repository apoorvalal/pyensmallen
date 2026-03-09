#!/usr/bin/env python3
"""Resolve the package version for Meson builds.

This supports three build contexts:

1. Tagged GitHub Actions builds via ``SETUPTOOLS_SCM_PRETEND_VERSION``.
2. Source distributions, which include ``PKG-INFO`` with the resolved version.
3. Git checkouts, using ``setuptools_scm`` directly.
"""

from __future__ import annotations

from pathlib import Path
import os
import re
import sys


ROOT = Path(__file__).resolve().parents[1]


def _version_from_env() -> str | None:
    version = os.environ.get("SETUPTOOLS_SCM_PRETEND_VERSION")
    if version:
        return version
    return None


def _version_from_pkg_info() -> str | None:
    pkg_info = ROOT / "PKG-INFO"
    if not pkg_info.exists():
        return None

    for line in pkg_info.read_text(encoding="utf-8").splitlines():
        if line.startswith("Version: "):
            return line.split("Version: ", 1)[1].strip()
    return None


def _version_from_generated_file() -> str | None:
    version_file = ROOT / "pyensmallen" / "_version.py"
    if not version_file.exists():
        return None

    match = re.search(
        r'^__version__\s*=\s*"([^"]+)"\s*$',
        version_file.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if match:
        return match.group(1)
    return None


def _version_from_scm() -> str | None:
    try:
        from setuptools_scm import get_version
    except ImportError:
        return None

    try:
        return get_version(root=str(ROOT), fallback_version="0.0.0")
    except Exception:
        return None


def main() -> int:
    for getter in (
        _version_from_env,
        _version_from_pkg_info,
        _version_from_generated_file,
        _version_from_scm,
    ):
        version = getter()
        if version:
            print(version)
            return 0

    print("0.0.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
