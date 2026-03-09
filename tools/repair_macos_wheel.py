#!/usr/bin/env python3
"""Repair macOS wheels so vendored BLAS dependencies resolve in-wheel.

This script is intended to run under cibuildwheel as the macOS repair command.
It performs the standard delocate pass, then patches the repaired wheel:

- rewrite any ``@rpath/libblas.3.dylib`` reference to the vendored OpenBLAS
- add a ``libblas.3.dylib`` symlink next to the vendored OpenBLAS as a fallback
- remove stale build-time rpaths that point into the GitHub Actions micromamba env
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


STALE_RPATH_PREFIXES = (
    "/Users/runner/micromamba/envs/pyensmallen",
    "/Users/runner/micromamba/envs/pyensmallen/lib",
)
OLD_BLAS_REFERENCE = "@rpath/libblas.3.dylib"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def capture(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True)


def unpack_wheel(wheel: Path, workdir: Path) -> Path:
    run([sys.executable, "-m", "wheel", "unpack", "--dest", str(workdir), str(wheel)])
    unpacked = [path for path in workdir.iterdir() if path.is_dir()]
    if len(unpacked) != 1:
        raise RuntimeError(f"Expected one unpacked wheel directory, found {len(unpacked)}")
    return unpacked[0]


def repack_wheel(unpacked_dir: Path, dest_dir: Path, original_wheel: Path) -> Path:
    original_wheel.unlink()
    run([sys.executable, "-m", "wheel", "pack", "--dest-dir", str(dest_dir), str(unpacked_dir)])
    rebuilt = dest_dir / original_wheel.name
    if not rebuilt.exists():
        raise RuntimeError(f"Expected repaired wheel at {rebuilt}")
    return rebuilt


def mach_o_dependencies(binary: Path) -> list[str]:
    lines = capture(["otool", "-L", str(binary)]).splitlines()[1:]
    deps: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        deps.append(stripped.split(" ", 1)[0])
    return deps


def mach_o_rpaths(binary: Path) -> list[str]:
    lines = capture(["otool", "-l", str(binary)]).splitlines()
    rpaths: list[str] = []
    for idx, line in enumerate(lines):
        if line.strip() != "cmd LC_RPATH":
            continue
        for follow in lines[idx + 1 : idx + 6]:
            stripped = follow.strip()
            if stripped.startswith("path "):
                rpaths.append(stripped.split(" ", 2)[1])
                break
    return rpaths


def relative_loader_reference(binary: Path, target: Path) -> str:
    relative = os.path.relpath(target, start=binary.parent)
    return f"@loader_path/{Path(relative).as_posix()}"


def patch_binary(binary: Path, vendored_openblas: Path) -> None:
    modified = False
    deps = mach_o_dependencies(binary)
    if OLD_BLAS_REFERENCE in deps:
        replacement = relative_loader_reference(binary, vendored_openblas)
        run(
            [
                "install_name_tool",
                "-change",
                OLD_BLAS_REFERENCE,
                replacement,
                str(binary),
            ]
        )
        modified = True

    for rpath in mach_o_rpaths(binary):
        if any(rpath.startswith(prefix) for prefix in STALE_RPATH_PREFIXES):
            run(["install_name_tool", "-delete_rpath", rpath, str(binary)])
            modified = True

    if modified:
        run(["codesign", "--force", "--sign", "-", str(binary)])


def ensure_blas_symlink(dylib_dir: Path, vendored_openblas: Path) -> None:
    alias = dylib_dir / "libblas.3.dylib"
    target_name = vendored_openblas.name
    if alias.exists() or alias.is_symlink():
        alias.unlink()
    alias.symlink_to(target_name)


def find_single(pattern: str, root: Path) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one match for {pattern}, found {len(matches)}")
    return matches[0]


def repair_wheel(input_wheel: Path, dest_dir: Path, delocate_archs: str) -> Path:
    run(
        [
            sys.executable,
            "-m",
            "delocate.cmd.delocate_wheel",
            "--require-archs",
            delocate_archs,
            "-w",
            str(dest_dir),
            "-v",
            str(input_wheel),
        ]
    )

    repaired_wheel = dest_dir / input_wheel.name
    if not repaired_wheel.exists():
        raise RuntimeError(f"Expected delocate output at {repaired_wheel}")

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        unpacked_dir = unpack_wheel(repaired_wheel, workdir)
        package_dir = unpacked_dir / "pyensmallen"
        dylib_dir = package_dir / ".dylibs"

        vendored_openblas = find_single("libopenblas*.dylib", dylib_dir)
        ensure_blas_symlink(dylib_dir, vendored_openblas)

        binaries = list(package_dir.glob("_pyensmallen*.so")) + [
            path for path in dylib_dir.glob("*.dylib") if path.name != "libblas.3.dylib"
        ]
        for binary in binaries:
            patch_binary(binary, vendored_openblas)

        repaired_wheel = repack_wheel(unpacked_dir, dest_dir, repaired_wheel)

    return repaired_wheel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, help="Path to the built wheel")
    parser.add_argument("dest_dir", type=Path, help="Directory for repaired wheels")
    parser.add_argument(
        "delocate_archs",
        help="Architecture list passed through from cibuildwheel's {delocate_archs}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repair_wheel(args.wheel.resolve(), args.dest_dir.resolve(), args.delocate_archs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
