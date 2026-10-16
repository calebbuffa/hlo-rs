"""Build a native scalar-conversion oracle using dependencies from extern/xla."""

import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
from urllib.parse import urlparse
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = ROOT.parent.parent
MAX_DOWNLOAD = 16 * 1024 * 1024


def download(url, path, expected):
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "xlahlo-numeric-oracle"})
    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read(MAX_DOWNLOAD + 1)
    if len(content) > MAX_DOWNLOAD:
        raise SystemExit(f"download exceeds {MAX_DOWNLOAD} bytes: {url}")
    if hashlib.sha256(content).hexdigest() != expected:
        raise SystemExit(f"source hash mismatch: {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def assignment(source, name):
    match = re.search(rf"^\s*{re.escape(name)}\s*=\s*\"([^\"]+)\"", source, re.MULTILINE)
    if not match:
        raise SystemExit(f"missing {name} in extern/xla workspace definition")
    return match.group(1)


def dependency(source, commit_name, checksum_name):
    commit = assignment(source, commit_name)
    checksum = assignment(source, checksum_name)
    url_match = re.search(
        rf'urls\s*=\s*tf_mirror_urls\("([^"]+)"\.format\(commit\s*=\s*{re.escape(commit_name)}\)\)',
        source,
    )
    prefix_match = re.search(
        rf'strip_prefix\s*=\s*"([^"]+)"\.format\(commit\s*=\s*{re.escape(commit_name)}\)',
        source,
    )
    if not url_match or not prefix_match:
        raise SystemExit("unsupported dependency URL or strip_prefix in extern/xla")
    return {
        "commit": commit,
        "checksum": checksum,
        "url": url_match.group(1).format(commit=commit),
        "prefix": prefix_match.group(1).format(commit=commit),
    }


def extract(archive, prefix, keep, external_root):
    with tarfile.open(archive) as source:
        members = [
            item for item in source.getmembers()
            if item.name.startswith(f"{prefix}/{keep}")
            and (item.isfile() or item.isdir())
        ]
        if not members:
            raise SystemExit(f"archive has no expected source directory: {prefix}/{keep}")
        source.extractall(external_root, members=members, filter="data")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default=shutil.which("clang++"),
                        help="Clang/GCC-compatible C++ compiler path")
    args = parser.parse_args()
    if not args.cxx:
        raise SystemExit("no clang++ found; specify a compiler with --cxx")
    work = ROOT / "target" / "numeric-oracle"
    work.mkdir(parents=True, exist_ok=True)
    xla_root = WORKSPACE_ROOT / "extern" / "xla"
    external_root = WORKSPACE_ROOT / "extern"
    eigen = dependency(
        (xla_root / "third_party" / "eigen3" / "workspace.bzl").read_text(),
        "EIGEN_COMMIT",
        "EIGEN_SHA256",
    )
    eigen_archive = work / Path(urlparse(eigen["url"]).path).name
    download(eigen["url"], eigen_archive, eigen["checksum"])
    eigen_root = external_root / eigen["prefix"]
    eigen_header = eigen_root / "Eigen" / "Core"
    if not eigen_header.is_file():
        extract(eigen_archive, eigen["prefix"], "Eigen/", external_root)

    ml_dtypes = dependency(
        (xla_root / "third_party" / "py" / "ml_dtypes" / "workspace.bzl").read_text(),
        "ML_DTYPES_COMMIT",
        "ML_DTYPES_SHA256",
    )
    ml_archive = work / Path(urlparse(ml_dtypes["url"]).path).name
    download(ml_dtypes["url"], ml_archive, ml_dtypes["checksum"])
    ml_dtypes_root = external_root / ml_dtypes["prefix"]
    ml_dtypes_header = ml_dtypes_root / "ml_dtypes" / "include" / "float8.h"
    if not ml_dtypes_header.is_file():
        extract(ml_archive, ml_dtypes["prefix"], "ml_dtypes/include/", external_root)

    executable = work / ("numeric-oracle.exe" if os.name == "nt" else "numeric-oracle")
    subprocess.run(
        [
            args.cxx,
            "-std=c++20",
            "-O2",
            "-DNDEBUG",
            "-I",
            str(eigen_root),
            "-I",
            str(ml_dtypes_root),
            str(ROOT / "tools" / "numeric_oracle.cc"),
            "-o",
            str(executable),
        ],
        check=True,
    )
    print(executable)


if __name__ == "__main__":
    main()
