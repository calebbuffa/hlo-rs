"""Build the hardware-independent HLO oracle for the current extern/xla checkout."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
from urllib.parse import urlparse
from urllib.request import url2pathname

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = ROOT.parent.parent


def git(directory, *arguments):
    return subprocess.check_output(
        ["git", "--no-pager", "-C", str(directory), *arguments], text=True
    ).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bazel", default=shutil.which("bazel"),
                        help="path to the XLA checkout's Bazel version")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if not args.bazel:
        raise SystemExit("Bazel is required; supply --bazel")
    if args.jobs < 1:
        raise SystemExit("--jobs must be positive")
    reference = (WORKSPACE_ROOT / "extern" / "xla").resolve(strict=True)
    revision = git(reference, "rev-parse", "HEAD")
    if git(reference, "status", "--porcelain"):
        raise SystemExit("extern/xla has local changes; commit them before building the oracle")
    bazel = str(Path(args.bazel).resolve(strict=True))
    expected_version = (reference / ".bazelversion").read_text().strip()
    if subprocess.check_output([bazel, "--version"], text=True).strip() != f"bazel {expected_version}":
        raise SystemExit("Bazel version does not match extern/xla/.bazelversion")
    work = ROOT / "target" / "hlo-oracle"
    work.mkdir(parents=True, exist_ok=True)
    snapshot = work / f"reference-{revision}"
    if not snapshot.exists():
        subprocess.run(["git", "clone", "--shared", "--no-checkout", "--quiet",
                        str(reference), str(snapshot)], check=True)
        subprocess.run(["git", "-C", str(snapshot), "checkout", "--quiet", "--detach", revision],
                       check=True)
    if git(snapshot, "rev-parse", "HEAD") != revision or git(snapshot, "status", "--porcelain"):
        raise SystemExit("isolated source snapshot is modified or at a different revision")
    package = snapshot / "xla" / "xlahlo_oracle"
    package.mkdir(exist_ok=True)
    for filename in ["BUILD.bazel", "main.cc"]:
        shutil.copyfile(ROOT / "tools" / "hlo_oracle" / filename, package / filename)
    environment = os.environ.copy()
    platform_options = []
    if os.name == "nt":
        compiler = shutil.which("clang++")
        if not compiler:
            raise SystemExit("native Windows oracle requires Clang")
        environment["BAZEL_USE_LLVM_NATIVE"] = "1"
        environment["BAZEL_LLVM"] = str(Path(compiler).resolve().parent.parent)
        shell = shutil.which("bash")
        if shell:
            environment["BAZEL_SH"] = shell
        platform_options.extend([
            "--compiler=clang-cl", "--host_compiler=clang-cl",
            "--features=compiler_param_file", "--features=archive_param_file",
            "--dynamic_mode=off",
            "--extra_toolchains=@@rules_cc++cc_configure_extension+local_config_cc//:cc-toolchain-x64_windows-clang-cl",
            "--host_platform=//xla/xlahlo_oracle:windows-clang",
            "--platforms=//xla/xlahlo_oracle:windows-clang",
        ])
        for flag in ["/Zc:__cplusplus", "/Zc:preprocessor", "/permissive-",
                     "-D_USE_MATH_DEFINES", "-D_ENABLE_EXTENDED_ALIGNED_STORAGE",
                     "-DWIN32_LEAN_AND_MEAN", "-DNOGDI"]:
            platform_options.extend([f"--copt={flag}", f"--host_copt={flag}"])
    cpp_flag = "/std:c++20" if os.name == "nt" else "-std=c++20"
    subprocess.run(
        [bazel, "--batch", f"--output_user_root={ROOT / 'target' / 'bzl-workspace'}",
         "--nosystem_rc", "--nohome_rc", "--noworkspace_rc",
         "build", "//xla/xlahlo_oracle:hlo-oracle",
         "--enable_bzlmod", "--noenable_workspace",
         "--incompatible_disallow_empty_glob=false", "--experimental_repo_remote_exec",
         "--lockfile_mode=off", f"--repository_cache={ROOT / 'target' / 'bazel-cache'}",
         "--experimental_repository_downloader_retries=1",
         "--repo_env=USE_HERMETIC_CC_TOOLCHAIN=0",
         "--@rules_ml_toolchain//common:enable_hermetic_cc=False",
         "--incompatible_enable_cc_toolchain_resolution",
         "--repo_env=TF_NEED_CUDA=0", "--repo_env=TF_NEED_ROCM=0",
         "--repo_env=HERMETIC_PYTHON_VERSION=3.12",
         "--define=framework_shared_object=false",
         "--define=tsl_protobuf_header_only=false",
         f"--build_event_json_file={work / 'events.json'}",
         f"--cxxopt={cpp_flag}", f"--host_cxxopt={cpp_flag}",
         "-c", "opt", f"--jobs={args.jobs}", "--local_resources=memory=4096",
         "--experimental_convenience_symlinks=ignore", "--noshow_progress",
         *platform_options],
        cwd=snapshot, env=environment, check=True,
    )
    filename = "hlo-oracle.exe" if os.name == "nt" else "hlo-oracle"
    for line in (work / "events.json").read_text().splitlines():
        event = json.loads(line)
        for output in event.get("namedSetOfFiles", {}).get("files", []):
            if Path(output["name"]).name == filename:
                uri = urlparse(output["uri"])
                if uri.scheme != "file":
                    raise SystemExit("oracle output is not a local file")
                executable = Path(url2pathname(uri.path))
                destination = work / filename
                shutil.copyfile(executable, destination)
                print(destination)
                return
    raise SystemExit("build did not report an oracle executable")


if __name__ == "__main__":
    main()
