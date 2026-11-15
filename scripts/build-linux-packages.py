#!/usr/bin/env python3
"""Build native Arch engine/GUI packages and an architecture-independent GUI .deb.

No host install, sudo, Cargo build, dependency download or distro ELF conversion.
Concrete PKGBUILD + checksum-pinned payloads are emitted beside every Arch package.
"""
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
BINARIES = ("nexus", "nexus-server", "nexus-convert")


def command(args, **kwargs):
    # ELF/package output is parsed as an interface. Never inherit translations,
    # including when a caller supplied its own environment for makepkg/dpkg-deb.
    environment = dict(kwargs.pop("env", os.environ))
    environment.update({"LC_ALL": "C", "LANG": "C", "LANGUAGE": "C"})
    return subprocess.run([str(x) for x in args], check=True, text=True, env=environment, **kwargs)


def output(args):
    return command(args, capture_output=True).stdout.strip()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render(template: Path, values: dict[str, str]):
    text = template.read_text()
    for key, value in values.items():
        text = text.replace(f"@{key}@", str(value))
    if re.search(r"@[A-Z_]+@", text):
        raise ValueError(f"Unexpanded placeholder in {template}")
    return text


def install_file(source: Path, rootfs: Path, relative: str, executable=False):
    target = rootfs / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    target.chmod(0o755 if executable else 0o644)
    return target


def copy_gui(rootfs: Path):
    module = ROOT / "desktop/nexus_desktop"
    if not (module / "__main__.py").is_file():
        raise ValueError("desktop/nexus_desktop/__main__.py is required before packaging the GUI")
    for source in sorted(module.rglob("*")):
        if not source.is_file() or "__pycache__" in source.parts or source.suffix in (".pyc", ".pyo"):
            continue
        if source.is_symlink():
            raise ValueError(f"GUI source symlinks are not supported: {source}")
        install_file(source, rootfs, f"usr/share/nexus-desktop/nexus_desktop/{source.relative_to(module)}")
    install_file(ROOT / "packaging/launcher.sh", rootfs, "usr/bin/nexus-desktop", True)
    install_file(ROOT / "packaging/nexus-desktop.desktop", rootfs, "usr/share/applications/nexus-desktop.desktop")
    install_file(module / "assets/nexus.svg", rootfs, "usr/share/icons/hicolor/scalable/apps/nexus-desktop.svg")
    install_file(ROOT / "LICENSE", rootfs, "usr/share/licenses/nexus-desktop/LICENSE")
    install_file(ROOT / "LICENSE", rootfs, "usr/share/doc/nexus-desktop/copyright")
    command(["desktop-file-validate", rootfs / "usr/share/applications/nexus-desktop.desktop"])


def normalize(rootfs: Path, epoch: int):
    for path in sorted(rootfs.rglob("*"), reverse=True):
        if path.is_dir():
            path.chmod(0o755)
        os.utime(path, (epoch, epoch), follow_symlinks=False)
    rootfs.chmod(0o755)
    os.utime(rootfs, (epoch, epoch))


def archive(rootfs: Path, target: Path, epoch: int):
    """Deterministic payload; no absolute source paths or current UID/time leak."""
    with target.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=epoch, compresslevel=9) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.GNU_FORMAT) as tar:
                for path in [rootfs, *sorted(rootfs.rglob("*"))]:
                    info = tar.gettarinfo(str(path), str(path.relative_to(rootfs.parent)))
                    info.uid = info.gid = 0
                    info.uname = info.gname = "root"
                    info.mtime = epoch
                    if path.is_file():
                        with path.open("rb") as source:
                            tar.addfile(info, source)
                    else:
                        tar.addfile(info)


@contextmanager
def arch_build_directory(name):
    # makepkg records startdir/builddir in .BUILDINFO. Fixed paths + flock make
    # identical inputs reproducible without concurrent builds sharing scratch.
    base = Path(os.environ.get("NEXUS_ARCH_BUILD_ROOT", "/tmp/nexus-arch-build"))
    if base.is_symlink():
        raise ValueError("Arch build root cannot be a symlink")
    base.mkdir(mode=0o700, parents=True, exist_ok=True)
    if base.stat().st_uid != os.getuid():
        raise ValueError("Arch build root must be owned by the current user")
    base.chmod(0o700)
    with (base / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        build = base / name
        if build.exists():
            if not (build / ".nexus-scratch").is_file():
                raise ValueError(f"Refusing to remove unrelated build directory {build}")
            shutil.rmtree(build)
        build.mkdir()
        (build / ".nexus-scratch").touch()
        try:
            yield build
        finally:
            shutil.rmtree(build)


def arch_package(rootfs, name, template, values, destination, epoch, work):
    package_dir = destination / "arch" / name
    package_dir.mkdir(parents=True, exist_ok=True)
    payload = package_dir / ("nexus-engine-rootfs.tar.gz" if name.startswith("nexus-engine") else "nexus-desktop-rootfs.tar.gz")
    normalize(rootfs, epoch)
    archive(rootfs, payload, epoch)
    values = {**values, "PAYLOAD_SHA256": sha256(payload)}
    pkgbuild = render(ROOT / template, values)
    (package_dir / "PKGBUILD").write_text(pkgbuild)
    environment = {**os.environ, "SOURCE_DATE_EPOCH": str(epoch), "PKGDEST": str(package_dir), "PACKAGER": "Nexus Contributors <noreply@users.noreply.github.com>"}
    # Fixed scratch paths keep .BUILDINFO reproducible; ownership and locking
    # prevent accidental deletion or mutation of another build/user's directory.
    with arch_build_directory(name) as build:
        environment["BUILDDIR"] = str(build)
        shutil.copyfile(package_dir / "PKGBUILD", build / "PKGBUILD")
        shutil.copyfile(payload, build / payload.name)
        config = build / "makepkg.conf"
        config.write_text("source /etc/makepkg.conf\nCOMPRESSZST=(zstd -c -T1 -10 -)\n")
        with (package_dir / ".SRCINFO").open("w") as out:
            command(["makepkg", "--config", config, "--printsrcinfo"], cwd=build, env=environment, stdout=out)
        with (package_dir / "build.log").open("w") as log:
            command(["makepkg", "--config", config, "--nodeps", "--nocheck", "--noconfirm", "--nosign", "--skippgpcheck", "--force"], cwd=build, env=environment, stdout=log, stderr=subprocess.STDOUT)
    files = sorted(package_dir.glob(f"{name}-{values['VERSION']}-{values['RELEASE']}-*.pkg.tar.zst"))
    if len(files) != 1:
        raise ValueError(f"Expected one Arch package for {name}, got {files}")
    # pacman validates native archive metadata without installation.
    metadata = output(["pacman", "-Qip", files[0]])
    (package_dir / "package-info.txt").write_text(metadata + "\n")
    return files[0]


def deb_package(rootfs, values, destination, epoch):
    normalize(rootfs, epoch)
    control = rootfs / "DEBIAN"
    control.mkdir()
    (control / "control").write_text(render(ROOT / "packaging/debian/control.in", values))
    installed_kib = sum(p.stat().st_size for p in rootfs.rglob("*") if p.is_file() and "DEBIAN" not in p.parts) // 1024 + 1
    with (control / "control").open("a") as f:
        f.write(f"Installed-Size: {installed_kib}\n")
    md5s = []
    for path in sorted(rootfs.rglob("*")):
        if path.is_file() and "DEBIAN" not in path.parts:
            md5s.append(f"{hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()}  {path.relative_to(rootfs)}")
    (control / "md5sums").write_text("\n".join(md5s) + "\n")
    normalize(rootfs, epoch)
    folder = destination / "debian"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"nexus-desktop_{values['VERSION']}-{values['RELEASE']}_all.deb"
    command(["dpkg-deb", "--root-owner-group", "--threads-max=1", "-Zxz", "--build", rootfs, path], env={**os.environ, "SOURCE_DATE_EPOCH": str(epoch)})
    (folder / "package-info.txt").write_text(output(["dpkg-deb", "--info", path]) + "\n")
    return path


def numeric_version(name):
    return tuple(int(n) for n in name.split("_", 1)[-1].split("."))


def inspect_elf(path):
    if not path.read_bytes().startswith(b"\x7fELF"):
        raise ValueError(f"Native engine is not ELF: {path}")
    header = output(["readelf", "-h", path])
    if "ELF64" not in header or "Advanced Micro Devices X86-64" not in header:
        raise ValueError("Only native x86_64 Arch engine packaging is supported")
    dynamic = output(["readelf", "-d", path])
    versions = output(["readelf", "--version-info", path])
    names = sorted(set(re.findall(r"Name: ((?:GLIBCXX|GLIBC|CXXABI|GCC)_[0-9.]+)", versions)))
    required = {}
    for family in ("GLIBC", "GLIBCXX", "CXXABI", "GCC"):
        matches = [n for n in names if n.startswith(family + "_")]
        required[family] = max(matches, key=numeric_version) if matches else None
    linked = output(["ldd", "-r", path])
    if "not found" in linked or "undefined symbol" in linked:
        raise ValueError(f"Unresolved ELF dependencies for {path.name}: {linked}")
    return {"sha256": sha256(path), "needed": re.findall(r"Shared library: \[(.*?)\]", dynamic), "required_versions": required, "ldd": re.sub(r"\s+\(0x[0-9a-fA-F]+\)", "", linked)}


def third_party_licenses(rootfs):
    """Preserve available registry and bundled llama.cpp notices without downloading."""
    registry = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo")) / "registry/src"
    entries = []
    lock = tomllib.loads((ROOT / "Cargo.lock").read_text())
    for package in lock.get("package", []):
        if not package.get("source", "").startswith("registry+"):
            continue
        crate_id = f"{package['name']}-{package['version']}"
        candidates = sorted(registry.glob(f"*/{crate_id}"))
        if not candidates:
            continue  # Target-specific unbuilt packages need not be fetched to repackage.
        crate = candidates[0]
        manifest = tomllib.loads((crate / "Cargo.toml").read_text()).get("package", {})
        notices = [p for p in crate.iterdir() if p.is_file() and p.name.upper().startswith(("LICENSE", "COPYING", "COPYRIGHT", "NOTICE"))]
        if package["name"] == "llama-cpp-sys-2":
            notices += [p for p in [crate / "llama.cpp/LICENSE", crate / "llama.cpp/ggml/LICENSE"] if p.is_file()]
        records = []
        for source in sorted(set(notices)):
            relative = source.relative_to(crate)
            target = f"usr/share/licenses/nexus-engine/third-party/{crate_id}/{relative}"
            install_file(source, rootfs, target)
            records.append({"file": str(relative), "sha256": sha256(source)})
        entries.append({"crate": crate_id, "license": manifest.get("license"), "notices": records})
    target = rootfs / "usr/share/licenses/nexus-engine/third-party-inventory.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n")


def engine_rootfs(rootfs, engine_dir, flavor):
    os_release = Path("/etc/os-release").read_text()
    if not re.search(r"^ID=\"?arch\"?$", os_release, re.M) or platform.machine() != "x86_64":
        raise ValueError("Native engine .pkg.tar.zst must be assembled on x86_64 Arch Linux")
    elf = {}
    for name in BINARIES:
        path = install_file(engine_dir / name, rootfs, "usr/bin/" + name, True)
        elf[name] = inspect_elf(path)
        output([path, "--help"])
    compiled_cuda = b"CUDA driver library is unavailable" in (rootfs / "usr/bin/nexus-server").read_bytes()
    actual_flavor = "cuda" if compiled_cuda else "cpu"
    if flavor != "auto" and flavor != actual_flavor:
        raise ValueError(f"Requested {flavor} engine but binary CUDA feature marker indicates {actual_flavor}")
    needed = set(n for entry in elf.values() for n in entry["needed"])
    dependencies = {"libgcc", "libstdc++"}
    glibc = max((e["required_versions"]["GLIBC"] for e in elf.values() if e["required_versions"]["GLIBC"]), key=numeric_version)
    dependencies.add("glibc>=" + glibc.split("_", 1)[1])
    for soname in needed:
        if soname.startswith(("libssl.so.", "libcrypto.so.")):
            base, abi = soname.split(".so.", 1)
            dependencies.add(f"{base}.so={abi}-64")
        elif soname in ("libstdc++.so.6", "libgcc_s.so.1", "libc.so.6", "libm.so.6", "libpthread.so.0", "libdl.so.2", "librt.so.1", "ld-linux-x86-64.so.2"):
            pass
        else:
            # Fail closed on a new native dependency instead of shipping incomplete metadata.
            raise ValueError(f"Unmapped native dependency {soname}; update the Arch recipe")
    # Native Arch C++ library version pin is intentionally conservative. The manifest
    # records the actual GLIBCXX/CXXABI requirement for downstream rebuild decisions.
    cxx = output(["pacman", "-Q", "libstdc++"]).split(maxsplit=1)[1]
    dependencies.discard("libstdc++")
    dependencies.add("libstdc++>=" + cxx)
    install_file(ROOT / "LICENSE", rootfs, "usr/share/licenses/nexus-engine/LICENSE")
    third_party_licenses(rootfs)
    return actual_flavor, sorted(dependencies), elf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formats", nargs="+", choices=("arch", "deb"), default=["arch", "deb"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist/linux")
    parser.add_argument("--engine-dir", type=Path, default=ROOT / "target/release")
    parser.add_argument("--engine-flavor", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-engine", action="store_true")
    parser.add_argument("--skip-gui", action="store_true")
    parser.add_argument("--version", default=tomllib.loads((ROOT / "Cargo.toml").read_text())["package"]["version"])
    parser.add_argument("--release", default="1")
    parser.add_argument("--source-date-epoch", type=int, default=int(os.environ.get("SOURCE_DATE_EPOCH", (ROOT / "packaging/source-date-epoch").read_text().strip())))
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)*(?:\+[a-z0-9.]+)?", args.version) or not args.release.isdigit():
        parser.error("Use a numeric version and release")
    if args.source_date_epoch < 0:
        parser.error("source-date-epoch must be nonnegative")
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    # A failed/partial rebuild must not leave an old success manifest consumable.
    (destination / "manifest.json").unlink(missing_ok=True)
    (destination / "SHA256SUMS").unlink(missing_ok=True)
    values = {"VERSION": args.version, "RELEASE": args.release, "EPOCH": str(args.source_date_epoch)}
    manifest = {"version": args.version, "release": args.release, "source_date_epoch": args.source_date_epoch, "artifacts": [], "engine": None, "gui": None}
    with tempfile.TemporaryDirectory(prefix="nexus-linux-packages-") as temporary:
        work = Path(temporary)
        if not args.skip_gui:
            gui = work / "gui/rootfs"
            gui.mkdir(parents=True)
            copy_gui(gui)
            manifest["gui"] = {"files": {str(p.relative_to(gui)): sha256(p) for p in sorted(gui.rglob("*")) if p.is_file()}, "runtime": "system Python >=3.11 and PySide6 >=6.8"}
            if "arch" in args.formats:
                artifact = arch_package(gui, "nexus-desktop", "packaging/arch/nexus-desktop.PKGBUILD.in", values, destination, args.source_date_epoch, work)
                manifest["artifacts"].append({"kind": "arch-gui", "path": str(artifact.relative_to(destination)), "sha256": sha256(artifact)})
            if "deb" in args.formats:
                artifact = deb_package(gui, values, destination, args.source_date_epoch)
                manifest["artifacts"].append({"kind": "deb-gui", "path": str(artifact.relative_to(destination)), "sha256": sha256(artifact)})
        if not args.skip_engine and "arch" in args.formats:
            engine = work / "engine/rootfs"
            engine.mkdir(parents=True)
            flavor, dependencies, elf = engine_rootfs(engine, args.engine_dir.resolve(), args.engine_flavor)
            name = "nexus-engine-cuda" if flavor == "cuda" else "nexus-engine"
            variant = "provides=('nexus-engine=@VERSION@')\nconflicts=('nexus-engine')\noptdepends=('nvidia-utils: optional CUDA driver; CPU remains available')" if flavor == "cuda" else "optdepends=()"
            variant = variant.replace("@VERSION@", args.version)
            variables = {**values, "ENGINE_PACKAGE": name, "ENGINE_FLAVOR": flavor, "ENGINE_DEPENDENCIES": " ".join(shlex.quote(dep) for dep in dependencies), "ENGINE_VARIANT_METADATA": variant}
            artifact = arch_package(engine, name, "packaging/arch/nexus-engine.PKGBUILD.in", variables, destination, args.source_date_epoch, work)
            manifest["artifacts"].append({"kind": "arch-engine", "path": str(artifact.relative_to(destination)), "sha256": sha256(artifact)})
            manifest["engine"] = {"flavor": flavor, "dependencies": dependencies, "elf": elf, "platform": "Arch Linux x86_64 native; not a Debian engine build"}
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (destination / "SHA256SUMS").write_text("".join(f"{a['sha256']}  {a['path']}\n" for a in manifest["artifacts"]))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
