# Linux desktop and packages

Nexus Desktop is a native Qt server manager with an English interface. It selects a GGUF model and CPU/CUDA backend, starts or stops the local server, manages model loading, and shows status, metrics and bounded logs. It can also connect to an existing loopback server. It does not provide a chat screen or download model weights.

## Packages and supported systems

| Package | Target | Contents |
|---|---|---|
| `nexus-desktop-…-any.pkg.tar.zst` | Arch Linux rolling | Python GUI, launcher, desktop entry and icon |
| `nexus-engine-cuda-…-x86_64.pkg.tar.zst` | Arch Linux x86_64 | Native `nexus`, `nexus-server`, `nexus-convert`; CPU and optional CUDA execution |
| `nexus-engine-…-x86_64.pkg.tar.zst` | Arch Linux x86_64 | Alternative CPU-only native engine build |
| `nexus-desktop_…_all.deb` | Debian 13+ / Ubuntu 26.04+ | Architecture-independent GUI; **no native engine ELF** |

The GUI uses Python 3.11+ and PySide6 6.8+. Debian 13 provides the split [`python3-pyside6.qtwidgets`](https://packages.debian.org/trixie/python3-pyside6.qtwidgets) package; Ubuntu 26.04 provides it in [`universe`](https://packages.ubuntu.com/resolute/python3-pyside6.qtwidgets). Core, Gui and Network modules, base QPA platform plugins, [`qt6-svg-plugins`](https://packages.debian.org/trixie/qt6-svg-plugins), the SVG library and a basic font are also declared dependencies; Debian-family builds recommend Qt Wayland integration. Arch provides [`pyside6`](https://archlinux.org/packages/extra/x86_64/pyside6/) with `qt6-svg` and `qt6-wayland`. Debian 12 and Ubuntu 24.04 are outside this ready-made package matrix. Later releases still need to satisfy these dependencies; future platform behavior is not guaranteed.

A `.deb` is not an Arch package. Arch-built engine binaries are not bundled into the `.deb`: their OpenSSL SONAME and glibc/libstdc++ symbol requirements belong to the Arch build environment. Debian-family users should build the engine on their target distribution or connect to an existing compatible local server. The native package manifest records the actual ELF `NEEDED`, GLIBC, GLIBCXX, CXXABI and GCC requirements. CUDA additionally needs an NVIDIA GPU/driver capable of loading the checked-in `sm_80` / PTX 9.4 kernel; CUDA toolkit/NVRTC is not required. The hardware acceptance in this work used an RTX 5070. CPU operation does not require an NVIDIA driver.

## Verified runtime matrix (15 November 2026)

| Environment | Packaged GUI checks |
|---|---|
| Arch Linux, Python 3.14, PySide6 6.12 | Extraction, entrypoints, non-null SVG, offscreen window and graceful idle SIGTERM; packaged native engine lifecycle with a real cached model |
| Ubuntu 26.04, Python 3.14, PySide6 6.10.2 | Apt dependency installation in the official container, SVG/window rendering, hostile-CWD launcher check and graceful idle SIGTERM |
| Debian 13, Python 3.13.5, PySide6 6.8.2.1 | Same package installation/window/shutdown checks plus all 19 GUI unit/HTTP/process tests, process exit 0 |

The GUI `.deb` tested in both containers matched the final artifact SHA-256. No packages were installed on the development host. Two builds from the same frozen binaries/GUI sources and packaging environment produced bit-identical hashes for the Arch GUI, Arch engine and Debian GUI artifacts. Native engine `ldd -r`, every command's `--help`, archive metadata, payload hashes and extracted GUI lifecycle were checked. CUDA/model correctness acceptance remains documented in [validation](validation.md). Final package and engine hashes are also recorded in the [release package record](validation/2026-11-15-packages.json).

## Install and launch

Arch, with generated package filenames substituted:

```sh
sudo pacman -U ./nexus-engine-cuda-0.1.0-1-x86_64.pkg.tar.zst \
  ./nexus-desktop-0.1.0-1-any.pkg.tar.zst
nexus-desktop
```

Install either the CPU engine or CUDA-capable engine package; they provide the same commands and conflict with each other. The GUI can be installed alone to connect to an already running local server.

Debian 13 / Ubuntu 26.04 with the required repositories enabled:

```sh
sudo apt install ./nexus-desktop_0.1.0-1_all.deb
nexus-desktop
```

In the GUI, select the `nexus-server` executable if it is not on `PATH`, choose a compatible GGUF model, and start the server. CUDA is an explicit backend choice. Use a context size that fits available memory. A quantized model's disk size is not its runtime RAM/VRAM usage; Nexus currently materializes supported model weights as F32.

Optional launch arguments:

```sh
nexus-desktop --server /path/to/nexus-server --model /path/to/model.gguf --port 8080
nexus-desktop --connect http://127.0.0.1:8080
nexus-desktop --help
```

Settings are stored under the XDG configuration directory. Authentication secrets stay in memory. Closing a window shuts down only the server process it started; connecting to another local server does not transfer process ownership.

## Build packages without installing anything

Build and validate engine binaries first. Packaging performs no Cargo/native build, dependency installation or model download:

```sh
cargo build --release --locked --workspace --features cuda
python3 scripts/build-linux-packages.py --formats arch deb \
  --engine-dir target/release --engine-flavor cuda
```

Run on x86_64 Arch Linux for native engine packages. Packaging tools are `python`, `makepkg`/`pacman`, `fakeroot`, `zstd`, `readelf`, `ldd`, `desktop-file-validate` and `dpkg-deb`. GUI-only Debian packaging can be run on a Debian-family build host:

```sh
python3 scripts/build-linux-packages.py --formats deb --skip-engine
```

Use `--engine-flavor cpu` with a separately built CPU-only binary directory. `auto` detects the CUDA feature marker in `nexus-server` and records the result. This detects compiled capability, not whether a GPU is available.

Generated artifacts go to `dist/linux/`, which is excluded from source control:

- `arch/<package>/`: native archive, concrete `PKGBUILD`, checksum-pinned rootfs source archive, `.SRCINFO` and build log.
- `debian/`: architecture-independent GUI `.deb` and control inspection.
- `manifest.json` and `SHA256SUMS`: artifact and original ELF hashes, dependency/ABI evidence and flavor.

The recipe templates, package builder, launcher and control definitions live in [`packaging/`](../packaging) and [`scripts/build-linux-packages.py`](../scripts/build-linux-packages.py). A concrete generated Arch recipe can be rebuilt in its artifact directory with `makepkg`; verify the source checksum first. It repackages the pinned native ELF payload, rather than silently recompiling a different engine.

`SOURCE_DATE_EPOCH` is pinned in `packaging/source-date-epoch` unless overridden. Rootfs archive ownership, modes, file order and timestamps are normalized. Package builds use one compression thread. Makepkg records build paths and the installed package set in `.BUILDINFO`, so the builder uses a fixed, user-owned locked scratch directory (`/tmp/nexus-arch-build`; override with `NEXUS_ARCH_BUILD_ROOT`). Bitwise reproduction requires matching input binaries, GUI sources, version/epoch, build paths and packaging tool/environment versions. This does not promise reproducibility of the Rust/llama.cpp native compilation itself.

## Check extracted packages

No system installation is needed for validation. Extract into a temporary directory with `bsdtar -xf package.pkg.tar.zst -C <root>` or `dpkg-deb -x package.deb <root>`. The launcher supports an extraction-only location override:

```sh
NEXUS_DESKTOP_APP_DIR=<root>/usr/share/nexus-desktop \
  <root>/usr/bin/nexus-desktop --version
```

The installed default remains `/usr/share/nexus-desktop`. The launcher uses Python safe-path mode (`-P`) so an unrelated `nexus_desktop.py` in the current directory cannot replace the installed module. Run the regression and source archive checks with:

```sh
python3 -m unittest discover -s packaging -p 'test_*.py'
desktop-file-validate packaging/nexus-desktop.desktop
python3 packaging/smoke_packages.py --artifacts dist/linux --output /tmp/nexus-package-smoke
```

For graphical smoke tests in headless environments, use `QT_QPA_PLATFORM=offscreen`; a working desktop session is still needed for an interactive window. Engine smoke tests should include each packaged command's `--help`, server lifecycle, the target host's `ldd -r` resolution, and CUDA correctness tests on actual supported hardware. Package metadata validation alone does not establish inference correctness or GPU support.

For a disposable Debian-family installation test, with Docker available:

```sh
mkdir -p /tmp/nexus-deb-smoke-evidence
docker run --rm --memory=1g --cpus=2 -e NEXUS_SMOKE_LABEL=DEBIAN13 \
  -v "$PWD/dist/linux:/packages:ro" \
  -v "$PWD/packaging:/checks:ro" -v "$PWD:/source:ro" \
  -v /tmp/nexus-deb-smoke-evidence:/evidence \
  debian:13 /bin/sh /checks/container-smoke.sh
```

Use `ubuntu:26.04` and `NEXUS_SMOKE_LABEL=UBUNTU26` for the other verified distribution. This installs dependencies only inside the disposable container. With `/source` mounted, the script adds the test-only QtTest binding and runs the GUI suite against the **installed** module. It checks the same-name hostile-CWD launcher separately, then returns to a neutral working directory for dev-mode child Python tests. It does not mount a model or launch GPU inference.
