# Nexus Desktop

An English-language Qt 6 desktop interface that starts `nexus-server` or connects to an existing local server. Requires Python **3.11+** and **PySide6 >=6.8,<7**. The interface provides model and executable selection, CPU/CUDA and KV format controls, context and port settings, model lifecycle actions, health and metrics, and the latest 600 log lines.

## Development

```bash
PYTHONPATH=desktop python3 -m nexus_desktop
PYTHONPATH=desktop python3 -m nexus_desktop --help
PYTHONPATH=desktop python3 -m nexus_desktop --version
```

`--help` and `--version` require neither a display nor Qt. `--server /path/to/nexus-server`, `--model /path/to/model.gguf` and `--port 8080` fill the initial fields without starting a process. `--connect http://127.0.0.1:8080` connects to an existing server and uses `NEXUS_API_KEY` from the environment when provided.

## Linux installation

On Arch Linux, install Qt dependencies with `sudo pacman -S pyside6 qt6-svg qt6-wayland` and use the Nexus Desktop Arch package. The `.deb` package targets Debian/Ubuntu; use the supplied Arch package on Arch. Packaging recipes are in [`packaging/`](../packaging) at the repository root. Running from source uses the same Python module. The official Arch dependency is [`pyside6`](https://archlinux.org/packages/extra/x86_64/pyside6/).

Package layout: the Python module is installed at `/usr/share/nexus-desktop/nexus_desktop`, with the launcher at `/usr/bin/nexus-desktop`. Server discovery checks `/usr/lib/nexus-desktop/nexus-server`, `/usr/share/nexus-desktop/bin/nexus-server`, `/usr/lib/nexus/nexus-server` and `PATH`. A development checkout also supports `target/release/nexus-server`. If discovery fails, select the executable using the file picker. The GUI does not download server dependencies.

CPU execution and F32 KV storage are the defaults. CUDA requires a CUDA-enabled server build and a compatible NVIDIA driver; initialization errors appear in the log. Source quantization is detected from the GGUF file; runtime weights use F32 storage. Choose a context size within the model's supported limit. Concurrent requests use separate KV caches.

## Connections and process ownership

Management connections are restricted to loopback HTTP addresses (`127.0.0.1`, `localhost`, `::1`). A server started by the application binds to `127.0.0.1`. The application stops only its own `QProcess`; **Disconnect** sends no signal to an external server. **Start local server** is disabled while connected to an external server; disconnect before starting a separate local process. Closing the window first requests termination of an owned server, then kills it if shutdown is delayed, without blocking the UI thread.

Each server started by the application receives a random API key through its process environment, never through command-line arguments. External connection keys remain in the masked field for the current session. With `--connect`, the key is read from `NEXUS_API_KEY` and removed from the application's environment. No API keys are saved in the settings file, and known keys are redacted from logs. To connect another client to a server started by the application, use **Copy API key**; the key changes whenever that server is restarted. XDG settings are written atomically with mode 0600 to `$XDG_CONFIG_HOME/nexus-desktop/settings.json` (default: `~/.config/nexus-desktop/settings.json`).

## Tests

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=desktop python3 -m unittest discover -s desktop/tests -v
```

Offline tests use mock HTTP servers and child processes. The real-server smoke test is separately enabled and downloads no models:

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=desktop python3 desktop/tests/real_server_smoke.py \
  --server target/release/nexus-server --model /path/to/model-f32.gguf \
  --screenshot /tmp/nexus-desktop.png
```

**Unload** removes the model from memory without stopping the server process. It also clears the server's remembered model path; use **Load** to open the selected file again. **Reload** reads the currently loaded model from the same file.
