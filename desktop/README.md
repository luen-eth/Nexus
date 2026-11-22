# Nexus Desktop

An English-language Qt 6 desktop interface for chatting with local models, finding and downloading GGUF files, and managing `nexus-server`. It can start a local server process or connect to an existing server, including one running in Docker. Requires Python **3.11+** and **PySide6 >=6.8,<7**.

The sidebar has **Chat**, **Models**, and **Server** pages; `Ctrl+1`, `Ctrl+2`, and `Ctrl+3` switch between them. Server controls include executable selection, CPU/CUDA and KV formats, context and port settings, model lifecycle actions, health and metrics, and the latest 600 log lines.

## Chat

Connect to a server and load a model, then choose the model reported by that server in **Chat**. This selector shows loaded models, not every file in the local library. Replies stream into the conversation while server polling remains responsive. **Enter** sends a message; **Shift+Enter** adds a line. **Max output** defaults to 128 tokens and is clamped to the model's context limit. The server checks whether the complete conversation and requested output fit; use **New chat** or reduce the output limit if the context is full.

Every assistant reply has a compact statistics line: server-reported completion tokens, elapsed seconds, and tokens/s. Elapsed time covers the entire request, including network, queueing, prompt processing, and generation; tokens/s divides the reported completion count by that time. It is not a decoder-only benchmark. Interrupted replies and servers that omit usage show **Tokens unavailable**; the application never estimates counts from words, characters, or stream events.

**Stop** cancels only the chat request and retains its partial text for inspection. Partial or failed replies are excluded from subsequent conversation history. **New chat** clears the conversation. History also resets when the connection, authentication, selected model, or model installation epoch changes, and when a load/reload/unload operation begins. Chat history is bounded and kept in memory; it is not written to settings or uploaded to Hugging Face. Messages are displayed as plain text, including model-generated links or HTML.

## Models and downloads

Use **Models** to search public GGUF repositories on Hugging Face, select a repository and file, then choose **Download selected file**. Results show file size and available architecture/integrity metadata. Gated and private repositories are not supported, and no Hugging Face token is requested. Search and download use separate background workers; they do not receive the local server's API key.

Search results, file variants and downloaded models show the publisher's public Hugging Face avatar when one is available. A bundled Hugging Face logo is used while loading, offline without a cached avatar, or when the publisher is unknown. Avatars are cached under `$XDG_CACHE_HOME/nexus-desktop/logos` (default `~/.cache/nexus-desktop/logos`).

Downloads are pinned to the repository commit returned by Hugging Face. Before a file appears in the local library, the downloader checks its expected byte length, SHA-256 when provided by Git LFS metadata, and bounded GGUF v2/v3 metadata. If the publisher does not provide a SHA-256, the UI explicitly reports that only size and GGUF structure can be checked. Downloads use temporary files and publish the result atomically; existing files are not overwritten. **Cancel download**, failures, and application shutdown discard incomplete downloads. Cancelled downloads restart from the beginning when retried.

Select a file under **Your local models** and choose **Load selected model**. With a connected server, this loads the file using the current Server settings. Without a connection, the application fills the model path and opens **Server** so you can start or connect explicitly. **Refresh** rescans the library; it does not download anything.

**Delete selected model** shows the file name, full path and size before asking to move it to the system Trash. Cancel is the default. Unload a model before deleting its file; deletion is also unavailable during model operations. Only regular GGUF files inside the library are eligible. Successful deletion refreshes the library and clears a matching saved model path. If Trash is unavailable, the file is left in place. Space is reclaimed when you empty Trash using your file manager.

The library directory is the saved `models_dir` setting, otherwise `NEXUS_MODELS_DIR`, otherwise `$XDG_DATA_HOME/nexus/models` (default: `~/.local/share/nexus/models`). Downloaded files are organized as `owner/repository/commit/path/to/model.gguf`. The selected directory is saved in the XDG settings file. For Docker, use a library directory mounted at the **same absolute path** inside the container, with permissions readable by the server's configured UID/GID. A read-only container mount can still see files downloaded by the desktop application on the host. See the [Docker guide](../docs/docker.md) for server setup.

A GGUF search result or successful download does not guarantee that Nexus can run the model. Current support requires a compatible LLaMA or Mistral architecture, full-head unscaled RoPE, tokenizer, and tensor formats; the server performs final validation when loading. Split GGUF models are rejected. Runtime weights expand to F32, so compressed download size is not a RAM or VRAM requirement. Check the [validation notes](../docs/validation.md) before choosing larger models.

Known unsupported architectures reported by Hugging Face, such as `gemma4`, cannot be downloaded through the model browser. Missing architecture metadata is shown as unknown, not compatible. Local files are inspected before a Models-page load request; an unsupported architecture is rejected with guidance. These checks do not add inference support for that architecture or guarantee all LLaMA variants will run.

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
