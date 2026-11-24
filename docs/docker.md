# Docker with NVIDIA GPU

Run the server in a Linux container and keep Nexus Desktop on the host. The image contains the server and its runtime libraries; mount your GGUF models at runtime. It does not contain model weights or a desktop session.

Use the native Linux Docker Engine. On a machine with Docker Desktop installed, `docker --context default` selects the native daemon without changing your global context. The commands below use that context explicitly.

## Build

From the repository root, build from source with Rust 1.88 and the locked dependencies:

```bash
docker --context default build --target runtime \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  -t nexus-server:0.2.0-cuda .
```

The base images are pinned by digest. Debian package installation follows the distribution repositories, so this is not a claim of bit-for-bit reproducible native compilation. No CUDA toolkit or GPU is needed to compile the checked-in PTX path.

To package an already tested CUDA-enabled Linux x86_64 server instead, place it at `target/release/nexus-server` and supply its expected SHA-256:

```bash
NEXUS_SERVER_SHA256="$(sha256sum target/release/nexus-server | cut -d ' ' -f 1)"
docker --context default build --target prebuilt \
  --build-arg NEXUS_SERVER_SHA256="$NEXUS_SERVER_SHA256" \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  -t nexus-server:0.2.0-cuda .
```

The build checks the binary hash and executes `--help` inside Debian 13 to detect missing libraries or incompatible libc/libstdc++ versions. The hash check verifies the selected file, not its origin; use a binary you have built and validated. The runtime executes as UID/GID 10001 by default. Compose overrides that with your host IDs for model-file access. Tini forwards Docker's stop signal to the server, which stops accepting new connections and drains in-flight requests within a bounded 25-second window measured from the signal; compose's `stop_grace_period: 30s` leaves headroom before SIGKILL. The process exits 0 after a completed (or window-elapsed) drain.

## GPU and model access

The host needs a compatible NVIDIA driver and NVIDIA Container Toolkit. The kernel currently requires SM 8.0 or newer and a driver that supports PTX 9.4. Runtime weights expand to F32; plan RAM and VRAM from the loaded weights, not the GGUF file size. See [CUDA behavior and validation](validation.md).

The Compose file uses native CDI device access (`nvidia.com/gpu=all`). Confirm that your installed Docker Engine supports CDI and that `nvidia-ctk cdi list` reports the NVIDIA devices. Host setup is described in the [NVIDIA CDI documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html). Do not combine CDI device injection with a second legacy NVIDIA injection mechanism in the same container.

Set the model directory to its real absolute path and use the same path inside the container. That lets the host GUI submit a model path that the server can also read:

```bash
export NEXUS_MODELS_DIR="$(realpath /path/to/your/models)"
export NEXUS_MODEL_FILE=model.gguf
export NEXUS_UID="$(id -u)" NEXUS_GID="$(id -g)"
export NEXUS_API_KEY="$(openssl rand -hex 32)"
docker --context default compose up -d
docker --context default compose ps
curl --fail http://127.0.0.1:8080/health
```

Keep the same API key when recreating the container and connecting the GUI. The health endpoint is public; management, metrics and inference requests need the key. Do not commit keys or machine-specific paths to the repository. The Compose readiness check requires a loaded model; unloading it intentionally changes readiness to unhealthy until another model is loaded.

The server listens on `0.0.0.0:8080` **inside** the container; Docker publishes it only on host `127.0.0.1`. The model directory is read-only, the container root filesystem is read-only, and no Docker socket or privileged mode is needed. `NEXUS_PORT`, `NEXUS_CTX_SIZE`, `NEXUS_KV_QUANT` and `NEXUS_BATCH_SIZE` can override the Compose defaults. Allow enough memory for context caches and temporary old/new weights during model replacement.

If your Docker installation uses the registered legacy NVIDIA runtime instead of CDI, run the image directly using `--gpus all`, as described in [NVIDIA's Docker configuration](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html). Preserve the loopback publication, read-only model bind and explicit `--backend cuda`; `auto` selects CPU.

## Connect the desktop

Install [Nexus Desktop](linux-desktop.md) on the host, then run:

```bash
nexus-desktop --connect http://127.0.0.1:8080
```

With `NEXUS_API_KEY` exported as above, `--connect` uses that key automatically. The connection section also accepts a key manually; enter it and click **Connect** to reconnect. The GUI can inspect status and metrics and load, reload or unload models. It does not own the Docker process: start/stop/restart the container with Docker Compose. Paths chosen in the GUI must be under the mounted model directory and readable by the configured UID/GID. This mode does not require forwarding X11 or Wayland into the container.

```bash
docker --context default compose logs --tail 100
docker --context default compose stop
docker --context default compose down
```

An HTTP health response alone does not prove GPU inference. Verify `/v1/model` reports the CUDA backend and that authenticated `/metrics` reports `runtime.gpu_kernel_launches` increasing after a completion request.
