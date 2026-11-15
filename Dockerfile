# syntax=docker/dockerfile:1
ARG RUST_IMAGE=rust:1.88.0-bookworm@sha256:af306cfa71d987911a781c37b59d7d67d934f49684058f96cf72079c3626bfe0
ARG RUNTIME_IMAGE=debian:13-slim@sha256:a72b4433de1cf7d6bae0c4cee74d0547f58601a643b3375f070c51600d63b6b2

FROM ${RUST_IMAGE} AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    clang cmake libclang-dev libssl-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY tools/nexus-convert ./tools/nexus-convert
ARG CARGO_BUILD_JOBS=2
RUN cargo build --release --locked --features cuda --bin nexus-server --jobs "${CARGO_BUILD_JOBS}"

FROM ${RUNTIME_IMAGE} AS runtime-base
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl libgcc-s1 libstdc++6 libssl3t64 tini \
    && rm -rf /var/lib/apt/lists/*
LABEL org.opencontainers.image.title="Nexus CUDA server" \
      org.opencontainers.image.source="https://github.com/luen-eth/Nexus" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.licenses="Apache-2.0"
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
WORKDIR /app
USER 10001:10001
EXPOSE 8080
STOPSIGNAL SIGTERM
# The server currently relies on default Unix signal handling. An init process
# forwards Docker's stop signal; a bare server as PID 1 would ignore SIGTERM.
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/nexus-server"]
CMD ["--host", "0.0.0.0", "--port", "8080", "--backend", "cuda"]

# Repackage an already validated Linux binary without compiling it again.
# The expected hash is mandatory and is checked inside the runtime image.
FROM runtime-base AS prebuilt
ARG NEXUS_SERVER_SHA256
ARG SOURCE_REVISION=unknown
COPY --chmod=0755 target/release/nexus-server /usr/local/bin/nexus-server
RUN test "${#NEXUS_SERVER_SHA256}" -eq 64 \
    && printf '%s  /usr/local/bin/nexus-server\n' "${NEXUS_SERVER_SHA256}" | sha256sum --check - \
    && /usr/local/bin/nexus-server --help >/dev/null
LABEL io.nexus.server.sha256="${NEXUS_SERVER_SHA256}" \
      org.opencontainers.image.revision="${SOURCE_REVISION}"

# Default target: compile from the checked-out source and Cargo.lock.
FROM runtime-base AS runtime
ARG SOURCE_REVISION=unknown
COPY --from=builder --chmod=0755 /build/target/release/nexus-server /usr/local/bin/nexus-server
RUN /usr/local/bin/nexus-server --help >/dev/null
LABEL org.opencontainers.image.revision="${SOURCE_REVISION}"
