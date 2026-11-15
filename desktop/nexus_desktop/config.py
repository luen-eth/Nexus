"""Small, atomic XDG settings. Authentication secrets are session-only."""
from __future__ import annotations
from dataclasses import asdict, dataclass, fields
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.parse import urlsplit


@dataclass
class Settings:
    binary: str = ""
    model: str = ""
    backend: str = "cpu"
    kv_quant: str = "none"
    context: int = 2048
    port: int = 8080
    batch_size: int = 1
    server_url: str = "http://127.0.0.1:8080"


def config_path() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "nexus-desktop" / "settings.json"


def load_settings(path: Path | None = None) -> Settings:
    path = path or config_path()
    result = Settings()
    try:
        if path.stat().st_size > 65536:
            return result
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return result
        for field in fields(result):
            value = data.get(field.name)
            expected = type(getattr(result, field.name))
            if type(value) is expected:
                setattr(result, field.name, value)
    except (OSError, ValueError):
        pass
    result.port = max(1, min(65535, result.port))
    result.context = max(1, min(1048576, result.context))
    result.batch_size = max(1, min(16, result.batch_size))
    if result.backend not in {"cpu", "cuda"}:
        result.backend = "cpu"
    if result.kv_quant not in {"none", "tq_3b2b", "tq_3b4b", "tq_4b4b", "block_q4", "block_q6"}:
        result.kv_quant = "none"
    try:
        result.server_url = local_url(result.server_url)
    except ValueError:
        result.server_url = Settings().server_url
    return result


def save_settings(settings: Settings, path: Path | None = None) -> None:
    path = path or config_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".settings-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(asdict(settings), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def local_url(value: str) -> str:
    """Restrict management/authentication to a loopback HTTP endpoint."""
    try:
        parts = urlsplit(value.strip())
        port = parts.port if parts.port is not None else 8080
    except ValueError as error:
        raise ValueError("Enter a valid local address and port.") from error
    if (parts.scheme != "http" or parts.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parts.username or parts.password or parts.query or parts.fragment or parts.path not in {"", "/"}
            or not 1 <= port <= 65535):
        raise ValueError("Only local HTTP addresses are supported: http://127.0.0.1:8080")
    host = "[::1]" if parts.hostname == "::1" else parts.hostname
    return f"http://{host}:{port}"


def discover_binary() -> str:
    root = Path(__file__).resolve().parents[2]
    candidates = [
        Path("/usr/lib/nexus-desktop/nexus-server"),
        Path("/usr/share/nexus-desktop/bin/nexus-server"),
        Path("/usr/lib/nexus/nexus-server"),
    ]
    found = shutil.which("nexus-server")
    if found:
        candidates.append(Path(found))
    candidates.extend([root / "target/release/nexus-server", root / "target/debug/nexus-server"])
    return next((str(p) for p in candidates if p.is_file() and os.access(p, os.X_OK)), "")
