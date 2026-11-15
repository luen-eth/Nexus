"""Public Hugging Face discovery and verified, atomic local GGUF downloads.

The Hub client deliberately does not import or share the local server API client.
API reference: https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import html
import json
import os
from pathlib import Path
import re
import ssl
import stat
import struct
import threading
import time
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit, urljoin
from urllib.request import HTTPRedirectHandler, HTTPSHandler, ProxyHandler, Request, build_opener
import uuid

from PySide6.QtCore import Qt, QObject, Signal, QSize, QBuffer, QByteArray, QIODevice, QThread
from PySide6.QtGui import QIcon, QImageReader, QPixmap
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QProgressBar, QPushButton, QVBoxLayout, QWidget,
)


class HubError(Exception):
    """A bounded, user-readable Hub operation failure."""


class Cancelled(HubError):
    pass


def _check_cancel(cancel: threading.Event):
    if cancel.is_set():
        raise Cancelled("Download cancelled.")


def _parts(value: str, *, repo=False) -> list[str]:
    if not isinstance(value, str) or not value or len(value) > 1024 or "\\" in value:
        raise HubError("Invalid repository or file path.")
    parts = value.split("/")
    if len(parts) > 16 or any(not p or p in {".", ".."} or len(p.encode()) > 255
                             or any(ord(c) < 32 or ord(c) == 127 for c in p) for p in parts):
        raise HubError("Invalid repository or file path.")
    if repo and (len(parts) != 2 or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", p) for p in parts)):
        raise HubError("Repository names must use owner/repository format.")
    return parts


def _allowed_url(url: str, allow_local_http=False):
    p = urlsplit(url)
    try:
        port = p.port
    except ValueError as error:
        raise HubError("Invalid download address.") from error
    if p.username or p.password or p.fragment:
        raise HubError("Invalid download address.")
    if allow_local_http and p.scheme == "http" and p.hostname == "127.0.0.1":
        return
    host = p.hostname or ""
    trusted = (host in {"huggingface.co", "cdn-lfs.huggingface.co", "cdn-lfs.hf.co"}
               or host.endswith(".cdn.hf.co") or host.endswith(".xethub.hf.co"))
    if p.scheme != "https" or port not in {None, 443} or not trusted:
        raise HubError("The download redirected outside the trusted Hugging Face HTTPS hosts.")


class _SafeRedirect(HTTPRedirectHandler):
    max_redirections = 5
    max_repeats = 2

    def __init__(self, allow_local_http=False, validator=_allowed_url):
        self.allow_local_http = allow_local_http
        self.validator = validator

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.validator(newurl, self.allow_local_http)
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected:
            for header in ("Authorization", "Cookie", "Proxy-Authorization"):
                redirected.remove_header(header)
        return redirected


@dataclass(frozen=True)
class HubFile:
    repo: str
    revision: str
    name: str
    size: int
    sha256: str | None = None
    architecture: str = ""

    @property
    def split(self) -> bool:
        return bool(re.search(r"-\d{5}-of-\d{5}\.gguf$", self.name, re.IGNORECASE))


def _inspect_stream(stream) -> dict:
    """Read only bounded GGUF metadata; full model validation belongs to Nexus."""
    total = os.fstat(stream.fileno()).st_size
    budget = min(total, 32 * 1024 * 1024)

    def read(n):
        if n < 0 or stream.tell() + n > budget:
            raise HubError("GGUF metadata is truncated or exceeds the inspection limit.")
        data = stream.read(n)
        if len(data) != n:
            raise HubError("The file is not a complete GGUF model.")
        return data

    def number(fmt):
        return struct.unpack("<" + fmt, read(struct.calcsize(fmt)))[0]

    def string(keep=False):
        length = number("Q")
        if length > 16 * 1024 * 1024:
            raise HubError("GGUF metadata string exceeds the inspection limit.")
        data = read(length)
        if keep:
            if length > 4096:
                raise HubError("GGUF metadata key is too long.")
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError as error:
                raise HubError("GGUF metadata contains invalid text.") from error
        return None

    formats = {0:"B", 1:"b", 2:"H", 3:"h", 4:"I", 5:"i", 6:"f", 7:"B", 10:"Q", 11:"q", 12:"d"}

    def value(kind, keep=False):
        if kind in formats:
            return number(formats[kind])
        if kind == 8:
            return string(keep)
        if kind == 9:
            element, count = number("I"), number("Q")
            if count > 2_000_000 or element == 9:
                raise HubError("GGUF metadata array exceeds the inspection limit.")
            if element in formats:
                read(count * struct.calcsize(formats[element]))
            elif element == 8:
                for _ in range(count):
                    string()
            else:
                raise HubError("Unsupported GGUF metadata type.")
            return None
        raise HubError("Unsupported GGUF metadata type.")

    if read(4) != b"GGUF" or number("I") not in {2, 3}:
        raise HubError("The downloaded file is not a supported GGUF v2/v3 model.")
    number("Q")  # Tensor table is checked by the server when loading.
    count = number("Q")
    if count > 100_000:
        raise HubError("GGUF metadata count exceeds the inspection limit.")
    result = {}
    for _ in range(count):
        key, kind = string(True), number("I")
        keep = key in {"general.architecture", "split.count"}
        item = value(kind, keep)
        if keep:
            result[key] = item
    if result.get("split.count", 1) != 1:
        raise HubError("Split GGUF models are not supported. Choose a single-file model.")
    return result


def inspect_gguf(path: Path) -> dict:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        return _inspect_stream(stream)


def _directory_fd(path: Path) -> int:
    """Create/open each component without following symlinks (Linux)."""
    absolute = path.absolute()
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in absolute.parts[1:]:
            if part in {".", ".."}:
                raise HubError("The model directory must use an absolute normalized path.")
            try:
                os.mkdir(part, mode=0o755, dir_fd=fd)
            except FileExistsError:
                pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


class HubClient:
    MAX_METADATA_BYTES = 4 * 1024 * 1024
    MAX_FILE_BYTES = 32 * 1024**3
    RESERVE_BYTES = 64 * 1024**2

    def __init__(self, base_url="https://huggingface.co", timeout=5, allow_local_http=False):
        _allowed_url(base_url, allow_local_http)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.allow_local_http = allow_local_http
        self._validate_url = _allowed_url
        # No environment proxies, cookies, netrc, HF tokens or Nexus API keys.
        self.opener = build_opener(ProxyHandler({}), _SafeRedirect(allow_local_http),
                                   HTTPSHandler(context=ssl.create_default_context()))

    def _open(self, url):
        self._validate_url(url, self.allow_local_http)
        request = Request(url, headers={"User-Agent":"NexusDesktop/0.1", "Accept-Encoding":"identity"})
        try:
            return self.opener.open(request, timeout=self.timeout)
        except HTTPError as error:
            error.close()
            if error.code in {401, 403}:
                raise HubError("This model requires Hugging Face access approval or authentication. Public models only are supported here.") from error
            if error.code == 429:
                raise HubError("Hugging Face is rate limiting requests. Try again later.") from error
            raise HubError(f"Hugging Face returned HTTP {error.code}.") from error
        except (URLError, TimeoutError, OSError) as error:
            raise HubError("Could not reach Hugging Face. Check your connection and try again.") from error

    def _json(self, path, cancel):
        _check_cancel(cancel)
        deadline = time.monotonic() + 30
        chunks, total = [], 0
        with self._open(self.base_url + path) as response:
            while True:
                _check_cancel(cancel)
                if time.monotonic() > deadline:
                    raise HubError("The metadata request timed out.")
                chunk = response.read1(min(65536, self.MAX_METADATA_BYTES + 1 - total))
                if not chunk:
                    break
                total += len(chunk)
                if total > self.MAX_METADATA_BYTES:
                    raise HubError("Hugging Face metadata exceeds the response size limit.")
                chunks.append(chunk)
        try:
            return json.loads(b"".join(chunks))
        except (ValueError, UnicodeDecodeError) as error:
            raise HubError("Hugging Face returned invalid metadata.") from error

    def search(self, query: str, cancel: threading.Event) -> list[dict]:
        params = urlencode({"search":query.strip()[:200], "filter":"gguf", "sort":"downloads", "direction":-1, "limit":20})
        data = self._json("/api/models?" + params, cancel)
        if not isinstance(data, list):
            raise HubError("Hugging Face returned an invalid model list.")
        found = []
        for item in data[:20]:
            if not isinstance(item, dict):
                continue
            repo = item.get("id")
            try:
                _parts(repo, repo=True)
            except HubError:
                continue
            count = item.get("downloads", 0)
            found.append({"id":repo, "downloads":count if type(count) is int and count >= 0 else 0})
        return found

    def files(self, repo_id: str, cancel: threading.Event) -> list[HubFile]:
        repo_path = "/".join(quote(p, safe="") for p in _parts(repo_id, repo=True))
        data = self._json(f"/api/models/{repo_path}/revision/main?blobs=true", cancel)
        if not isinstance(data, dict):
            raise HubError("Hugging Face returned invalid repository metadata.")
        if data.get("private") or data.get("gated"):
            raise HubError("This repository is private or gated. Choose a public model that does not require approval.")
        revision = data.get("sha", "")
        if not isinstance(revision, str) or not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
            raise HubError("The repository did not provide an immutable revision.")
        gguf = data.get("gguf") or {}
        architecture = gguf.get("architecture", "") if isinstance(gguf, dict) else ""
        if not isinstance(architecture, str):
            architecture = ""
        siblings = data.get("siblings")
        if not isinstance(siblings, list) or len(siblings) > 10_000:
            raise HubError("The repository file list is missing or too large.")
        result = []
        for item in siblings:
            if not isinstance(item, dict):
                continue
            name = item.get("rfilename", "")
            if not isinstance(name, str) or not name.lower().endswith(".gguf"):
                continue
            _parts(name)
            size = item.get("size", 0)
            lfs = item.get("lfs") or {}
            if not isinstance(lfs, dict):
                raise HubError("Invalid file integrity metadata.")
            digest = lfs.get("sha256")
            if digest is not None and (not isinstance(digest, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", digest)):
                raise HubError("Invalid file SHA-256 metadata.")
            if lfs.get("size", size) != size:
                raise HubError("The repository reports inconsistent file sizes.")
            result.append(HubFile(repo_id, revision, name, size if type(size) is int else 0,
                                  digest.lower() if digest else None, architecture[:128]))
            if len(result) >= 200:
                break
        return result

    def download(self, file: HubFile, root: Path, cancel: threading.Event,
                 progress: Callable[[int, int], None]) -> Path:
        _check_cancel(cancel)
        repo_parts, file_parts = _parts(file.repo, repo=True), _parts(file.name)
        if not re.fullmatch(r"[a-fA-F0-9]{40}", file.revision):
            raise HubError("An immutable repository revision is required.")
        if file.split:
            raise HubError("Split GGUF models are not supported. Choose a single-file model.")
        if not file.name.lower().endswith(".gguf") or type(file.size) is not int or not 24 <= file.size <= self.MAX_FILE_BYTES:
            raise HubError("A known GGUF file size between 24 bytes and 32 GiB is required.")
        if file.sha256 is not None and not re.fullmatch(r"[a-fA-F0-9]{64}", file.sha256):
            raise HubError("Invalid file SHA-256 metadata.")
        root = Path(root).absolute()
        relative = repo_parts + [file.revision] + file_parts
        folder = root.joinpath(*relative[:-1])
        destination = folder / relative[-1]
        try:
            directory = _directory_fd(folder)
        except OSError as error:
            raise HubError("The model directory is inaccessible or contains a symlink.") from error
        temporary = ".nexus-download-" + uuid.uuid4().hex + ".part"
        created = False
        try:
            try:
                os.stat(relative[-1], dir_fd=directory, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise HubError("This file already exists locally. It will not be overwritten.")
            space = os.fstatvfs(directory)
            if space.f_bavail * space.f_frsize < file.size + self.RESERVE_BYTES:
                raise HubError("There is not enough free disk space for this model and a 64 MiB reserve.")
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
            created = True
            url_path = "/".join(quote(p, safe="") for p in repo_parts + ["resolve", file.revision] + file_parts)
            digest, written = hashlib.sha256(), 0
            deadline = time.monotonic() + 24 * 60 * 60
            with os.fdopen(fd, "wb") as output, self._open(self.base_url + "/" + url_path) as response:
                content_length = response.headers.get("Content-Length")
                if content_length is not None and (not content_length.isdigit() or int(content_length) != file.size):
                    raise HubError("Download size does not match the pinned file metadata.")
                progress(0, file.size)
                while True:
                    _check_cancel(cancel)
                    if time.monotonic() > deadline:
                        raise HubError("The download exceeded the time limit.")
                    chunk = response.read1(min(256 * 1024, file.size + 1 - written))
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > file.size:
                        raise HubError("Download exceeded the expected file size.")
                    output.write(chunk)
                    digest.update(chunk)
                    progress(written, file.size)
                output.flush()
                os.fsync(output.fileno())
            _check_cancel(cancel)
            if written != file.size:
                raise HubError("Download was incomplete. No model file was installed.")
            if file.sha256 and digest.hexdigest() != file.sha256.lower():
                raise HubError("SHA-256 verification failed. No model file was installed.")
            check_fd = os.open(temporary, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
            with os.fdopen(check_fd, "rb") as stream:
                _inspect_stream(stream)
            _check_cancel(cancel)
            # A hard-link publication is atomic and fails if ANY destination exists.
            os.link(temporary, relative[-1], src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
            os.unlink(temporary, dir_fd=directory)
            created = False
            os.fsync(directory)
            return destination
        except OSError as error:
            raise HubError("The download could not be saved. Check directory permissions, symlinks and free space.") from error
        finally:
            if created:
                try:
                    os.unlink(temporary, dir_fd=directory)
                except FileNotFoundError:
                    pass
            os.close(directory)


def list_local_models(root: Path) -> list[Path]:
    root = Path(root).absolute()
    if root.is_symlink() or not root.is_dir():
        return []
    found, pending, visited = [], [(root, 0)], 0
    while pending and visited < 5000 and len(found) < 500:
        folder, depth = pending.pop()
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    visited += 1
                    if visited > 5000:
                        break
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False) and depth < 20:
                        pending.append((Path(entry.path), depth + 1))
                    elif entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(".gguf"):
                        found.append(Path(entry.path))
                        if len(found) >= 500:
                            break
        except OSError:
            continue
    return sorted(found, key=lambda p: str(p).lower())


def _allowed_logo_url(url: str, allow_local_http=False):
    """Avatar URLs must come from a profile API and stay on these HF hosts."""
    p = urlsplit(url)
    try:
        port = p.port
    except ValueError as error:
        raise HubError("Invalid publisher image address.") from error
    if p.username or p.password or p.fragment:
        raise HubError("Invalid publisher image address.")
    if allow_local_http and p.scheme == "http" and p.hostname == "127.0.0.1":
        return
    if (p.scheme != "https" or port not in {None, 443}
            or p.hostname not in {"huggingface.co", "cdn-avatars.huggingface.co", "cdn-uploads.huggingface.co"}):
        raise HubError("Publisher images must use a trusted Hugging Face HTTPS host.")


def _author(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", value))


class PublisherLogoClient(HubClient):
    """Only public HF profile avatars; never inspect model cards or README images.

    User overview schema: https://huggingface.co/.well-known/openapi.json
    Organization overview uses the corresponding public organization endpoint.
    """
    MAX_METADATA_BYTES = 64 * 1024
    MAX_IMAGE_BYTES = 1024 * 1024

    def __init__(self, base_url="https://huggingface.co", timeout=5, allow_local_http=False):
        super().__init__(base_url, timeout, allow_local_http)
        self._validate_url = _allowed_logo_url
        self.opener = build_opener(ProxyHandler({}),
                                   _SafeRedirect(allow_local_http, _allowed_logo_url),
                                   HTTPSHandler(context=ssl.create_default_context()))

    def avatar(self, author: str, cancel: threading.Event) -> bytes | None:
        if not _author(author):
            return None
        profile = None
        for kind, key in (("users", "user"), ("organizations", "name")):
            _check_cancel(cancel)
            try:
                profile = self._json(f"/api/{kind}/{quote(author, safe='')}/overview", cancel)
            except HubError as error:
                if isinstance(error.__cause__, HTTPError) and error.__cause__.code == 404:
                    continue
                raise
            if not isinstance(profile, dict) or str(profile.get(key, "")).casefold() != author.casefold():
                return None
            break
        if not isinstance(profile, dict):
            return None
        avatar = profile.get("avatarUrl")
        if not isinstance(avatar, str) or not avatar or len(avatar) > 2048:
            return None
        url = urljoin(self.base_url + "/", avatar)
        _allowed_logo_url(url, self.allow_local_http)
        # HF's generated default avatars are SVG; use our trusted bundled fallback.
        if urlsplit(url).path.lower().endswith(".svg"):
            return None
        deadline = time.monotonic() + 15
        chunks, total = [], 0
        with self._open(url) as response:
            mime = response.headers.get("Content-Type", "").partition(";")[0].strip().lower()
            if mime not in {"image/png", "image/jpeg", "image/webp"}:
                return None
            length = response.headers.get("Content-Length")
            if length is not None and (not length.isdigit() or int(length) > self.MAX_IMAGE_BYTES):
                return None
            while True:
                _check_cancel(cancel)
                if time.monotonic() > deadline:
                    raise HubError("Publisher image request timed out.")
                chunk = response.read1(min(65536, self.MAX_IMAGE_BYTES + 1 - total))
                if not chunk:
                    break
                total += len(chunk)
                if total > self.MAX_IMAGE_BYTES:
                    return None
                chunks.append(chunk)
        _check_cancel(cancel)
        return b"".join(chunks)


def _decode_logo(data: bytes):
    """Decode bounded raster data only on the GUI thread and normalize to PNG."""
    app = QApplication.instance()
    if app is None or QThread.currentThread() != app.thread():
        raise RuntimeError("Publisher images must be decoded on the GUI thread.")
    if not isinstance(data, bytes) or not data or len(data) > PublisherLogoClient.MAX_IMAGE_BYTES:
        return None
    buffer = QBuffer()
    buffer.setData(QByteArray(data))
    buffer.open(QIODevice.OpenModeFlag.ReadOnly)
    reader = QImageReader(buffer)
    reader.setDecideFormatFromContent(True)
    if bytes(reader.format()).lower() not in {b"png", b"jpeg", b"webp"}:
        return None
    size = reader.size()
    if not size.isValid() or size.width() > 1024 or size.height() > 1024:
        return None
    reader.setScaledSize(size.scaled(QSize(32, 32), Qt.AspectRatioMode.KeepAspectRatio))
    image = reader.read()
    if image.isNull():
        return None
    image = image.scaled(32, 32, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    encoded = QBuffer()
    encoded.open(QIODevice.OpenModeFlag.WriteOnly)
    if not image.save(encoded, "PNG"):
        return None
    return QIcon(QPixmap.fromImage(image)), bytes(encoded.data())


class _LogoSignals(QObject):
    received = Signal(str, object, bool)
    stored = Signal(str)


class LogoCache(QObject):
    """Two separate workers, bounded requests, and a small private raster cache."""
    updated = Signal(str)
    MAX_PENDING = 32
    MAX_ICONS = 128
    AUTHOR_ROLE = int(Qt.ItemDataRole.UserRole) + 1
    STATE_ROLE = int(Qt.ItemDataRole.UserRole) + 2

    def __init__(self, cache_dir: Path | None = None, parent=None):
        super().__init__(parent)
        fallback = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        self.cache_dir = Path(cache_dir) if cache_dir is not None else fallback / "nexus-desktop/logos"
        self.client = PublisherLogoClient()
        self.fallback_icon = QIcon(str(Path(__file__).parent / "assets/huggingface.svg"))
        self._icons = OrderedDict()
        self._failed = OrderedDict()
        self._pending = {}
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="nexus-logos")
        self._signals = _LogoSignals(self)
        self._signals.received.connect(self._received)
        self._signals.stored.connect(self._stored)

    @staticmethod
    def _filename(author):
        return hashlib.sha256(author.casefold().encode()).hexdigest() + ".png"

    def has_logo(self, author):
        return author.casefold() in self._icons if isinstance(author, str) else False

    def icon(self, author: str) -> QIcon:
        if not _author(author):
            return self.fallback_icon
        author = author.casefold()
        if author in self._icons:
            self._icons.move_to_end(author)
            return self._icons[author]
        failed = self._failed.get(author, 0)
        if (not self._closed and author not in self._pending
                and len(self._pending) < self.MAX_PENDING and time.monotonic() >= failed):
            cancel = threading.Event()
            self._pending[author] = cancel
            self._executor.submit(self._load, author, cancel)
        return self.fallback_icon

    def _read_cache(self, author):
        # Opening components without following links also keeps the cache private.
        directory = _directory_fd(self.cache_dir)
        try:
            os.fchmod(directory, 0o700)
            fd = os.open(self._filename(author), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            with os.fdopen(fd, "rb") as source:
                metadata = os.fstat(source.fileno())
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > self.client.MAX_IMAGE_BYTES:
                    return None
                return source.read(self.client.MAX_IMAGE_BYTES + 1)
        finally:
            os.close(directory)

    def _load(self, author, cancel, skip_cache=False):
        payload, cached = None, False
        try:
            _check_cancel(cancel)
            if not skip_cache:
                try:
                    payload = self._read_cache(author)
                    cached = payload is not None
                except OSError:
                    pass
            if payload is None:
                payload = self.client.avatar(author, cancel)
            _check_cancel(cancel)
        except Exception:
            payload = None
        try:
            self._signals.received.emit(author, payload, cached)
        except RuntimeError:
            pass

    def _received(self, author, payload, cached):
        if self._closed or author not in self._pending:
            return
        decoded = _decode_logo(payload) if isinstance(payload, bytes) else None
        if decoded is None and cached:
            self._executor.submit(self._load, author, self._pending[author], True)
            return
        if decoded is None:
            self._pending.pop(author, None)
            self._failed[author] = time.monotonic() + 60
            while len(self._failed) > self.MAX_ICONS:
                self._failed.popitem(last=False)
        else:
            icon, normalized = decoded
            self._icons[author] = icon
            while len(self._icons) > self.MAX_ICONS:
                self._icons.popitem(last=False)
            if cached:
                self._pending.pop(author, None)
            else:
                self._executor.submit(self._store, author, normalized, self._pending[author])
        self.updated.emit(author)

    def _store(self, author, data, cancel):
        directory, temporary = None, None
        try:
            _check_cancel(cancel)
            directory = _directory_fd(self.cache_dir)
            os.fchmod(directory, 0o700)
            temporary = ".logo-" + uuid.uuid4().hex + ".part"
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
            with os.fdopen(fd, "wb") as target:
                target.write(data)
            _check_cancel(cancel)
            os.replace(temporary, self._filename(author), src_dir_fd=directory, dst_dir_fd=directory)
            temporary = None
            entries = []
            with os.scandir(directory) as scan:
                for item in scan:
                    if len(entries) >= 512:
                        break
                    if re.fullmatch(r"[a-f0-9]{64}\.png", item.name) and item.is_file(follow_symlinks=False):
                        entries.append((item.stat(follow_symlinks=False).st_mtime_ns, item.name))
            for _, name in sorted(entries)[:-self.MAX_ICONS]:
                os.unlink(name, dir_fd=directory)
        except (HubError, OSError):
            pass  # A read-only/unavailable cache must not hide an in-memory logo.
        finally:
            if directory is not None:
                if temporary:
                    try:
                        os.unlink(temporary, dir_fd=directory)
                    except OSError:
                        pass
                os.close(directory)
            try:
                self._signals.stored.emit(author)
            except RuntimeError:
                pass

    def _stored(self, author):
        self._pending.pop(author, None)

    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        for cancel in self._pending.values():
            cancel.set()
        self._pending.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)


def _local_repository(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root).parts
        if len(relative) < 4 or not re.fullmatch(r"[a-fA-F0-9]{40}", relative[2]):
            return ""
        repo = "/".join(relative[:2])
        _parts(repo, repo=True)
        _parts("/".join(relative[3:]))
        return repo
    except (ValueError, HubError):
        return ""


def _size(value):
    if not isinstance(value, int) or value <= 0:
        return "size unavailable"
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024


class _WorkerSignals(QObject):
    done = Signal(str, int, object, str)
    progress = Signal(int, object, object)  # Byte counts can exceed Qt's 32-bit int.


class ModelHubPanel(QWidget):
    model_selected = Signal(str)

    def __init__(self, controller, download_dir: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("hubPanel")
        self.controller = controller
        fallback = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share"))) / "nexus/models"
        self.download_dir = Path(download_dir or os.environ.get("NEXUS_MODELS_DIR") or fallback).absolute()
        self.client = HubClient()
        self.logo_cache = LogoCache(parent=self)
        self.logo_cache.updated.connect(self._logo_updated)
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="nexus-hub")
        self._signals = _WorkerSignals(self)
        self._signals.done.connect(self._finished)
        self._signals.progress.connect(self._progress)
        self._jobs = {}
        self._serial = 0
        self._closed = False
        self._download_file = None
        self._local_rescan = False
        self._build_ui()
        self.refresh_local()

    @staticmethod
    def _label(text, name=""):
        label = QLabel(text)
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setWordWrap(True)
        if name:
            label.setObjectName(name)
        return label

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        layout.addWidget(self._label("Find a model", "cardTitle"))
        layout.addWidget(self._label("Search public GGUF repositories on Hugging Face. Choose a file to download, then load it from your local library.", "subtitle"))
        search_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setObjectName("hubSearch")
        self.search_input.setPlaceholderText("Search models, e.g. SmolLM2-135M")
        self.search_input.setMaxLength(200)
        self.search_button = QPushButton("Search")
        self.search_button.setProperty("primary", True)
        self.search_button.clicked.connect(self.search)
        self.search_input.returnPressed.connect(self.search)
        search_row.addWidget(self.search_input, 1)
        search_row.addWidget(self.search_button)
        layout.addLayout(search_row)
        columns = QHBoxLayout()
        for title, attr, name in [("Repositories", "results", "hubResults"), ("GGUF files", "file_list", "hubFiles")]:
            column = QVBoxLayout()
            column.addWidget(self._label(title, "fieldLabel"))
            widget = QListWidget()
            widget.setObjectName(name)
            widget.setMinimumHeight(150)
            widget.setIconSize(QSize(32, 32))
            setattr(self, attr, widget)
            column.addWidget(widget)
            columns.addLayout(column, 1)
        layout.addLayout(columns)
        self.results.currentItemChanged.connect(self._repository_selected)
        self.file_list.currentItemChanged.connect(self._file_selected)
        self.file_detail = self._label("Select a repository to see file sizes and available downloads.", "subtitle")
        layout.addWidget(self.file_detail)
        row = QHBoxLayout()
        self.download_button = QPushButton("Download selected file")
        self.download_button.setProperty("primary", True)
        self.download_button.setEnabled(False)
        self.download_button.clicked.connect(self.start_download)
        self.cancel_button = QPushButton("Cancel download")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_download)
        row.addWidget(self.download_button)
        row.addWidget(self.cancel_button)
        row.addStretch()
        layout.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setObjectName("hubProgress")
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setFormat("No download in progress")
        layout.addWidget(self.progress)
        self.status = self._label("Public repositories only. No Hugging Face account is required.", "hubStatus")
        layout.addWidget(self.status)
        layout.addWidget(self._label("Compatibility: Nexus currently supports LLaMA architecture with full-head, unscaled RoPE. GGUF search results are not a compatibility guarantee. Weights expand to F32 in RAM/VRAM; a small download may need much more memory. Split GGUF files are not supported.", "notice"))
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(line)
        library_row = QHBoxLayout()
        library_row.addWidget(self._label("Your local models", "cardTitle"), 1)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_local)
        library_row.addWidget(refresh)
        layout.addLayout(library_row)
        self.local_files = QListWidget()
        self.local_files.setMinimumHeight(110)
        self.local_files.setObjectName("hubLocalFiles")
        self.local_files.setIconSize(QSize(32, 32))
        self.local_files.currentItemChanged.connect(lambda *_: self.load_button.setEnabled(self.local_files.currentItem() is not None))
        layout.addWidget(self.local_files)
        self.load_button = QPushButton("Load selected model")
        self.load_button.setProperty("primary", True)
        self.load_button.setEnabled(False)
        self.load_button.clicked.connect(self._load_selected)
        layout.addWidget(self.load_button)
        self.directory_label = self._label(str(self.download_dir), "subtitle")
        self.directory_label.setToolTip("Model library directory. For Docker, mount this same absolute path in the server container.")
        layout.addWidget(self.directory_label)

    def _item_logo(self, item, author="", path=""):
        item.setData(LogoCache.AUTHOR_ROLE, author)
        item.setIcon(self.logo_cache.icon(author))
        item.setData(LogoCache.STATE_ROLE, "publisher" if self.logo_cache.has_logo(author) else "fallback")
        tooltip = f"Publisher: {author}" if author else "Hugging Face · publisher unknown"
        if path:
            tooltip += "\n" + str(path)
        # Tooltips may interpret rich text; escape all file/repository-derived text.
        item.setToolTip("<qt>" + html.escape(tooltip).replace("\n", "<br>") + "</qt>")

    def _logo_updated(self, author):
        if self._closed:
            return
        # Find current rows instead of retaining deleted QListWidgetItem wrappers.
        for widget in (self.results, self.file_list, self.local_files):
            for index in range(widget.count()):
                item = widget.item(index)
                owner = item.data(LogoCache.AUTHOR_ROLE) or ""
                if owner.casefold() == author:
                    item.setIcon(self.logo_cache.icon(owner))
                    item.setData(LogoCache.STATE_ROLE, "publisher" if self.logo_cache.has_logo(owner) else "fallback")

    def _submit(self, kind, operation):
        if self._closed or kind in self._jobs or len(self._jobs) >= 2:
            return False
        self._serial += 1
        serial, cancel, signals = self._serial, threading.Event(), self._signals
        self._jobs[kind] = (serial, cancel)

        def work():
            try:
                result, error = operation(cancel, serial), ""
            except Cancelled:
                result, error = None, "Operation cancelled."
            except (HubError, OSError, ValueError) as failure:
                result, error = None, str(failure)[:500]
            except Exception:
                result, error = None, "The operation could not be completed. Try again."
            try:
                signals.done.emit(kind, serial, result, error)
            except RuntimeError:
                pass  # Widget was deleted after shutdown; no Qt worker thread exists.
        self._executor.submit(work)
        return True

    def search(self):
        if "search" in self._jobs or "files" in self._jobs:
            return
        self.results.clear()
        self.file_list.clear()
        self.status.setText("Searching Hugging Face…")
        query = self.search_input.text()
        if self._submit("search", lambda cancel, _: self.client.search(query, cancel)):
            self.search_button.setEnabled(False)

    def _repository_selected(self, item, _previous=None):
        self.file_list.clear()
        if item is None:
            return
        repo = item.data(Qt.ItemDataRole.UserRole)
        if "files" in self._jobs:
            self._jobs["files"][1].set()
            return
        self.status.setText(f"Fetching files for {repo}…")
        if self._submit("files", lambda cancel, _: self.client.files(repo, cancel)):
            self.results.setEnabled(False)
            self.search_button.setEnabled(False)

    def _file_selected(self, item, _previous=None):
        file = item.data(Qt.ItemDataRole.UserRole) if item else None
        allowed = bool(file and not file.split and 24 <= file.size <= self.client.MAX_FILE_BYTES)
        self.download_button.setEnabled(allowed and "download" not in self._jobs)
        if not file:
            return
        architecture = file.architecture or "unknown (not verified)"
        if file.architecture and file.architecture != "llama":
            architecture += " (not currently supported by Nexus)"
        integrity = "SHA-256 available" if file.sha256 else "Publisher SHA-256 unavailable; size and GGUF structure will be checked"
        suffix = " · Split file: unsupported" if file.split else ""
        self.file_detail.setText(f"{file.name}\n{_size(file.size)} · Architecture: {architecture} · {integrity}{suffix}")

    def start_download(self):
        item = self.file_list.currentItem()
        if not item or "download" in self._jobs:
            return
        file = item.data(Qt.ItemDataRole.UserRole)
        self._download_file = file
        last_update = [0.0]
        signals = self._signals

        def operation(cancel, serial):
            def report(done, total):
                now = time.monotonic()
                if done == total or now - last_update[0] >= .1:
                    last_update[0] = now
                    try:
                        signals.progress.emit(serial, done, total)
                    except RuntimeError:
                        cancel.set()
            return self.client.download(file, self.download_dir, cancel, report)
        if self._submit("download", operation):
            self.download_button.setEnabled(False)
            self.cancel_button.setEnabled(True)
            self.progress.setValue(0)
            self.progress.setFormat("Starting download…")
            self.status.setText(f"Downloading {file.name}…")

    def cancel_download(self):
        job = self._jobs.get("download")
        if job:
            job[1].set()
            self.cancel_button.setEnabled(False)
            self.status.setText("Cancelling download…")

    def _progress(self, serial, done, total):
        if not self._closed and self._jobs.get("download", (None,))[0] == serial:
            self.progress.setValue(int(done * 1000 / total) if total else 0)
            self.progress.setFormat(f"{_size(done)} / {_size(total)}")

    def _finished(self, kind, serial, result, error):
        if self._closed or self._jobs.get(kind, (None,))[0] != serial:
            return
        self._jobs.pop(kind)
        self.search_button.setEnabled("search" not in self._jobs and "files" not in self._jobs)
        self.results.setEnabled("files" not in self._jobs)
        if error:
            self.status.setText(error)
        elif kind == "search":
            for repo in result:
                item = QListWidgetItem(f"{repo['id']}\n{repo['downloads']:,} downloads")
                item.setData(Qt.ItemDataRole.UserRole, repo["id"])
                self._item_logo(item, repo["id"].split("/", 1)[0])
                self.results.addItem(item)
            self.status.setText(f"{len(result)} repositories found. Select one to browse files." if result else "No GGUF repositories found. Try a different search.")
        elif kind == "files":
            for file in result:
                item = QListWidgetItem(f"{file.name}\n{_size(file.size)}" + (" · split file, unsupported" if file.split else ""))
                item.setData(Qt.ItemDataRole.UserRole, file)
                self._item_logo(item, file.repo.split("/", 1)[0])
                self.file_list.addItem(item)
            self.status.setText(f"{len(result)} GGUF files shown (up to 200). Select a file to download." if result else "This repository has no GGUF files.")
        elif kind == "download":
            self.progress.setValue(1000)
            self.progress.setFormat("Download verified")
            self.status.setText("Download verified. Select the model in your local library to load it.")
            self.refresh_local(select=str(result))
        elif kind == "local":
            self.local_files.clear()
            for path in result:
                repo = _local_repository(path, self.download_dir)
                item = QListWidgetItem(path.name + ("\n" + repo if repo else "\nLocal model"))
                item.setData(Qt.ItemDataRole.UserRole, str(path))
                self._item_logo(item, repo.split("/", 1)[0] if repo else "", path)
                self.local_files.addItem(item)
                if str(path) == getattr(self, "_select_local", ""):
                    self.local_files.setCurrentItem(item)
            if self.local_files.count() and self.local_files.currentRow() < 0:
                self.local_files.setCurrentRow(0)
            if self._local_rescan:
                self.refresh_local(select=self._select_local)
        elif kind == "loadcheck":
            self.model_selected.emit(str(result))
        if kind == "loadcheck":
            self.load_button.setEnabled(self.local_files.currentItem() is not None)
        if kind == "download":
            self.cancel_button.setEnabled(False)
            if error:
                self.progress.setFormat("Download not installed")
            self._file_selected(self.file_list.currentItem())

    def refresh_local(self, _checked=False, *, select=""):
        self._select_local = select
        if "local" in self._jobs:
            self._local_rescan = True
            return
        self._local_rescan = False
        self._submit("local", lambda cancel, _: list_local_models(self.download_dir))

    def _load_selected(self):
        item = self.local_files.currentItem()
        if item:
            path = Path(item.data(Qt.ItemDataRole.UserRole))
            if path.is_file() and not path.is_symlink():
                def checked(cancel, _serial):
                    _check_cancel(cancel)
                    if re.search(r"-\d{5}-of-\d{5}\.gguf$", path.name, re.IGNORECASE):
                        raise HubError("Split GGUF models are not supported. Choose a single-file model.")
                    inspect_gguf(path)
                    _check_cancel(cancel)
                    return path
                if self._submit("loadcheck", checked):
                    self.load_button.setEnabled(False)
            else:
                self.status.setText("The selected local model is no longer available. Refresh the library.")

    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        self.logo_cache.shutdown()
        for _, cancel in self._jobs.values():
            cancel.set()
        self._jobs.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)
