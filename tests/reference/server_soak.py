#!/usr/bin/env python3
"""Real-model local HTTP soak. No third-party Python packages or global process cleanup."""
import argparse
import collections
import concurrent.futures
import datetime
import hashlib
import http.client
import json
import os
import pathlib
import secrets
import signal
import shutil
import socket
import subprocess
import threading
import time


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def quantiles(values):
    values = sorted(values)
    if not values:
        return {"count": 0}
    return {"count": len(values), "min": values[0], "max": values[-1],
            **{f"p{p}": values[min(len(values)-1, round((len(values)-1)*p/100))]
               for p in (50, 90, 95, 99)}}


class Soak:
    def __init__(self, args):
        self.args = args
        self.out = pathlib.Path(args.output).resolve()
        if self.out.exists() and any(self.out.iterdir()):
            raise SystemExit("--output must be a new or empty evidence directory")
        self.out.mkdir(parents=True, exist_ok=True)
        self.key = secrets.token_hex(20)  # Test-only credential, never a user/account secret.
        self.process = None
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.phase = "startup"
        self.drain_index = 0
        self.records = []
        self.samples = []
        self.failures = []
        self.work_started = None
        self.request_log = open(self.out / "requests.jsonl", "w", buffering=1)
        self.sample_log = open(self.out / "samples.jsonl", "w", buffering=1)
        self.server_log = open(self.out / "server.log", "w")
        self.gpu_query = shutil.which("nvidia-smi")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]

    def http(self, method, path, payload=None, mode="json", auth=True, timeout=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port,
                                         timeout=timeout or self.args.request_timeout)
        if mode == "slow":
            conn.connect()
            conn.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
        headers = {"Content-Type": "application/json"}
        if auth:
            headers["Authorization"] = "Bearer " + self.key
        encoded = None if payload is None else json.dumps(payload).encode()
        began = time.monotonic()
        first_event = None
        first_token = None
        events = 0
        finished = False
        reason = None
        choice_text = ""
        response_model = None
        try:
            conn.request(method, path, body=encoded, headers=headers)
            response = conn.getresponse()
            status = response.status
            if mode == "cancel" and status == 200 and self.args.cancel_after == "headers":
                value = {"events": 0, "done": False, "model": None,
                         "headers_only_cancel": True}
            elif mode in ("sse", "slow", "cancel") and status == 200:
                while True:
                    raw = response.readline()
                    if not raw:
                        break
                    if not raw.startswith(b"data: "):
                        continue
                    if first_event is None:
                        first_event = time.monotonic() - began
                    data = raw[6:].strip()
                    if data == b"[DONE]":
                        finished = True
                        break
                    event = json.loads(data)
                    if "error" in event:
                        raise RuntimeError("SSE error: " + json.dumps(event))
                    response_model = event.get("model", response_model)
                    for choice in event.get("choices", []):
                        delta = choice.get("text", "") + choice.get("delta", {}).get("content", "")
                        if delta and first_token is None:
                            first_token = time.monotonic() - began
                        choice_text += delta
                        reason = choice.get("finish_reason") or reason
                    events += 1
                    if mode == "cancel":
                        # Closing only this connection drops the server response body.
                        break
                    if mode == "slow":
                        time.sleep(self.args.slow_read_delay)
                value = {"events": events, "done": finished, "finish_reason": reason,
                         "model": response_model, "text_chars": len(choice_text)}
            else:
                raw = response.read()
                value = json.loads(raw) if raw else None
            return {"status": status, "latency_s": time.monotonic()-began,
                    "ttft_s": first_token, "first_event_s": first_event, "value": value}
        finally:
            conn.close()

    @property
    def completion_path(self):
        return "/v1/chat/completions" if self.args.api_format == "chat" else "/v1/completions"

    def completion_body(self, tokens, stream=False, prompt=None, model=None):
        prompt = self.args.prompt if prompt is None else prompt
        value = {"model": self.model_name if model is None else model,
                 "max_tokens": tokens, "temperature": 0, "stream": stream}
        if self.args.api_format == "chat":
            value["messages"] = [{"role": "user", "content": prompt}]
        else:
            value["prompt"] = prompt
        return value

    def record_request(self, kind, concurrency, expected, path, payload=None, mode="json", auth=True):
        began = time.monotonic()
        record = {"utc": stamp(), "elapsed_s": None if self.work_started is None else began-self.work_started,
                  "kind": kind, "concurrency": concurrency, "phase": self.phase}
        try:
            result = self.http("POST" if payload is not None else "GET",
                               path, payload, mode, auth)
            record.update({k: v for k, v in result.items() if k != "value"})
            record["expected_status"] = expected
            value = result["value"]
            record["value"] = value
            if result["status"] != expected:
                raise RuntimeError(f"HTTP {result['status']}, expected {expected}: {value}")
            if expected == 200 and mode in ("sse", "slow") and not value.get("done"):
                raise RuntimeError("SSE closed without [DONE]")
            if expected == 200 and path.endswith("completions"):
                response_model = value.get("model")
                if mode != "cancel" and response_model != self.model_name:
                    raise RuntimeError(f"wrong response model {response_model!r}")
                if mode == "json":
                    if not value.get("choices"):
                        raise RuntimeError("successful completion without choices")
                    if any(c.get("finish_reason") not in ("stop", "length") for c in value["choices"]):
                        raise RuntimeError("invalid completion finish_reason")
                    usage = value.get("usage", {})
                    prompt_count = usage.get("prompt_tokens", -1)
                    completion_count = usage.get("completion_tokens", -1)
                    if (prompt_count < 0 or completion_count < 0
                        or completion_count > payload["max_tokens"]
                        or usage.get("total_tokens") != prompt_count + completion_count):
                        raise RuntimeError("inconsistent completion token usage")
                elif mode in ("sse", "slow") and value.get("finish_reason") not in ("stop", "length"):
                    raise RuntimeError("SSE missing a valid terminal finish_reason")
            record["ok"] = True
        except Exception as error:
            record.update(ok=False, error=f"{type(error).__name__}: {error}",
                          latency_s=time.monotonic()-began)
            with self.lock:
                self.failures.append(record.copy())
        with self.lock:
            self.records.append(record)
            self.request_log.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def resources(self, tag="periodic"):
        sample = {"utc": stamp(), "elapsed_s": None if self.work_started is None else time.monotonic()-self.work_started,
                  "phase": self.phase, "tag": tag, "pid": self.process.pid}
        try:
            status = pathlib.Path(f"/proc/{self.process.pid}/status").read_text()
            for line in status.splitlines():
                if line.startswith(("VmRSS:", "VmHWM:", "VmSize:", "VmSwap:", "Threads:")):
                    name, value = line.split(":", 1)
                    sample[name] = int(value.strip().split()[0])
            stat = pathlib.Path(f"/proc/{self.process.pid}/stat").read_text().rsplit(")", 1)[1].split()
            sample["cpu_s"] = (int(stat[11])+int(stat[12])) / os.sysconf("SC_CLK_TCK")
            result = self.http("GET", "/metrics", timeout=2)
            if result["status"] != 200:
                raise RuntimeError(f"metrics HTTP {result['status']}")
            sample["metrics"] = result["value"]
            runtime = result["value"].get("runtime", {})
            if runtime.get("worker_alive") is False or runtime.get("backend_healthy") is False:
                raise RuntimeError("runtime worker or backend reports unhealthy")
            sample.update(self.gpu_resources())
        except Exception as error:
            sample["error"] = f"{type(error).__name__}: {error}"
        with self.lock:
            self.samples.append(sample)
            self.sample_log.write(json.dumps(sample, ensure_ascii=False) + "\n")
        return sample

    def gpu_resources(self):
        if not self.gpu_query:
            return {"gpu_query_ok": False, "own_pid_gpu_memory_mib": None,
                    "gpu_query_error": "nvidia-smi unavailable"}
        try:
            gpu = subprocess.run([self.gpu_query,
                "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2)
            if gpu.returncode:
                return {"gpu_query_ok": False, "own_pid_gpu_memory_mib": None,
                        "gpu_query_error": gpu.stderr.strip()}
            values = []
            for line in gpu.stdout.splitlines():
                fields = [x.strip() for x in line.split(",")]
                if len(fields) == 2 and fields[0] == str(self.process.pid):
                    try:
                        values.append(int(fields[1]))
                    except ValueError:
                        return {"gpu_query_ok": False, "own_pid_gpu_memory_mib": None,
                                "gpu_query_error": "own PID memory is N/A"}
            return {"gpu_query_ok": True, "own_pid_gpu_memory_mib": sum(values),
                    "gpu_pid_present": bool(values)}
        except (OSError, subprocess.SubprocessError) as error:
            return {"gpu_query_ok": False, "own_pid_gpu_memory_mib": None,
                    "gpu_query_error": f"{type(error).__name__}: {error}"}

    def sampling_loop(self):
        while not self.stop.wait(self.args.sample_interval):
            self.resources()

    def drain(self):
        self.drain_index += 1
        deadline = time.monotonic() + self.args.drain_timeout
        last = None
        while time.monotonic() < deadline:
            last = self.resources(tag=f"drain-{self.drain_index}")
            metrics = last.get("metrics", {})
            runtime = metrics.get("runtime", {})
            scheduler = metrics.get("scheduler", {})
            if (runtime.get("active_sequences") == 0 and runtime.get("queued_requests") == 0
                and scheduler.get("running") == 0 and scheduler.get("pending") == 0
                and scheduler.get("kv_pages_used") == 0 and runtime.get("active_kv_payload_bytes") == 0):
                return self.resources(tag=f"drained-{self.drain_index}")
            time.sleep(.1)
        raise RuntimeError("drain timeout: " + json.dumps(last))

    def start(self):
        env = os.environ.copy()
        for key in ("NEXUS_API_KEY", "NEXUS_RATE_LIMIT_PER_MINUTE", "NEXUS_CORS_ORIGIN"):
            env.pop(key, None)
        # Compatibility throttle for the first-package global-Rayon binary only.
        # Current production CpuBackend owns a private default-one pool; this env
        # variable is not treated as its configuration interface.
        env["RAYON_NUM_THREADS"] = "1"
        command = [str(pathlib.Path(self.args.binary).resolve()), "--host", "127.0.0.1",
                   "--port", str(self.port), "--api-key", self.key,
                   "--model", str(pathlib.Path(self.args.model).resolve()),
                   "--backend", self.args.backend, "--ctx-size", str(self.args.ctx_size),
                   "--batch-size", "8", "--kv-quant", self.args.kv_quant,
                   "--weight-quant", "f32"]
        self.process = subprocess.Popen(command, stdout=self.server_log,
                                        stderr=subprocess.STDOUT, env=env)
        ready_until = time.monotonic()+self.args.startup_timeout
        while time.monotonic() < ready_until:
            if self.process.poll() is not None:
                raise RuntimeError(f"server exited at startup: {self.process.returncode}")
            try:
                result = self.http("GET", "/v1/model", timeout=2)
                if result["status"] == 200 and result["value"].get("loaded"):
                    self.model_name = result["value"]["id"]
                    return
            except (OSError, http.client.HTTPException):
                pass
            time.sleep(.1)
        raise RuntimeError("server startup timeout")

    def lifecycle(self, cycle):
        self.phase = f"lifecycle-{cycle}"
        self.drain()
        self.record_request("reload", 0, 200, "/v1/model/reload", {})
        self.drain()
        self.record_request("unload", 0, 200, "/v1/model/unload", {})
        self.record_request("unloaded-completion", 0, 503, self.completion_path,
                            self.completion_body(1))
        self.record_request("load", 0, 200, "/v1/model/load",
                            {"model": str(pathlib.Path(self.args.model).resolve()),
                             "ctx_size": self.args.ctx_size, "backend": self.args.backend,
                             "weight_quant": "f32", "kv_quant": self.args.kv_quant})
        self.drain()

    def run(self):
        manifest = {"started_utc": stamp(), "binary": str(pathlib.Path(self.args.binary).resolve()),
                    "binary_sha256": sha256(self.args.binary),
                    "harness_sha256": sha256(__file__), "model": str(pathlib.Path(self.args.model).resolve()),
                    "model_sha256": sha256(self.args.model), "arguments": vars(self.args),
                    "effective_cpu_threads": 1,
                    "cpu_thread_policy": "Server CLI exposes no pool control; private CpuBackend default is one. --cpu-threads only accepts 1.",
                    "warning": "Local soak; latency figures are not a controlled throughput benchmark."}
        shutil.copy2(__file__, self.out/"harness.py")
        (self.out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        sampler = None
        fatal = None
        baseline = final = None
        cycle = 0
        workload_wall = 0
        try:
            self.start()
            manifest["server_pid"] = self.process.pid
            manifest["running_binary_sha256"] = sha256(f"/proc/{self.process.pid}/exe")
            manifest["listen_address"] = f"127.0.0.1:{self.port}"
            (self.out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
            if manifest["running_binary_sha256"] != manifest["binary_sha256"]:
                raise RuntimeError("release binary changed between manifest hashing and process startup")
            self.work_started = time.monotonic()
            self.phase = "baseline"
            baseline = self.drain()
            sampler = threading.Thread(target=self.sampling_loop, daemon=True)
            sampler.start()
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                while time.monotonic()-self.work_started < self.args.duration_seconds:
                    for concurrency in self.args.concurrency:
                        self.phase = f"cycle-{cycle}-concurrency-{concurrency}"
                        jobs = []
                        for index in range(concurrency):
                            mode = "sse" if index % 2 else "json"
                            # Each concurrency level covers both formats over consecutive cycles.
                            if cycle % 2:
                                mode = "sse" if mode == "json" else "json"
                            payload = self.completion_body(self.args.tokens, mode == "sse")
                            jobs.append(pool.submit(self.record_request, mode, concurrency, 200,
                                                    self.completion_path, payload, mode))
                        for job in jobs:
                            job.result()
                        self.record_request("cancel", concurrency, 200, self.completion_path,
                            self.completion_body(self.args.cancel_tokens, True), "cancel")
                        self.record_request("slow", concurrency, 200, self.completion_path,
                            self.completion_body(self.args.slow_tokens, True), "slow")
                        self.drain()
                        if self.process.poll() is not None:
                            raise RuntimeError("server exited during workload")
                    self.record_request("wrong-model", 0, 400, self.completion_path,
                                        self.completion_body(1, prompt="x", model="not-installed-soak"))
                    self.record_request("invalid-budget", 0, 400, self.completion_path,
                                        self.completion_body(2**64-1, prompt="x"))
                    self.record_request("unauthorized", 0, 401, "/v1/models", auth=False)
                    cycle += 1
                    print(f"[{stamp()}] cycles={cycle} elapsed={time.monotonic()-self.work_started:.1f}s "
                          f"requests={len(self.records)} unexpected_failures={len(self.failures)}", flush=True)
                    if cycle % self.args.lifecycle_every == 0:
                        self.lifecycle(cycle)
                    self.stop.wait(self.args.cycle_pause)
            workload_wall = time.monotonic()-self.work_started
            # Always include at least one complete lifecycle even in a short pilot.
            if cycle < self.args.lifecycle_every:
                self.lifecycle(cycle)
            self.phase = "final-drain"
            final = self.drain()
        except (Exception, KeyboardInterrupt) as error:
            fatal = f"{type(error).__name__}: {error}"
        finally:
            if self.work_started is not None and not workload_wall:
                workload_wall = time.monotonic()-self.work_started
            self.stop.set()
            if sampler:
                sampler.join(timeout=5)
            if self.process is not None and self.process.poll() is None:
                # Popen identifies only the child started above. Never pkill/killall or a process group.
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            self.server_log.close()
            self.request_log.close()
            self.sample_log.close()
        summary = self.summary(manifest, baseline, final, fatal, workload_wall, cycle)
        (self.out/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n")
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        return 0 if summary["healthy_run"] else 1

    def summary(self, manifest, baseline, final, fatal, wall, cycles):
        groups = collections.defaultdict(list)
        ttft = collections.defaultdict(list)
        statuses = collections.Counter()
        for record in self.records:
            group = f"{record['kind']}/c{record['concurrency']}"
            groups[group].append(record["latency_s"])
            statuses[f"{record['kind']}:{record.get('status', 'exception')}"] += 1
            if record.get("ttft_s") is not None:
                ttft[group].append(record["ttft_s"])
        rss = [s["VmRSS"] for s in self.samples if "VmRSS" in s]
        swaps = [s["VmSwap"] for s in self.samples if "VmSwap" in s]
        rss_swaps = [s["VmRSS"] + s["VmSwap"] for s in self.samples if "VmRSS" in s and "VmSwap" in s]
        gpu = [s["own_pid_gpu_memory_mib"] for s in self.samples if isinstance(s.get("own_pid_gpu_memory_mib"), int)]
        sample_errors = [s for s in self.samples if "error" in s]
        runtime_final = (final or {}).get("metrics", {}).get("runtime", {})
        lifecycle_count = sum(r["kind"] == "load" and r["ok"] for r in self.records)
        counters = runtime_final or next((s["metrics"]["runtime"] for s in reversed(self.samples)
                                         if "runtime" in s.get("metrics", {})), {})
        cancellation_count = counters.get("cancelled_total", 0)
        nonempty = 0
        for record in self.records:
            value = record.get("value") or {}
            if record.get("ok") and record["kind"] in ("json", "sse", "slow"):
                choices = value.get("choices", [])
                text = (choices[0].get("text", "") + choices[0].get("message", {}).get("content", "")) if choices else ""
                nonempty += bool(text or value.get("text_chars", 0))
        coverage = {str(c): {kind: sum(r["kind"] == kind and r["concurrency"] == c and r.get("ok", False)
                                     for r in self.records)
                            for kind in ("json", "sse", "slow", "cancel")}
                    for c in (1, 2, 4, 8)}
        healthy = (fatal is None and not self.failures and final is not None
                   and not sample_errors and lifecycle_count > 0 and cancellation_count > 0 and nonempty > 0)
        full_coverage = all(all(count > 0 for count in kinds.values()) for kinds in coverage.values())
        return {"finished_utc": stamp(), "workload_wall_seconds": wall, "cycles": cycles,
                "requested_duration_seconds": self.args.duration_seconds,
                "minimum_30_minutes_met": wall >= 1800,
                "healthy_run": healthy,
                "a9_duration_and_protocol_pass": healthy and wall >= 1800 and full_coverage and lifecycle_count >= 2,
                "leak_verdict": "No automatic no-leak claim: inspect steady drained RSS/VRAM series; healthy_run checks protocol/cleanup only.",
                "coverage": coverage, "nonempty_successful_completions": nonempty,
                "fatal_error": fatal, "unexpected_failures": self.failures,
                "sample_error_count": len(sample_errors), "sample_errors": sample_errors[:10],
                "binary_sha256": manifest["binary_sha256"],
                "request_count": len(self.records), "status_counts": dict(statuses),
                "latency_seconds": {k: quantiles(v) for k,v in sorted(groups.items())},
                "ttft_seconds": {k: quantiles(v) for k,v in sorted(ttft.items())},
                "rss_kib": {"min": min(rss, default=None), "max": max(rss, default=None),
                            "baseline": (baseline or {}).get("VmRSS"), "final": (final or {}).get("VmRSS")},
                "swap_kib": {"min": min(swaps, default=None), "max": max(swaps, default=None),
                             "baseline": (baseline or {}).get("VmSwap"), "final": (final or {}).get("VmSwap")},
                "rss_plus_swap_kib": {"min": min(rss_swaps, default=None), "max": max(rss_swaps, default=None)},
                "own_pid_gpu_memory_mib": {"max": max(gpu, default=None), "final": (final or {}).get("own_pid_gpu_memory_mib")},
                "baseline_drain": baseline, "final_drain": final,
                "successful_lifecycle_loads": lifecycle_count,
                "confirmed_server_cancellations": cancellation_count,
                "note": "Expected 400/401/503 responses are separated by kind; RSS includes allocator retention and is not payload bytes."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration-seconds", type=float, default=1800)
    parser.add_argument("--api-format", choices=("chat", "text"), default="chat")
    parser.add_argument("--backend", choices=("cpu", "cpu-simd", "cuda"), default="cpu")
    parser.add_argument("--ctx-size", type=int, default=256)
    parser.add_argument("--kv-quant", default="none")
    parser.add_argument("--cpu-threads", type=int, default=1,
                        help="Only 1 is supported: the server CLI does not expose its private CPU pool size.")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--cancel-tokens", type=int, default=192)
    parser.add_argument("--slow-tokens", type=int, default=32)
    parser.add_argument("--cancel-after", choices=("headers", "first-event"), default="headers")
    parser.add_argument("--slow-read-delay", type=float, default=.05)
    parser.add_argument("--request-timeout", type=float, default=90)
    parser.add_argument("--startup-timeout", type=float, default=120)
    parser.add_argument("--drain-timeout", type=float, default=90)
    parser.add_argument("--sample-interval", type=float, default=5)
    parser.add_argument("--cycle-pause", type=float, default=1)
    parser.add_argument("--lifecycle-every", type=int, default=2)
    parser.add_argument("--concurrency", type=lambda x:[int(i) for i in x.split(",")], default=[1,2,4,8])
    parser.add_argument("--prompt", default="Explain why the sky appears blue in three short sentences.")
    args = parser.parse_args()
    if args.cpu_threads != 1:
        parser.error("--cpu-threads must be 1; changing RAYON_NUM_THREADS does not control the server's private CPU pool")
    if args.duration_seconds <= 0 or args.sample_interval <= 0 or args.lifecycle_every < 1:
        parser.error("duration/sample interval/lifecycle cadence must be positive")
    if not args.concurrency or any(x not in (1,2,4,8) for x in args.concurrency):
        parser.error("concurrency must select from 1,2,4,8")
    if not pathlib.Path(args.binary).is_file() or not pathlib.Path(args.model).is_file():
        parser.error("binary and model must be existing files")
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    return Soak(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
