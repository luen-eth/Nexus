mod common;

use common::{TinyModel, Value};
use nexus::engine::{InferenceConfig, InferenceEngine};
use nexus::quant::KvQuantType;
use nexus::server::{
    serve, GenerationRequest, PromptInput, RuntimeOptions, SamplingParams, ServerConfig,
    ServerState,
};
use serde_json::{json, Value as Json};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn engine(name: &str, utf8: bool) -> InferenceEngine {
    let mut model = TinyModel::new();
    model.resize_vocab(256);
    model
        .metadata
        .insert("general.name".into(), Value::Text(name.into()));
    model
        .metadata
        .insert("llama.context_length".into(), Value::Uint(512));
    for (name, shape, data) in &mut model.tensors {
        let dim = shape[0] as usize;
        if name == "token_embd.weight" {
            data.fill(0.);
            for row in data.chunks_exact_mut(dim) {
                row[0] = 1.;
            }
            if utf8 {
                data[195 * dim] = 0.;
                data[195 * dim + 1] = 1.;
            }
        } else if name == "output.weight" {
            data.fill(0.);
            if utf8 {
                data[195 * dim] = 2.;
                data[169 * dim + 1] = 2.;
            } else {
                data[b'a' as usize * dim] = 2.;
            }
        } else if !name.contains("norm") {
            data.fill(0.);
        }
    }
    let file = tempfile::NamedTempFile::new().unwrap();
    model.write(file.path());
    let mut engine = InferenceEngine::new(Some(InferenceConfig {
        max_seq_len: 512,
        allow_byte_tokenizer: true,
        kv_quant: KvQuantType::None,
        ..Default::default()
    }));
    engine.load_model(file.path()).unwrap();
    engine
}

struct TestServer {
    base: String,
    state: ServerState,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for TestServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}
impl TestServer {
    async fn new(name: &str, utf8: bool, options: RuntimeOptions) -> Self {
        Self::new_with_config(
            name,
            utf8,
            options,
            ServerConfig {
                api_key: None,
                cors_origin: None,
                rate_limit_per_minute: None,
            },
        )
        .await
    }
    async fn new_with_config(
        name: &str,
        utf8: bool,
        options: RuntimeOptions,
        config: ServerConfig,
    ) -> Self {
        let state = ServerState::new(engine(name, utf8), None, config, options).unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let copy = state.clone();
        let task = tokio::spawn(async move {
            serve(listener, copy).await.unwrap();
        });
        Self { base, state, task }
    }
    async fn post(&self, path: &str, value: Json) -> reqwest::Response {
        reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(3))
            .build()
            .unwrap()
            .post(format!("{}{path}", self.base))
            .header("content-type", "application/json")
            .body(value.to_string())
            .send()
            .await
            .unwrap()
    }
}
async fn body(response: reqwest::Response) -> Json {
    let text = response.text().await.unwrap();
    serde_json::from_str(&text).unwrap_or_else(|err| panic!("{err}: {text}"))
}
fn request(id: &str, tokens: usize) -> GenerationRequest {
    GenerationRequest {
        id: id.into(),
        prompt: PromptInput::Text("x".into()),
        stop: vec![],
        sampling: SamplingParams {
            max_tokens: Some(tokens),
            ..Default::default()
        },
    }
}
async fn until(mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(3);
    while !condition() {
        assert!(
            Instant::now() < deadline,
            "worker did not reach expected state"
        );
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
}
fn options() -> RuntimeOptions {
    RuntimeOptions {
        max_active: 1,
        max_pending: 2,
        stream_buffer: 4,
        stream_stall_timeout: Duration::from_secs(2),
        ..Default::default()
    }
}
fn sse_text(text: &str, chat: bool) -> String {
    assert!(text.lines().any(|line| line == "data: [DONE]"), "{text}");
    text.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|payload| *payload != "[DONE]")
        .map(|payload| {
            let event: Json = serde_json::from_str(payload).unwrap();
            assert!(event.get("error").is_none(), "{event}");
            if chat {
                event["choices"][0]["delta"]["content"]
                    .as_str()
                    .unwrap_or("")
                    .to_owned()
            } else {
                event["choices"][0]["text"]
                    .as_str()
                    .unwrap_or("")
                    .to_owned()
            }
        })
        .collect()
}

#[tokio::test]
async fn unknown_models_are_rejected_before_json_or_sse_generation() {
    let server = TestServer::new("audit-A", false, options()).await;
    for stream in [false, true] {
        let response = server
            .post(
                "/v1/completions",
                json!({
                    "model":"not-installed","prompt":"x","max_tokens":2,"stream":stream
                }),
            )
            .await;
        assert_eq!(response.status(), 400);
        assert!(body(response).await["error"]["message"]
            .as_str()
            .unwrap()
            .contains("not installed"));
        let response = server
            .post(
                "/v1/chat/completions",
                json!({
                    "model":"not-installed","messages":[{"role":"user","content":"x"}],
                    "max_tokens":2,"stream":stream
                }),
            )
            .await;
        assert_eq!(response.status(), 400);
    }
    assert_eq!(server.state.scheduler().stats(), (0, 0, 0));
    let response = server
        .post(
            "/v1/completions",
            json!({"model":"audit-A","prompt":"x","max_tokens":2}),
        )
        .await;
    assert_eq!(response.status(), 200);
    assert_eq!(body(response).await["choices"][0]["text"], "aa");
}

#[tokio::test]
async fn invalid_budgets_and_sampling_leave_the_service_healthy() {
    let server = TestServer::new("audit-A", false, options()).await;
    for extra in [
        json!({"max_tokens":u64::MAX}),
        json!({"max_tokens":512}),
        json!({"temperature":-1}),
        json!({"top_p":0}),
        json!({"top_p":2}),
        json!({"repetition_penalty":0}),
    ] {
        let mut value = json!({"model":"audit-A","prompt":"x"});
        value
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        assert_eq!(server.post("/v1/completions", value).await.status(), 400);
    }
    let response = server
        .post(
            "/v1/completions",
            json!({"model":"audit-A","prompt":"x","max_tokens":2}),
        )
        .await;
    assert_eq!(response.status(), 200);
    assert_eq!(body(response).await["choices"][0]["text"], "aa");
    until(|| server.state.runtime.stats().active == 0).await;
    assert_eq!(server.state.scheduler().kv_page_stats().0, 0);
}

#[tokio::test]
async fn unicode_and_multitoken_stops_agree_in_chat_text_json_and_sse() {
    for utf8 in [false, true] {
        let server = TestServer::new("audit-A", utf8, options()).await;
        for chat in [false, true] {
            let path = if chat {
                "/v1/chat/completions"
            } else {
                "/v1/completions"
            };
            let mut value = if chat {
                json!({"model":"audit-A","messages":[{"role":"user","content":"x"}],"max_tokens":4})
            } else {
                json!({"model":"audit-A","prompt":"x","max_tokens":4})
            };
            let expected = if utf8 { "éé" } else { "" };
            if !utf8 {
                value["stop"] = json!("aa");
            }
            let response = server.post(path, value.clone()).await;
            assert_eq!(response.status(), 200);
            let json_body = body(response).await;
            let text = if chat {
                &json_body["choices"][0]["message"]["content"]
            } else {
                &json_body["choices"][0]["text"]
            };
            assert_eq!(text, expected);
            value["stream"] = json!(true);
            let response = server.post(path, value).await;
            assert_eq!(response.status(), 200);
            assert_eq!(sse_text(&response.text().await.unwrap(), chat), expected);
        }
        // A bad stream must never poison later work.
        assert!(server.state.runtime.is_alive());
    }
}

#[tokio::test]
async fn backpressure_cancellation_and_health_work_on_a_single_async_worker() {
    let server = TestServer::new("audit-A", false, options()).await;
    let runtime = &server.state.runtime;
    let stream = runtime
        .submit_stream_for_model(request("held", 400), "audit-A".into())
        .unwrap();
    until(|| stream.len() >= 3).await; // full enough to pause generation
    let payload = runtime.stats().active_kv_payload_bytes;
    assert!(payload > 0);
    assert!(
        runtime.snapshot().engine_metrics["host_weight_payload_bytes"]
            .as_u64()
            .unwrap()
            > 0
    );
    let queued = runtime
        .submit_for_model(request("queued", 2), "audit-A".into())
        .unwrap();
    let queued2 = runtime
        .submit_for_model(request("queued-2", 2), "audit-A".into())
        .unwrap();
    assert!(runtime.submit(request("overflow", 2)).is_err());
    let health = reqwest::Client::new().get(format!("{}/health", server.base));
    let response = tokio::time::timeout(Duration::from_secs(1), health.send())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(response.status(), 200);
    assert!(stream.len() <= options().stream_buffer);
    drop(queued);
    drop(queued2);
    drop(stream);
    until(|| {
        runtime.stats().active == 0
            && runtime.stats().queued == 0
            && runtime.stats().cancelled_total == 3
    })
    .await;
    assert_eq!(server.state.scheduler().kv_page_stats().0, 0);
    assert_eq!(runtime.stats().active_kv_payload_bytes, 0);
    let complete = runtime
        .submit_for_model(request("next", 2), "audit-A".into())
        .unwrap()
        .await
        .unwrap()
        .unwrap();
    assert_eq!(complete.text, "aa");
}

#[tokio::test]
async fn stalled_stream_releases_the_slot_and_kv_budget() {
    let mut opts = options();
    opts.stream_stall_timeout = Duration::from_millis(20);
    let server = TestServer::new("audit-A", false, opts).await;
    let stream = server
        .state
        .runtime
        .submit_stream(request("held", 400))
        .unwrap();
    until(|| {
        server.state.runtime.stats().cancelled_total == 1
            && server.state.runtime.stats().active == 0
    })
    .await;
    assert_eq!(server.state.runtime.stats().active, 0);
    assert_eq!(server.state.scheduler().kv_page_stats().0, 0);
    assert!(stream.len() <= 4);
}

#[tokio::test]
async fn queued_model_id_is_rechecked_after_an_atomic_replacement() {
    let server = TestServer::new("audit-A", false, options()).await;
    let runtime = server.state.runtime.clone();
    let held = runtime
        .submit_stream_for_model(request("held", 400), "audit-A".into())
        .unwrap();
    until(|| held.len() >= 3).await;
    let queued = runtime
        .submit_for_model(request("queued-old", 2), "audit-A".into())
        .unwrap();
    let replacement = engine("audit-B", false);
    let swap_runtime = Arc::clone(&runtime);
    let install =
        tokio::spawn(async move { swap_runtime.install(replacement, None).await.unwrap() });
    until(|| runtime.stats().model_swap_pending).await;
    assert_eq!(runtime.snapshot().name, "audit-A");
    drop(held);
    let installed = install.await.unwrap();
    assert_eq!(installed.name, "audit-B");
    let err = queued.await.unwrap().unwrap_err();
    assert!(err.message.contains("not installed"));
    let output = runtime
        .submit_for_model(request("new", 2), "audit-B".into())
        .unwrap()
        .await
        .unwrap()
        .unwrap();
    assert_eq!(output.model, "audit-B");
    assert_eq!(output.text, "aa");
}

#[tokio::test]
async fn normalized_runtime_and_scheduler_options_agree() {
    let server = TestServer::new(
        "audit-A",
        false,
        RuntimeOptions {
            max_active: 0,
            max_pending: 0,
            stream_buffer: 0,
            kv_page_size_tokens: 0,
            ..Default::default()
        },
    )
    .await;
    assert_eq!(server.state.scheduler().max_batch_size(), 1);
    assert_eq!(server.state.scheduler().page_size_tokens(), 1);
    let output = server
        .state
        .runtime
        .submit_for_model(request("normal", 1), "audit-A".into())
        .unwrap()
        .await
        .unwrap()
        .unwrap();
    assert_eq!(output.text, "a");
}

#[tokio::test]
async fn unsupported_runtime_weight_storage_is_rejected_before_replacing_the_model() {
    let server = TestServer::new("audit-A", false, options()).await;
    let epoch = server.state.runtime.snapshot().epoch;
    let response = server
        .post(
            "/v1/model/load",
            json!({
                "model":"/this-path-is-not-read.gguf", "weight_quant":"q4_k",
            }),
        )
        .await;
    assert_eq!(response.status(), 400);
    assert!(body(response).await["error"]["message"]
        .as_str()
        .unwrap()
        .contains("only f32"));
    assert_eq!(server.state.runtime.snapshot().epoch, epoch);
    assert_eq!(server.state.runtime.snapshot().name, "audit-A");
}

#[tokio::test]
async fn an_empty_configured_api_key_never_authorizes_an_empty_client_key() {
    // `--api-key "$NEXUS_API_KEY"` with an unset variable produces Some("");
    // a client sending an empty key header must not be authorized by it.
    let server = TestServer::new_with_config(
        "audit-A",
        false,
        options(),
        ServerConfig {
            api_key: Some(String::new()),
            cors_origin: None,
            rate_limit_per_minute: None,
        },
    )
    .await;
    let http = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(3))
        .build()
        .unwrap();
    let url = format!("{}/v1/models", server.base);
    let status = |r: reqwest::Response| r.status().as_u16();
    assert_eq!(
        status(http.get(&url).header("x-api-key", "").send().await.unwrap()),
        401
    );
    assert_eq!(
        status(
            http.get(&url)
                .header("authorization", "Bearer ")
                .send()
                .await
                .unwrap()
        ),
        401
    );
    // The public path stays reachable and unauthenticated.
    let health = http
        .get(format!("{}/health", server.base))
        .send()
        .await
        .unwrap();
    assert_eq!(health.status(), 200);
}

#[tokio::test]
async fn bearer_scheme_matching_is_case_insensitive() {
    let server = TestServer::new_with_config(
        "audit-A",
        false,
        options(),
        ServerConfig {
            api_key: Some("s3cret".into()),
            cors_origin: None,
            rate_limit_per_minute: None,
        },
    )
    .await;
    let http = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(3))
        .build()
        .unwrap();
    let url = format!("{}/v1/models", server.base);
    for scheme in ["bearer", "BEARER", "BeArEr"] {
        let response = http
            .get(&url)
            .header("authorization", format!("{scheme} s3cret"))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200, "scheme {scheme}");
    }
    let response = http
        .get(&url)
        .header("authorization", "bearer wrong")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 401);
}

#[tokio::test]
async fn cors_headers_require_an_explicit_origin() {
    // A permissive `*` default made the no-auth loopback server readable
    // from any web page in browsers that do not enforce Private Network
    // Access; without NEXUS_CORS_ORIGIN there must be no CORS headers at all.
    let server = TestServer::new("audit-A", false, options()).await;
    let http = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(3))
        .build()
        .unwrap();
    let response = http
        .get(format!("{}/v1/models", server.base))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert!(response
        .headers()
        .get("access-control-allow-origin")
        .is_none());
    let response = http
        .request(
            reqwest::Method::OPTIONS,
            format!("{}/v1/chat/completions", server.base),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 204);
    assert!(
        response
            .headers()
            .get("access-control-allow-origin")
            .is_none(),
        "an unconfigured preflight must not invite browsers in"
    );

    // An explicitly configured origin still gets full CORS support.
    let server = TestServer::new_with_config(
        "audit-A",
        false,
        options(),
        ServerConfig {
            api_key: None,
            cors_origin: Some("https://example.test".into()),
            rate_limit_per_minute: None,
        },
    )
    .await;
    let response = http
        .get(format!("{}/v1/models", server.base))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(
        response.headers()["access-control-allow-origin"],
        "https://example.test"
    );
}

#[tokio::test]
async fn fifo_model_paths_are_rejected_before_wedging_the_loader() {
    // Opening a FIFO blocks the loader thread forever and wedges every
    // model-lifecycle endpoint behind the lifecycle guard; the precheck must
    // reject it before enqueueing.
    let server = TestServer::new("audit-A", false, options()).await;
    let dir = tempfile::tempdir().unwrap();
    let fifo = dir.path().join("trap.fifo");
    let status = std::process::Command::new("mkfifo")
        .arg(&fifo)
        .status()
        .unwrap();
    assert!(status.success(), "this test requires mkfifo");
    let response = server
        .post(
            "/v1/model/load",
            json!({"model": fifo.display().to_string()}),
        )
        .await;
    assert_eq!(response.status(), 400);
    assert!(body(response).await["error"]["message"]
        .as_str()
        .unwrap()
        .contains("regular file"));
    // The lifecycle and the serving model stay healthy afterwards.
    let response = server
        .post(
            "/v1/completions",
            json!({"model":"audit-A","prompt":"x","max_tokens":2}),
        )
        .await;
    assert_eq!(response.status(), 200);
    assert_eq!(body(response).await["choices"][0]["text"], "aa");
}

#[tokio::test]
async fn a_whitespace_api_key_is_never_a_livable_secret() {
    // `--api-key " "` (a pasted space) previously enabled auth with " " as
    // the secret; a client sending a matching whitespace key had to pass.
    for bad_key in ["", " "] {
        let server = TestServer::new_with_config(
            "audit-A",
            false,
            options(),
            ServerConfig {
                api_key: Some(bad_key.into()),
                cors_origin: None,
                rate_limit_per_minute: None,
            },
        )
        .await;
        let http = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(3))
            .build()
            .unwrap();
        let url = format!("{}/v1/models", server.base);
        assert_eq!(
            http.get(&url)
                .header("x-api-key", bad_key)
                .send()
                .await
                .unwrap()
                .status()
                .as_u16(),
            401,
            "key {bad_key:?}"
        );
        assert_eq!(
            http.get(&url)
                .header("authorization", format!("Bearer {bad_key}"))
                .send()
                .await
                .unwrap()
                .status()
                .as_u16(),
            401,
            "key {bad_key:?}"
        );
    }
}

#[tokio::test]
async fn requests_during_a_model_swap_fail_fast_instead_of_hanging() {
    // Measured before the fix: a request addressed during a pending swap
    // hung silently behind the drain of active sequences (12+ s in the
    // smoke audit) with no status, no Retry-After and no deadline.
    let server = TestServer::new("audit-A", false, options()).await;
    let runtime = server.state.runtime.clone();
    let held = runtime.submit_stream(request("held", 400)).unwrap();
    until(|| held.len() >= 3).await;
    let replacement = engine("audit-B", false);
    let swap_runtime = Arc::clone(&runtime);
    let install =
        tokio::spawn(async move { swap_runtime.install(replacement, None).await.unwrap() });
    until(|| runtime.stats().model_swap_pending).await;

    let started = Instant::now();
    let response = server
        .post(
            "/v1/completions",
            json!({"model":"audit-A","prompt":"x","max_tokens":2}),
        )
        .await;
    assert_eq!(response.status(), 503);
    assert_eq!(response.headers()[reqwest::header::RETRY_AFTER], "1");
    let message = body(response).await["error"]["message"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(message.contains("swap"), "message: {message}");
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "the swap window must reject immediately, took {:?}",
        started.elapsed()
    );
    assert_eq!(runtime.stats().queued, 0);

    drop(held);
    install.await.unwrap();
    // After the swap completes the new model serves normally.
    let response = server
        .post(
            "/v1/completions",
            json!({"model":"audit-B","prompt":"x","max_tokens":2}),
        )
        .await;
    assert_eq!(response.status(), 200);
}

#[tokio::test]
async fn oversized_stop_payloads_are_rejected_before_reaching_the_worker() {
    let server = TestServer::new("audit-A", false, options()).await;
    let seventeen_stops: Vec<&str> = vec!["a"; 17];
    let huge_stop = "a".repeat(1025);
    for (stop, detail) in [
        (json!(seventeen_stops), "too many stops"),
        (json!(huge_stop), "too long stop"),
        (json!(["ok", huge_stop]), "too long stop among many"),
    ] {
        let response = server
            .post(
                "/v1/completions",
                json!({"model":"audit-A","prompt":"x","max_tokens":2,"stop":stop}),
            )
            .await;
        assert_eq!(response.status(), 400, "{detail}");
        let response = server
            .post(
                "/v1/chat/completions",
                json!({"model":"audit-A","messages":[{"role":"user","content":"x"}],"max_tokens":2,"stop":stop}),
            )
            .await;
        assert_eq!(response.status(), 400, "{detail}");
    }
    // Rejected payloads never reached the scheduler.
    assert_eq!(server.state.scheduler().stats(), (0, 0, 0));
    // The in-bounds case (16 stops, 1024 bytes) still works.
    let stops: Vec<String> = (0..16).map(|i| format!("stop-{i}")).collect();
    let response = server
        .post(
            "/v1/completions",
            json!({"model":"audit-A","prompt":"x","max_tokens":2,"stop":stops}),
        )
        .await;
    assert_eq!(response.status(), 200);
    assert_eq!(body(response).await["choices"][0]["text"], "aa");
    until(|| server.state.runtime.stats().active == 0).await;
}
