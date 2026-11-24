//! OpenAI-compatible API server for Nexus inference engine.
//!
//! Generation runs on a dedicated worker thread (see [`runtime`]), never on a
//! Tokio worker. Handlers only enqueue jobs and await channels, so `/health`,
//! `/metrics` and model metadata stay responsive while models generate or load.

pub mod runtime;

use axum::{
    body::Body,
    extract::{ConnectInfo, Json, State},
    http::{
        header::{self, HeaderValue},
        Method, Request, StatusCode,
    },
    middleware::{from_fn_with_state, Next},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, options, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::convert::Infallible;
use std::net::{IpAddr, Ipv6Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use uuid::Uuid;

use crate::backend::{BackendFactory, BackendType};
use crate::engine::{InferenceConfig, InferenceEngine};
use crate::quant::{KvQuantType, QuantFormat};
use crate::scheduler::Scheduler;

pub use runtime::{
    ChatTurn, GenerationOutput, GenerationRequest, ModelSnapshot, PromptInput, Runtime,
    RuntimeError, RuntimeErrorKind, RuntimeOptions, RuntimeStats, SamplingParams, StreamEvent,
    DEFAULT_MODEL_NAME,
};

/// API request body (OpenAI-compatible)
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub stop: Option<StopSpec>,
    pub stream: Option<bool>,
}

/// OpenAI-compatible text completion request body.
#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: PromptSpec,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub stop: Option<StopSpec>,
    pub stream: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum PromptSpec {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StopSpec {
    One(String),
    Many(Vec<String>),
}

impl StopSpec {
    fn into_vec(self) -> Vec<String> {
        match self {
            StopSpec::One(stop) => vec![stop],
            StopSpec::Many(stops) => stops,
        }
    }
}

/// Stop-string limits. Stop matching runs on the single generation worker for
/// every decoded token; without a bound, one request could pin that thread
/// with quadratic scans over megabyte-long stop strings.
const MAX_STOP_SEQUENCES: usize = 16;
const MAX_STOP_BYTES: usize = 1024;

fn validate_stops(stops: &[String]) -> Result<(), ApiError> {
    if stops.len() > MAX_STOP_SEQUENCES {
        return Err(ApiError::bad_request(format!(
            "at most {MAX_STOP_SEQUENCES} stop sequences are accepted per request"
        )));
    }
    if let Some(stop) = stops.iter().find(|stop| stop.len() > MAX_STOP_BYTES) {
        return Err(ApiError::bad_request(format!(
            "each stop sequence must be at most {MAX_STOP_BYTES} bytes; this one is {} bytes",
            stop.len()
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// API response (OpenAI-compatible)
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: UsageStats,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: UsageStats,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct UsageStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub api_key: Option<String>,
    pub cors_origin: Option<String>,
    /// Requests per minute per client IP. The client IP is the TCP peer
    /// address; `X-Forwarded-For` and similar headers are ignored because
    /// clients can forge them.
    pub rate_limit_per_minute: Option<usize>,
}

impl ServerConfig {
    pub fn from_env() -> Self {
        let api_key = std::env::var("NEXUS_API_KEY")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        let cors_origin = std::env::var("NEXUS_CORS_ORIGIN")
            .ok()
            .filter(|value| !value.is_empty());
        let rate_limit_per_minute = std::env::var("NEXUS_RATE_LIMIT_PER_MINUTE")
            .ok()
            .and_then(|value| value.parse().ok());

        ServerConfig {
            api_key,
            cors_origin,
            rate_limit_per_minute,
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

#[derive(Debug, Deserialize)]
pub struct ModelLoadRequest {
    pub model: PathBuf,
    pub ctx_size: Option<usize>,
    pub kv_quant: Option<String>,
    pub weight_quant: Option<String>,
    pub backend: Option<String>,
    pub max_loaded_tensor_elements: Option<usize>,
}

/// Server state shared by all handlers.
#[derive(Clone)]
pub struct ServerState {
    pub runtime: Arc<Runtime>,
    pub server_config: Arc<ServerConfig>,
    rate_limiter: Arc<RateLimiter>,
    model_loader: Arc<ModelLoader>,
}

impl ServerState {
    /// Start the generation worker for `engine` and build handler state.
    pub fn new(
        engine: InferenceEngine,
        model_path: Option<PathBuf>,
        server_config: ServerConfig,
        options: RuntimeOptions,
    ) -> std::io::Result<Self> {
        let requested_config = engine.config.clone();
        Self::new_with_requested_config(
            engine,
            model_path,
            server_config,
            options,
            requested_config,
        )
    }

    pub fn new_with_requested_config(
        engine: InferenceEngine,
        model_path: Option<PathBuf>,
        server_config: ServerConfig,
        options: RuntimeOptions,
        requested_config: InferenceConfig,
    ) -> std::io::Result<Self> {
        let model_loader = Arc::new(ModelLoader::start()?);
        let options = options.normalized();
        let scheduler = Arc::new(Scheduler::with_kv_pages(
            options.max_active,
            options.kv_page_size_tokens,
            usize::MAX / 2,
        ));
        let runtime = Runtime::start_with_requested_config(
            engine,
            model_path,
            scheduler,
            options,
            requested_config,
        )?;
        Ok(ServerState {
            runtime: Arc::new(runtime),
            model_loader,
            rate_limiter: Arc::new(RateLimiter::new(
                server_config.rate_limit_per_minute,
                Duration::from_secs(60),
            )),
            server_config: Arc::new(server_config),
        })
    }

    pub fn scheduler(&self) -> &Arc<Scheduler> {
        self.runtime.scheduler()
    }
}

/// One long-lived thread owns model construction work. Tokio's blocking
/// pool expires idle threads; repeatedly loading large models on new allocator
/// arenas can retain substantial anonymous memory after model replacement.
type ModelLoadJob = Box<dyn FnOnce() + Send + 'static>;

struct ModelLoader {
    jobs: tokio::sync::mpsc::Sender<ModelLoadJob>,
    alive: Arc<std::sync::atomic::AtomicBool>,
}

impl ModelLoader {
    fn start() -> std::io::Result<Self> {
        use std::sync::atomic::{AtomicBool, Ordering};
        let (jobs, mut pending) = tokio::sync::mpsc::channel::<ModelLoadJob>(1);
        let alive = Arc::new(AtomicBool::new(true));
        let worker_alive = alive.clone();
        // Deliberately detach: the final Sender drop closes blocking_recv.
        // ServerState destruction must never join a model-loading thread.
        std::thread::Builder::new()
            .name("nexus-model-loader".into())
            .spawn(move || {
                struct Exit(Arc<AtomicBool>);
                impl Drop for Exit {
                    fn drop(&mut self) {
                        self.0.store(false, Ordering::Release);
                    }
                }
                let _exit = Exit(worker_alive);
                while let Some(job) = pending.blocking_recv() {
                    // run() reports normal task panics through its oneshot. The
                    // outer boundary also keeps the worker alive if cleanup panics.
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
                }
            })?;
        Ok(Self { jobs, alive })
    }

    async fn run<T, F>(&self, task: F) -> Result<T, ApiError>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T, ApiError> + Send + 'static,
    {
        use std::sync::atomic::Ordering;
        if !self.alive.load(Ordering::Acquire) {
            return Err(ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "Model loader is not running",
            ));
        }
        let (reply, result) = tokio::sync::oneshot::channel();
        let job: ModelLoadJob = Box::new(move || {
            if reply.is_closed() {
                return;
            } // Skip queued work whose HTTP request went away.
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(task))
                .unwrap_or_else(|panic| {
                    let message = panic
                        .downcast_ref::<&str>()
                        .copied()
                        .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
                        .unwrap_or("unknown panic");
                    Err(ApiError::new(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Model loader panicked: {message}"),
                    ))
                });
            // If the request was cancelled while loading, dropping the unsent
            // engine here frees its buffers on this same persistent thread.
            let _ = reply.send(outcome);
        });
        self.jobs.send(job).await.map_err(|_| {
            ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "Model loader is not running",
            )
        })?;
        result.await.map_err(|_| {
            ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Model loader stopped before replying",
            )
        })?
    }
}

/// Sliding-window rate limiter keyed by client IP (IPv6 grouped by /64).
const MAX_RATE_LIMIT_CLIENTS: usize = 16_384;

struct RateLimiter {
    limit: Option<usize>,
    window: Duration,
    state: Mutex<RateState>,
}

struct RateState {
    buckets: HashMap<IpAddr, VecDeque<Instant>>,
    last_sweep: Instant,
}

impl RateLimiter {
    fn new(limit: Option<usize>, window: Duration) -> Self {
        RateLimiter {
            limit,
            window,
            state: Mutex::new(RateState {
                buckets: HashMap::new(),
                last_sweep: Instant::now(),
            }),
        }
    }

    /// Record a request from `peer` and report whether it is allowed.
    /// Without a known peer address all such requests share one bucket.
    fn check(&self, peer: Option<IpAddr>, now: Instant) -> bool {
        // `0` means "no limit", matching the documented compose guidance
        // (set `NEXUS_RATE_LIMIT_PER_MINUTE=0` when a fronting proxy enforces
        // its own limit); the previous hard deny-all made the documented
        // value lock every client out.
        let Some(limit) = self.limit.filter(|limit| *limit > 0) else {
            return true;
        };
        let key = peer
            .map(rate_limit_key)
            .unwrap_or(IpAddr::V6(Ipv6Addr::UNSPECIFIED));
        let window = self.window;
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());

        // Evict idle clients so the table does not grow without bound.
        if now.duration_since(state.last_sweep) >= window {
            state.buckets.retain(|_, bucket| {
                bucket
                    .back()
                    .is_some_and(|seen| now.duration_since(*seen) < window)
            });
            state.last_sweep = now;
        }

        if !state.buckets.contains_key(&key) && state.buckets.len() >= MAX_RATE_LIMIT_CLIENTS {
            return false; // Fail closed for a new identity until idle entries expire.
        }
        let bucket = state.buckets.entry(key).or_default();
        while bucket
            .front()
            .is_some_and(|seen| now.duration_since(*seen) >= window)
        {
            bucket.pop_front();
        }
        if bucket.len() >= limit {
            return false;
        }
        bucket.push_back(now);
        true
    }

    #[cfg(test)]
    fn tracked_clients(&self) -> usize {
        self.state.lock().unwrap().buckets.len()
    }
}

fn rate_limit_key(ip: IpAddr) -> IpAddr {
    match ip.to_canonical() {
        IpAddr::V4(v4) => IpAddr::V4(v4),
        IpAddr::V6(v6) => {
            // One client usually controls a whole /64.
            let s = v6.segments();
            IpAddr::V6(Ipv6Addr::new(s[0], s[1], s[2], s[3], 0, 0, 0, 0))
        }
    }
}

/// JSON error response.
#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        ApiError {
            status,
            message: message.into(),
        }
    }

    fn bad_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, message)
    }
}

impl ApiError {
    pub fn status(&self) -> StatusCode {
        self.status
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ApiError {}

impl From<RuntimeError> for ApiError {
    fn from(err: RuntimeError) -> Self {
        ApiError::new(err.status_code(), err.message)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        status_error(self.status, &self.message)
    }
}

/// Create the API router. Serve it with
/// `into_make_service_with_connect_info::<SocketAddr>()` so rate limiting can
/// see the peer address (see [`serve`]).
pub fn create_router(state: ServerState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/backends", get(list_backends))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/model", get(model_info))
        .route("/v1/model/load", post(load_model))
        .route("/v1/model/reload", post(reload_model))
        .route("/v1/model/unload", post(unload_model))
        .route("/v1/models", get(list_models))
        .route("/*path", options(preflight))
        .layer(from_fn_with_state(state.clone(), request_middleware))
        .with_state(state)
}

/// Serve the API on an already-bound listener, exposing peer addresses.
/// Runs until the process is torn down (tests use this form).
pub async fn serve(listener: tokio::net::TcpListener, state: ServerState) -> std::io::Result<()> {
    serve_with_shutdown(listener, state, std::future::pending::<()>()).await
}

/// Serve until `shutdown` resolves: the listener stops accepting, idle
/// connections close, and in-flight requests get the axum grace window to
/// complete. A SIGTERM can therefore drain instead of cutting every stream
/// mid-token.
pub async fn serve_with_shutdown(
    listener: tokio::net::TcpListener,
    state: ServerState,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    let app = create_router(state);
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown)
    .await
}

async fn request_middleware(
    State(state): State<ServerState>,
    req: Request<Body>,
    next: Next,
) -> Response {
    if req.method() == Method::OPTIONS {
        let mut response = StatusCode::NO_CONTENT.into_response();
        add_cors_headers(&state, &mut response);
        return response;
    }

    if !is_public_path(req.uri().path()) {
        let peer = req
            .extensions()
            .get::<ConnectInfo<SocketAddr>>()
            .map(|info| info.0.ip());
        if !state.rate_limiter.check(peer, Instant::now()) {
            let mut response = status_error(StatusCode::TOO_MANY_REQUESTS, "Rate limit exceeded");
            add_cors_headers(&state, &mut response);
            return response;
        }
        if let Some(api_key) = &state.server_config.api_key {
            if !authorized(&req, api_key) {
                let mut response = status_error(StatusCode::UNAUTHORIZED, "Unauthorized");
                add_cors_headers(&state, &mut response);
                return response;
            }
        }
    }

    let mut response = next.run(req).await;
    add_cors_headers(&state, &mut response);
    response
}

async fn preflight(State(state): State<ServerState>) -> Response {
    let mut response = StatusCode::NO_CONTENT.into_response();
    add_cors_headers(&state, &mut response);
    response
}

fn is_public_path(path: &str) -> bool {
    path == "/health"
}

fn authorized(req: &Request<Body>, api_key: &str) -> bool {
    if api_key.trim().is_empty() {
        // Defense in depth: an empty or whitespace-only configured secret
        // must never authorize a matching client key, even if a caller
        // bypassed startup validation.
        return false;
    }
    // RFC 7235: the auth scheme is case-insensitive (`bearer`, `Bearer`, ...).
    // Credentials are trimmed: clients that paste keys with stray
    // whitespace must not be locked out by an invisible trailing space.
    let bearer = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| {
            let (scheme, credentials) = value.split_once(char::is_whitespace)?;
            scheme
                .eq_ignore_ascii_case("bearer")
                .then(|| credentials.trim())
        });
    let x_api_key = req
        .headers()
        .get("x-api-key")
        .and_then(|value| value.to_str().ok())
        .map(str::trim);
    bearer
        .or(x_api_key)
        .is_some_and(|candidate| constant_time_eq(candidate.as_bytes(), api_key.as_bytes()))
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    let mut diff = a.len() ^ b.len();
    for i in 0..a.len().max(b.len()) {
        let x = a.get(i).copied().unwrap_or(0);
        let y = b.get(i).copied().unwrap_or(0);
        diff |= usize::from(x ^ y);
    }
    diff == 0
}

fn add_cors_headers(state: &ServerState, response: &mut Response) {
    // Emit CORS headers only when an operator explicitly configured an
    // origin. A permissive `*` default lets any web page drive a no-auth
    // loopback server from browsers that do not enforce Private/Local
    // Network Access: silent compute theft, model unload, remote path
    // probing. Native clients (CLI, desktop app) never need CORS.
    let Some(origin) = state.server_config.cors_origin.as_deref() else {
        return;
    };
    if let Ok(value) = HeaderValue::from_str(origin) {
        response
            .headers_mut()
            .insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, value);
    }
    response.headers_mut().insert(
        header::ACCESS_CONTROL_ALLOW_METHODS,
        HeaderValue::from_static("GET,POST,OPTIONS"),
    );
    response.headers_mut().insert(
        header::ACCESS_CONTROL_ALLOW_HEADERS,
        HeaderValue::from_static("authorization,content-type,x-api-key"),
    );
}

fn status_error(status: StatusCode, message: &str) -> Response {
    let kind = match status {
        StatusCode::UNAUTHORIZED => "authentication_error",
        StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
        s if s.is_client_error() => "invalid_request_error",
        _ => "server_error",
    };
    let mut response = (
        status,
        Json(serde_json::json!({
            "error": {
                "message": message,
                "type": kind
            }
        })),
    )
        .into_response();
    // OpenAI SDKs back off on 429/503 much more smoothly with a hint.
    if status == StatusCode::TOO_MANY_REQUESTS || status == StatusCode::SERVICE_UNAVAILABLE {
        response
            .headers_mut()
            .insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
    }
    response
}

fn sampling_params(
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    repetition_penalty: Option<f32>,
    seed: Option<u64>,
) -> SamplingParams {
    let defaults = SamplingParams::default();
    SamplingParams {
        max_tokens,
        temperature: temperature.unwrap_or(defaults.temperature),
        top_p: top_p.unwrap_or(defaults.top_p),
        top_k: top_k.unwrap_or(defaults.top_k),
        repetition_penalty: repetition_penalty.unwrap_or(defaults.repetition_penalty),
        seed,
    }
}

fn now_unix() -> u64 {
    chrono::Utc::now().timestamp().max(0) as u64
}

async fn chat_completions(
    State(state): State<ServerState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    if req.messages.is_empty() {
        return Err(ApiError::bad_request("messages cannot be empty"));
    }
    let request = GenerationRequest {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        prompt: PromptInput::Chat(
            req.messages
                .into_iter()
                .map(|message| ChatTurn {
                    role: message.role,
                    content: message.content,
                })
                .collect(),
        ),
        stop: {
            let stops = req.stop.map(StopSpec::into_vec).unwrap_or_default();
            validate_stops(&stops)?;
            stops
        },
        sampling: sampling_params(
            req.max_tokens,
            req.temperature,
            req.top_p,
            req.top_k,
            req.repetition_penalty,
            req.seed,
        ),
    };
    let created = now_unix();

    if req.stream.unwrap_or(false) {
        return stream_response(&state, request, req.model, created, StreamKind::Chat).await;
    }

    let id = request.id.clone();
    let rx = state.runtime.submit_for_model(request, req.model)?;
    let output = await_output(rx).await?;
    Ok(Json(ChatCompletionResponse {
        id,
        object: "chat.completion".to_string(),
        created,
        model: output.model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: output.text,
            },
            finish_reason: output.finish_reason,
        }],
        usage: UsageStats {
            prompt_tokens: output.prompt_tokens,
            completion_tokens: output.completion_tokens,
            total_tokens: output.prompt_tokens + output.completion_tokens,
        },
    })
    .into_response())
}

async fn completions(
    State(state): State<ServerState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
    let prompts = match req.prompt {
        PromptSpec::One(prompt) => vec![prompt],
        PromptSpec::Many(prompts) => prompts,
    };
    if prompts.is_empty() {
        return Err(ApiError::bad_request("prompt cannot be empty"));
    }
    let max_prompts = state.runtime.options().max_pending.max(1);
    if prompts.len() > max_prompts {
        return Err(ApiError::bad_request(format!(
            "at most {max_prompts} prompts are accepted per request"
        )));
    }
    let stop = req.stop.map(StopSpec::into_vec).unwrap_or_default();
    validate_stops(&stop)?;
    let sampling = sampling_params(
        req.max_tokens,
        req.temperature,
        req.top_p,
        req.top_k,
        req.repetition_penalty,
        req.seed,
    );
    let request_id = format!("cmpl-{}", Uuid::new_v4());
    let created = now_unix();

    if req.stream.unwrap_or(false) {
        if prompts.len() != 1 {
            return Err(ApiError::bad_request(
                "stream=true only supports a single prompt",
            ));
        }
        let request = GenerationRequest {
            id: request_id,
            prompt: PromptInput::Text(prompts.into_iter().next().unwrap_or_default()),
            stop,
            sampling,
        };
        return stream_response(&state, request, req.model, created, StreamKind::Completion).await;
    }

    // Every prompt becomes its own sequence; they decode concurrently.
    // Dropping the receivers (on early return or client disconnect) cancels them.
    let multi = prompts.len() > 1;
    let mut receivers = Vec::with_capacity(prompts.len());
    for (index, prompt) in prompts.into_iter().enumerate() {
        let id = if multi {
            format!("{request_id}-{index}")
        } else {
            request_id.clone()
        };
        receivers.push(state.runtime.submit_for_model(
            GenerationRequest {
                id,
                prompt: PromptInput::Text(prompt),
                stop: stop.clone(),
                sampling: sampling.clone(),
            },
            req.model.clone(),
        )?);
    }

    let results = futures_util::future::join_all(receivers.into_iter().map(await_output)).await;
    let mut outputs = Vec::with_capacity(results.len());
    for result in results {
        outputs.push(result?);
    }
    let model = outputs[0].model.clone();
    if outputs.iter().any(|output| output.model != model) {
        return Err(ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "The model changed while this request was being served; retry",
        ));
    }

    let prompt_tokens: usize = outputs.iter().map(|o| o.prompt_tokens).sum();
    let completion_tokens: usize = outputs.iter().map(|o| o.completion_tokens).sum();
    let choices = outputs
        .into_iter()
        .enumerate()
        .map(|(index, output)| CompletionChoice {
            index,
            text: output.text,
            finish_reason: output.finish_reason,
        })
        .collect();

    Ok(Json(CompletionResponse {
        id: request_id,
        object: "text_completion".to_string(),
        created,
        model,
        choices,
        usage: UsageStats {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
    .into_response())
}

async fn await_output(
    rx: tokio::sync::oneshot::Receiver<Result<GenerationOutput, RuntimeError>>,
) -> Result<GenerationOutput, ApiError> {
    match rx.await {
        Ok(result) => result.map_err(ApiError::from),
        Err(_) => Err(ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "Generation worker stopped",
        )),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StreamKind {
    Chat,
    Completion,
}

struct ChunkContext {
    kind: StreamKind,
    id: String,
    created: u64,
    model: String,
}

impl ChunkContext {
    fn chunk(
        &self,
        text: Option<&str>,
        finish_reason: Option<&str>,
        usage: Option<UsageStats>,
    ) -> String {
        let mut chunk = match self.kind {
            StreamKind::Chat => {
                let delta = match text {
                    Some(text) => serde_json::json!({ "content": text }),
                    None => serde_json::json!({}),
                };
                serde_json::json!({
                    "id": self.id,
                    "object": "chat.completion.chunk",
                    "created": self.created,
                    "model": self.model,
                    "choices": [{
                        "index": 0,
                        "delta": delta,
                        "finish_reason": finish_reason
                    }]
                })
            }
            StreamKind::Completion => serde_json::json!({
                "id": self.id,
                "object": "text_completion.chunk",
                "created": self.created,
                "model": self.model,
                "choices": [{
                    "index": 0,
                    "text": text.unwrap_or(""),
                    "finish_reason": finish_reason
                }]
            }),
        };
        if let Some(usage) = usage {
            chunk["usage"] = serde_json::json!(usage);
        }
        chunk.to_string()
    }

    fn role_chunk(&self) -> String {
        serde_json::json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": { "role": "assistant" },
                "finish_reason": null
            }]
        })
        .to_string()
    }
}

fn error_event(message: &str) -> String {
    serde_json::json!({
        "error": {
            "message": message,
            "type": "generation_error"
        }
    })
    .to_string()
}

/// Queue a streaming job and wait for admission, so admission failures (busy,
/// invalid prompt, no model) are reported with a proper HTTP status.
async fn stream_response(
    state: &ServerState,
    request: GenerationRequest,
    expected_model: String,
    created: u64,
    kind: StreamKind,
) -> Result<Response, ApiError> {
    let id = request.id.clone();
    let mut rx = state
        .runtime
        .submit_stream_for_model(request, expected_model)?;
    let model = match rx.recv().await {
        Some(StreamEvent::Started { model, .. }) => model,
        Some(StreamEvent::Error(err)) => return Err(err.into()),
        Some(_) | None => {
            return Err(ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "Generation worker stopped",
            ))
        }
    };

    let ctx = ChunkContext {
        kind,
        id,
        created,
        model,
    };
    let mut pending = VecDeque::new();
    if kind == StreamKind::Chat {
        pending.push_back(ctx.role_chunk());
    }

    struct StreamState {
        rx: tokio::sync::mpsc::Receiver<StreamEvent>,
        ctx: ChunkContext,
        pending: VecDeque<String>,
        done: bool,
    }

    // The body pulls from the bounded channel only as fast as the client
    // reads; when the client disconnects the receiver is dropped and the
    // worker cancels the sequence.
    let events = futures_util::stream::unfold(
        StreamState {
            rx,
            ctx,
            pending,
            done: false,
        },
        |mut st| async move {
            loop {
                if let Some(data) = st.pending.pop_front() {
                    return Some((Ok::<_, Infallible>(Event::default().data(data)), st));
                }
                if st.done {
                    return None;
                }
                match st.rx.recv().await {
                    Some(StreamEvent::Delta(text)) => {
                        st.pending.push_back(st.ctx.chunk(Some(&text), None, None));
                    }
                    Some(StreamEvent::Finished {
                        finish_reason,
                        prompt_tokens,
                        completion_tokens,
                    }) => {
                        st.pending.push_back(st.ctx.chunk(
                            None,
                            Some(finish_reason.as_str()),
                            Some(UsageStats {
                                prompt_tokens,
                                completion_tokens,
                                total_tokens: prompt_tokens + completion_tokens,
                            }),
                        ));
                        st.pending.push_back("[DONE]".to_string());
                        st.done = true;
                    }
                    Some(StreamEvent::Error(err)) => {
                        st.pending.push_back(error_event(&err.message));
                        st.pending
                            .push_back(st.ctx.chunk(None, Some("error"), None));
                        st.pending.push_back("[DONE]".to_string());
                        st.done = true;
                    }
                    Some(StreamEvent::Started { .. }) => {}
                    None => {
                        st.pending
                            .push_back(error_event("Generation terminated before completion"));
                        st.pending.push_back("[DONE]".to_string());
                        st.done = true;
                    }
                }
            }
        },
    );
    Ok(Sse::new(events)
        .keep_alive(KeepAlive::default())
        .into_response())
}

/// Load and validate a replacement off the reactor. CPU and optional hybrid
/// CUDA execution are supported; failed replacements preserve the old model.
async fn build_engine(
    loader: &ModelLoader,
    path: PathBuf,
    config: InferenceConfig,
) -> Result<InferenceEngine, ApiError> {
    loader.run(move || {
        let mut engine = InferenceEngine::new(Some(config));
        engine
            .load_model(&path)
            .map_err(|err| ApiError::bad_request(format!("Failed to load model: {err}")))?;
        if engine.model.is_none() {
            return Err(ApiError::bad_request(
                "Failed to load model: no weights were loaded",
            ));
        }
        if !matches!(
            engine.backend.backend_type(),
            BackendType::CpuSimd | BackendType::Cuda
        ) {
            return Err(ApiError::bad_request(format!(
                "Backend '{}' is not implemented for generation; use cpu or enable the cuda feature for CUDA inference",
                engine.backend.name()
            )));
        }
        Ok(engine)
    }).await
}

fn lifecycle_response(status: &str, snapshot: &ModelSnapshot) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": status,
        "id": snapshot.name,
        "path": snapshot.path,
        "loaded": snapshot.loaded,
        "backend": snapshot.backend.name(),
        "max_seq_len": snapshot.max_seq_len,
        "epoch": snapshot.epoch,
    }))
}

/// Load a model. The new engine is built off the reactor; it replaces the
/// current one only after it loaded successfully (a failed load keeps the
/// existing model) and only once in-flight sequences on the old model finish.
/// Model paths cross a trust boundary: opening a FIFO blocks the loader
/// thread forever (wedging every model-lifecycle endpoint behind
/// `lifecycle_guard`), and other special files fail with confusing
/// mid-parse errors. Only regular files (symlinks to them included) may be
/// enqueued to the loader.
fn ensure_regular_model_file(path: &std::path::Path) -> Result<(), ApiError> {
    let metadata = std::fs::metadata(path)
        .map_err(|err| ApiError::bad_request(format!("Model file is not accessible: {err}")))?;
    if !metadata.is_file() {
        return Err(ApiError::bad_request(
            "Model path must be a regular file; FIFOs and devices are rejected",
        ));
    }
    Ok(())
}

async fn load_model(
    State(state): State<ServerState>,
    Json(req): Json<ModelLoadRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _lifecycle = state.runtime.lifecycle_guard().await;
    let config = config_from_load_request(&req, state.runtime.snapshot().requested_config.clone())?;
    // Reject special files before the loader thread ever opens them: a FIFO
    // would block it forever and wedge every lifecycle endpoint.
    ensure_regular_model_file(&req.model)?;
    let engine = build_engine(&state.model_loader, req.model.clone(), config.clone()).await?;
    let snapshot = state
        .runtime
        .install_with_requested_config(engine, Some(req.model), config)
        .await?;
    Ok(lifecycle_response("loaded", &snapshot))
}

async fn reload_model(
    State(state): State<ServerState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _lifecycle = state.runtime.lifecycle_guard().await;
    let current = state.runtime.snapshot();
    let Some(path) = current.path.clone() else {
        return Err(ApiError::bad_request(
            "No previously loaded model path to reload",
        ));
    };
    // The stored path was operator-supplied once; the file behind it may
    // since have been replaced by a FIFO or other special file.
    ensure_regular_model_file(&path)?;
    let config = current.requested_config.clone();
    let engine = build_engine(&state.model_loader, path.clone(), config.clone()).await?;
    let snapshot = state
        .runtime
        .install_with_requested_config(engine, Some(path), config)
        .await?;
    Ok(lifecycle_response("reloaded", &snapshot))
}

async fn unload_model(
    State(state): State<ServerState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _lifecycle = state.runtime.lifecycle_guard().await;
    let config = state.runtime.snapshot().requested_config.clone();
    let empty_config = config.clone();
    let engine = state
        .model_loader
        .run(move || Ok(InferenceEngine::new(Some(empty_config))))
        .await?;
    let snapshot = state
        .runtime
        .install_with_requested_config(engine, None, config)
        .await?;
    Ok(lifecycle_response("unloaded", &snapshot))
}

fn config_from_load_request(
    req: &ModelLoadRequest,
    mut config: InferenceConfig,
) -> Result<InferenceConfig, ApiError> {
    if let Some(ctx_size) = req.ctx_size {
        if ctx_size == 0 {
            return Err(ApiError::bad_request("ctx_size must be greater than 0"));
        }
        config.max_seq_len = ctx_size;
    }
    if let Some(kv_quant) = &req.kv_quant {
        let normalized = kv_quant.to_ascii_lowercase().replace('-', "_");
        config.kv_quant = KvQuantType::from_str(&normalized).ok_or_else(|| {
            ApiError::bad_request(format!("Unsupported KV quantization: {kv_quant}"))
        })?;
    }
    if let Some(weight_quant) = &req.weight_quant {
        let normalized = weight_quant.to_ascii_uppercase().replace('-', "_");
        config.weight_format = QuantFormat::from_str(if normalized == "F32" {
            "FP32"
        } else {
            &normalized
        })
        .ok_or_else(|| {
            ApiError::bad_request(format!(
                "Unsupported runtime weight storage: {weight_quant}"
            ))
        })?;
    }
    if config.weight_format != QuantFormat::FP32 {
        return Err(ApiError::bad_request(
            "Runtime weight storage supports only f32; GGUF file codecs are detected automatically",
        ));
    }
    config.backend = match &req.backend {
        Some(backend) => parse_backend(backend)?,
        None if cfg!(feature = "cuda") && config.backend == BackendType::Cuda => BackendType::Cuda,
        None => BackendType::CpuSimd,
    };
    if let Some(max_loaded_tensor_elements) = req.max_loaded_tensor_elements {
        config.max_loaded_tensor_elements = if max_loaded_tensor_elements == 0 {
            None
        } else {
            Some(max_loaded_tensor_elements)
        };
    }
    Ok(config)
}

/// Unsupported backends are rejected rather than silently falling back.
pub fn parse_backend(value: &str) -> Result<BackendType, ApiError> {
    match value.to_ascii_lowercase().as_str() {
        "auto" | "cpu" | "cpu-simd" | "cpu_simd" => Ok(BackendType::CpuSimd),
        "cuda" if cfg!(feature = "cuda") => Ok(BackendType::Cuda),
        "metal" | "cuda" | "vulkan" | "webgpu" => Err(ApiError::bad_request(format!(
            "Backend '{value}' is not implemented for generation; use cpu or enable the cuda feature for CUDA inference"
        ))),
        other => Err(ApiError::bad_request(format!(
            "Unsupported backend: {other}"
        ))),
    }
}

async fn health(State(state): State<ServerState>) -> Response {
    let snapshot = state.runtime.snapshot();
    let stats = state.runtime.stats();
    let worker_ok = state.runtime.is_alive() && stats.backend_healthy;
    let (status, code) = health_status(worker_ok, snapshot.loaded);
    (
        code,
        Json(serde_json::json!({
            "status": status,
            "model_loaded": snapshot.loaded,
            "model": snapshot.name,
            "active_sequences": stats.active,
            "queued_requests": stats.queued,
        })),
    )
        .into_response()
}

/// Health status label and HTTP code for one snapshot. Code-based probes
/// (Kubernetes, load balancers) must see a dead worker as a failure instead
/// of routing traffic to a 200-with-error-body instance.
fn health_status(worker_ok: bool, loaded: bool) -> (&'static str, StatusCode) {
    if !worker_ok {
        ("error", StatusCode::SERVICE_UNAVAILABLE)
    } else if loaded {
        ("ok", StatusCode::OK)
    } else {
        ("degraded", StatusCode::OK)
    }
}

async fn metrics(State(state): State<ServerState>) -> Json<serde_json::Value> {
    let scheduler = state.scheduler();
    let (pending, running, completed) = scheduler.stats();
    let (kv_pages_used, kv_pages_max) = scheduler.kv_page_stats();
    let stats = state.runtime.stats();
    let snapshot = state.runtime.snapshot();

    Json(serde_json::json!({
        "scheduler": {
            "pending": pending,
            "running": running,
            "completed": completed,
            "kv_pages_used": kv_pages_used,
            "kv_pages_max": kv_pages_max,
            "kv_page_size_tokens": scheduler.page_size_tokens(),
        },
        "runtime": {
            "mode": if snapshot.backend==BackendType::Cuda {"cuda-matvec-cpu-attention"} else {"cpu-token-interleaving"},
            "gpu_kernel_launches":stats.gpu_kernel_launches,
            "backend_healthy":stats.backend_healthy,
            "active_sequences": stats.active,
            "active_kv_payload_bytes": stats.active_kv_payload_bytes,
            "queued_requests": stats.queued,
            "max_active_sequences": stats.max_active,
            "max_queued_requests": stats.max_pending,
            "model_swap_pending": stats.model_swap_pending,
            "cancelled_total": stats.cancelled_total,
            "rejected_busy_total": stats.rejected_busy_total,
            "worker_alive": state.runtime.is_alive(),
        },
        "engine": snapshot.engine_metrics,
        "model": {
            "id": snapshot.name,
            "epoch": snapshot.epoch,
        },
        "server": {
            "auth_enabled": state.server_config.api_key.is_some(),
            "cors_origin": state.server_config.cors_origin.clone(),
            "rate_limit_per_minute": state.server_config.rate_limit_per_minute,
        }
    }))
}

async fn list_backends() -> Result<Json<serde_json::Value>, ApiError> {
    let capabilities = tokio::task::spawn_blocking(BackendFactory::available_backends)
        .await
        .map_err(|e| {
            ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Backend probe failed: {e}"),
            )
        })?;
    let data = capabilities
        .into_iter()
        .map(|capability| {
            serde_json::json!({
                "id": capability.backend.name(),
                "available": capability.available,
                // Whether the server can run generation on this backend today.
                "generation_supported": capability.available && matches!(capability.backend, BackendType::CpuSimd | BackendType::Cuda),
                "features": capability.features,
                "supported_ops": capability.supported_ops,
            })
        })
        .collect::<Vec<_>>();

    Ok(Json(serde_json::json!({
        "object": "list",
        "data": data,
    })))
}

async fn model_info(State(state): State<ServerState>) -> Json<serde_json::Value> {
    let snapshot = state.runtime.snapshot();
    let mut info = snapshot.info.clone();
    if let Some(object) = info.as_object_mut() {
        object.insert("id".to_string(), serde_json::json!(snapshot.name));
        object.insert("path".to_string(), serde_json::json!(snapshot.path));
        object.insert("epoch".to_string(), serde_json::json!(snapshot.epoch));
    }
    Json(info)
}

async fn list_models(State(state): State<ServerState>) -> Json<serde_json::Value> {
    let snapshot = state.runtime.snapshot();
    let data = if snapshot.loaded {
        vec![serde_json::json!({
            "id": snapshot.name,
            "object": "model",
            "created": now_unix(),
            "owned_by": "nexus",
        })]
    } else {
        Vec::new()
    };
    Json(serde_json::json!({
        "object": "list",
        "data": data,
    }))
}

/// Start the API server
pub async fn run_server(engine: InferenceEngine, host: &str, port: u16) -> anyhow::Result<()> {
    run_server_with_config(engine, host, port, ServerConfig::default(), None).await
}

/// Start the API server. Up to `engine.config.batch_size` sequences decode
/// concurrently (see [`RuntimeOptions::from_batch_size`]).
pub async fn run_server_with_config(
    engine: InferenceEngine,
    host: &str,
    port: u16,
    server_config: ServerConfig,
    initial_model_path: Option<PathBuf>,
) -> anyhow::Result<()> {
    let requested_config = engine.config.clone();
    run_server_with_requested_config(
        engine,
        host,
        port,
        server_config,
        initial_model_path,
        requested_config,
    )
    .await
}

/// Serve until the process is asked to stop, then drain within a bounded
/// window. Long SSE streams do not end on their own, so the drain window
/// bounds total shutdown time (docker's `stop_grace_period: 30s` matches
/// the 25-second window with headroom before the runtime escalates).
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut stream) => {
                stream.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
}

pub async fn run_server_with_requested_config(
    engine: InferenceEngine,
    host: &str,
    port: u16,
    server_config: ServerConfig,
    initial_model_path: Option<PathBuf>,
    requested_config: InferenceConfig,
) -> anyhow::Result<()> {
    let options = RuntimeOptions::from_batch_size(engine.config.batch_size);
    let max_active = options.max_active;
    let max_pending = options.max_pending;
    let state = ServerState::new_with_requested_config(
        engine,
        initial_model_path,
        server_config,
        options,
        requested_config,
    )?;
    // Tuple-form bind accepts both IPv4 and IPv6 literals ("::1"), which the
    // previous `format!("{host}:{port}")` string could not represent.
    let display_addr = if host.contains(':') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    };

    println!("Nexus API server starting on {}", display_addr);
    println!(
        "Generation: token-interleaved worker, up to {} concurrent sequences, {} queued",
        max_active, max_pending
    );
    println!("OpenAI-compatible endpoints:");
    println!("  POST {}/v1/chat/completions", display_addr);
    println!("  POST {}/v1/completions", display_addr);
    println!("  GET  {}/v1/models", display_addr);
    println!("  GET  {}/v1/model", display_addr);
    println!("  POST {}/v1/model/load", display_addr);
    println!("  POST {}/v1/model/reload", display_addr);
    println!("  POST {}/v1/model/unload", display_addr);
    println!("  GET  {}/v1/backends", display_addr);
    println!("  GET  {}/health", display_addr);
    println!("  GET  {}/metrics", display_addr);
    println!();

    let listener = tokio::net::TcpListener::bind((host, port)).await?;
    // The drain window must start when the shutdown SIGNAL resolves, never
    // at startup: wrapping the whole serve future in a timeout capped the
    // binary's uptime at the window length (the server self-terminated
    // 25 s after start with exit 0 and no signal anywhere).
    serve_with_bounded_drain(
        listener,
        state,
        shutdown_signal(),
        std::time::Duration::from_secs(25),
    )
    .await?;

    Ok(())
}

/// Serve until `signal` resolves, then bound only the post-signal drain by
/// `window`. Before the signal the server runs without any time cap. After
/// the window elapses the caller's process exit closes the remaining
/// connections (detached tasks survive a dropped serve future; the message
/// below says so honestly instead of implying an active close).
async fn serve_with_bounded_drain(
    listener: tokio::net::TcpListener,
    state: ServerState,
    signal: impl std::future::Future<Output = ()> + Send + 'static,
    window: std::time::Duration,
) -> std::io::Result<()> {
    let (fired_tx, mut fired_rx) = tokio::sync::watch::channel(());
    let shutdown = async move {
        signal.await;
        let _ = fired_tx.send(());
    };
    let mut server = tokio::spawn(serve_with_shutdown(listener, state, shutdown));
    // No cap before the signal: this arm waits for the server to end on
    // its own (rare) while the watch stays silent.
    tokio::select! {
        result = &mut server => {
            return join_serve(result);
        }
        _ = fired_rx.changed() => {}
    }
    bound_post_signal_drain(server, window).await
}

/// Wait for the (already signaled) serve task, bounded by `window`.
async fn bound_post_signal_drain(
    mut server: tokio::task::JoinHandle<std::io::Result<()>>,
    window: std::time::Duration,
) -> std::io::Result<()> {
    match tokio::time::timeout(window, &mut server).await {
        Ok(result) => join_serve(result),
        Err(_) => {
            println!("Shutdown drain window elapsed; process exit closes remaining connections.");
            Ok(())
        }
    }
}

fn join_serve(result: Result<std::io::Result<()>, tokio::task::JoinError>) -> std::io::Result<()> {
    match result {
        Ok(result) => result,
        Err(err) => Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, err)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{GenerationConfig, ModelArch, ModelWeights};
    use crate::gguf::{GgufByteOrder, GgufDataType, GgufValue, GgufValueType, ModelMetadata};
    use crate::memory::UnifiedMemory;
    use crate::quant::QuantEngine;
    use crate::tokenizer::{ByteTokenizer, Tokenizer};
    use std::io::Write;

    /// `reqwest` is built without its `json` feature; minimal equivalents.
    trait JsonBody {
        fn json_body(self, value: &serde_json::Value) -> Self;
    }

    impl JsonBody for reqwest::RequestBuilder {
        fn json_body(self, value: &serde_json::Value) -> Self {
            self.header("content-type", "application/json")
                .body(value.to_string())
        }
    }

    trait JsonResponse {
        async fn json_value(self) -> Result<serde_json::Value, String>;
    }

    impl JsonResponse for reqwest::Response {
        async fn json_value(self) -> Result<serde_json::Value, String> {
            let text = self.text().await.map_err(|e| e.to_string())?;
            serde_json::from_str(&text).map_err(|e| format!("{e}: {text}"))
        }
    }

    // ---------------------------------------------------------------------
    // Deterministic fixture engines (no downloaded models).
    // ---------------------------------------------------------------------

    const VOCAB: usize = 128; // ASCII only: every token decodes to one char.
    const HIDDEN: usize = 8;
    const HEADS: usize = 2;
    const KV_HEADS: usize = 1;
    const HEAD_DIM: usize = 4;
    const FF: usize = 16;

    fn lcg(seed: u64, n: usize, scale: f32) -> Vec<f32> {
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let unit = ((state >> 33) as f32) / ((1u64 << 31) as f32);
                (unit * 2.0 - 1.0) * scale
            })
            .collect()
    }

    fn layered_tensors(seed: u64) -> Vec<(&'static str, Vec<usize>, Vec<f32>)> {
        let kv = KV_HEADS * HEAD_DIM;
        vec![
            (
                "token_embd.weight",
                vec![HIDDEN, VOCAB],
                lcg(seed, VOCAB * HIDDEN, 1.0),
            ),
            ("blk.0.attn_norm.weight", vec![HIDDEN], vec![1.0; HIDDEN]),
            (
                "blk.0.attn_q.weight",
                vec![HIDDEN, HIDDEN],
                lcg(seed + 1, HIDDEN * HIDDEN, 0.8),
            ),
            (
                "blk.0.attn_k.weight",
                vec![HIDDEN, kv],
                lcg(seed + 2, kv * HIDDEN, 0.8),
            ),
            (
                "blk.0.attn_v.weight",
                vec![HIDDEN, kv],
                lcg(seed + 3, kv * HIDDEN, 0.8),
            ),
            (
                "blk.0.attn_output.weight",
                vec![HIDDEN, HIDDEN],
                lcg(seed + 4, HIDDEN * HIDDEN, 0.8),
            ),
            ("blk.0.ffn_norm.weight", vec![HIDDEN], vec![1.0; HIDDEN]),
            (
                "blk.0.ffn_gate.weight",
                vec![HIDDEN, FF],
                lcg(seed + 5, FF * HIDDEN, 0.5),
            ),
            (
                "blk.0.ffn_up.weight",
                vec![HIDDEN, FF],
                lcg(seed + 6, FF * HIDDEN, 0.5),
            ),
            (
                "blk.0.ffn_down.weight",
                vec![FF, HIDDEN],
                lcg(seed + 7, HIDDEN * FF, 0.5),
            ),
            ("output_norm.weight", vec![HIDDEN], vec![1.0; HIDDEN]),
            (
                "output.weight",
                vec![HIDDEN, VOCAB],
                lcg(seed + 8, VOCAB * HIDDEN, 2.0),
            ),
        ]
    }

    fn fixture_config(max_seq_len: usize) -> InferenceConfig {
        InferenceConfig {
            max_seq_len,
            allow_byte_tokenizer: true,
            kv_quant: KvQuantType::None,
            weight_format: QuantFormat::FP32,
            backend: BackendType::CpuSimd,
            ..Default::default()
        }
    }

    fn engine_from(
        name: &str,
        tensors: HashMap<String, Vec<f32>>,
        max_seq_len: usize,
    ) -> InferenceEngine {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.name".to_string(),
            GgufValue::String(name.to_string()),
        );
        let model = ModelWeights {
            metadata: ModelMetadata {
                version: 3,
                tensor_count: tensors.len() as u64,
                kv_count: 1,
                byte_order: GgufByteOrder::LittleEndian,
                alignment: 32,
                tensor_data_offset: 0,
                metadata,
                tensors: Vec::new(),
            },
            arch: ModelArch::Llama,
            tensors,
            quant_engine: QuantEngine::new(QuantFormat::FP32, KvQuantType::None),
            memory: UnifiedMemory::new(),
        };
        let mut engine = InferenceEngine::new(Some(fixture_config(max_seq_len)));
        engine.model = Some(model);
        engine
    }

    /// One transformer layer: outputs depend on attention over the whole
    /// prefix, so any KV cross-talk between sequences changes the text.
    fn layered_fixture(name: &str, seed: u64) -> InferenceEngine {
        let tensors = layered_tensors(seed)
            .into_iter()
            .map(|(name, _, data)| (name.to_string(), data))
            .collect();
        let mut engine = engine_from(name, tensors, 128);
        engine.vocab_size = VOCAB;
        engine.num_layers = 1;
        engine.num_heads = HEADS;
        engine.num_kv_heads = KV_HEADS;
        engine.head_dim = HEAD_DIM;
        engine.hidden_size = HIDDEN;
        engine.ff_dim = FF;
        engine
    }

    /// Layer-free model whose greedy next token is `transitions[token]`
    /// (one-hot embeddings), for exact byte-level scripting.
    fn scripted_fixture(
        name: &str,
        transitions: &[(u8, u8)],
        max_seq_len: usize,
    ) -> InferenceEngine {
        const V: usize = 256;
        let mut embed = vec![0.0; V * V];
        for t in 0..V {
            embed[t * V + t] = 1.0;
        }
        let mut output = vec![0.0; V * V];
        for &(from, to) in transitions {
            output[to as usize * V + from as usize] = 1.0;
        }
        let mut tensors = HashMap::new();
        tensors.insert("token_embd.weight".to_string(), embed);
        tensors.insert("output_norm.weight".to_string(), vec![1.0; V]);
        tensors.insert("output.weight".to_string(), output);
        let mut engine = engine_from(name, tensors, max_seq_len);
        engine.vocab_size = V;
        engine.num_layers = 0;
        engine.num_heads = 1;
        engine.num_kv_heads = 1;
        engine.head_dim = V;
        engine.hidden_size = V;
        engine.ff_dim = V;
        engine
    }

    /// Endless cycle over `text` (ASCII), entered from its first byte.
    fn cycle(text: &str) -> Vec<(u8, u8)> {
        let bytes = text.as_bytes();
        (0..bytes.len())
            .map(|i| (bytes[i], bytes[(i + 1) % bytes.len()]))
            .collect()
    }

    fn options(max_active: usize, max_pending: usize) -> RuntimeOptions {
        RuntimeOptions {
            max_active,
            max_pending,
            stream_buffer: 4,
            stream_stall_timeout: Duration::from_secs(60),
            kv_page_size_tokens: 16,
        }
    }

    fn open_config() -> ServerConfig {
        ServerConfig {
            api_key: None,
            cors_origin: None,
            rate_limit_per_minute: None,
        }
    }

    async fn start(
        engine: InferenceEngine,
        opts: RuntimeOptions,
        config: ServerConfig,
    ) -> (String, ServerState) {
        let state = ServerState::new(engine, None, config, opts).unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(serve(listener, state.clone()));
        (base, state)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn serve_with_shutdown_stops_accepting_on_signal() {
        // SIGTERM previously killed in-flight requests instantly (measured
        // < 1 s in the smoke audit, with no [DONE]); the shutdown future
        // must let the listener drain instead.
        let (base, state) =
            start(layered_fixture("shutdown", 7), options(2, 8), open_config()).await;
        let http = client();
        let health = http.get(format!("{base}/health")).send().await.unwrap();
        assert_eq!(health.status(), 200);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let second = format!("http://{}", listener.local_addr().unwrap());
        let (trigger, shutdown) = tokio::sync::oneshot::channel::<()>();
        let task = tokio::spawn(serve_with_shutdown(listener, state, async move {
            let _ = shutdown.await;
        }));
        let probe = http.get(format!("{second}/health")).send().await.unwrap();
        assert_eq!(probe.status(), 200);
        trigger.send(()).unwrap();
        let outcome = tokio::time::timeout(Duration::from_secs(5), task).await;
        assert!(
            outcome.is_ok(),
            "serve must return after the shutdown signal"
        );
        assert!(outcome.unwrap().unwrap().is_ok());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_drain_window_starts_at_the_signal_not_at_startup() {
        // NEW-10 regression: the bounded-drain wrapper previously wrapped
        // the WHOLE serve future in a timeout, so the binary self-terminated
        // at the window length with exit 0 and no signal anywhere (verified
        // empirically as a 25-second uptime cap on the wire).
        let (_base, state) =
            start(layered_fixture("uptime", 7), options(2, 8), open_config()).await;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let mut task = tokio::spawn(serve_with_bounded_drain(
            listener,
            state,
            std::future::pending::<()>(),
            Duration::from_millis(60),
        ));
        // Well past the 60 ms window: the server must still be serving.
        tokio::time::sleep(Duration::from_millis(200)).await;
        let health = client()
            .get(format!("http://127.0.0.1:{port}/health"))
            .send()
            .await
            .unwrap();
        assert_eq!(health.status(), 200, "the window must not cap uptime");
        // And the serve future itself has not resolved.
        if tokio::time::timeout(Duration::from_millis(50), &mut task)
            .await
            .is_ok()
        {
            panic!("serve resolved before any signal");
        }
        task.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_post_signal_drain_is_bounded_and_a_clean_drain_returns_promptly() {
        // A serve task that ignores the shutdown (a stream held open past
        // the grace) is released by the caller's process exit after the
        // window — never before it, and not forever.
        let never = tokio::spawn(async { std::future::pending::<std::io::Result<()>>().await });
        let started = Instant::now();
        let result = bound_post_signal_drain(never, Duration::from_millis(80)).await;
        assert!(result.is_ok());
        assert!(started.elapsed() >= Duration::from_millis(60));
        assert!(started.elapsed() < Duration::from_secs(5));
        // A serve task that completes drains immediately.
        let done = tokio::spawn(async { Ok::<_, std::io::Error>(()) });
        let result = bound_post_signal_drain(done, Duration::from_secs(30)).await;
        assert!(result.is_ok());
    }

    fn client() -> reqwest::Client {
        reqwest::Client::builder().no_proxy().build().unwrap()
    }

    async fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !cond() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    }

    fn text_request(id: &str, prompt: &str, max_tokens: usize) -> GenerationRequest {
        GenerationRequest {
            id: id.to_string(),
            prompt: PromptInput::Text(prompt.to_string()),
            stop: Vec::new(),
            sampling: SamplingParams {
                max_tokens: Some(max_tokens),
                ..Default::default()
            },
        }
    }

    /// Isolated reference: a fresh fixture engine generating one prompt alone.
    fn reference(
        mut engine: InferenceEngine,
        prompt: &str,
        generation: &GenerationConfig,
    ) -> String {
        let tokens = ByteTokenizer.encode(prompt);
        let output = engine.generate_with_config(&tokens, generation).unwrap();
        ByteTokenizer.decode(&output[tokens.len()..])
    }

    fn sse_payloads(body: &str) -> Vec<String> {
        body.lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(str::to_string)
            .collect()
    }

    async fn collect_stream(mut rx: tokio::sync::mpsc::Receiver<StreamEvent>) -> (String, String) {
        let mut text = String::new();
        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::Delta(delta) => text.push_str(&delta),
                StreamEvent::Finished { finish_reason, .. } => return (text, finish_reason),
                StreamEvent::Error(err) => panic!("stream error: {err}"),
                StreamEvent::Started { .. } => {}
            }
        }
        panic!("stream closed without a terminal event");
    }

    #[test]
    fn model_loader_reuses_thread_after_tokio_blocking_idle_expiry_and_exits_on_drop() {
        let reactor = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .max_blocking_threads(1)
            .thread_keep_alive(Duration::from_millis(10))
            .enable_all()
            .build()
            .unwrap();
        reactor.block_on(async {
            let loader = ModelLoader::start().unwrap();
            let first = loader
                .run(|| Ok(std::thread::current().id()))
                .await
                .unwrap();
            tokio::task::spawn_blocking(|| ()).await.unwrap();
            tokio::time::sleep(Duration::from_millis(60)).await;
            let second = loader
                .run(|| Ok(std::thread::current().id()))
                .await
                .unwrap();
            assert_eq!(
                first, second,
                "loader must not use expiring Tokio blocking threads"
            );
            let alive = loader.alive.clone();
            drop(loader); // No blocking join; the receiver closes by sender ownership.
            tokio::time::timeout(Duration::from_secs(1), async {
                while alive.load(std::sync::atomic::Ordering::Acquire) {
                    tokio::time::sleep(Duration::from_millis(2)).await;
                }
            })
            .await
            .unwrap();
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn model_loader_bounds_queue_skips_cancelled_jobs_and_serializes_execution() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        let loader = Arc::new(ModelLoader::start().unwrap());
        let active = Arc::new(AtomicUsize::new(0));
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let first_loader = loader.clone();
        let active_first = active.clone();
        let first = tokio::spawn(async move {
            first_loader
                .run(move || {
                    assert_eq!(active_first.fetch_add(1, Ordering::SeqCst), 0);
                    let _ = started_tx.send(());
                    release_rx.recv().unwrap();
                    active_first.fetch_sub(1, Ordering::SeqCst);
                    Ok(std::thread::current().id())
                })
                .await
                .unwrap()
        });
        started_rx.await.unwrap();
        let ran_cancelled = Arc::new(AtomicBool::new(false));
        let flag = ran_cancelled.clone();
        let cancel_loader = loader.clone();
        let cancelled = tokio::spawn(async move {
            cancel_loader
                .run(move || {
                    flag.store(true, Ordering::SeqCst);
                    Ok(())
                })
                .await
        });
        wait_until("loader queue full", || loader.jobs.capacity() == 0).await;
        assert_eq!(active.load(Ordering::SeqCst), 1);
        cancelled.abort();
        assert!(cancelled.await.unwrap_err().is_cancelled());
        let next_loader = loader.clone();
        let active_next = active.clone();
        let next = tokio::spawn(async move {
            next_loader
                .run(move || {
                    assert_eq!(active_next.fetch_add(1, Ordering::SeqCst), 0);
                    active_next.fetch_sub(1, Ordering::SeqCst);
                    Ok(std::thread::current().id())
                })
                .await
                .unwrap()
        });
        // The queue has one cancelled job; another sender cannot add a second.
        assert_eq!(loader.jobs.max_capacity(), 1);
        assert!(!next.is_finished());
        release_tx.send(()).unwrap();
        assert_eq!(first.await.unwrap(), next.await.unwrap());
        assert!(!ran_cancelled.load(Ordering::SeqCst));
        assert_eq!(active.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn model_loader_recovers_from_a_job_panic() {
        let loader = ModelLoader::start().unwrap();
        let failed = loader
            .run::<(), _>(|| panic!("controlled loader panic"))
            .await
            .unwrap_err();
        assert_eq!(failed.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert!(failed.message().contains("controlled loader panic"));
        assert_eq!(loader.run(|| Ok(17)).await.unwrap(), 17);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn model_loader_drop_does_not_join_a_running_cancelled_job() {
        let loader = Arc::new(ModelLoader::start().unwrap());
        let alive = loader.alive.clone();
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let borrowed = loader.clone();
        let task = tokio::spawn(async move {
            borrowed
                .run(move || {
                    let _ = started_tx.send(());
                    release_rx.recv().unwrap();
                    Ok(())
                })
                .await
        });
        started_rx.await.unwrap();
        drop(loader);
        task.abort();
        tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .unwrap()
            .unwrap_err();
        assert!(alive.load(std::sync::atomic::Ordering::Acquire));
        release_tx.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            while alive.load(std::sync::atomic::Ordering::Acquire) {
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        })
        .await
        .unwrap();
    }

    // ---------------------------------------------------------------------
    // Serialization / config parsing
    // ---------------------------------------------------------------------

    #[test]
    fn test_request_deserialize() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 100
        }"#;

        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "test");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].content, "hello");
        assert_eq!(req.max_tokens, Some(100));
    }

    #[test]
    fn test_response_serialize() {
        let response = ChatCompletionResponse {
            id: "test-123".to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: "nexus".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: "Hello!".to_string(),
                },
                finish_reason: "stop".to_string(),
            }],
            usage: UsageStats {
                prompt_tokens: 5,
                completion_tokens: 2,
                total_tokens: 7,
            },
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("test-123"));
        assert!(json.contains("Hello!"));
    }

    #[test]
    fn test_completion_request_deserialize() {
        let json = r#"{
            "model": "test",
            "prompt": ["hello", "goodbye"],
            "max_tokens": 8
        }"#;

        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        match req.prompt {
            PromptSpec::Many(prompts) => assert_eq!(prompts.len(), 2),
            PromptSpec::One(_) => panic!("expected prompt array"),
        }
        assert_eq!(req.max_tokens, Some(8));
    }

    #[test]
    fn test_model_load_request_overrides_config() {
        let req: ModelLoadRequest = serde_json::from_str(
            r#"{
                "model": "model.gguf",
                "ctx_size": 8192,
                "kv_quant": "none",
                "weight_quant": "f32",
                "backend": "cpu-simd",
                "max_loaded_tensor_elements": 0
            }"#,
        )
        .unwrap();

        let config = config_from_load_request(&req, InferenceConfig::default()).unwrap();
        assert_eq!(config.max_seq_len, 8192);
        assert_eq!(config.kv_quant, KvQuantType::None);
        assert_eq!(config.weight_format, QuantFormat::FP32);
        assert_eq!(config.backend, BackendType::CpuSimd);
        assert_eq!(config.max_loaded_tensor_elements, None);
    }

    #[test]
    fn load_rejects_unimplemented_runtime_storage_without_restricting_file_codecs() {
        for requested in ["q4_k", "q8_0", "fp16"] {
            let req: ModelLoadRequest = serde_json::from_value(serde_json::json!({
                "model": "a-q4-k.gguf", "weight_quant": requested,
            }))
            .unwrap();
            let error = config_from_load_request(&req, InferenceConfig::default()).unwrap_err();
            assert_eq!(error.status(), StatusCode::BAD_REQUEST);
            assert!(error.message().contains("only f32"));
        }
        for requested in ["f32", "FP32"] {
            let req: ModelLoadRequest = serde_json::from_value(serde_json::json!({
                "model": "a-q4-k.gguf", "weight_quant": requested,
            }))
            .unwrap();
            assert_eq!(
                config_from_load_request(&req, InferenceConfig::default())
                    .unwrap()
                    .weight_format,
                QuantFormat::FP32
            );
        }
    }

    #[test]
    fn test_parse_unimplemented_backend_errors() {
        for backend in ["metal", "vulkan", "webgpu"] {
            let err = parse_backend(backend).unwrap_err();
            assert_eq!(err.status(), StatusCode::BAD_REQUEST);
            assert!(err.message().contains("not implemented"));
        }
        if cfg!(feature = "cuda") {
            assert_eq!(parse_backend("cuda").unwrap(), BackendType::Cuda);
        } else {
            assert!(parse_backend("cuda").is_err());
        }
        assert_eq!(parse_backend("auto").unwrap(), BackendType::CpuSimd);
        // Omitting the backend never inherits a GPU default.
        let req: ModelLoadRequest = serde_json::from_str(r#"{"model": "m.gguf"}"#).unwrap();
        let base = InferenceConfig {
            backend: BackendType::Metal,
            ..Default::default()
        };
        assert_eq!(
            config_from_load_request(&req, base).unwrap().backend,
            BackendType::CpuSimd
        );
    }

    #[test]
    fn test_constant_time_eq() {
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"secreT"));
        assert!(!constant_time_eq(b"secret", b"secret2"));
        assert!(!constant_time_eq(b"", b"x"));
    }

    // ---------------------------------------------------------------------
    // Rate limiting / auth
    // ---------------------------------------------------------------------

    #[test]
    fn a_zero_rate_limit_means_unlimited() {
        // Compose documents `NEXUS_RATE_LIMIT_PER_MINUTE=0` as the way to
        // disable the limiter behind a fronting proxy; the limiter must not
        // turn that documented value into a deny-all for every client.
        let limiter = RateLimiter::new(Some(0), Duration::from_secs(60));
        let peer = Some(IpAddr::V4(std::net::Ipv4Addr::LOCALHOST));
        let t0 = Instant::now();
        for _ in 0..100 {
            assert!(limiter.check(peer, t0), "0 must disable the limit");
        }
    }

    #[test]
    fn rate_limiter_keys_by_peer_and_evicts_idle_clients() {
        let limiter = RateLimiter::new(Some(2), Duration::from_secs(60));
        let t0 = Instant::now();

        let a: IpAddr = "10.0.0.1".parse().unwrap();
        let b: IpAddr = "10.0.0.2".parse().unwrap();
        assert!(limiter.check(Some(a), t0));
        assert!(limiter.check(Some(a), t0));
        assert!(!limiter.check(Some(a), t0));
        assert!(limiter.check(Some(b), t0));

        // IPv4-mapped IPv6 is the same client as IPv4.
        let mapped: IpAddr = "::ffff:10.0.0.1".parse().unwrap();
        assert!(!limiter.check(Some(mapped), t0));

        // Addresses inside one IPv6 /64 share a bucket.
        let v6a: IpAddr = "2001:db8:1:2::1".parse().unwrap();
        let v6b: IpAddr = "2001:db8:1:2::ffff".parse().unwrap();
        assert!(limiter.check(Some(v6a), t0));
        assert!(limiter.check(Some(v6b), t0));
        assert!(!limiter.check(Some(v6a), t0));
        assert_eq!(limiter.tracked_clients(), 3);

        // After the window, idle buckets are evicted and the limit resets.
        let later = t0 + Duration::from_secs(61);
        assert!(limiter.check(Some(a), later));
        assert_eq!(limiter.tracked_clients(), 1);
    }

    #[test]
    fn rate_limiter_bounds_live_identities_and_reclaims_expired_ones() {
        let limiter = RateLimiter::new(Some(2), Duration::from_secs(60));
        let now = Instant::now();
        for i in 0..MAX_RATE_LIMIT_CLIENTS {
            assert!(limiter.check(Some(IpAddr::V4((i as u32).into())), now));
        }
        let extra = IpAddr::V4((MAX_RATE_LIMIT_CLIENTS as u32).into());
        assert!(!limiter.check(Some(extra), now));
        assert!(limiter.check(Some(IpAddr::V4(0u32.into())), now));
        assert_eq!(limiter.tracked_clients(), MAX_RATE_LIMIT_CLIENTS);
        assert!(limiter.check(Some(extra), now + Duration::from_secs(61)));
        assert_eq!(limiter.tracked_clients(), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn http_rate_limit_ignores_forged_forwarding_headers() {
        let config = ServerConfig {
            rate_limit_per_minute: Some(2),
            ..open_config()
        };
        let (base, _state) = start(layered_fixture("m", 1), options(1, 4), config).await;
        let http = client();
        let mut statuses = Vec::new();
        for i in 0..4 {
            let response = http
                .get(format!("{base}/v1/models"))
                .header("x-forwarded-for", format!("203.0.113.{i}"))
                .header("x-real-ip", format!("198.51.100.{i}"))
                .header("forwarded", format!("for=192.0.2.{i}"))
                .send()
                .await
                .unwrap();
            statuses.push(response.status().as_u16());
        }
        assert_eq!(statuses, vec![200, 200, 429, 429]);

        // /health is public and not rate limited.
        let health = http.get(format!("{base}/health")).send().await.unwrap();
        assert_eq!(health.status(), 200);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn http_auth_is_enforced() {
        let config = ServerConfig {
            api_key: Some("s3cret".to_string()),
            ..open_config()
        };
        let (base, _state) = start(layered_fixture("m", 1), options(1, 4), config).await;
        let http = client();
        let url = format!("{base}/v1/models");
        let status = |r: reqwest::Response| r.status().as_u16();
        assert_eq!(status(http.get(&url).send().await.unwrap()), 401);
        assert_eq!(
            status(http.get(&url).bearer_auth("wrong").send().await.unwrap()),
            401
        );
        assert_eq!(
            status(http.get(&url).bearer_auth("s3cret").send().await.unwrap()),
            200
        );
        assert_eq!(
            status(
                http.get(&url)
                    .header("x-api-key", "s3cret")
                    .send()
                    .await
                    .unwrap()
            ),
            200
        );
        let unauthorized = http
            .post(format!("{base}/v1/model/unload"))
            .send()
            .await
            .unwrap();
        assert_eq!(unauthorized.status(), 401);
        assert_eq!(
            status(http.get(format!("{base}/health")).send().await.unwrap()),
            200
        );
    }

    // ---------------------------------------------------------------------
    // Continuous batching correctness
    // ---------------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_http_completions_match_isolated_generation() {
        let (base, state) =
            start(layered_fixture("fixture", 7), options(4, 16), open_config()).await;
        let http = client();
        let cases: Vec<(String, Option<u64>)> = (0..8)
            .map(|i| {
                let prompt = format!("prompt number {i} {}", "z".repeat(i));
                let seed = (i % 2 == 1).then_some(100 + i as u64);
                (prompt, seed)
            })
            .collect();

        let tasks: Vec<_> = cases
            .iter()
            .cloned()
            .map(|(prompt, seed)| {
                let http = http.clone();
                let url = format!("{base}/v1/completions");
                tokio::spawn(async move {
                    let mut body = serde_json::json!({
                        "model": "fixture",
                        "prompt": prompt,
                        "max_tokens": 24,
                    });
                    if let Some(seed) = seed {
                        body["temperature"] = serde_json::json!(0.9);
                        body["seed"] = serde_json::json!(seed);
                    }
                    let response = http.post(url).json_body(&body).send().await.unwrap();
                    assert_eq!(response.status(), 200);
                    response.json_value().await.unwrap()
                })
            })
            .collect();

        for ((prompt, seed), task) in cases.iter().zip(tasks) {
            let body = task.await.unwrap();
            let generation = GenerationConfig {
                max_tokens: 24,
                temperature: if seed.is_some() { 0.9 } else { 0.0 },
                seed: *seed,
                ..Default::default()
            };
            let expected = reference(layered_fixture("fixture", 7), prompt, &generation);
            assert_eq!(body["choices"][0]["text"], expected, "prompt {prompt:?}");
            assert_eq!(body["choices"][0]["finish_reason"], "length");
            assert_eq!(body["usage"]["completion_tokens"], 24);
            assert_eq!(body["usage"]["prompt_tokens"], prompt.len());
            assert_eq!(body["model"], "fixture");
        }

        // A multi-prompt request runs its prompts as concurrent sequences.
        let prompts = ["alpha", "beta beta", "gamma gamma gamma"];
        let body: serde_json::Value = http
            .post(format!("{base}/v1/completions"))
            .json_body(
                &serde_json::json!({"model": "fixture", "prompt": prompts, "max_tokens": 10}),
            )
            .send()
            .await
            .unwrap()
            .json_value()
            .await
            .unwrap();
        for (index, prompt) in prompts.iter().enumerate() {
            let generation = GenerationConfig {
                max_tokens: 10,
                temperature: 0.0,
                ..Default::default()
            };
            let expected = reference(layered_fixture("fixture", 7), prompt, &generation);
            assert_eq!(body["choices"][index]["index"], index);
            assert_eq!(body["choices"][index]["text"], expected);
        }

        // Everything was released.
        let scheduler = state.scheduler();
        wait_until("idle scheduler", || {
            scheduler.stats().0 == 0 && scheduler.stats().1 == 0
        })
        .await;
        assert_eq!(scheduler.kv_page_stats().0, 0);
        assert_eq!(scheduler.stats().2, 11);
        assert!(scheduler
            .get_completed()
            .iter()
            .all(|r| r.tokens.is_empty()));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sequences_interleave_and_streams_match_isolated_generation() {
        let scheduler = Arc::new(Scheduler::new(2));
        let runtime = Runtime::start(
            layered_fixture("fixture", 3),
            None,
            scheduler,
            options(2, 4),
        )
        .unwrap();
        let mut a = runtime
            .submit_stream(text_request("a", "first sequence", 40))
            .unwrap();
        let mut b = runtime
            .submit_stream(text_request("b", "second one", 40))
            .unwrap();

        // Neither client reads, so each sequence pauses on backpressure after a
        // couple of tokens. Both being active at once proves they interleave
        // rather than run to completion one after another.
        wait_until("both sequences active", || runtime.stats().active == 2).await;
        wait_until("both paused on backpressure", || {
            runtime.scheduler().stats().1 == 2 && a.len() >= 3 && b.len() >= 3
        })
        .await;
        assert!(matches!(a.recv().await, Some(StreamEvent::Started { .. })));
        assert!(matches!(b.recv().await, Some(StreamEvent::Started { .. })));

        let (text_a, reason_a) = collect_stream(a).await;
        let (text_b, reason_b) = collect_stream(b).await;
        let generation = GenerationConfig {
            max_tokens: 40,
            temperature: 0.0,
            ..Default::default()
        };
        assert_eq!(
            text_a,
            reference(layered_fixture("fixture", 3), "first sequence", &generation)
        );
        assert_eq!(
            text_b,
            reference(layered_fixture("fixture", 3), "second one", &generation)
        );
        assert_eq!((reason_a.as_str(), reason_b.as_str()), ("length", "length"));
    }

    // ---------------------------------------------------------------------
    // Streaming: stop sequences and UTF-8
    // ---------------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bare_byte_fallback_streams_complete_unicode_and_stops_before_a_revision() {
        let mut engine = scripted_fixture("byte-fallback", &[(0, 1), (1, 2), (2, 1)], 64);
        let metadata = &mut engine.model.as_mut().unwrap().metadata;
        metadata.metadata.insert(
            "nexus.tokenizer.json".into(),
            GgufValue::String(
                include_str!("../../tests/fixtures/tokenizers/bare-byte-fallback.json").into(),
            ),
        );
        let tokenizer = crate::tokenizer::RuntimeTokenizer::from_model(
            std::path::Path::new("unused"),
            metadata,
            256,
            false,
        )
        .unwrap();
        assert!(!tokenizer.prefix_stable());
        assert_eq!(tokenizer.try_decode(&[1, 2]).unwrap(), "é");
        assert_eq!(tokenizer.try_decode(&[1, 2, 1]).unwrap(), "���");
        engine.tokenizer = Some(tokenizer);
        let (base, state) = start(engine, options(2, 4), open_config()).await;
        let (text, reason) = collect_stream(
            state
                .runtime
                .submit_stream(text_request("fallback", "x", 4))
                .unwrap(),
        )
        .await;
        assert_eq!((text.as_str(), reason.as_str()), ("éé", "length"));
        // A third byte revises the full decode and would erase the stop match.
        // Matching must terminate at token 2, before that extra work occurs.
        let result: serde_json::Value = client().post(format!("{base}/v1/completions"))
            .json_body(&serde_json::json!({"model":"byte-fallback","prompt":"x","max_tokens":3,"temperature":0,"stop":"é"}))
            .send().await.unwrap().json_value().await.unwrap();
        assert_eq!(result["choices"][0]["text"], "");
        assert_eq!(result["choices"][0]["finish_reason"], "stop");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn exact_tokenizer_buffers_revising_decoders_and_reports_template_errors() {
        let mut engine = scripted_fixture("wordpiece", &[(2, 3), (3, 10), (10, 5), (5, 4)], 64);
        let metadata = &mut engine.model.as_mut().unwrap().metadata;
        metadata.metadata.insert(
            "nexus.tokenizer.json".into(),
            GgufValue::String(
                include_str!("../../tests/fixtures/tokenizers/wordpiece.json").into(),
            ),
        );
        metadata.metadata.insert(
            "tokenizer.chat_template".into(),
            GgufValue::String("{{ raise_exception('invalid role order') }}".into()),
        );
        engine.tokenizer = Some(
            crate::tokenizer::RuntimeTokenizer::from_model(
                std::path::Path::new("unused"),
                metadata,
                256,
                false,
            )
            .unwrap(),
        );
        let (base, state) = start(engine, options(2, 4), open_config()).await;
        let mut request = text_request("buffered", "world", 4);
        request.sampling.temperature = 0.0;
        let mut stream = state.runtime.submit_stream(request).unwrap();
        let mut deltas = Vec::new();
        loop {
            match stream.recv().await.unwrap() {
                StreamEvent::Started { .. } => {}
                StreamEvent::Delta(text) => deltas.push(text),
                StreamEvent::Finished { .. } => break,
                StreamEvent::Error(error) => panic!("{error}"),
            }
        }
        assert_eq!(deltas, vec!["hellos! world"]);
        let http = client();
        let result: serde_json::Value = http.post(format!("{base}/v1/completions"))
            .json_body(&serde_json::json!({"model":"wordpiece","prompt":"world","max_tokens":4,"temperature":0,"stop":"los"}))
            .send().await.unwrap().json_value().await.unwrap();
        assert_eq!(result["choices"][0]["text"], "hel");
        assert_eq!(result["choices"][0]["finish_reason"], "stop");
        let error = http.post(format!("{base}/v1/chat/completions"))
            .json_body(&serde_json::json!({"model":"wordpiece","messages":[{"role":"user","content":"hello"}],"max_tokens":4}))
            .send().await.unwrap();
        assert_eq!(error.status(), reqwest::StatusCode::BAD_REQUEST);
        assert!(error.text().await.unwrap().contains("invalid role order"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn streaming_holds_partial_stop_and_incomplete_utf8() {
        // x -> H -> 0xC3 0xA9 ("é") -> E -> N -> D -> x ...
        let transitions = [
            (b'x', b'H'),
            (b'H', 0xC3),
            (0xC3, 0xA9),
            (0xA9, b'E'),
            (b'E', b'N'),
            (b'N', b'D'),
            (b'D', b'x'),
        ];
        let engine = scripted_fixture("script", &transitions, 64);
        let (base, _state) = start(engine, options(2, 4), open_config()).await;
        let http = client();

        let body = http
            .post(format!("{base}/v1/completions"))
            .json_body(&serde_json::json!({
                "model": "script", "prompt": "x", "max_tokens": 20,
                "stop": ["END"], "stream": true
            }))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        let payloads = sse_payloads(&body);
        assert_eq!(payloads.last().map(String::as_str), Some("[DONE]"));
        let chunks: Vec<serde_json::Value> = payloads[..payloads.len() - 1]
            .iter()
            .map(|p| serde_json::from_str(p).unwrap())
            .collect();
        let deltas: Vec<&str> = chunks
            .iter()
            .filter(|c| c["choices"][0]["finish_reason"].is_null())
            .map(|c| c["choices"][0]["text"].as_str().unwrap())
            .collect();
        // "é" is only sent once both bytes exist; "E"/"EN" (a stop prefix) are
        // never sent.
        assert_eq!(deltas, vec!["H", "é"]);
        assert_eq!(
            chunks.last().unwrap()["choices"][0]["finish_reason"],
            "stop"
        );
        assert_eq!(chunks.last().unwrap()["model"], "script");

        // The non-streaming path agrees.
        let body: serde_json::Value = http
            .post(format!("{base}/v1/completions"))
            .json_body(&serde_json::json!({
                "model": "script", "prompt": "x", "max_tokens": 20, "stop": "END"
            }))
            .send()
            .await
            .unwrap()
            .json_value()
            .await
            .unwrap();
        assert_eq!(body["choices"][0]["text"], "Hé");
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        // Usage counts generated tokens at the moment the stop matched,
        // including the withheld stop tail — the llama.cpp/OpenAI convention
        // for `finish_reason: "stop"` (the previous implementation excluded
        // the tail via a quadratic re-decode scan).
        assert_eq!(body["usage"]["prompt_tokens"], 1);
        assert_eq!(body["usage"]["completion_tokens"], 6);
        assert_eq!(body["usage"]["total_tokens"], 7);
        assert_eq!(chunks.last().unwrap()["usage"], body["usage"]);
        assert!(chunks[..chunks.len() - 1]
            .iter()
            .all(|chunk| chunk.get("usage").is_none()));

        // Chat streaming: role chunk first, [DONE] last, and the concatenated
        // deltas equal the non-streaming chat answer.
        let chat = |stream: bool| {
            serde_json::json!({
                "model": "script",
                "messages": [{"role": "user", "content": "x"}],
                "max_tokens": 5, "stream": stream
            })
        };
        let body = http
            .post(format!("{base}/v1/chat/completions"))
            .json_body(&chat(true))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        let payloads = sse_payloads(&body);
        let first: serde_json::Value = serde_json::from_str(&payloads[0]).unwrap();
        assert_eq!(first["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(payloads.last().map(String::as_str), Some("[DONE]"));
        let mut streamed = String::new();
        for payload in &payloads[1..payloads.len() - 1] {
            let chunk: serde_json::Value = serde_json::from_str(payload).unwrap();
            if let Some(text) = chunk["choices"][0]["delta"]["content"].as_str() {
                streamed.push_str(text);
            }
        }
        let whole = http
            .post(format!("{base}/v1/chat/completions"))
            .json_body(&chat(false))
            .send()
            .await
            .unwrap()
            .json_value()
            .await
            .unwrap();
        assert_eq!(whole["choices"][0]["message"]["content"], streamed);
        assert_eq!(whole["usage"]["completion_tokens"], 5);
        let final_chunk: serde_json::Value =
            serde_json::from_str(&payloads[payloads.len() - 2]).unwrap();
        assert_eq!(final_chunk["choices"][0]["finish_reason"], "length");
        assert_eq!(final_chunk["usage"], whole["usage"]);
        assert!(payloads[..payloads.len() - 2].iter().all(|payload| {
            serde_json::from_str::<serde_json::Value>(payload)
                .unwrap()
                .get("usage")
                .is_none()
        }));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn streaming_emits_multibyte_chars_whole() {
        // Each step adds one byte of "é€"; partial chars must never be sent.
        let text = "é€!";
        let bytes = text.as_bytes();
        let mut transitions = vec![(b'q', bytes[0])];
        transitions.extend(bytes.windows(2).map(|w| (w[0], w[1])));
        transitions.push((bytes[bytes.len() - 1], b'q'));
        let engine = scripted_fixture("utf8", &transitions, 64);
        let runtime = Runtime::start(
            engine,
            None,
            Arc::new(Scheduler::new(1)),
            RuntimeOptions {
                stream_buffer: 64,
                ..options(1, 1)
            },
        )
        .unwrap();
        let mut rx = runtime
            .submit_stream(text_request("u", "q", bytes.len()))
            .unwrap();
        let mut deltas = Vec::new();
        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::Delta(delta) => deltas.push(delta),
                StreamEvent::Finished { finish_reason, .. } => {
                    assert_eq!(finish_reason, "length");
                    break;
                }
                StreamEvent::Error(err) => panic!("{err}"),
                StreamEvent::Started { .. } => {}
            }
        }
        assert_eq!(deltas, vec!["é", "€", "!"]);
    }

    // ---------------------------------------------------------------------
    // Capacity, cancellation, health
    // ---------------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn capacity_returns_503_cancellation_releases_and_health_stays_responsive() {
        let engine = scripted_fixture("loop", &cycle("abc"), 512);
        let (base, state) = start(engine, options(1, 1), open_config()).await;
        let http = client();
        let runtime = state.runtime.clone();

        // Occupy the only slot with a stream nobody reads (paused).
        let stream = runtime
            .submit_stream(text_request("held", "a", 400))
            .unwrap();
        wait_until("held stream active", || runtime.stats().active == 1).await;
        wait_until("held stream paused", || stream.len() >= 3).await;
        // Fill the queue.
        let queued = runtime.submit(text_request("queued", "a", 5)).unwrap();
        assert_eq!(runtime.stats().queued, 1);

        // Further work is rejected immediately with 503.
        let response = http
            .post(format!("{base}/v1/completions"))
            .json_body(&serde_json::json!({"model": "loop", "prompt": "a", "max_tokens": 3}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 503);
        let error: serde_json::Value = response.json_value().await.unwrap();
        assert!(error["error"]["message"]
            .as_str()
            .unwrap()
            .contains("capacity"));

        // Health and metrics answer promptly while the worker is saturated.
        let started = Instant::now();
        let health: serde_json::Value = http
            .get(format!("{base}/health"))
            .send()
            .await
            .unwrap()
            .json_value()
            .await
            .unwrap();
        assert!(started.elapsed() < Duration::from_secs(1));
        assert_eq!(health["status"], "ok");
        assert_eq!(health["active_sequences"], 1);
        assert_eq!(health["queued_requests"], 1);
        let metrics: serde_json::Value = http
            .get(format!("{base}/metrics"))
            .send()
            .await
            .unwrap()
            .json_value()
            .await
            .unwrap();
        assert_eq!(metrics["scheduler"]["running"], 1);
        assert!(metrics["scheduler"]["kv_pages_used"].as_u64().unwrap() > 0);
        assert_eq!(metrics["runtime"]["rejected_busy_total"], 1);

        // Clients going away cancels both the active and the queued job.
        drop(queued);
        drop(stream);
        wait_until("cancellations", || {
            let stats = runtime.stats();
            stats.active == 0 && stats.queued == 0 && stats.cancelled_total == 2
        })
        .await;
        let scheduler = state.scheduler();
        assert_eq!(scheduler.kv_page_stats().0, 0);
        assert_eq!((scheduler.stats().0, scheduler.stats().1), (0, 0));

        // The server recovered.
        let body: serde_json::Value = http
            .post(format!("{base}/v1/completions"))
            .json_body(&serde_json::json!({"model": "loop", "prompt": "a", "max_tokens": 4}))
            .send()
            .await
            .unwrap()
            .json_value()
            .await
            .unwrap();
        assert_eq!(body["choices"][0]["text"], "bcab");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn invalid_requests_are_rejected_and_cleaned_up() {
        let engine = scripted_fixture("loop", &cycle("abc"), 16);
        let (base, state) = start(engine, options(2, 4), open_config()).await;
        let http = client();
        let post = |body: serde_json::Value| {
            let http = http.clone();
            let url = format!("{base}/v1/completions");
            async move { http.post(url).json_body(&body).send().await.unwrap() }
        };

        // Prompt longer than the context.
        let response = post(serde_json::json!({"model": "loop", "prompt": "a".repeat(17)})).await;
        assert_eq!(response.status(), 400);
        // Explicit max_tokens beyond the remaining context.
        let response =
            post(serde_json::json!({"model": "loop", "prompt": "a", "max_tokens": 16})).await;
        assert_eq!(response.status(), 400);
        // Invalid sampling parameters.
        let response =
            post(serde_json::json!({"model": "loop", "prompt": "a", "top_p": 0.0})).await;
        assert_eq!(response.status(), 400);
        let response = post(serde_json::json!({"model": "loop", "prompt": []})).await;
        assert_eq!(response.status(), 400);
        // Streaming admission errors are real HTTP errors, not 200 + SSE.
        let response = post(serde_json::json!({
            "model": "loop", "prompt": "a".repeat(20), "stream": true
        }))
        .await;
        assert_eq!(response.status(), 400);

        // Without max_tokens the default is clamped to the remaining context.
        let body: serde_json::Value = post(serde_json::json!({"model": "loop", "prompt": "a"}))
            .await
            .json_value()
            .await
            .unwrap();
        assert_eq!(body["usage"]["completion_tokens"], 15);
        assert_eq!(body["choices"][0]["finish_reason"], "length");

        let scheduler = state.scheduler();
        assert_eq!(scheduler.kv_page_stats().0, 0);
        assert_eq!((scheduler.stats().0, scheduler.stats().1), (0, 0));
        assert_eq!(state.runtime.stats().active, 0);
    }

    // ---------------------------------------------------------------------
    // Model lifecycle
    // ---------------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn model_swap_waits_for_active_sequences_and_updates_metadata_atomically() {
        let scheduler = Arc::new(Scheduler::new(1));
        let runtime = Arc::new(
            Runtime::start(
                scripted_fixture("model-a", &cycle("a"), 64),
                Some(PathBuf::from("/models/a.gguf")),
                scheduler,
                options(1, 4),
            )
            .unwrap(),
        );
        assert_eq!(runtime.snapshot().name, "model-a");
        assert_eq!(runtime.snapshot().epoch, 0);

        let mut held = runtime
            .submit_stream(text_request("held", "a", 12))
            .unwrap();
        wait_until("held stream paused", || held.len() >= 3).await;

        let installer = {
            let runtime = runtime.clone();
            tokio::spawn(async move {
                runtime
                    .install(
                        scripted_fixture("model-b", &cycle("b"), 32),
                        Some(PathBuf::from("/models/b.gguf")),
                    )
                    .await
            })
        };
        wait_until("swap pending", || runtime.stats().model_swap_pending).await;

        // New submissions during a pending swap fail fast with Busy instead
        // of queueing indefinitely behind the drain of active sequences.
        // (This is the inverted gate: the test previously pinned the
        // queue-and-hang behavior as expected.)
        let rejection = runtime.submit(text_request("after", "b", 3));
        let err = rejection.expect_err("the swap window must reject instead of queueing");
        assert_eq!(err.status_code(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(
            err.message.contains("Model swap in progress"),
            "message: {err}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            !installer.is_finished(),
            "swap must wait for the active sequence"
        );
        let snapshot = runtime.snapshot();
        assert_eq!((snapshot.name.as_str(), snapshot.epoch), ("model-a", 0));
        assert_eq!(runtime.stats().queued, 0);

        // The in-flight sequence completes entirely on model A.
        assert!(
            matches!(held.recv().await, Some(StreamEvent::Started { ref model, .. }) if model == "model-a")
        );
        let (text, reason) = collect_stream(held).await;
        assert_eq!((text.as_str(), reason.as_str()), ("aaaaaaaaaaaa", "length"));

        let installed = installer.await.unwrap().unwrap();
        assert_eq!(installed.name, "model-b");
        assert_eq!(installed.epoch, 1);
        assert_eq!(installed.max_seq_len, 32);
        assert_eq!(installed.path, Some(PathBuf::from("/models/b.gguf")));
        assert_eq!(installed.info["max_seq_len"], 32);
        assert!(!runtime.stats().model_swap_pending);
        assert_eq!(runtime.scheduler().kv_page_stats().1, 2); // 32 tokens / 16 per page * 1 slot

        // After the swap completes, a fresh request is tokenized, generated
        // and labelled by model B.
        let after = runtime
            .submit(text_request("after", "b", 3))
            .unwrap()
            .await
            .unwrap()
            .unwrap();
        assert_eq!(after.model, "model-b");
        assert_eq!(after.text, "bbb");
    }

    fn push_u32(out: &mut Vec<u8>, value: u32) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(out: &mut Vec<u8>, value: u64) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn push_str(out: &mut Vec<u8>, value: &str) {
        push_u64(out, value.len() as u64);
        out.extend_from_slice(value.as_bytes());
    }

    /// Write a small but complete F32 llama GGUF for the layered fixture.
    fn write_tiny_gguf(name: &str, context: u32) -> tempfile::NamedTempFile {
        let tensors = layered_tensors(11);
        let kv_u32 = [
            ("general.alignment", 32u32),
            ("llama.context_length", context),
            ("llama.embedding_length", HIDDEN as u32),
            ("llama.block_count", 1),
            ("llama.feed_forward_length", FF as u32),
            ("llama.attention.head_count", HEADS as u32),
            ("llama.attention.head_count_kv", KV_HEADS as u32),
            ("llama.vocab_size", VOCAB as u32),
        ];
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, tensors.len() as u64);
        push_u64(&mut bytes, kv_u32.len() as u64 + 2);
        for (key, value) in [("general.architecture", "llama"), ("general.name", name)] {
            push_str(&mut bytes, key);
            push_u32(&mut bytes, GgufValueType::String.id());
            push_str(&mut bytes, value);
        }
        for (key, value) in kv_u32 {
            push_str(&mut bytes, key);
            push_u32(&mut bytes, GgufValueType::U32.id());
            push_u32(&mut bytes, value);
        }
        let mut offset = 0u64;
        for (tensor, dims, data) in &tensors {
            push_str(&mut bytes, tensor);
            push_u32(&mut bytes, dims.len() as u32);
            for &dim in dims {
                push_u64(&mut bytes, dim as u64);
            }
            push_u32(&mut bytes, GgufDataType::F32.id());
            push_u64(&mut bytes, offset);
            offset += (data.len() as u64 * 4).div_ceil(32) * 32;
        }
        let pad = |bytes: &mut Vec<u8>| {
            while bytes.len() & 31 != 0 {
                bytes.push(0);
            }
        };
        pad(&mut bytes);
        for (_, _, data) in &tensors {
            for value in data {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            pad(&mut bytes);
        }
        let file = tempfile::Builder::new().suffix(".gguf").tempfile().unwrap();
        file.as_file().write_all(&bytes).unwrap();
        file.as_file().flush().unwrap();
        file
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn http_model_load_reload_unload_lifecycle() {
        let empty = InferenceEngine::new(Some(fixture_config(128)));
        let (base, _state) = start(empty, options(2, 4), open_config()).await;
        let http = client();
        let get = |path: &str| {
            let http = http.clone();
            let url = format!("{base}{path}");
            async move {
                let response = http.get(url).send().await.unwrap();
                (
                    response.status().as_u16(),
                    response.json_value().await.unwrap(),
                )
            }
        };
        let post = |path: &str, body: serde_json::Value| {
            let http = http.clone();
            let url = format!("{base}{path}");
            async move {
                let response = http.post(url).json_body(&body).send().await.unwrap();
                (
                    response.status().as_u16(),
                    response.json_value().await.unwrap(),
                )
            }
        };
        let complete =
            serde_json::json!({"model": "tiny-gguf", "prompt": "hello", "max_tokens": 6});

        // No model yet.
        assert_eq!(get("/health").await.1["status"], "degraded");
        assert_eq!(get("/v1/models").await.1["data"], serde_json::json!([]));
        assert_eq!(get("/v1/model").await.1["epoch"], 0);
        assert_eq!(post("/v1/completions", complete.clone()).await.0, 503);
        assert_eq!(post("/v1/model/reload", serde_json::json!({})).await.0, 400);

        // Load a real GGUF from disk.
        let file = write_tiny_gguf("tiny-gguf", 64);
        let (status, body) = post(
            "/v1/model/load",
            serde_json::json!({"model": file.path(), "kv_quant": "none"}),
        )
        .await;
        assert_eq!(status, 200, "{body}");
        assert_eq!(body["id"], "tiny-gguf");
        assert_eq!(body["max_seq_len"], 64);
        assert_eq!(body["backend"], "cpu-simd");
        assert_eq!(body["epoch"], 1);

        let (_, info) = get("/v1/model").await;
        assert_eq!(info["id"], "tiny-gguf");
        assert_eq!(info["loaded"], true);
        assert_eq!(info["max_seq_len"], 64);
        assert_eq!(info["dimensions"]["hidden_size"], HIDDEN);
        assert_eq!(info["dimensions"]["vocab_size"], VOCAB);
        assert_eq!(info["path"], serde_json::json!(file.path()));
        assert_eq!(info["epoch"], body["epoch"]);
        assert_eq!(get("/v1/models").await.1["data"][0]["id"], "tiny-gguf");
        assert_eq!(get("/health").await.1["status"], "ok");

        let (status, first) = post("/v1/completions", complete.clone()).await;
        assert_eq!(status, 200, "{first}");
        assert_eq!(first["model"], "tiny-gguf");
        assert_eq!(first["usage"]["completion_tokens"], 6);

        // Rejected loads keep the current model.
        let (status, _) = post(
            "/v1/model/load",
            serde_json::json!({"model": "/nonexistent/model.gguf"}),
        )
        .await;
        assert_eq!(status, 400);
        let (status, body) = post(
            "/v1/model/load",
            serde_json::json!({"model": file.path(), "backend": "metal"}),
        )
        .await;
        assert_eq!(status, 400);
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("not implemented"));
        assert_eq!(get("/v1/model").await.1["id"], "tiny-gguf");
        assert_eq!(get("/v1/model").await.1["epoch"], 1);
        let (status, again) = post("/v1/completions", complete.clone()).await;
        assert_eq!(status, 200);
        assert_eq!(again["choices"][0]["text"], first["choices"][0]["text"]);

        // Reload swaps in a fresh engine for the same file.
        let epoch = get("/metrics").await.1["model"]["epoch"].as_u64().unwrap();
        let (status, body) = post("/v1/model/reload", serde_json::json!({})).await;
        assert_eq!(status, 200, "{body}");
        assert_eq!(body["status"], "reloaded");
        assert_eq!(body["id"], "tiny-gguf");
        assert_eq!(get("/metrics").await.1["model"]["epoch"], epoch + 1);
        assert_eq!(body["epoch"], epoch + 1);
        let (_, info) = get("/v1/model").await;
        assert_eq!(info["epoch"], epoch + 1);
        assert_eq!(info["path"], serde_json::json!(file.path()));
        let (_, reloaded) = post("/v1/completions", complete.clone()).await;
        assert_eq!(reloaded["choices"][0]["text"], first["choices"][0]["text"]);

        // Unload.
        let (status, body) = post("/v1/model/unload", serde_json::json!({})).await;
        assert_eq!(status, 200);
        assert_eq!(body["id"], DEFAULT_MODEL_NAME);
        assert_eq!(body["loaded"], false);
        assert_eq!(body["epoch"], epoch + 2);
        assert_eq!(post("/v1/completions", complete.clone()).await.0, 503);
        assert_eq!(get("/health").await.1["status"], "degraded");
        let (_, info) = get("/v1/model").await;
        assert_eq!(info["loaded"], false);
        assert_eq!(info["path"], serde_json::Value::Null);
        assert_eq!(info["epoch"], body["epoch"]);
        assert_eq!(post("/v1/model/reload", serde_json::json!({})).await.0, 400);
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unauthorized_requests_consume_peer_rate_limit() {
        let config = ServerConfig {
            api_key: Some("secret".into()),
            cors_origin: None,
            rate_limit_per_minute: Some(2),
        };
        let (base, _) = start(
            scripted_fixture("loop", &cycle("abc"), 16),
            options(1, 4),
            config,
        )
        .await;
        let http = client();
        for expected in [401, 401, 429] {
            let response = http
                .get(format!("{base}/v1/models"))
                .header("x-forwarded-for", format!("198.51.100.{expected}"))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status().as_u16(), expected);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn full_context_zero_output_and_invalid_tokens_have_correct_status() {
        let mut engine = scripted_fixture("loop", &cycle("abc"), 16);
        engine.vocab_size = 128;
        let (base, _) = start(engine, options(1, 4), open_config()).await;
        let http = client();
        let response = http
            .post(format!("{base}/v1/completions"))
            .json_body(&serde_json::json!({"model":"loop","prompt":"a".repeat(16),"max_tokens":0}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        assert_eq!(
            response.json_value().await.unwrap()["usage"]["completion_tokens"],
            0
        );
        let response = http
            .post(format!("{base}/v1/completions"))
            .json_body(&serde_json::json!({"model":"loop","prompt":"a".repeat(16)}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 400);
        let response = http
            .post(format!("{base}/v1/completions"))
            .json_body(&serde_json::json!({"model":"loop","prompt":"é","max_tokens":1}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 400);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn requested_context_survives_clamping_reloads_and_failed_replacements() {
        let small = write_tiny_gguf("small", 2048);
        let large = write_tiny_gguf("large", 8192);
        let requested = fixture_config(4096);
        let mut engine = InferenceEngine::new(Some(requested.clone()));
        engine.load_model(small.path()).unwrap();
        assert_eq!(engine.config.max_seq_len, 2048);
        let state = ServerState::new_with_requested_config(
            engine,
            Some(small.path().into()),
            open_config(),
            options(2, 4),
            requested,
        )
        .unwrap();
        let runtime = state.runtime.clone();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(serve(listener, state));
        let http = client();
        for (body, expected) in [
            (serde_json::json!({"model":large.path()}), 4096),
            (
                serde_json::json!({"model":small.path(),"ctx_size":1024}),
                1024,
            ),
            (serde_json::json!({"model":large.path()}), 1024),
        ] {
            let response = http
                .post(format!("{base}/v1/model/load"))
                .json_body(&body)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), 200);
            assert_eq!(
                response.json_value().await.unwrap()["max_seq_len"],
                expected
            );
            assert_eq!(runtime.snapshot().requested_config.max_seq_len, expected);
        }
        let response = http
            .post(format!("{base}/v1/model/load"))
            .json_body(&serde_json::json!({"model":"missing-model.gguf","ctx_size":512}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 400);
        assert_eq!(runtime.snapshot().requested_config.max_seq_len, 1024);
        assert_eq!(runtime.snapshot().config.max_seq_len, 1024);
        for endpoint in ["reload", "unload"] {
            let response = http
                .post(format!("{base}/v1/model/{endpoint}"))
                .json_body(&serde_json::json!({}))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), 200);
            assert_eq!(runtime.snapshot().requested_config.max_seq_len, 1024);
        }
        task.abort();
    }

    #[test]
    fn health_reports_a_dead_worker_as_503_for_code_based_probes() {
        assert_eq!(
            health_status(false, false),
            ("error", StatusCode::SERVICE_UNAVAILABLE)
        );
        assert_eq!(
            health_status(false, true),
            ("error", StatusCode::SERVICE_UNAVAILABLE)
        );
        assert_eq!(health_status(true, true), ("ok", StatusCode::OK));
        assert_eq!(health_status(true, false), ("degraded", StatusCode::OK));
    }
}
