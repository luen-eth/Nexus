//! OpenAI-compatible API server for Nexus inference engine.

use axum::{
    body::Body,
    extract::{Json, State},
    http::{
        header::{self, HeaderValue},
        Method, Request, StatusCode,
    },
    middleware::{from_fn_with_state, Next},
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    routing::{get, options, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use uuid::Uuid;

use crate::backend::{BackendFactory, BackendType};
use crate::engine::{GenerationConfig, InferenceConfig, InferenceEngine};
use crate::quant::{KvQuantType, QuantFormat};
use crate::scheduler::Scheduler;
use crate::tokenizer::{RuntimeTokenizer, Tokenizer};

const DEFAULT_MODEL_NAME: &str = "nexus-model";

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
    pub rate_limit_per_minute: Option<usize>,
}

impl ServerConfig {
    pub fn from_env() -> Self {
        let api_key = std::env::var("NEXUS_API_KEY")
            .ok()
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

/// Server state
#[derive(Clone)]
pub struct ServerState {
    pub engine: Arc<Mutex<Option<InferenceEngine>>>,
    pub scheduler: Arc<Scheduler>,
    pub model_name: Arc<Mutex<String>>,
    pub model_path: Arc<Mutex<Option<PathBuf>>>,
    pub inference_config: Arc<Mutex<InferenceConfig>>,
    pub server_config: Arc<ServerConfig>,
    rate_limits: Arc<Mutex<HashMap<String, VecDeque<Instant>>>>,
}

impl ServerState {
    fn current_model_name(&self) -> String {
        self.model_name.lock().unwrap().clone()
    }

    fn set_model_name(&self, name: String) {
        *self.model_name.lock().unwrap() = name;
    }

    fn current_inference_config(&self) -> InferenceConfig {
        self.inference_config.lock().unwrap().clone()
    }

    fn set_inference_config(&self, config: InferenceConfig) {
        *self.inference_config.lock().unwrap() = config;
    }
}

/// Create the API router
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

    let path = req.uri().path().to_string();
    if !is_public_path(&path) {
        if let Some(api_key) = &state.server_config.api_key {
            if !authorized(&req, api_key) {
                let mut response = status_error(StatusCode::UNAUTHORIZED, "Unauthorized");
                add_cors_headers(&state, &mut response);
                return response;
            }
        }
        if !within_rate_limit(&state, &req) {
            let mut response = status_error(StatusCode::TOO_MANY_REQUESTS, "Rate limit exceeded");
            add_cors_headers(&state, &mut response);
            return response;
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
    let bearer = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "));
    let x_api_key = req
        .headers()
        .get("x-api-key")
        .and_then(|value| value.to_str().ok());
    bearer
        .or(x_api_key)
        .is_some_and(|candidate| candidate == api_key)
}

fn within_rate_limit(state: &ServerState, req: &Request<Body>) -> bool {
    let Some(limit) = state.server_config.rate_limit_per_minute else {
        return true;
    };
    if limit == 0 {
        return false;
    }

    let identity = req
        .headers()
        .get("x-forwarded-for")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(',').next())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("local")
        .to_string();
    let now = Instant::now();
    let window = Duration::from_secs(60);
    let mut limits = state.rate_limits.lock().unwrap();
    let bucket = limits.entry(identity).or_default();
    while bucket
        .front()
        .is_some_and(|seen| now.duration_since(*seen) > window)
    {
        bucket.pop_front();
    }
    if bucket.len() >= limit {
        return false;
    }
    bucket.push_back(now);
    true
}

fn add_cors_headers(state: &ServerState, response: &mut Response) {
    let origin = state.server_config.cors_origin.as_deref().unwrap_or("*");
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
    (
        status,
        Json(serde_json::json!({
            "error": {
                "message": message,
                "type": "server_error"
            }
        })),
    )
        .into_response()
}

async fn chat_completions(
    State(state): State<ServerState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, (StatusCode, String)> {
    let tokenizer = {
        let engine = state.engine.lock().unwrap();
        let model = engine.as_ref().ok_or_else(|| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "Model not loaded".to_string(),
            )
        })?;
        if model.model.is_none() {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                "Model not loaded".to_string(),
            ));
        }
        RuntimeTokenizer::from_metadata(model.model.as_ref().map(|model| model.metadata()))
    };

    let prompt_text = tokenizer.render_chat_messages(
        req.messages
            .iter()
            .map(|message| (message.role.as_str(), message.content.as_str())),
        true,
    );
    let prompt_tokens = tokenizer.encode(&prompt_text);

    let max_tokens = req.max_tokens.unwrap_or(256);
    let stop_sequences = req
        .stop
        .as_ref()
        .map(|stop| match stop {
            StopSpec::One(stop) => vec![tokenizer.encode(stop)],
            StopSpec::Many(stops) => stops.iter().map(|stop| tokenizer.encode(stop)).collect(),
        })
        .unwrap_or_default();
    let generation = GenerationConfig {
        max_tokens,
        temperature: req.temperature.unwrap_or(0.0),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: req.top_k.unwrap_or(0),
        repetition_penalty: req.repetition_penalty.unwrap_or(1.0),
        seed: req.seed,
        stop_sequences,
    };

    let request_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = chrono::Utc::now().timestamp() as u64;
    state.scheduler.add_request(
        request_id.clone(),
        prompt_tokens.clone(),
        generation.max_tokens,
    );
    state.scheduler.get_next_batch();

    if req.stream.unwrap_or(false) {
        return Ok(stream_live_response(
            state,
            request_id,
            created,
            tokenizer,
            prompt_tokens,
            generation,
        )
        .into_response());
    }

    let mut engine = state.engine.lock().unwrap();
    let model = engine.as_mut().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Model not loaded".to_string(),
        )
    })?;

    // Generate response
    match model.generate_with_config(&prompt_tokens, &generation) {
        Ok(output) => {
            let generated = tokenizer.decode(&output[prompt_tokens.len()..]);
            let completion_tokens = output.len() - prompt_tokens.len();
            let finish_reason = if completion_tokens >= generation.max_tokens {
                "length"
            } else {
                "stop"
            };
            state.scheduler.complete_request(&request_id);

            let response = ChatCompletionResponse {
                id: request_id,
                object: "chat.completion".to_string(),
                created,
                model: state.current_model_name(),
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatMessage {
                        role: "assistant".to_string(),
                        content: generated,
                    },
                    finish_reason: finish_reason.to_string(),
                }],
                usage: UsageStats {
                    prompt_tokens: prompt_tokens.len(),
                    completion_tokens,
                    total_tokens: output.len(),
                },
            };

            Ok(Json(response).into_response())
        }
        Err(e) => {
            state.scheduler.complete_request(&request_id);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Generation error: {}", e),
            ))
        }
    }
}

async fn completions(
    State(state): State<ServerState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Response, (StatusCode, String)> {
    let tokenizer = tokenizer_for_loaded_model(&state)?;
    let prompts = match req.prompt.clone() {
        PromptSpec::One(prompt) => vec![prompt],
        PromptSpec::Many(prompts) => prompts,
    };
    if prompts.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "prompt cannot be empty".to_string(),
        ));
    }

    let stop_sequences = req
        .stop
        .as_ref()
        .map(|stop| match stop {
            StopSpec::One(stop) => vec![tokenizer.encode(stop)],
            StopSpec::Many(stops) => stops.iter().map(|stop| tokenizer.encode(stop)).collect(),
        })
        .unwrap_or_default();
    let generation = GenerationConfig {
        max_tokens: req.max_tokens.unwrap_or(256),
        temperature: req.temperature.unwrap_or(0.0),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: req.top_k.unwrap_or(0),
        repetition_penalty: req.repetition_penalty.unwrap_or(1.0),
        seed: req.seed,
        stop_sequences,
    };

    let request_id = format!("cmpl-{}", Uuid::new_v4());
    let created = chrono::Utc::now().timestamp() as u64;

    if req.stream.unwrap_or(false) {
        if prompts.len() != 1 {
            return Err((
                StatusCode::BAD_REQUEST,
                "stream=true only supports a single prompt".to_string(),
            ));
        }
        let prompt_tokens = tokenizer.encode(&prompts[0]);
        state.scheduler.add_request(
            request_id.clone(),
            prompt_tokens.clone(),
            generation.max_tokens,
        );
        state.scheduler.get_next_batch();
        return Ok(stream_completion_response(
            state,
            request_id,
            created,
            tokenizer,
            prompt_tokens,
            generation,
        )
        .into_response());
    }

    let mut choices = Vec::with_capacity(prompts.len());
    let mut prompt_token_count = 0usize;
    let mut completion_token_count = 0usize;
    let mut engine = state.engine.lock().unwrap();
    let model = engine.as_mut().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Model not loaded".to_string(),
        )
    })?;

    for (index, prompt) in prompts.iter().enumerate() {
        let prompt_tokens = tokenizer.encode(prompt);
        prompt_token_count += prompt_tokens.len();
        let request_part_id = format!("{}-{}", request_id, index);
        state.scheduler.add_request(
            request_part_id.clone(),
            prompt_tokens.clone(),
            generation.max_tokens,
        );
        state.scheduler.get_next_batch();

        let output = match model.generate_with_config(&prompt_tokens, &generation) {
            Ok(output) => output,
            Err(e) => {
                state.scheduler.complete_request(&request_part_id);
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Generation error: {}", e),
                ));
            }
        };
        state.scheduler.complete_request(&request_part_id);

        let completion_tokens = output.len() - prompt_tokens.len();
        completion_token_count += completion_tokens;
        let finish_reason = if completion_tokens >= generation.max_tokens {
            "length"
        } else {
            "stop"
        };
        choices.push(CompletionChoice {
            index,
            text: tokenizer.decode(&output[prompt_tokens.len()..]),
            finish_reason: finish_reason.to_string(),
        });
    }

    Ok(Json(CompletionResponse {
        id: request_id,
        object: "text_completion".to_string(),
        created,
        model: state.current_model_name(),
        choices,
        usage: UsageStats {
            prompt_tokens: prompt_token_count,
            completion_tokens: completion_token_count,
            total_tokens: prompt_token_count + completion_token_count,
        },
    })
    .into_response())
}

fn tokenizer_for_loaded_model(
    state: &ServerState,
) -> Result<RuntimeTokenizer, (StatusCode, String)> {
    let engine = state.engine.lock().unwrap();
    let model = engine.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Model not loaded".to_string(),
        )
    })?;
    if model.model.is_none() {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "Model not loaded".to_string(),
        ));
    }
    Ok(RuntimeTokenizer::from_metadata(
        model.model.as_ref().map(|model| model.metadata()),
    ))
}

async fn load_model(
    State(state): State<ServerState>,
    Json(req): Json<ModelLoadRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let base_config = state.current_inference_config();
    let (engine, config, model_name) = build_loaded_engine(&req.model, &req, base_config)?;

    state.set_model_name(model_name.clone());
    state.set_inference_config(config);
    *state.model_path.lock().unwrap() = Some(req.model.clone());
    *state.engine.lock().unwrap() = Some(engine);

    Ok(Json(serde_json::json!({
        "status": "loaded",
        "id": model_name,
        "path": req.model,
    })))
}

async fn reload_model(
    State(state): State<ServerState>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let Some(path) = state.model_path.lock().unwrap().clone() else {
        return Err((
            StatusCode::BAD_REQUEST,
            "No previously loaded model path to reload".to_string(),
        ));
    };

    let config = state.current_inference_config();
    let req = ModelLoadRequest {
        model: path.clone(),
        ctx_size: Some(config.max_seq_len),
        kv_quant: Some(config.kv_quant.name().to_string()),
        weight_quant: Some(config.weight_format.name().to_string()),
        backend: Some(config.backend.name().to_string()),
        max_loaded_tensor_elements: config.max_loaded_tensor_elements,
    };
    let (engine, config, model_name) = build_loaded_engine(&path, &req, config)?;

    state.set_model_name(model_name.clone());
    state.set_inference_config(config);
    *state.engine.lock().unwrap() = Some(engine);

    Ok(Json(serde_json::json!({
        "status": "reloaded",
        "id": model_name,
        "path": path,
    })))
}

async fn unload_model(
    State(state): State<ServerState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let config = state.current_inference_config();
    *state.engine.lock().unwrap() = Some(InferenceEngine::new(Some(config)));
    *state.model_path.lock().unwrap() = None;
    state.set_model_name(DEFAULT_MODEL_NAME.to_string());

    Ok(Json(serde_json::json!({
        "status": "unloaded",
        "id": DEFAULT_MODEL_NAME,
    })))
}

fn build_loaded_engine(
    path: &PathBuf,
    req: &ModelLoadRequest,
    base_config: InferenceConfig,
) -> Result<(InferenceEngine, InferenceConfig, String), (StatusCode, String)> {
    let config = config_from_load_request(req, base_config)?;
    let mut engine = InferenceEngine::new(Some(config.clone()));
    engine.load_model(path).map_err(|err| {
        (
            StatusCode::BAD_REQUEST,
            format!("Failed to load model: {}", err),
        )
    })?;
    let model_name = engine
        .model
        .as_ref()
        .and_then(|model| model.metadata.get_str("general.name"))
        .or_else(|| path.file_stem().and_then(|name| name.to_str()))
        .unwrap_or(DEFAULT_MODEL_NAME)
        .to_string();
    Ok((engine, config, model_name))
}

fn config_from_load_request(
    req: &ModelLoadRequest,
    mut config: InferenceConfig,
) -> Result<InferenceConfig, (StatusCode, String)> {
    if let Some(ctx_size) = req.ctx_size {
        config.max_seq_len = ctx_size;
    }
    if let Some(kv_quant) = &req.kv_quant {
        let normalized = kv_quant.to_ascii_lowercase().replace('-', "_");
        config.kv_quant = KvQuantType::from_str(&normalized).ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                format!("Unsupported KV quantization: {}", kv_quant),
            )
        })?;
    }
    if let Some(weight_quant) = &req.weight_quant {
        let normalized = weight_quant.to_ascii_uppercase().replace('-', "_");
        config.weight_format = QuantFormat::from_str(&normalized).ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                format!("Unsupported weight quantization: {}", weight_quant),
            )
        })?;
    }
    if let Some(backend) = &req.backend {
        config.backend = parse_backend(backend)?;
    }
    if let Some(max_loaded_tensor_elements) = req.max_loaded_tensor_elements {
        config.max_loaded_tensor_elements = if max_loaded_tensor_elements == 0 {
            None
        } else {
            Some(max_loaded_tensor_elements)
        };
    }
    Ok(config)
}

fn parse_backend(value: &str) -> Result<BackendType, (StatusCode, String)> {
    match value.to_ascii_lowercase().as_str() {
        "auto" => Ok(BackendType::default()),
        "cpu" | "cpu-simd" => Ok(BackendType::CpuSimd),
        "metal" => Ok(BackendType::Metal),
        "cuda" | "vulkan" | "webgpu" => Err((
            StatusCode::BAD_REQUEST,
            format!(
                "Backend '{}' is exposed as a capability but not implemented yet",
                value
            ),
        )),
        other => Err((
            StatusCode::BAD_REQUEST,
            format!("Unsupported backend: {}", other),
        )),
    }
}

fn stream_live_response(
    state: ServerState,
    request_id: String,
    created: u64,
    tokenizer: RuntimeTokenizer,
    prompt_tokens: Vec<u32>,
    generation: GenerationConfig,
) -> Sse<impl futures_util::Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let engine = state.engine.clone();
    let scheduler = state.scheduler.clone();
    let model_name = state.current_model_name();
    let request_id_for_worker = request_id;
    let prompt_len = prompt_tokens.len();
    let max_tokens = generation.max_tokens;

    tokio::task::spawn_blocking(move || {
        send_sse_json(
            &tx,
            serde_json::json!({
                "id": &request_id_for_worker,
                "object": "chat.completion.chunk",
                "created": created,
                "model": &model_name,
                "choices": [{
                    "index": 0,
                    "delta": { "role": "assistant" },
                    "finish_reason": null
                }]
            }),
        );

        let mut streamed_text = String::new();
        let result = {
            let mut engine = engine.lock().unwrap();
            match engine.as_mut() {
                Some(model) => model.generate_with_config_streaming(
                    &prompt_tokens,
                    &generation,
                    |_, output| {
                        let generated = tokenizer.decode(&output[prompt_len..]);
                        if generated.len() > streamed_text.len() {
                            let delta = generated[streamed_text.len()..].to_string();
                            streamed_text = generated;
                            send_sse_json(
                                &tx,
                                serde_json::json!({
                                    "id": &request_id_for_worker,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": &model_name,
                                    "choices": [{
                                        "index": 0,
                                        "delta": { "content": delta },
                                        "finish_reason": null
                                    }]
                                }),
                            );
                        }
                        Ok(())
                    },
                ),
                None => Err("Model not loaded".to_string()),
            }
        };

        scheduler.complete_request(&request_id_for_worker);

        let finish_reason = match &result {
            Ok(output) if output.len().saturating_sub(prompt_len) >= max_tokens => "length",
            Ok(_) => "stop",
            Err(err) => {
                send_sse_json(
                    &tx,
                    serde_json::json!({
                        "error": {
                            "message": err,
                            "type": "generation_error"
                        }
                    }),
                );
                "error"
            }
        };

        send_sse_json(
            &tx,
            serde_json::json!({
                "id": &request_id_for_worker,
                "object": "chat.completion.chunk",
                "created": created,
                "model": &model_name,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason
                }]
            }),
        );
        let _ = tx.send("[DONE]".to_string());
    });

    let events = futures_util::stream::unfold(rx, |mut rx| async {
        rx.recv()
            .await
            .map(|data| (Ok(Event::default().data(data)), rx))
    });
    Sse::new(events)
}

fn stream_completion_response(
    state: ServerState,
    request_id: String,
    created: u64,
    tokenizer: RuntimeTokenizer,
    prompt_tokens: Vec<u32>,
    generation: GenerationConfig,
) -> Sse<impl futures_util::Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let engine = state.engine.clone();
    let scheduler = state.scheduler.clone();
    let model_name = state.current_model_name();
    let request_id_for_worker = request_id;
    let prompt_len = prompt_tokens.len();
    let max_tokens = generation.max_tokens;

    tokio::task::spawn_blocking(move || {
        let mut streamed_text = String::new();
        let result = {
            let mut engine = engine.lock().unwrap();
            match engine.as_mut() {
                Some(model) => model.generate_with_config_streaming(
                    &prompt_tokens,
                    &generation,
                    |_, output| {
                        let generated = tokenizer.decode(&output[prompt_len..]);
                        if generated.len() > streamed_text.len() {
                            let delta = generated[streamed_text.len()..].to_string();
                            streamed_text = generated;
                            send_sse_json(
                                &tx,
                                serde_json::json!({
                                    "id": &request_id_for_worker,
                                    "object": "text_completion.chunk",
                                    "created": created,
                                    "model": &model_name,
                                    "choices": [{
                                        "index": 0,
                                        "text": delta,
                                        "finish_reason": null
                                    }]
                                }),
                            );
                        }
                        Ok(())
                    },
                ),
                None => Err("Model not loaded".to_string()),
            }
        };

        scheduler.complete_request(&request_id_for_worker);

        let finish_reason = match &result {
            Ok(output) if output.len().saturating_sub(prompt_len) >= max_tokens => "length",
            Ok(_) => "stop",
            Err(err) => {
                send_sse_json(
                    &tx,
                    serde_json::json!({
                        "error": {
                            "message": err,
                            "type": "generation_error"
                        }
                    }),
                );
                "error"
            }
        };

        send_sse_json(
            &tx,
            serde_json::json!({
                "id": &request_id_for_worker,
                "object": "text_completion.chunk",
                "created": created,
                "model": &model_name,
                "choices": [{
                    "index": 0,
                    "text": "",
                    "finish_reason": finish_reason
                }]
            }),
        );
        let _ = tx.send("[DONE]".to_string());
    });

    let events = futures_util::stream::unfold(rx, |mut rx| async {
        rx.recv()
            .await
            .map(|data| (Ok(Event::default().data(data)), rx))
    });
    Sse::new(events)
}

fn send_sse_json(tx: &tokio::sync::mpsc::UnboundedSender<String>, value: serde_json::Value) {
    let _ = tx.send(value.to_string());
}

async fn health(State(state): State<ServerState>) -> Json<serde_json::Value> {
    let model_loaded = state
        .engine
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|engine| engine.model.as_ref())
        .is_some();
    Json(serde_json::json!({
        "status": if model_loaded { "ok" } else { "degraded" },
        "model_loaded": model_loaded,
    }))
}

async fn metrics(State(state): State<ServerState>) -> Json<serde_json::Value> {
    let (pending, running, completed) = state.scheduler.stats();
    let (kv_pages_used, kv_pages_max) = state.scheduler.kv_page_stats();
    let engine = state.engine.lock().unwrap();
    let engine_metrics = engine.as_ref().map(|engine| {
        let capabilities = engine.backend.capabilities();
        serde_json::json!({
            "model_loaded": engine.model.is_some(),
            "backend": engine.backend.name(),
            "backend_available": engine.backend.is_available(),
            "backend_features": capabilities.features,
            "kv_quant": engine.config.kv_quant.name(),
            "weight_quant": engine.config.weight_format.name(),
            "max_seq_len": engine.config.max_seq_len,
            "max_loaded_tensor_elements": engine.config.max_loaded_tensor_elements,
            "num_layers": engine.num_layers,
            "num_heads": engine.num_heads,
            "num_kv_heads": engine.num_kv_heads,
            "hidden_size": engine.hidden_size,
            "vocab_size": engine.vocab_size,
        })
    });

    Json(serde_json::json!({
        "scheduler": {
            "pending": pending,
            "running": running,
            "completed": completed,
            "kv_pages_used": kv_pages_used,
            "kv_pages_max": kv_pages_max,
        },
        "engine": engine_metrics,
        "server": {
            "auth_enabled": state.server_config.api_key.is_some(),
            "cors_origin": state.server_config.cors_origin.clone(),
            "rate_limit_per_minute": state.server_config.rate_limit_per_minute,
        }
    }))
}

async fn list_backends() -> Json<serde_json::Value> {
    let data = BackendFactory::available_backends()
        .into_iter()
        .map(|capability| {
            serde_json::json!({
                "id": capability.backend.name(),
                "available": capability.available,
                "features": capability.features,
                "supported_ops": capability.supported_ops,
            })
        })
        .collect::<Vec<_>>();

    Json(serde_json::json!({
        "object": "list",
        "data": data,
    }))
}

async fn model_info(
    State(state): State<ServerState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let engine = state.engine.lock().unwrap();
    let Some(engine) = engine.as_ref() else {
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    };

    let metadata = engine.model.as_ref().map(|model| {
        serde_json::json!({
            "name": model.metadata.get_str("general.name"),
            "architecture": model.architecture().to_string(),
            "tensor_count": model.metadata.tensor_count,
            "metadata_entries": model.metadata.kv_count,
        })
    });

    Ok(Json(serde_json::json!({
        "id": state.current_model_name(),
        "loaded": engine.model.is_some(),
        "path": state.model_path.lock().unwrap().clone(),
        "backend": engine.backend.name(),
        "kv_quant": engine.config.kv_quant.name(),
        "weight_quant": engine.config.weight_format.name(),
        "max_seq_len": engine.config.max_seq_len,
        "max_loaded_tensor_elements": engine.config.max_loaded_tensor_elements,
        "dimensions": {
            "layers": engine.num_layers,
            "attention_heads": engine.num_heads,
            "kv_heads": engine.num_kv_heads,
            "head_dim": engine.head_dim,
            "hidden_size": engine.hidden_size,
            "ff_dim": engine.ff_dim,
            "vocab_size": engine.vocab_size,
        },
        "metadata": metadata,
    })))
}

async fn list_models(
    State(state): State<ServerState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    Ok(Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": state.current_model_name(),
            "object": "model",
            "created": chrono::Utc::now().timestamp(),
            "owned_by": "nexus",
        }]
    })))
}

/// Start the API server
pub async fn run_server(engine: InferenceEngine, host: &str, port: u16) -> anyhow::Result<()> {
    run_server_with_config(engine, host, port, ServerConfig::default(), None).await
}

pub async fn run_server_with_config(
    engine: InferenceEngine,
    host: &str,
    port: u16,
    server_config: ServerConfig,
    initial_model_path: Option<PathBuf>,
) -> anyhow::Result<()> {
    let model_name = engine
        .model
        .as_ref()
        .and_then(|model| model.metadata.get_str("general.name"))
        .unwrap_or(DEFAULT_MODEL_NAME)
        .to_string();
    let inference_config = engine.config.clone();
    let state = ServerState {
        engine: Arc::new(Mutex::new(Some(engine))),
        scheduler: Arc::new(Scheduler::new(8)),
        model_name: Arc::new(Mutex::new(model_name)),
        model_path: Arc::new(Mutex::new(initial_model_path)),
        inference_config: Arc::new(Mutex::new(inference_config)),
        server_config: Arc::new(server_config),
        rate_limits: Arc::new(Mutex::new(HashMap::new())),
    };

    let app = create_router(state);
    let addr = format!("{}:{}", host, port);

    println!("Nexus API server starting on {}", addr);
    println!("OpenAI-compatible endpoints:");
    println!("  POST {}/v1/chat/completions", addr);
    println!("  POST {}/v1/completions", addr);
    println!("  GET  {}/v1/models", addr);
    println!("  GET  {}/v1/model", addr);
    println!("  POST {}/v1/model/load", addr);
    println!("  POST {}/v1/model/reload", addr);
    println!("  POST {}/v1/model/unload", addr);
    println!("  GET  {}/v1/backends", addr);
    println!("  GET  {}/health", addr);
    println!("  GET  {}/metrics", addr);
    println!();

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
                "weight_quant": "q8_0",
                "backend": "cpu-simd",
                "max_loaded_tensor_elements": 0
            }"#,
        )
        .unwrap();

        let config = config_from_load_request(&req, InferenceConfig::default()).unwrap();
        assert_eq!(config.max_seq_len, 8192);
        assert_eq!(config.kv_quant, KvQuantType::None);
        assert_eq!(config.weight_format, QuantFormat::Q8_0);
        assert_eq!(config.backend, BackendType::CpuSimd);
        assert_eq!(config.max_loaded_tensor_elements, None);
    }

    #[test]
    fn test_parse_unimplemented_backend_errors() {
        let err = parse_backend("cuda").unwrap_err();
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        assert!(err.1.contains("not implemented"));
    }
}
