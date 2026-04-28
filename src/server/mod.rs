//! OpenAI-compatible API server for Nexus inference engine.

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use uuid::Uuid;

use crate::engine::{GenerationConfig, InferenceEngine};
use crate::scheduler::Scheduler;
use crate::tokenizer::{RuntimeTokenizer, Tokenizer};

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

/// Server state
#[derive(Clone)]
pub struct ServerState {
    pub engine: std::sync::Arc<std::sync::Mutex<Option<InferenceEngine>>>,
    pub scheduler: Arc<Scheduler>,
    pub model_name: String,
}

/// Create the API router
pub fn create_router(state: ServerState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/model", get(model_info))
        .route("/v1/models", get(list_models))
        .with_state(state)
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
                model: state.model_name.clone(),
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
    let model_name = state.model_name.clone();
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
        "id": state.model_name,
        "loaded": engine.model.is_some(),
        "backend": engine.backend.name(),
        "kv_quant": engine.config.kv_quant.name(),
        "weight_quant": engine.config.weight_format.name(),
        "max_seq_len": engine.config.max_seq_len,
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
            "id": state.model_name,
            "object": "model",
            "created": chrono::Utc::now().timestamp(),
            "owned_by": "nexus",
        }]
    })))
}

/// Start the API server
pub async fn run_server(engine: InferenceEngine, host: &str, port: u16) -> anyhow::Result<()> {
    let model_name = engine
        .model
        .as_ref()
        .and_then(|model| model.metadata.get_str("general.name"))
        .unwrap_or("nexus-model")
        .to_string();
    let state = ServerState {
        engine: std::sync::Arc::new(std::sync::Mutex::new(Some(engine))),
        scheduler: Arc::new(Scheduler::new(8)),
        model_name,
    };

    let app = create_router(state);
    let addr = format!("{}:{}", host, port);

    println!("Nexus API server starting on {}", addr);
    println!("OpenAI-compatible endpoints:");
    println!("  POST {}/v1/chat/completions", addr);
    println!("  GET  {}/v1/models", addr);
    println!("  GET  {}/v1/model", addr);
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
}
