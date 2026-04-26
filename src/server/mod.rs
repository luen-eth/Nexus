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
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .with_state(state)
}

async fn chat_completions(
    State(state): State<ServerState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, (StatusCode, String)> {
    let mut engine = state.engine.lock().unwrap();
    let model = engine.as_mut().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Model not loaded".to_string(),
        )
    })?;

    let tokenizer =
        RuntimeTokenizer::from_metadata(model.model.as_ref().map(|model| model.metadata()));
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
    state.scheduler.add_request(
        request_id.clone(),
        prompt_tokens.clone(),
        generation.max_tokens,
    );
    state.scheduler.get_next_batch();

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
                created: chrono::Utc::now().timestamp() as u64,
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

            if req.stream.unwrap_or(false) {
                Ok(stream_response(&response).into_response())
            } else {
                Ok(Json(response).into_response())
            }
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

fn stream_response(
    response: &ChatCompletionResponse,
) -> Sse<impl futures_util::Stream<Item = Result<Event, Infallible>>> {
    let choice = &response.choices[0];
    let mut chunks = vec![serde_json::json!({
        "id": response.id,
        "object": "chat.completion.chunk",
        "created": response.created,
        "model": response.model,
        "choices": [{
            "index": choice.index,
            "delta": { "role": choice.message.role },
            "finish_reason": null
        }]
    })];

    for content in choice.message.content.chars().map(|c| c.to_string()) {
        chunks.push(serde_json::json!({
            "id": response.id,
            "object": "chat.completion.chunk",
            "created": response.created,
            "model": response.model,
            "choices": [{
                "index": choice.index,
                "delta": { "content": content },
                "finish_reason": null
            }]
        }));
    }

    chunks.push(serde_json::json!({
        "id": response.id,
        "object": "chat.completion.chunk",
        "created": response.created,
        "model": response.model,
        "choices": [{
            "index": choice.index,
            "delta": {},
            "finish_reason": choice.finish_reason
        }]
    }));

    let events = chunks
        .into_iter()
        .map(|chunk| Ok(Event::default().data(chunk.to_string())))
        .chain(std::iter::once(Ok(Event::default().data("[DONE]"))));
    Sse::new(futures_util::stream::iter(events))
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
    let state = ServerState {
        engine: std::sync::Arc::new(std::sync::Mutex::new(Some(engine))),
        scheduler: Arc::new(Scheduler::new(8)),
        model_name: "nexus-model".to_string(),
    };

    let app = create_router(state);
    let addr = format!("{}:{}", host, port);

    println!("Nexus API server starting on {}", addr);
    println!("OpenAI-compatible endpoints:");
    println!("  POST {}/v1/chat/completions", addr);
    println!("  GET  {}/v1/models", addr);
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
