//! Generation runtime for the API server.
//!
//! A single dedicated OS thread (never a Tokio worker) owns the
//! [`InferenceEngine`]. HTTP handlers submit jobs through a bounded queue and
//! receive results over channels. The worker performs CPU continuous batching
//! by token interleaving: up to `max_active` sequences are live at once, each
//! with its own [`GenerationSession`] (private KV caches and RNG), and every
//! scheduling round advances each runnable sequence by at most one token.
//! This is not a fused batched matmul; it is fair round-robin decoding that
//! lets short requests finish without waiting for long ones.
//!
//! Model lifecycle: a replacement engine is built off the reactor by the caller
//! and handed to the worker. The worker stops admitting new sequences, lets the
//! active ones finish on the model they started with, then swaps the engine and
//! publishes a new [`ModelSnapshot`] in one step. Prompts are tokenized by the
//! worker at admission time, so a request is always tokenized, generated and
//! labelled with the same model.

use std::collections::VecDeque;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use tokio::sync::{mpsc, oneshot};

use crate::backend::BackendType;
use crate::engine::{GenerationConfig, GenerationSession, InferenceEngine};
use crate::scheduler::Scheduler;
use crate::tokenizer::{filter_stop_text, RuntimeTokenizer};

pub const DEFAULT_MODEL_NAME: &str = "nexus-model";
/// Completion length used when a request does not specify `max_tokens`
/// (clamped to the remaining context).
pub const DEFAULT_MAX_TOKENS: usize = 256;
/// Upper bound on concurrently decoding sequences.
pub const MAX_ACTIVE_LIMIT: usize = 64;
/// Smallest per-stream buffer; a step needs room for a delta and a final event.
const MIN_STREAM_BUFFER: usize = 4;
/// Free slots a stream must have before its sequence is advanced.
const STREAM_HEADROOM: usize = 2;
/// Poll interval while every active sequence is blocked on client backpressure.
const BACKPRESSURE_POLL: Duration = Duration::from_millis(2);

/// Tunables for the generation worker.
#[derive(Debug, Clone)]
pub struct RuntimeOptions {
    /// Maximum sequences decoding concurrently (each owns a KV cache).
    pub max_active: usize,
    /// Maximum jobs waiting for a slot; further submissions get 503.
    pub max_pending: usize,
    /// Per-stream event buffer; a full buffer pauses that sequence.
    pub stream_buffer: usize,
    /// A stream whose client has not drained its buffer for this long is cancelled.
    pub stream_stall_timeout: Duration,
    /// Tokens per KV page for scheduler accounting.
    pub kv_page_size_tokens: usize,
}

impl RuntimeOptions {
    /// Options derived from an engine `batch_size` (number of parallel sequences).
    pub fn from_batch_size(batch_size: usize) -> Self {
        let max_active = batch_size.clamp(1, MAX_ACTIVE_LIMIT);
        RuntimeOptions {
            max_active,
            max_pending: max_active * 4,
            ..Default::default()
        }
    }

    pub(crate) fn normalized(mut self) -> Self {
        self.max_active = self.max_active.clamp(1, MAX_ACTIVE_LIMIT);
        // Every job passes through the queue, so it needs at least one slot.
        self.max_pending = self.max_pending.max(1);
        self.stream_buffer = self.stream_buffer.max(MIN_STREAM_BUFFER);
        self.kv_page_size_tokens = self.kv_page_size_tokens.max(1);
        self
    }
}

impl Default for RuntimeOptions {
    fn default() -> Self {
        RuntimeOptions {
            max_active: 4,
            max_pending: 16,
            stream_buffer: 16,
            stream_stall_timeout: Duration::from_secs(30),
            kv_page_size_tokens: 16,
        }
    }
}

/// Immutable view of the currently installed model. Replaced atomically on
/// load/reload/unload, so every field always describes the same engine.
#[derive(Debug, Clone)]
pub struct ModelSnapshot {
    /// Increments on every install.
    pub epoch: u64,
    pub loaded: bool,
    pub name: String,
    pub path: Option<PathBuf>,
    pub backend: BackendType,
    pub max_seq_len: usize,
    /// Body of `GET /v1/model` (without `id`/`path`).
    pub info: serde_json::Value,
    /// Engine section of `GET /metrics`.
    pub engine_metrics: serde_json::Value,
    /// Engine configuration as actually applied by the loader.
    pub config: crate::engine::InferenceConfig,
    /// Operator settings before model-specific context clamping.
    pub requested_config: crate::engine::InferenceConfig,
}

impl ModelSnapshot {
    fn from_engine(
        engine: &InferenceEngine,
        path: Option<PathBuf>,
        epoch: u64,
        requested_config: crate::engine::InferenceConfig,
    ) -> Self {
        let loaded = engine.model.is_some();
        let name = model_name(engine, path.as_ref());
        let capabilities = engine.backend.capabilities();
        let config = &engine.config;
        let metadata = engine.model.as_ref().map(|model| {
            serde_json::json!({
                "name": model.metadata.get_str("general.name"),
                "architecture": model.architecture().to_string(),
                "tensor_count": model.metadata.tensor_count,
                "metadata_entries": model.metadata.kv_count,
            })
        });
        let info = serde_json::json!({
            "loaded": loaded,
            "backend": engine.backend.name(),
            "device_weight_bytes":engine.backend.resident_weight_bytes(),
            "host_weight_payload_bytes": engine.weight_memory_usage(),
            "engine_kv_payload_bytes": engine.kv_cache_memory_usage(),
            "tokenizer":engine.tokenizer.as_ref().map(|t|t.kind()),
            "execution_mode":if engine.backend.backend_type()==BackendType::Cuda {"cuda-matvec-cpu-attention"} else {"cpu"},
            "device_ops":if engine.backend.backend_type()==BackendType::Cuda {capabilities.supported_ops.clone()} else {vec![]},
            "kv_quant": config.kv_quant.name(),
            "weight_quant": config.weight_format.name(),
            "max_seq_len": config.max_seq_len,
            "max_loaded_tensor_elements": config.max_loaded_tensor_elements,
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
        });
        let engine_metrics = serde_json::json!({
            "model_loaded": loaded,
            "backend": engine.backend.name(),
            "backend_available": engine.backend.is_available(),
            "backend_features": capabilities.features,
            "device_weight_bytes":engine.backend.resident_weight_bytes(),
            "host_weight_payload_bytes": engine.weight_memory_usage(),
            "engine_kv_payload_bytes": engine.kv_cache_memory_usage(),
            "tokenizer":engine.tokenizer.as_ref().map(|t|t.kind()),
            "execution_mode":if engine.backend.backend_type()==BackendType::Cuda {"cuda-matvec-cpu-attention"} else {"cpu"},
            "device_ops":if engine.backend.backend_type()==BackendType::Cuda {capabilities.supported_ops.clone()} else {vec![]},
            "kv_quant": config.kv_quant.name(),
            "weight_quant": config.weight_format.name(),
            "max_seq_len": config.max_seq_len,
            "max_loaded_tensor_elements": config.max_loaded_tensor_elements,
            "num_layers": engine.num_layers,
            "num_heads": engine.num_heads,
            "num_kv_heads": engine.num_kv_heads,
            "hidden_size": engine.hidden_size,
            "vocab_size": engine.vocab_size,
        });
        ModelSnapshot {
            epoch,
            loaded,
            name,
            path,
            backend: engine.backend.backend_type(),
            max_seq_len: config.max_seq_len,
            info,
            engine_metrics,
            config: config.clone(),
            requested_config,
        }
    }
}

/// Model id reported by the API: GGUF `general.name`, else the file stem.
pub fn model_name(engine: &InferenceEngine, path: Option<&PathBuf>) -> String {
    let Some(model) = engine.model.as_ref() else {
        return DEFAULT_MODEL_NAME.to_string();
    };
    model
        .metadata
        .get_str("general.name")
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .or_else(|| {
            path.and_then(|path| path.file_stem())
                .and_then(|stem| stem.to_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| DEFAULT_MODEL_NAME.to_string())
}

/// One chat turn.
#[derive(Debug, Clone)]
pub struct ChatTurn {
    pub role: String,
    pub content: String,
}

/// Prompt to be tokenized by the worker with the model it will run on.
#[derive(Debug, Clone)]
pub enum PromptInput {
    Text(String),
    Chat(Vec<ChatTurn>),
}

/// Sampling parameters as supplied by the client.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// `None` means [`DEFAULT_MAX_TOKENS`] clamped to the remaining context.
    pub max_tokens: Option<usize>,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            max_tokens: None,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            seed: None,
        }
    }
}

impl SamplingParams {
    /// Reject parameters the engine would refuse, so the client gets a 400.
    pub fn validate(&self) -> Result<(), RuntimeError> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(RuntimeError::invalid(
                "temperature must be a finite number >= 0",
            ));
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(RuntimeError::invalid("top_p must be in (0, 1]"));
        }
        if !self.repetition_penalty.is_finite() || self.repetition_penalty <= 0.0 {
            return Err(RuntimeError::invalid(
                "repetition_penalty must be a finite number > 0",
            ));
        }
        Ok(())
    }
}

/// A generation job.
#[derive(Debug, Clone)]
pub struct GenerationRequest {
    /// Unique id (also used as the scheduler request id).
    pub id: String,
    pub prompt: PromptInput,
    /// Stop strings; empty strings are ignored.
    pub stop: Vec<String>,
    pub sampling: SamplingParams,
}

/// Result of a finished non-streaming job.
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    pub id: String,
    /// Model that tokenized and generated this output.
    pub model: String,
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeErrorKind {
    /// No model is installed.
    NotLoaded,
    /// Queue or KV budget exhausted.
    Busy,
    /// The request cannot be served as specified.
    InvalidRequest,
    /// Engine failure.
    Internal,
    /// Worker is gone.
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeError {
    pub kind: RuntimeErrorKind,
    pub message: String,
}

impl RuntimeError {
    pub fn new(kind: RuntimeErrorKind, message: impl Into<String>) -> Self {
        RuntimeError {
            kind,
            message: message.into(),
        }
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(RuntimeErrorKind::InvalidRequest, message)
    }

    fn not_loaded() -> Self {
        Self::new(RuntimeErrorKind::NotLoaded, "Model not loaded")
    }

    fn unavailable() -> Self {
        Self::new(
            RuntimeErrorKind::Unavailable,
            "Generation worker is not running",
        )
    }

    pub fn status_code(&self) -> StatusCode {
        match self.kind {
            RuntimeErrorKind::NotLoaded
            | RuntimeErrorKind::Busy
            | RuntimeErrorKind::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
            RuntimeErrorKind::InvalidRequest => StatusCode::BAD_REQUEST,
            RuntimeErrorKind::Internal => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RuntimeError {}

/// Events delivered to a streaming client, in order:
/// `Started`, zero or more `Delta`, then exactly one `Finished` or `Error`.
/// The channel closes without a terminal event only if the stream was
/// cancelled for stalling or the worker shut down.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    Started {
        model: String,
        prompt_tokens: usize,
    },
    Delta(String),
    Finished {
        finish_reason: String,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
    Error(RuntimeError),
}

/// Live counters for `/metrics`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeStats {
    pub active: usize,
    pub queued: usize,
    pub max_active: usize,
    pub max_pending: usize,
    pub model_swap_pending: bool,
    pub cancelled_total: usize,
    pub rejected_busy_total: usize,
    pub gpu_kernel_launches: u64,
    pub backend_healthy: bool,
    /// Allocated host KV payload of all live sessions, excluding RSS/allocator/GPU data.
    pub active_kv_payload_bytes: usize,
}

type OnceReply = oneshot::Sender<Result<GenerationOutput, RuntimeError>>;

enum Reply {
    Once(OnceReply),
    Stream(mpsc::Sender<StreamEvent>),
}

impl Reply {
    fn is_closed(&self) -> bool {
        match self {
            Reply::Once(tx) => tx.is_closed(),
            Reply::Stream(tx) => tx.is_closed(),
        }
    }

    fn fail(self, err: RuntimeError) {
        match self {
            Reply::Once(tx) => {
                let _ = tx.send(Err(err));
            }
            Reply::Stream(tx) => {
                let _ = tx.try_send(StreamEvent::Error(err));
            }
        }
    }
}

struct Job {
    request: GenerationRequest,
    // Checked again at admission: a queued job must never silently switch models.
    expected_model: Option<String>,
    reply: Reply,
}

struct Install {
    requested_config: crate::engine::InferenceConfig,
    engine: Box<InferenceEngine>,
    path: Option<PathBuf>,
    reply: oneshot::Sender<Arc<ModelSnapshot>>,
}

#[derive(Default)]
struct Queue {
    jobs: VecDeque<Job>,
    installs: VecDeque<Install>,
    shutdown: bool,
}

struct Shared {
    queue: Mutex<Queue>,
    wake: Condvar,
    snapshot: RwLock<Arc<ModelSnapshot>>,
    scheduler: Arc<Scheduler>,
    options: RuntimeOptions,
    active: AtomicUsize,
    swap_pending: AtomicBool,
    alive: AtomicBool,
    cancelled_total: AtomicUsize,
    rejected_busy_total: AtomicUsize,
    gpu_kernel_launches: AtomicU64,
    backend_healthy: AtomicBool,
    active_kv_payload_bytes: AtomicUsize,
}

impl Shared {
    fn lock_queue(&self) -> MutexGuard<'_, Queue> {
        self.queue
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn snapshot(&self) -> Arc<ModelSnapshot> {
        self.snapshot
            .read()
            .unwrap_or_else(|poison| poison.into_inner())
            .clone()
    }

    fn publish(&self, snapshot: Arc<ModelSnapshot>) {
        *self
            .snapshot
            .write()
            .unwrap_or_else(|poison| poison.into_inner()) = snapshot;
    }
}

/// Handle to the generation worker. Dropping the last handle stops the worker;
/// queued and active jobs then fail with 503.
pub struct Runtime {
    shared: Arc<Shared>,
    lifecycle: tokio::sync::Mutex<()>,
}

impl Runtime {
    /// Spawn the worker thread that owns `engine`.
    pub fn start(
        engine: InferenceEngine,
        model_path: Option<PathBuf>,
        scheduler: Arc<Scheduler>,
        options: RuntimeOptions,
    ) -> std::io::Result<Self> {
        let requested_config = engine.config.clone();
        Self::start_with_requested_config(engine, model_path, scheduler, options, requested_config)
    }

    pub fn start_with_requested_config(
        engine: InferenceEngine,
        model_path: Option<PathBuf>,
        scheduler: Arc<Scheduler>,
        options: RuntimeOptions,
        requested_config: crate::engine::InferenceConfig,
    ) -> std::io::Result<Self> {
        let options = options.normalized();
        let model_path = model_path.filter(|_| engine.model.is_some());
        let snapshot = Arc::new(ModelSnapshot::from_engine(
            &engine,
            model_path,
            0,
            requested_config,
        ));
        scheduler.set_kv_page_budget(kv_page_budget(&scheduler, &options, &snapshot));
        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue::default()),
            wake: Condvar::new(),
            snapshot: RwLock::new(snapshot.clone()),
            scheduler,
            options,
            active: AtomicUsize::new(0),
            swap_pending: AtomicBool::new(false),
            alive: AtomicBool::new(true),
            cancelled_total: AtomicUsize::new(0),
            rejected_busy_total: AtomicUsize::new(0),
            gpu_kernel_launches: AtomicU64::new(0),
            backend_healthy: AtomicBool::new(engine.backend.is_available()),
            active_kv_payload_bytes: AtomicUsize::new(0),
        });

        let worker_shared = shared.clone();
        thread::Builder::new()
            .name("nexus-generation".to_string())
            .spawn(move || {
                let guard = WorkerExitGuard(worker_shared.clone());
                let tokenizer = tokenizer_for(&engine);
                Worker {
                    shared: worker_shared,
                    engine,
                    tokenizer,
                    snapshot,
                    active: Vec::new(),
                }
                .run();
                drop(guard);
            })?;

        Ok(Runtime {
            shared,
            lifecycle: tokio::sync::Mutex::new(()),
        })
    }

    /// Current model snapshot (never blocks on generation).
    pub fn snapshot(&self) -> Arc<ModelSnapshot> {
        self.shared.snapshot()
    }

    pub fn options(&self) -> &RuntimeOptions {
        &self.shared.options
    }

    /// Whether the generation worker thread is running.
    pub fn is_alive(&self) -> bool {
        self.shared.alive.load(Ordering::SeqCst)
    }

    pub fn scheduler(&self) -> &Arc<Scheduler> {
        &self.shared.scheduler
    }

    pub fn stats(&self) -> RuntimeStats {
        let queued = self.shared.lock_queue().jobs.len();
        RuntimeStats {
            active: self.shared.active.load(Ordering::SeqCst),
            queued,
            max_active: self.shared.options.max_active,
            max_pending: self.shared.options.max_pending,
            model_swap_pending: self.shared.swap_pending.load(Ordering::SeqCst),
            cancelled_total: self.shared.cancelled_total.load(Ordering::SeqCst),
            rejected_busy_total: self.shared.rejected_busy_total.load(Ordering::SeqCst),
            gpu_kernel_launches: self.shared.gpu_kernel_launches.load(Ordering::Relaxed),
            backend_healthy: self.shared.backend_healthy.load(Ordering::Relaxed),
            active_kv_payload_bytes: self.shared.active_kv_payload_bytes.load(Ordering::Relaxed),
        }
    }

    /// Serialize model lifecycle operations (load/reload/unload). Hold the
    /// guard while building the replacement engine and installing it.
    pub async fn lifecycle_guard(&self) -> tokio::sync::MutexGuard<'_, ()> {
        self.lifecycle.lock().await
    }

    /// Queue a non-streaming job. Dropping the receiver cancels the job.
    pub fn submit(
        &self,
        request: GenerationRequest,
    ) -> Result<oneshot::Receiver<Result<GenerationOutput, RuntimeError>>, RuntimeError> {
        let (tx, rx) = oneshot::channel();
        self.enqueue(request, None, Reply::Once(tx))?;
        Ok(rx)
    }

    /// Queue a streaming job. Dropping the receiver cancels the job; a full
    /// receiver pauses the sequence (bounded backpressure).
    pub fn submit_stream(
        &self,
        request: GenerationRequest,
    ) -> Result<mpsc::Receiver<StreamEvent>, RuntimeError> {
        let (tx, rx) = mpsc::channel(self.shared.options.stream_buffer);
        self.enqueue(request, None, Reply::Stream(tx))?;
        Ok(rx)
    }

    /// Submit an HTTP request pinned to the client's canonical model id.
    pub fn submit_for_model(
        &self,
        request: GenerationRequest,
        model: String,
    ) -> Result<oneshot::Receiver<Result<GenerationOutput, RuntimeError>>, RuntimeError> {
        let (tx, rx) = oneshot::channel();
        self.enqueue(request, Some(model), Reply::Once(tx))?;
        Ok(rx)
    }

    pub fn submit_stream_for_model(
        &self,
        request: GenerationRequest,
        model: String,
    ) -> Result<mpsc::Receiver<StreamEvent>, RuntimeError> {
        let (tx, rx) = mpsc::channel(self.shared.options.stream_buffer);
        self.enqueue(request, Some(model), Reply::Stream(tx))?;
        Ok(rx)
    }

    fn enqueue(
        &self,
        request: GenerationRequest,
        expected_model: Option<String>,
        reply: Reply,
    ) -> Result<(), RuntimeError> {
        request.sampling.validate()?;
        let snapshot = self.shared.snapshot();
        if !snapshot.loaded {
            return Err(RuntimeError::not_loaded());
        }
        validate_model_id(expected_model.as_deref(), &snapshot.name)?;
        let mut queue = self.shared.lock_queue();
        if queue.shutdown || !self.shared.alive.load(Ordering::SeqCst) {
            return Err(RuntimeError::unavailable());
        }
        if queue.jobs.len() >= self.shared.options.max_pending {
            self.shared
                .rejected_busy_total
                .fetch_add(1, Ordering::SeqCst);
            return Err(RuntimeError::new(
                RuntimeErrorKind::Busy,
                format!(
                    "Server at capacity ({} active, {} queued); retry later",
                    self.shared.options.max_active, self.shared.options.max_pending
                ),
            ));
        }
        queue.jobs.push_back(Job {
            request,
            expected_model,
            reply,
        });
        drop(queue);
        self.shared.wake.notify_one();
        Ok(())
    }

    /// Replace the engine. Waits until all sequences admitted on the current
    /// model have finished, then swaps and publishes the new snapshot. The
    /// caller should build `engine` off the async reactor (`spawn_blocking`).
    pub async fn install(
        &self,
        engine: InferenceEngine,
        path: Option<PathBuf>,
    ) -> Result<Arc<ModelSnapshot>, RuntimeError> {
        let requested_config = engine.config.clone();
        self.install_with_requested_config(engine, path, requested_config)
            .await
    }

    pub async fn install_with_requested_config(
        &self,
        engine: InferenceEngine,
        path: Option<PathBuf>,
        requested_config: crate::engine::InferenceConfig,
    ) -> Result<Arc<ModelSnapshot>, RuntimeError> {
        let (tx, rx) = oneshot::channel();
        {
            let mut queue = self.shared.lock_queue();
            if queue.shutdown || !self.shared.alive.load(Ordering::SeqCst) {
                return Err(RuntimeError::unavailable());
            }
            self.shared.swap_pending.store(true, Ordering::SeqCst);
            queue.installs.push_back(Install {
                engine: Box::new(engine),
                requested_config,
                path,
                reply: tx,
            });
        }
        self.shared.wake.notify_one();
        rx.await.map_err(|_| RuntimeError::unavailable())
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        self.shared.lock_queue().shutdown = true;
        self.shared.wake.notify_all();
    }
}

fn validate_model_id(expected: Option<&str>, installed: &str) -> Result<(), RuntimeError> {
    if expected.is_some_and(|model| model != installed) {
        return Err(RuntimeError::invalid(format!(
            "Requested model '{}' is not installed (current model: '{}')",
            expected.unwrap_or_default(),
            installed
        )));
    }
    Ok(())
}

fn kv_page_budget(scheduler: &Scheduler, options: &RuntimeOptions, snap: &ModelSnapshot) -> usize {
    scheduler
        .pages_needed(snap.max_seq_len.max(1), 0)
        .saturating_mul(options.max_active)
}

fn tokenizer_for(engine: &InferenceEngine) -> RuntimeTokenizer {
    engine.tokenizer.clone().unwrap_or_else(|| {
        RuntimeTokenizer::from_metadata(engine.model.as_ref().map(|model| model.metadata()))
    })
}

/// Marks the worker dead and fails anything still queued, even on panic.
struct WorkerExitGuard(Arc<Shared>);

impl Drop for WorkerExitGuard {
    fn drop(&mut self) {
        self.0.alive.store(false, Ordering::SeqCst);
        self.0.active.store(0, Ordering::SeqCst);
        self.0.active_kv_payload_bytes.store(0, Ordering::Relaxed);
        let (jobs, installs) = {
            let mut queue = self.0.lock_queue();
            queue.shutdown = true;
            (
                std::mem::take(&mut queue.jobs),
                std::mem::take(&mut queue.installs),
            )
        };
        for job in jobs {
            self.0.scheduler.complete_request(&job.request.id);
            job.reply.fail(RuntimeError::unavailable());
        }
        drop(installs);
    }
}

struct ActiveSeq {
    id: String,
    session: GenerationSession,
    reply: Reply,
    /// Text already delivered to a streaming client.
    sent_text: String,
    stalled_since: Option<Instant>,
    stop_strings: Vec<String>,
    stopped_text: Option<String>,
}

enum StepOutcome {
    Keep { progressed: bool },
    Remove,
}

struct Worker {
    shared: Arc<Shared>,
    engine: InferenceEngine,
    tokenizer: RuntimeTokenizer,
    snapshot: Arc<ModelSnapshot>,
    active: Vec<ActiveSeq>,
}

impl Worker {
    fn run(mut self) {
        loop {
            let Some(next) = self.wait_for_work() else {
                self.shutdown();
                return;
            };
            match next {
                Work::Install(install) => self.install(install),
                Work::Admit(job) => self.admit(job),
                Work::Step => {}
            }

            if !self.step_all() && !self.active.is_empty() {
                // Every sequence is waiting on its client; avoid spinning while
                // staying responsive to new jobs and installs.
                let queue = self.shared.lock_queue();
                if !self.has_actionable_work(&queue) {
                    let _ = self.shared.wake.wait_timeout(queue, BACKPRESSURE_POLL);
                }
            }
        }
    }

    /// Whether the queue holds something the worker can act on right now
    /// (as opposed to work blocked behind backpressure or a draining swap).
    fn has_actionable_work(&self, queue: &Queue) -> bool {
        if queue.shutdown {
            return true;
        }
        if !queue.installs.is_empty() {
            return self.active.is_empty();
        }
        !queue.jobs.is_empty() && self.active.len() < self.shared.options.max_active
    }

    /// Block until there is something to do. Returns `None` on shutdown.
    fn wait_for_work(&mut self) -> Option<Work> {
        let mut queue = self.shared.lock_queue();
        loop {
            if queue.shutdown {
                return None;
            }
            if !queue.installs.is_empty() {
                // Drain: admit nothing new until the current model is idle.
                if self.active.is_empty() {
                    return queue.installs.pop_front().map(Work::Install);
                }
                return Some(Work::Step);
            }
            // Admit one job per round; step_session advances one prefill or decode token.
            if self.active.len() < self.shared.options.max_active {
                if let Some(job) = queue.jobs.pop_front() {
                    return Some(Work::Admit(job));
                }
            }
            if !self.active.is_empty() {
                return Some(Work::Step);
            }
            queue = self
                .shared
                .wake
                .wait(queue)
                .unwrap_or_else(|poison| poison.into_inner());
        }
    }

    fn install(&mut self, install: Install) {
        let Install {
            engine,
            requested_config,
            path,
            reply,
        } = install;
        let path = path.filter(|_| engine.model.is_some());
        let old = std::mem::replace(&mut self.engine, *engine);
        drop(old);
        self.tokenizer = tokenizer_for(&self.engine);
        self.shared.gpu_kernel_launches.store(0, Ordering::Relaxed);
        self.shared
            .backend_healthy
            .store(self.engine.backend.is_available(), Ordering::Relaxed);
        let snapshot = Arc::new(ModelSnapshot::from_engine(
            &self.engine,
            path,
            self.snapshot.epoch + 1,
            requested_config,
        ));
        self.shared.scheduler.set_kv_page_budget(kv_page_budget(
            &self.shared.scheduler,
            &self.shared.options,
            &snapshot,
        ));
        self.snapshot = snapshot.clone();
        {
            // Publish under the queue lock so `swap_pending` and the snapshot
            // change together from the point of view of `stats()`.
            let queue = self.shared.lock_queue();
            self.shared.publish(snapshot.clone());
            self.shared
                .swap_pending
                .store(!queue.installs.is_empty(), Ordering::SeqCst);
        }
        tracing::info!(
            "Installed model '{}' (epoch {})",
            snapshot.name,
            snapshot.epoch
        );
        let _ = reply.send(snapshot);
    }

    fn admit(&mut self, job: Job) {
        let Job {
            request,
            expected_model,
            reply,
        } = job;
        if reply.is_closed() {
            self.shared.cancelled_total.fetch_add(1, Ordering::SeqCst);
            return;
        }
        if !self.snapshot.loaded {
            reply.fail(RuntimeError::not_loaded());
            return;
        }
        if let Err(err) = validate_model_id(expected_model.as_deref(), &self.snapshot.name) {
            reply.fail(err);
            return;
        }
        let (prompt, generation) = match self.prepare(&request) {
            Ok(prepared) => prepared,
            Err(err) => {
                reply.fail(err);
                return;
            }
        };

        let scheduler = &self.shared.scheduler;
        if !scheduler.can_ever_fit(prompt.len(), generation.max_tokens) {
            reply.fail(RuntimeError::invalid(
                "Request exceeds the per-sequence KV budget",
            ));
            return;
        }
        scheduler.add_request(request.id.clone(), prompt.clone(), generation.max_tokens);
        let admitted = scheduler
            .get_next_batch()
            .iter()
            .any(|req| req.id == request.id);
        if !admitted {
            scheduler.complete_request(&request.id);
            self.shared
                .rejected_busy_total
                .fetch_add(1, Ordering::SeqCst);
            reply.fail(RuntimeError::new(
                RuntimeErrorKind::Busy,
                "KV cache budget exhausted; retry later",
            ));
            return;
        }

        let engine = &mut self.engine;
        let started = catch_unwind(AssertUnwindSafe(|| {
            engine.start_session(&prompt, &generation)
        }));
        let session = match flatten_panic(started) {
            Ok(session) => session,
            Err(err) => {
                scheduler.complete_request(&request.id);
                reply.fail(RuntimeError::new(
                    RuntimeErrorKind::Internal,
                    format!("Generation error: {err}"),
                ));
                return;
            }
        };

        if let Reply::Stream(tx) = &reply {
            // Fresh channel with capacity >= MIN_STREAM_BUFFER; cannot be full.
            let _ = tx.try_send(StreamEvent::Started {
                model: self.snapshot.name.clone(),
                prompt_tokens: session.prompt_len,
            });
        }
        self.active.push(ActiveSeq {
            id: request.id,
            session,
            reply,
            sent_text: String::new(),
            stalled_since: None,
            stop_strings: request.stop,
            stopped_text: None,
        });
        self.shared
            .active
            .store(self.active.len(), Ordering::SeqCst);
        self.shared.active_kv_payload_bytes.store(
            self.active
                .iter()
                .map(|seq| seq.session.kv_cache_memory_usage())
                .sum(),
            Ordering::Relaxed,
        );
    }

    /// Tokenize with the installed model and resolve limits.
    fn prepare(
        &self,
        request: &GenerationRequest,
    ) -> Result<(Vec<u32>, GenerationConfig), RuntimeError> {
        request.sampling.validate()?;
        let (text, is_chat) = match &request.prompt {
            PromptInput::Text(text) => (text.clone(), false),
            PromptInput::Chat(turns) => (
                self.tokenizer
                    .try_render_chat_messages(
                        turns
                            .iter()
                            .map(|turn| (turn.role.as_str(), turn.content.as_str())),
                        true,
                    )
                    .map_err(RuntimeError::invalid)?,
                true,
            ),
        };
        let prompt = if is_chat {
            self.tokenizer.encode_fragment(&text)
        } else {
            self.tokenizer.try_encode(&text)
        }
        .map_err(RuntimeError::invalid)?;
        if prompt.is_empty() {
            return Err(RuntimeError::invalid("Prompt is empty"));
        }
        let ctx = self.snapshot.max_seq_len;
        if prompt
            .iter()
            .any(|&token| token as usize >= self.engine.vocab_size)
        {
            return Err(RuntimeError::invalid(
                "Prompt token is outside the model vocabulary",
            ));
        }
        if prompt.len() > ctx {
            return Err(RuntimeError::invalid(format!(
                "Prompt has {} tokens but the model context length is {}",
                prompt.len(),
                ctx
            )));
        }
        let remaining = ctx - prompt.len();
        if remaining == 0 && request.sampling.max_tokens != Some(0) {
            return Err(RuntimeError::invalid(
                "Prompt fills the context; request max_tokens=0 explicitly or shorten the prompt",
            ));
        }
        let max_tokens = match request.sampling.max_tokens {
            Some(requested) if requested > remaining => {
                return Err(RuntimeError::invalid(format!(
                    "max_tokens ({requested}) exceeds the remaining context ({remaining} tokens \
                     after a {}-token prompt)",
                    prompt.len()
                )));
            }
            Some(requested) => requested,
            None => DEFAULT_MAX_TOKENS.min(remaining),
        };
        let stop_sequences = Vec::new(); // API stops are matched on decoded text.
        let sampling = &request.sampling;
        Ok((
            prompt,
            GenerationConfig {
                max_tokens,
                temperature: sampling.temperature,
                top_p: sampling.top_p,
                top_k: sampling.top_k,
                repetition_penalty: sampling.repetition_penalty,
                seed: sampling.seed,
                stop_sequences,
            },
        ))
    }

    /// One scheduling round: advance each runnable sequence by at most one
    /// token. Returns whether any sequence made progress.
    fn step_all(&mut self) -> bool {
        let mut any_progress = false;
        let mut index = 0;
        while index < self.active.len() {
            match self.step_one(index) {
                StepOutcome::Keep { progressed } => {
                    any_progress |= progressed;
                    index += 1;
                }
                StepOutcome::Remove => {
                    any_progress = true;
                    let seq = self.active.swap_remove(index);
                    self.shared.scheduler.complete_request(&seq.id);
                }
            }
        }
        self.shared
            .active
            .store(self.active.len(), Ordering::SeqCst);
        self.shared.active_kv_payload_bytes.store(
            self.active
                .iter()
                .map(|seq| seq.session.kv_cache_memory_usage())
                .sum(),
            Ordering::Relaxed,
        );
        self.shared
            .gpu_kernel_launches
            .store(self.engine.backend.kernel_launches(), Ordering::Relaxed);
        self.shared
            .backend_healthy
            .store(self.engine.backend.is_available(), Ordering::Relaxed);
        any_progress
    }

    fn step_one(&mut self, index: usize) -> StepOutcome {
        let stall_timeout = self.shared.options.stream_stall_timeout;
        let seq = &mut self.active[index];

        if seq.reply.is_closed() {
            self.shared.cancelled_total.fetch_add(1, Ordering::SeqCst);
            return StepOutcome::Remove;
        }
        if let Reply::Stream(tx) = &seq.reply {
            if tx.capacity() < STREAM_HEADROOM {
                let since = *seq.stalled_since.get_or_insert_with(Instant::now);
                if since.elapsed() >= stall_timeout {
                    tracing::warn!(
                        "Cancelling stream {}: client did not read for {:?}",
                        seq.id,
                        stall_timeout
                    );
                    self.shared.cancelled_total.fetch_add(1, Ordering::SeqCst);
                    return StepOutcome::Remove;
                }
                return StepOutcome::Keep { progressed: false };
            }
            seq.stalled_since = None;
        }

        if !seq.session.finished {
            let engine = &mut self.engine;
            let session = &mut seq.session;
            let stepped = catch_unwind(AssertUnwindSafe(|| engine.step_session(session)));
            match flatten_panic(stepped) {
                Ok(Some(token)) => {
                    self.shared.scheduler.append_generated(&seq.id, token);
                }
                Ok(None) => {}
                Err(err) => {
                    take_reply(&mut seq.reply).fail(RuntimeError::new(
                        RuntimeErrorKind::Internal,
                        format!("Generation error: {err}"),
                    ));
                    return StepOutcome::Remove;
                }
            }
        }

        let seq = &mut self.active[index];
        let needs_intermediate_text = !seq.stop_strings.is_empty()
            || (matches!(&seq.reply, Reply::Stream(_)) && self.tokenizer.prefix_stable());
        if !seq.session.finished && !needs_intermediate_text {
            // Non-streaming jobs without text stops need one final decode only.
            // Revision-prone streaming decoders also flush only at completion.
            return StepOutcome::Keep { progressed: true };
        }
        let decoded = match safe_text(
            &self.tokenizer,
            seq.session.visible_output(),
            seq.session.finished,
        ) {
            Ok(text) => text,
            Err(err) => {
                take_reply(&mut seq.reply).fail(RuntimeError::new(RuntimeErrorKind::Internal, err));
                return StepOutcome::Remove;
            }
        };
        // Stops terminate on the current decoded prefix, even if streaming is
        // buffered. Once matched, no future token can revise that prefix.
        let (visible, matched) =
            filter_stop_text(&decoded, &seq.stop_strings, seq.session.finished);
        if matched {
            seq.stopped_text = Some(visible.to_string());
            seq.session.finished = true;
            seq.session.finish_reason = "stop";
        }
        if seq.session.finished {
            self.finish(index);
            return StepOutcome::Remove;
        }
        if self.tokenizer.prefix_stable() {
            if let Reply::Stream(tx) = &seq.reply {
                match take_delta(&mut seq.sent_text, visible.to_string()) {
                    Ok(Some(delta)) => {
                        let _ = tx.try_send(StreamEvent::Delta(delta));
                    }
                    Ok(None) => {}
                    Err(err) => {
                        take_reply(&mut seq.reply)
                            .fail(RuntimeError::new(RuntimeErrorKind::Internal, err));
                        return StepOutcome::Remove;
                    }
                }
            }
        }
        StepOutcome::Keep { progressed: true }
    }

    /// Deliver the final result. The sequence is removed by the caller.
    fn finish(&mut self, index: usize) {
        let model = self.snapshot.name.clone();
        let seq = &mut self.active[index];
        let session = &seq.session;
        let completion = session.visible_output();
        let completion_tokens = if let Some(text) = &seq.stopped_text {
            if text.is_empty() {
                0
            } else {
                (1..=completion.len())
                    .find(|&n| {
                        safe_text(&self.tokenizer, &completion[..n], false)
                            .is_ok_and(|decoded| decoded.len() >= text.len())
                    })
                    .unwrap_or(completion.len())
            }
        } else {
            completion.len()
        };
        let prompt_tokens = session.prompt_len;
        let finish_reason = session.finish_reason.to_string();
        let text = if let Some(text) = &seq.stopped_text {
            text.clone()
        } else {
            match safe_text(&self.tokenizer, completion, true) {
                Ok(text) => text,
                Err(err) => {
                    take_reply(&mut seq.reply)
                        .fail(RuntimeError::new(RuntimeErrorKind::Internal, err));
                    return;
                }
            }
        };
        match take_reply(&mut seq.reply) {
            Reply::Once(tx) => {
                let _ = tx.send(Ok(GenerationOutput {
                    id: seq.id.clone(),
                    model,
                    text,
                    prompt_tokens,
                    completion_tokens,
                    finish_reason,
                }));
            }
            Reply::Stream(tx) => {
                match take_delta(&mut seq.sent_text, text) {
                    Ok(Some(delta)) => {
                        let _ = tx.try_send(StreamEvent::Delta(delta));
                    }
                    Ok(None) => {}
                    Err(err) => {
                        Reply::Stream(tx).fail(RuntimeError::new(RuntimeErrorKind::Internal, err));
                        return;
                    }
                }
                let _ = tx.try_send(StreamEvent::Finished {
                    finish_reason,
                    prompt_tokens,
                    completion_tokens,
                });
            }
        }
    }

    fn shutdown(&mut self) {
        for seq in self.active.drain(..) {
            self.shared.scheduler.complete_request(&seq.id);
            seq.reply.fail(RuntimeError::unavailable());
        }
        self.shared.active.store(0, Ordering::SeqCst);
        self.shared
            .active_kv_payload_bytes
            .store(0, Ordering::Relaxed);
        // Queued jobs are failed by WorkerExitGuard.
    }
}

/// Move the reply out of a sequence that is about to be removed.
fn take_reply(reply: &mut Reply) -> Reply {
    std::mem::replace(reply, Reply::Once(oneshot::channel().0))
}

enum Work {
    Install(Install),
    Admit(Job),
    Step,
}

fn flatten_panic<T>(result: std::thread::Result<Result<T, String>>) -> Result<T, String> {
    match result {
        Ok(inner) => inner,
        Err(panic) => {
            let message = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".to_string());
            Err(format!("engine panicked: {message}"))
        }
    }
}

/// Decode `tokens`; while a sequence is still running, drop a trailing
/// replacement character produced by an incomplete multi-byte UTF-8 sequence
/// so it is never streamed as garbage.
fn safe_text(
    tokenizer: &RuntimeTokenizer,
    tokens: &[u32],
    finished: bool,
) -> Result<String, String> {
    let mut text = tokenizer.try_decode(tokens)?;
    if !finished {
        while text.ends_with('\u{FFFD}') {
            text.pop();
        }
    }
    Ok(text)
}

/// Return the new suffix; a decoder revision is an explicit stream failure.
fn take_delta(sent: &mut String, text: String) -> Result<Option<String>, String> {
    if !text.starts_with(sent.as_str()) {
        return Err("Tokenizer revised text already delivered to the stream".into());
    }
    if text.len() == sent.len() {
        return Ok(None);
    }
    let delta = text[sent.len()..].to_string();
    *sent = text;
    Ok(Some(delta))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_delta_is_prefix_safe() {
        let mut sent = String::new();
        assert_eq!(
            take_delta(&mut sent, "he".into()).unwrap(),
            Some("he".into())
        );
        assert_eq!(take_delta(&mut sent, "he".into()).unwrap(), None);
        assert_eq!(
            take_delta(&mut sent, "hex".into()).unwrap(),
            Some("x".into())
        );
        // Revisions must terminate the stream with a visible error.
        assert!(take_delta(&mut sent, "hey!".into()).is_err());
        assert_eq!(sent, "hex");
    }

    #[test]
    fn safe_text_withholds_incomplete_utf8() {
        let tokenizer = RuntimeTokenizer::default();
        let bytes: Vec<u32> = "é".bytes().map(u32::from).collect();
        assert_eq!(safe_text(&tokenizer, &bytes[..1], false).unwrap(), "");
        assert_eq!(safe_text(&tokenizer, &bytes, false).unwrap(), "é");
        // Once finished, nothing is hidden.
        assert_eq!(
            safe_text(&tokenizer, &bytes[..1], true).unwrap(),
            "\u{FFFD}"
        );
    }

    #[test]
    fn sampling_validation_rejects_bad_values() {
        let bad = SamplingParams {
            top_p: 0.0,
            ..Default::default()
        };
        assert_eq!(
            bad.validate().unwrap_err().kind,
            RuntimeErrorKind::InvalidRequest
        );
        let bad = SamplingParams {
            temperature: f32::NAN,
            ..Default::default()
        };
        assert!(bad.validate().is_err());
        assert!(SamplingParams::default().validate().is_ok());
    }
}
