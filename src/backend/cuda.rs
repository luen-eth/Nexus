//! CUDA Driver API matvec execution, with model weights resident on the device.
//! PTX is checked in, so a CUDA toolkit/NVRTC installation is not required.
use super::{Backend, BackendCapabilities, BackendType, KernelOp, KernelOutput, MemoryInfo};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

/// Retained buffers are bounded to 32 MiB; staged growth needs at most twice this.
/// This preserves the existing 64 MiB workspace reserve when preparing weights.
pub const DEFAULT_WORKSPACE_LIMIT_BYTES: usize = 32 * 1024 * 1024;

/// Per-backend counters. Allocations exclude immutable weights and host Vecs.
/// Bytes transferred count successfully submitted copies; a successful matvec
/// synchronizes those copies before returning its owned host result.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct CudaWorkspaceStats {
    pub limit_bytes: usize,
    pub retained_bytes: usize,
    /// Maximum bytes in live workspace CudaSlice objects, including staged growth.
    /// This is not a measurement of the driver's allocation-pool reservation.
    pub peak_live_bytes: usize,
    pub input_capacity_elements: usize,
    pub output_capacity_elements: usize,
    pub input_allocations: u64,
    pub output_allocations: u64,
    pub input_upload_bytes: u64,
    pub output_download_bytes: u64,
    pub weight_upload_bytes: u64,
}

struct Workspace {
    input: Option<CudaSlice<f32>>,
    output: Option<CudaSlice<f32>>,
    stats: CudaWorkspaceStats,
}
impl Workspace {
    fn new(limit_bytes: usize) -> Self {
        Self {
            input: None,
            output: None,
            stats: CudaWorkspaceStats {
                limit_bytes,
                ..Default::default()
            },
        }
    }

    fn ensure(&mut self, stream: &Arc<CudaStream>, cols: usize, rows: usize) -> Result<(), String> {
        let old_input = self.stats.input_capacity_elements;
        let old_output = self.stats.output_capacity_elements;
        // First check exact high-water sizes. A rejected request changes nothing.
        let exact_input = old_input.max(cols);
        let exact_output = old_output.max(rows);
        let bytes =
            |input: usize, output: usize| input.checked_add(output).and_then(|n| n.checked_mul(4));
        if bytes(exact_input, exact_output).is_none_or(|n| n > self.stats.limit_bytes) {
            return Err(format!(
                "CUDA workspace limit exceeded: need {cols} input and {rows} output elements; retained limit {} bytes",
                self.stats.limit_bytes
            ));
        }
        if old_input >= cols && old_output >= rows {
            return Ok(());
        }
        // Geometric growth avoids repeated allocations for slowly increasing shapes.
        // Fall back to exact sizes when rounding would exceed the retained budget.
        let mut input_capacity = if old_input >= cols {
            old_input
        } else {
            cols.next_power_of_two()
        };
        let mut output_capacity = if old_output >= rows {
            old_output
        } else {
            rows.next_power_of_two()
        };
        if bytes(input_capacity, output_capacity).is_none_or(|n| n > self.stats.limit_bytes) {
            input_capacity = exact_input;
            output_capacity = exact_output;
        }
        // Allocate replacements before touching existing buffers. Partial allocation
        // failure leaves the old workspace usable. Old + replacements <= 2 * limit.
        let mut input = None;
        let mut output = None;
        let mut live_bytes = self.stats.retained_bytes;
        if input_capacity > old_input {
            input = Some(
                stream
                    .alloc_zeros::<f32>(input_capacity)
                    .map_err(|e| format!("CUDA input workspace allocation: {e}"))?,
            );
            self.stats.input_allocations = self.stats.input_allocations.saturating_add(1);
            live_bytes += input_capacity * 4;
            self.stats.peak_live_bytes = self.stats.peak_live_bytes.max(live_bytes);
        }
        if output_capacity > old_output {
            output = Some(
                stream
                    .alloc_zeros::<f32>(output_capacity)
                    .map_err(|e| format!("CUDA output workspace allocation: {e}"))?,
            );
            self.stats.output_allocations = self.stats.output_allocations.saturating_add(1);
            live_bytes += output_capacity * 4;
            self.stats.peak_live_bytes = self.stats.peak_live_bytes.max(live_bytes);
        }
        if let Some(input) = input {
            self.input = Some(input);
        }
        if let Some(output) = output {
            self.output = Some(output);
        }
        self.stats.input_capacity_elements = input_capacity;
        self.stats.output_capacity_elements = output_capacity;
        self.stats.retained_bytes = (input_capacity + output_capacity) * 4;
        Ok(())
    }
}
struct CudaState {
    weights: HashMap<usize, CudaSlice<f32>>,
    workspace: Workspace,
}

pub struct CudaBackend {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    function: CudaFunction,
    /// Serializes weight replacement and the entire upload/launch/download/sync.
    state: Mutex<CudaState>,
    launches: AtomicU64,
    resident_bytes: AtomicU64,
    failure: Mutex<Option<String>>,
}
impl CudaBackend {
    pub fn new() -> Result<Self, String> {
        Self::new_with_workspace_limit_bytes(DEFAULT_WORKSPACE_LIMIT_BYTES)
    }

    /// A smaller limit is useful for bounded deployments and deterministic tests.
    /// Growth temporarily retains old buffers, so reserve twice this budget.
    pub fn new_with_workspace_limit_bytes(limit_bytes: usize) -> Result<Self, String> {
        if !(8..=DEFAULT_WORKSPACE_LIMIT_BYTES).contains(&limit_bytes) {
            return Err(format!(
                "CUDA workspace limit must be 8..={DEFAULT_WORKSPACE_LIMIT_BYTES} bytes"
            ));
        }
        // cudarc's dynamic loader panics if libcuda is absent. Turn this into an
        // unavailable backend, including on CPU-only machines with --all-features.
        std::panic::catch_unwind(|| {
            let context = CudaContext::new(0).map_err(|e| format!("CUDA context: {e}"))?;
            let stream = context.default_stream();
            let module = context
                .load_module(Ptx::from_src(include_str!("kernels/matvec.ptx")))
                .map_err(|e| format!("CUDA PTX module: {e}"))?;
            let function = module
                .load_function("nexus_matvec")
                .map_err(|e| format!("CUDA kernel: {e}"))?;
            Ok(Self {
                context,
                stream,
                function,
                state: Mutex::new(CudaState {
                    weights: HashMap::new(),
                    workspace: Workspace::new(limit_bytes),
                }),
                launches: AtomicU64::new(0),
                resident_bytes: AtomicU64::new(0),
                failure: Mutex::new(None),
            })
        })
        .map_err(|_| "CUDA driver library is unavailable".to_string())?
    }

    fn record_failure(&self, message: String) -> String {
        if let Ok(mut failure) = self.failure.lock() {
            // Preserve the first failure; it is usually the actionable cause.
            if failure.is_none() {
                *failure = Some(message.clone());
            }
        }
        message
    }
    fn check_status(&self) -> Result<(), String> {
        if let Some(reason) = self
            .failure
            .lock()
            .map_err(|_| "CUDA status poisoned")?
            .as_ref()
        {
            return Err(format!("CUDA backend failed: {reason}"));
        }
        Ok(())
    }
    fn lock_state(&self) -> Result<MutexGuard<'_, CudaState>, String> {
        self.state
            .lock()
            .map_err(|_| self.record_failure("CUDA state/workspace mutex poisoned".into()))
    }

    /// Coherent snapshot under the same lock as execution, without a Backend API change.
    pub fn workspace_stats(&self) -> Result<CudaWorkspaceStats, String> {
        Ok(self.lock_state()?.workspace.stats)
    }
}
impl Backend for CudaBackend {
    fn name(&self) -> &str {
        "cuda"
    }
    fn backend_type(&self) -> BackendType {
        BackendType::Cuda
    }
    fn is_available(&self) -> bool {
        !self.state.is_poisoned() && self.failure.lock().is_ok_and(|f| f.is_none())
    }
    fn memory_info(&self) -> MemoryInfo {
        let (free, total) = self.context.mem_get_info().unwrap_or((0, 0));
        MemoryInfo {
            total_memory: total,
            free_memory: free,
            used_memory: total.saturating_sub(free),
        }
    }
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            backend: BackendType::Cuda,
            available: self.is_available(),
            features: vec![
                "cuda-driver",
                "persistent-f32-weights",
                "bounded-reusable-workspace",
                "hybrid-cpu-attention",
            ],
            supported_ops: vec!["matvec"],
        }
    }
    fn prepare_weights(&self, tensors: &HashMap<String, Vec<f32>>) -> Result<(), String> {
        let mut state = self.lock_state()?;
        self.check_status()?;
        self.context
            .bind_to_thread()
            .map_err(|e| self.record_failure(format!("CUDA context binding: {e}")))?;
        let required = tensors
            .values()
            .try_fold(0usize, |total, v| {
                v.len()
                    .checked_mul(4)
                    .and_then(|bytes| total.checked_add(bytes))
            })
            .ok_or("CUDA weight size overflow")?;
        let (free, _) = self
            .context
            .mem_get_info()
            .map_err(|e| format!("CUDA memory query: {e}"))?;
        let reserve = 2 * state.workspace.stats.limit_bytes;
        if required.saturating_add(reserve) > free {
            return Err(format!("Insufficient CUDA memory: model requires {required} bytes plus {reserve} workspace reserve; {free} bytes free"));
        }
        // A rejected/failed staged upload must not discard the previous cache.
        // Keep the original recoverable preparation-error policy; context faults
        // still surface through cudarc or subsequent execution synchronization.
        let mut uploaded = HashMap::new();
        // Upload immutable model buffers, including embeddings that may serve
        // as a tied output head. Normalization still executes on the CPU.
        for values in tensors.values().filter(|v| !v.is_empty()) {
            let device = self
                .stream
                .clone_htod(values.as_slice())
                .map_err(|e| format!("CUDA weight upload: {e}"))?;
            state.workspace.stats.weight_upload_bytes = state
                .workspace
                .stats
                .weight_upload_bytes
                .saturating_add((values.len() * 4) as u64);
            uploaded.insert(values.as_ptr() as usize, device);
        }
        self.stream
            .synchronize()
            .map_err(|e| format!("CUDA upload synchronization: {e}"))?;
        state.weights = uploaded;
        // Workspace intentionally survives model replacement: it has no model data
        // identity, and each logical input is fully overwritten before launch.
        self.resident_bytes
            .store(required as u64, Ordering::Relaxed);
        Ok(())
    }
    fn matvec(
        &self,
        weight: &[f32],
        rows: usize,
        cols: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        self.check_status()?;
        if input.len() != cols
            || rows.checked_mul(cols) != Some(weight.len())
            || rows == 0
            || cols == 0
            || rows > i32::MAX as usize
            || cols > u32::MAX as usize
        {
            return Err("Invalid CUDA matvec dimensions".into());
        }
        let mut state = self.lock_state()?;
        // Recheck after waiting: the previous caller might have failed on the GPU.
        self.check_status()?;
        self.context
            .bind_to_thread()
            .map_err(|e| self.record_failure(format!("CUDA context binding: {e}")))?;
        let CudaState { weights, workspace } = &mut *state;
        let device_weights = weights
            .get(&(weight.as_ptr() as usize))
            .ok_or("CUDA weight was not prepared for this model")?;
        if device_weights.len() != weight.len() {
            return Err("CUDA cached weight shape mismatch".into());
        }
        workspace.ensure(&self.stream, cols, rows)?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(rows)
            .map_err(|e| format!("CUDA host output allocation: {e}"))?;
        result.resize(rows, 0f32);
        let input_buffer = workspace
            .input
            .as_mut()
            .ok_or("CUDA input workspace missing")?;
        {
            let mut logical_input = input_buffer
                .try_slice_mut(..cols)
                .ok_or("CUDA input workspace slice mismatch")?;
            self.stream
                .memcpy_htod(input, &mut logical_input)
                .map_err(|e| format!("CUDA input upload: {e}"))?;
        }
        workspace.stats.input_upload_bytes = workspace
            .stats
            .input_upload_bytes
            .saturating_add((cols * 4) as u64);
        let device_input = input_buffer
            .try_slice(..cols)
            .ok_or("CUDA input workspace slice mismatch")?;
        let mut output = workspace
            .output
            .as_mut()
            .ok_or("CUDA output workspace missing")?
            .try_slice_mut(..rows)
            .ok_or("CUDA output workspace slice mismatch")?;
        let rows_arg = rows as u32;
        let cols_arg = cols as u32;
        {
            let mut args = self.stream.launch_builder(&self.function);
            args.arg(device_weights)
                .arg(&device_input)
                .arg(&mut output)
                .arg(&rows_arg)
                .arg(&cols_arg);
            // SAFETY: the same F32 kernel and 256-thread block as before. Views
            // cover exactly cols/rows elements. Weights have rows*cols elements;
            // every block writes its distinct row before the logical output is read.
            // The state mutex and stream order exclude simultaneous workspace reuse.
            unsafe {
                args.launch(LaunchConfig {
                    grid_dim: (rows_arg, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| self.record_failure(format!("CUDA launch: {e}")))?;
        }
        self.launches.fetch_add(1, Ordering::Relaxed);
        self.stream
            .memcpy_dtoh(&output, result.as_mut_slice())
            .map_err(|e| self.record_failure(format!("CUDA output download: {e}")))?;
        workspace.stats.output_download_bytes = workspace
            .stats
            .output_download_bytes
            .saturating_add((rows * 4) as u64);
        // cudarc HostSlice<[T]> uses no SyncOnDrop. Make the CPU read and the
        // next thread's buffer reuse explicitly wait for all submitted work.
        self.stream
            .synchronize()
            .map_err(|e| self.record_failure(format!("CUDA matvec synchronization: {e}")))?;
        Ok(result)
    }
    fn resident_weight_bytes(&self) -> u64 {
        self.resident_bytes.load(Ordering::Relaxed)
    }
    fn failure_reason(&self) -> Option<String> {
        self.failure.lock().ok().and_then(|v| v.clone())
    }
    fn kernel_launches(&self) -> u64 {
        self.launches.load(Ordering::Relaxed)
    }
    fn execute(&self, _: &KernelOp) -> Result<KernelOutput, String> {
        Err("CUDA implements matvec only; use Backend::matvec".into())
    }
    fn synchronize(&self) {
        if let Ok(_state) = self.lock_state() {
            if let Err(error) = self.context.synchronize() {
                self.record_failure(format!("CUDA synchronization: {error}"));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn device() -> Option<CudaBackend> {
        match CudaBackend::new() {
            Ok(device) => Some(device),
            Err(error) => {
                assert!(
                    std::env::var_os("NEXUS_REQUIRE_CUDA").is_none(),
                    "CUDA required: {error}"
                );
                None
            }
        }
    }
    #[test]
    fn poisoned_workspace_rejects_work_without_launching() {
        let Some(device) = device() else { return };
        let device = Arc::new(device);
        let worker = device.clone();
        assert!(std::thread::spawn(move || {
            let _guard = worker.state.lock().unwrap();
            panic!("test-only mutex poison");
        })
        .join()
        .is_err());
        assert!(!device.is_available());
        assert!(device.prepare_weights(&HashMap::new()).is_err());
        assert!(device.matvec(&[1.], 1, 1, &[1.]).is_err());
        assert_eq!(device.kernel_launches(), 0);
        assert!(device.workspace_stats().is_err());
        assert!(device.failure_reason().unwrap().contains("poisoned"));
    }
    #[test]
    fn terminal_failure_is_preserved_and_never_reuses_workspace() {
        let Some(device) = device() else { return };
        let tensors = HashMap::from([("w".into(), vec![2.])]);
        device.prepare_weights(&tensors).unwrap();
        assert_eq!(device.matvec(&tensors["w"], 1, 1, &[3.]).unwrap(), [6.]);
        let stats = device.workspace_stats().unwrap();
        device.record_failure("test-only terminal failure".into());
        device.record_failure("later error".into());
        assert!(!device.is_available());
        assert!(device.matvec(&tensors["w"], 1, 1, &[4.]).is_err());
        assert!(device.prepare_weights(&tensors).is_err());
        assert_eq!(device.kernel_launches(), 1);
        assert_eq!(device.workspace_stats().unwrap(), stats);
        assert_eq!(
            device.failure_reason().unwrap(),
            "test-only terminal failure"
        );
    }
}
