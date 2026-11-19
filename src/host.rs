//! The model node's inference side: a `weirdnet-agent` [`InferHost`] over the [`Supervisor`],
//! serving raw infer/2 (assembled messages in, tokens out; cancel aborts the engine request).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use futures::StreamExt;
use weirdnet_agent::remote::infer::InferHost;
use weirdnet_agent::remote::infer::v2::{EmbedBackend, Finish, InferBackend, Output, Request};
use weirdnet_agent::remote::{Authorize, Caller, RemoteError};
use weirdnet_agent::{CancelToken, ModelBackend};

use crate::error::{bad_request, engine_error, unavailable};
use crate::sse::{Chunk, ToolCalls};
use crate::supervisor::{ModelState, Supervisor};

/// Measured speed, for the capability record.
#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Speed {
    /// Exponentially weighted decode tokens per second.
    pub decode_tps: Option<f32>,
    /// Exponentially weighted prefill tokens per second (prompt tokens over time to first token).
    pub prefill_tps: Option<f32>,
    /// The last time to first token.
    pub ttft_ms: Option<u64>,
}

impl Speed {
    fn ewma(old: Option<f32>, new: f32) -> Option<f32> {
        Some(match old {
            Some(o) => o * 0.8 + new * 0.2,
            None => new,
        })
    }

    fn record(&mut self, ttft: Option<f64>, total: f64, prompt: u64, completion: u64) {
        if let Some(t) = ttft {
            self.ttft_ms = Some((t * 1000.0) as u64);
            if t > 0.0 && prompt > 0 {
                self.prefill_tps = Self::ewma(self.prefill_tps, prompt as f32 / t as f32);
            }
            let decode = total - t;
            if decode > 0.0 && completion > 1 {
                self.decode_tps = Self::ewma(self.decode_tps, (completion - 1) as f32 / decode as f32);
            }
        }
    }
}

/// Serves the supervisor's models over infer/2. Requests name a model by its launch id (or get
/// the default). A model that is not ready is refused as unavailable unless the request has
/// `wait_for_model` or the host loads on demand. While draining, new requests are refused.
pub struct ModelHost {
    supervisor: Supervisor,
    auth: Arc<dyn Authorize>,
    default_model: Option<String>,
    load_on_demand: bool,
    draining: AtomicBool,
    speed: Arc<Mutex<std::collections::HashMap<String, Speed>>>,
}

impl ModelHost {
    pub fn new(supervisor: Supervisor, auth: Arc<dyn Authorize>) -> Self {
        Self { supervisor, auth, default_model: None, load_on_demand: false, draining: AtomicBool::new(false), speed: Default::default() }
    }

    /// The model for requests that name none.
    pub fn with_default(mut self, model: impl Into<String>) -> Self {
        self.default_model = Some(model.into());
        self
    }

    /// Load a model that is not ready instead of refusing (a single box with no coordinator).
    pub fn with_load_on_demand(mut self, on: bool) -> Self {
        self.load_on_demand = on;
        self
    }

    pub fn supervisor(&self) -> &Supervisor {
        &self.supervisor
    }

    /// Stop (or resume) taking new requests; requests under way finish.
    pub fn set_draining(&self, on: bool) {
        self.draining.store(on, Ordering::SeqCst);
    }

    pub fn draining(&self) -> bool {
        self.draining.load(Ordering::SeqCst)
    }

    /// Measured speed per model.
    pub fn speed(&self, model: &str) -> Speed {
        self.speed.lock().unwrap_or_else(|e| e.into_inner()).get(model).copied().unwrap_or_default()
    }

    fn pick(&self, model: Option<&str>, wait: bool) -> Result<(String, bool), RemoteError> {
        if self.draining() {
            return Err(unavailable("this node is draining"));
        }
        let id = model.map(str::to_string).or_else(|| self.default_model.clone()).ok_or_else(|| bad_request("no model named and no default"))?;
        match self.supervisor.state(&id) {
            None => Err(crate::error::unknown_model(&id)),
            Some(ModelState::Ready) => Ok((id, false)),
            Some(_) if wait || self.load_on_demand => Ok((id, true)),
            Some(state) => Err(unavailable(format!("{id} is not loaded ({state:?})"))),
        }
    }

    fn backend(&self, id: String, load: bool) -> EngineModel {
        EngineModel { supervisor: self.supervisor.clone(), id, load, speed: self.speed.clone() }
    }
}

impl InferHost for ModelHost {
    fn authorize(&self, caller: &Caller) -> Result<(), RemoteError> {
        self.auth.authorize(caller)
    }

    /// Model nodes serve infer/2 only.
    fn open(&self, _caller: &Caller, _model: Option<&str>) -> Result<Arc<dyn ModelBackend>, RemoteError> {
        Err(RemoteError::Unsupported { message: "model nodes serve /weirdnet-agent/infer/2".into() })
    }

    fn open_request(&self, _caller: &Caller, request: &Request) -> Result<Arc<dyn InferBackend>, RemoteError> {
        let (id, load) = self.pick(request.model.as_deref(), request.wait_for_model)?;
        Ok(Arc::new(self.backend(id, load)))
    }

    fn open_embed(&self, _caller: &Caller, model: Option<&str>) -> Result<Arc<dyn EmbedBackend>, RemoteError> {
        let (id, load) = self.pick(model, false)?;
        Ok(Arc::new(self.backend(id, load)))
    }
}

/// One supervised model as an infer/2 backend.
pub struct EngineModel {
    supervisor: Supervisor,
    id: String,
    load: bool,
    speed: Arc<Mutex<std::collections::HashMap<String, Speed>>>,
}

impl EngineModel {
    async fn ready(&self) -> Result<(), RemoteError> {
        if self.load { self.supervisor.load(&self.id).await } else { Ok(()) }
    }
}

#[async_trait]
impl InferBackend for EngineModel {
    fn name(&self) -> String {
        self.id.clone()
    }

    /// Both engines continue an assistant prefix.
    fn resumes(&self) -> bool {
        true
    }

    async fn infer(&self, request: &Request, out: &Output, cancel: &CancelToken) -> Result<Finish, RemoteError> {
        self.ready().await?;
        let started = Instant::now();
        let mut stream = self.supervisor.chat(&self.id, request).await?;
        let mut calls = ToolCalls::default();
        let mut first: Option<f64> = None;
        loop {
            let next = tokio::select! {
                _ = cancel.cancelled() => return Err(RemoteError::Cancelled),
                n = stream.next() => n,
            };
            let chunk = match next {
                Some(c) => c?,
                None => return Err(engine_error("engine stream ended early")),
            };
            if first.is_none() && matches!(chunk, Chunk::Text(_) | Chunk::Reasoning(_) | Chunk::ToolCall { .. }) {
                first = Some(started.elapsed().as_secs_f64());
            }
            match chunk {
                Chunk::Text(t) => out.text(t),
                Chunk::Reasoning(t) => out.reasoning(t),
                Chunk::ToolCall { index, id, name, arguments } => calls.push(index, id, name, &arguments),
                Chunk::Done { finish_reason, usage } => {
                    let had_calls = !calls.is_empty();
                    for call in calls.finish() {
                        out.tool_call(call);
                    }
                    let reason = finish_reason.unwrap_or_else(|| if had_calls { "tool_calls".into() } else { "stop".into() });
                    let mut finish = Finish::new(reason);
                    if let Some(u) = usage {
                        out.usage(u);
                        finish = finish.with_usage(u);
                        let total = started.elapsed().as_secs_f64();
                        self.speed.lock().unwrap_or_else(|e| e.into_inner()).entry(self.id.clone()).or_default().record(
                            first,
                            total,
                            u.prompt_tokens,
                            u.completion_tokens,
                        );
                    }
                    return Ok(finish);
                }
            }
        }
    }
}

#[async_trait]
impl EmbedBackend for EngineModel {
    fn name(&self) -> String {
        self.id.clone()
    }

    async fn embed(&self, input: &str, cancel: &CancelToken) -> Result<Vec<f32>, RemoteError> {
        self.ready().await?;
        let inputs = [input.to_string()];
        let call = self.supervisor.embed(&self.id, &inputs);
        let (mut vectors, _) = tokio::select! {
            _ = cancel.cancelled() => return Err(RemoteError::Cancelled),
            r = call => r?,
        };
        vectors.pop().ok_or_else(|| engine_error("no embedding"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speed_is_smoothed() {
        let mut s = Speed::default();
        s.record(Some(0.5), 1.5, 100, 11);
        assert_eq!((s.ttft_ms, s.prefill_tps, s.decode_tps), (Some(500), Some(200.0), Some(10.0)));
        s.record(Some(0.5), 0.5, 100, 11);
        assert_eq!(s.decode_tps, Some(10.0), "no decode time: unchanged");
    }
}
