//! The model node's inference side: a `weirdnet-agent` [`InferHost`] over the [`Supervisor`],
//! serving raw infer/2 (assembled messages in, tokens out; cancel aborts the engine request),
//! transcription (infer/2 `Transcribe`) and speech out (infer/2 `Speak`).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use futures::StreamExt;
use weirdnet_agent::remote::infer::InferHost;
use weirdnet_agent::remote::infer::v2::{
    self, AudioStream, EmbedBackend, Finish, InferBackend, Input, Output, Request, SpeakBackend, SpeakRequest, TranscribeBackend, TranscribeRequest,
    Transcript, Usage,
};
use weirdnet_agent::remote::{Authorize, Caller, Channel, Dialer, RemoteError};
use weirdnet_agent::{CancelToken, Modalities, ModelBackend};

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
/// the default). A request with image or audio parts is refused (`Unsupported`, so a pool tries
/// elsewhere) unless the model takes them ([`ModelLaunch::capabilities`](crate::ModelLaunch::capabilities));
/// remote image URLs are a `BadRequest`. A model that is not ready is refused as unavailable unless the request has
/// `wait_for_model` or the host loads on demand. While draining or reserved
/// ([`Supervisor::reserve`](crate::Supervisor::reserve)), new requests are refused.
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

    fn pick(&self, model: Option<&str>, wait: bool, needed: Modalities) -> Result<(String, bool), RemoteError> {
        if self.draining() {
            return Err(unavailable("this node is draining"));
        }
        if let Some(r) = self.supervisor.reservation() {
            return Err(unavailable(format!("this node is reserved by {}", r.holder)));
        }
        let id = model.map(str::to_string).or_else(|| self.default_model.clone()).ok_or_else(|| bad_request("no model named and no default"))?;
        if let Some(what) = self.supervisor.capabilities(&id).and_then(|c| c.lacking(needed)) {
            let fallback = if needed.video { format!("; {}", weirdnet_agent::remote::catalog::VIDEO_FALLBACK) } else { String::new() };
            return Err(RemoteError::Unsupported { message: format!("model `{id}` does not take {what} on this node{fallback}") });
        }
        match self.supervisor.state(&id) {
            None => Err(crate::error::unknown_model(&id)),
            Some(ModelState::Ready) => Ok((id, false)),
            Some(_) if wait || self.load_on_demand => Ok((id, true)),
            Some(state) => Err(unavailable(format!("{id} is not loaded ({state:?})"))),
        }
    }

    /// The model for a transcription request: a named one must transcribe here
    /// ([`Capabilities::transcribe`](crate::Capabilities::transcribe)); with none named, the
    /// ready one first by id, else (loading on demand) the first by id.
    fn pick_transcriber(&self, model: Option<&str>) -> Result<(String, bool), RemoteError> {
        let transcribes = |id: &str| self.supervisor.capabilities(id).is_some_and(|c| c.transcribe);
        let id = match model {
            Some(id) => {
                if self.supervisor.state(id).is_none() {
                    return Err(crate::error::unknown_model(id));
                }
                if !transcribes(id) {
                    return Err(RemoteError::Unsupported { message: format!("model `{id}` does not transcribe on this node") });
                }
                id.to_string()
            }
            None => {
                let ids: Vec<String> = self.supervisor.launches().into_iter().map(|l| l.id).filter(|id| transcribes(id)).collect();
                let ready = ids.iter().find(|id| self.supervisor.state(id) == Some(ModelState::Ready));
                match ready.or(ids.first().filter(|_| self.load_on_demand)) {
                    Some(id) => id.clone(),
                    None if ids.is_empty() => return Err(unavailable("no transcription model on this node")),
                    None => return Err(unavailable(format!("no transcription model is loaded on this node ({})", ids.join(", ")))),
                }
            }
        };
        self.pick(Some(&id), false, Modalities::TEXT)
    }

    /// The model for a speech request: a named one must speak here
    /// ([`Capabilities::speak`](crate::Capabilities::speak)) and its profile cover the request
    /// (what it lacks is a `BadRequest`; an output format this engine doesn't make is
    /// `Unsupported`, as are models that don't speak here, so a pool tries elsewhere). With none
    /// named, the speakers whose profile covers the request, the ready one first by id, else
    /// (loading on demand) the first by id.
    fn pick_speaker(&self, request: &SpeakRequest) -> Result<(String, bool), RemoteError> {
        let speaks = |id: &str| self.supervisor.capabilities(id).is_some_and(|c| c.speak);
        let needs = request.needs();
        let covers = |id: &str| -> Result<(), String> {
            match self.supervisor.speech(id) {
                Some(p) => p.covers(&needs),
                None => Err("has speech settings that widen what its engine reports".into()),
            }
        };
        let id = match request.model.as_deref() {
            Some(id) => {
                if self.supervisor.state(id).is_none() {
                    return Err(crate::error::unknown_model(id));
                }
                if !speaks(id) {
                    return Err(RemoteError::Unsupported { message: format!("model `{id}` does not speak on this node") });
                }
                if let Err(why) = covers(id) {
                    let mut other_format = needs.clone();
                    other_format.format = None;
                    let format_only = needs.format.is_some() && self.supervisor.speech(id).is_some_and(|p| p.covers(&other_format).is_ok());
                    return Err(if format_only {
                        RemoteError::Unsupported { message: format!("model `{id}` {why} on this node") }
                    } else {
                        bad_request(format!("model `{id}` {why}"))
                    });
                }
                id.to_string()
            }
            None => {
                let all: Vec<String> = self.supervisor.launches().into_iter().map(|l| l.id).filter(|id| speaks(id)).collect();
                if all.is_empty() {
                    return Err(unavailable("no speech model on this node"));
                }
                let mut why = None;
                let ids: Vec<&String> = all
                    .iter()
                    .filter(|id| match covers(id) {
                        Ok(()) => true,
                        Err(w) => {
                            why.get_or_insert_with(|| format!("`{id}` {w}"));
                            false
                        }
                    })
                    .collect();
                let ready = ids.iter().find(|id| self.supervisor.state(id) == Some(ModelState::Ready));
                match ready.or(ids.first().filter(|_| self.load_on_demand)) {
                    Some(id) => (*id).clone(),
                    None if ids.is_empty() => {
                        return Err(RemoteError::Unsupported {
                            message: format!("no speech model on this node covers the request ({})", why.unwrap_or_default()),
                        });
                    }
                    None => {
                        let names: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
                        return Err(unavailable(format!("no speech model is loaded on this node ({})", names.join(", "))));
                    }
                }
            }
        };
        self.pick(Some(&id), false, Modalities::TEXT)
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
        let needed = request.needs();
        if let Some(Err(message)) = request.inline_messages().unwrap_or_default().iter().map(|m| m.content.check()).find(Result::is_err) {
            return Err(bad_request(message));
        }
        let (id, load) = self.pick(request.model.as_deref(), request.wait_for_model, needed)?;
        Ok(Arc::new(self.backend(id, load)))
    }

    fn open_embed(&self, _caller: &Caller, model: Option<&str>) -> Result<Arc<dyn EmbedBackend>, RemoteError> {
        let (id, load) = self.pick(model, false, Modalities::TEXT)?;
        Ok(Arc::new(self.backend(id, load)))
    }

    /// A speech-to-text model on this node ([`Capabilities::transcribe`](crate::Capabilities::transcribe)):
    /// the named one, or with none named the loaded one (first by id; with load on demand, the
    /// first by id even when not loaded). A named model that doesn't transcribe here is
    /// `Unsupported` (a pool tries elsewhere); none at all is unavailable. Served by
    /// [`Supervisor::transcribe`](crate::Supervisor::transcribe).
    fn open_transcribe(&self, _caller: &Caller, model: Option<&str>) -> Result<Arc<dyn TranscribeBackend>, RemoteError> {
        let (id, load) = self.pick_transcriber(model)?;
        Ok(Arc::new(self.backend(id, load)))
    }

    /// A speaking model on this node ([`Capabilities::speak`](crate::Capabilities::speak))
    /// whose speech profile covers the request: the named one, or with none named the loaded one
    /// (first by id; with load on demand, the first by id even when not loaded). A named model
    /// that lacks what the request needs is a `BadRequest` that says what; one that doesn't speak
    /// here is `Unsupported` (a pool tries elsewhere); none at all is unavailable. Served by
    /// [`Supervisor::speak`](crate::Supervisor::speak). Since 0.4.10.
    fn open_speak(&self, _caller: &Caller, request: &SpeakRequest) -> Result<Arc<dyn SpeakBackend>, RemoteError> {
        let (id, load) = self.pick_speaker(request)?;
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

    /// Both engines continue an assistant prefix (chat) or the prompt plus the prefix (a raw
    /// prompt, through `/v1/completions`).
    fn resumes(&self) -> bool {
        true
    }

    async fn infer(&self, request: &Request, out: &Output, cancel: &CancelToken) -> Result<Finish, RemoteError> {
        self.ready().await?;
        let started = Instant::now();
        let mut stream = match &request.input {
            Input::Prompt(prompt) => {
                let mut prompt = prompt.clone();
                if let Some(p) = &request.assistant_prefix {
                    prompt.push_str(&p.text);
                }
                self.supervisor.complete_scaled(&self.id, &prompt, &request.sampling, request.seed, request.adapter_scale).await?
            }
            _ => self.supervisor.chat(&self.id, request).await?,
        };
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
        Ok(self.embed_counted(input, cancel).await?.0)
    }

    /// The vector and the engine's `usage.prompt_tokens`.
    async fn embed_counted(&self, input: &str, cancel: &CancelToken) -> Result<(Vec<f32>, Option<u64>), RemoteError> {
        self.ready().await?;
        let inputs = [input.to_string()];
        let call = self.supervisor.embed(&self.id, &inputs);
        let (mut vectors, tokens) = tokio::select! {
            _ = cancel.cancelled() => return Err(RemoteError::Cancelled),
            r = call => r?,
        };
        Ok((vectors.pop().ok_or_else(|| engine_error("no embedding"))?, Some(tokens)))
    }
}

#[async_trait]
impl TranscribeBackend for EngineModel {
    fn name(&self) -> String {
        self.id.clone()
    }

    async fn transcribe(&self, request: &TranscribeRequest, cancel: &CancelToken) -> Result<Transcript, RemoteError> {
        Ok(self.transcribe_counted(request, cancel).await?.0)
    }

    /// The transcript and the engine's token counts ([`Supervisor::transcribe`](crate::Supervisor::transcribe)).
    async fn transcribe_counted(&self, request: &TranscribeRequest, cancel: &CancelToken) -> Result<(Transcript, Option<Usage>), RemoteError> {
        self.ready().await?;
        tokio::select! {
            _ = cancel.cancelled() => Err(RemoteError::Cancelled),
            r = self.supervisor.transcribe(&self.id, request) => r,
        }
    }
}

#[async_trait]
impl SpeakBackend for EngineModel {
    fn name(&self) -> String {
        self.id.clone()
    }

    /// The speech from [`Supervisor::speak`](crate::Supervisor::speak), loading the model first
    /// when it was picked cold. Cancelling stops the engine request.
    async fn speak(&self, request: &SpeakRequest, cancel: &CancelToken) -> Result<AudioStream, RemoteError> {
        tokio::select! {
            _ = cancel.cancelled() => return Err(RemoteError::Cancelled),
            r = self.ready() => r?,
        }
        tokio::select! {
            _ = cancel.cancelled() => Err(RemoteError::Cancelled),
            r = self.supervisor.speak(&self.id, request) => r,
        }
    }
}

/// The protocol id model nodes serve raw infer on: the `/weirdnet-agent/infer/2` framing under
/// its own id, so one node can be an inference node (which serves infer/2 to apps) and a model
/// node at once. `NodeServer` (`node` feature) always serves it; [`RawInferDialer`] prefers it.
pub const RAW_INFER_PROTOCOL: &str = "/weirdnet-backends/infer/1";

/// A [`Dialer`] for inference nodes talking to model nodes: opening
/// `/weirdnet-agent/infer/2` tries [`RAW_INFER_PROTOCOL`] first and falls back to infer/2 on
/// nodes that don't accept it (backends 0.1.0). Nodes that fell back are remembered. Other
/// protocols pass through. Wrap the dialer given to the `Pool` of model nodes.
pub struct RawInferDialer {
    inner: Arc<dyn Dialer>,
    legacy: Mutex<std::collections::HashSet<String>>,
}

impl RawInferDialer {
    pub fn new(inner: Arc<dyn Dialer>) -> Self {
        Self { inner, legacy: Mutex::new(Default::default()) }
    }

    /// Whether `node` answered only on infer/2.
    pub fn is_legacy(&self, node: &str) -> bool {
        self.legacy.lock().unwrap_or_else(|e| e.into_inner()).contains(node)
    }
}

#[async_trait]
impl Dialer for RawInferDialer {
    async fn open(&self, node: &str, protocol: &str) -> Result<Channel, RemoteError> {
        if protocol != v2::PROTOCOL {
            return self.inner.open(node, protocol).await;
        }
        if !self.is_legacy(node) {
            match self.inner.open(node, RAW_INFER_PROTOCOL).await {
                Err(RemoteError::Unsupported { .. }) => {}
                other => return other,
            }
        }
        let ch = self.inner.open(node, v2::PROTOCOL).await?;
        self.legacy.lock().unwrap_or_else(|e| e.into_inner()).insert(node.to_string());
        Ok(ch)
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
