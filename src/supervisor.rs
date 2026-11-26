//! Starts, probes, drains and stops engine processes for one model node: swap groups, idle
//! unload, crash detection with restart backoff, and readiness probes.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::{broadcast, oneshot, watch};
use tokio_stream::wrappers::ReceiverStream;
use weirdnet_agent::remote::RemoteError;
use weirdnet_agent::remote::infer::v2::{Request, Sampling, TranscribeRequest, Transcript, Usage};
use weirdnet_model::ModelStore;
use weirdnet_model::limits::LimitsRecord;

use crate::engine::{Capabilities, EngineConfig, EngineKind, LoraAdapter, ModelLaunch};

/// An engine's answer as it streams.
pub type ChunkStream = std::pin::Pin<Box<dyn futures::Stream<Item = Result<Chunk, RemoteError>> + Send>>;

/// A model's state on this node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", content = "detail", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ModelState {
    Unloaded,
    Loading,
    Ready,
    Draining,
    Failed(String),
}
use crate::error::{bad_request, engine_error, unavailable, unknown_model};
use crate::openai::{self, Target};
use crate::sse::{Chunk, ChunkDecoder, SseParser};

#[derive(Debug, Clone)]
pub struct SupervisorOptions {
    /// Per-model engine logs (`<id>.log`); `None` discards engine output.
    pub log_dir: Option<PathBuf>,
    /// How long `unload` waits for in-flight requests.
    pub drain_timeout: Duration,
    /// How long a stopped engine gets after SIGTERM before SIGKILL.
    pub stop_grace: Duration,
    /// Restart an engine that crashed while serving.
    pub restart: bool,
    /// First restart delay; doubles per crash in the window, up to `backoff_max`.
    pub backoff_base: Duration,
    pub backoff_max: Duration,
    /// This many crashes within `crash_window` is a crash loop: no more restarts until an explicit
    /// (forced) load.
    pub max_crashes: usize,
    pub crash_window: Duration,
}

impl Default for SupervisorOptions {
    fn default() -> Self {
        Self {
            log_dir: None,
            drain_timeout: Duration::from_secs(30),
            stop_grace: Duration::from_secs(10),
            restart: true,
            backoff_base: Duration::from_secs(1),
            backoff_max: Duration::from_secs(60),
            max_crashes: 5,
            crash_window: Duration::from_secs(600),
        }
    }
}

/// Something that happened to an engine; the node's alerting and metrics listen to these.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum SupervisorEvent {
    Starting {
        model: String,
    },
    Ready {
        model: String,
        load_secs: f32,
    },
    LoadFailed {
        model: String,
        why: String,
    },
    Crashed {
        model: String,
        why: String,
        crashes: usize,
        restart_in_secs: Option<f32>,
    },
    CrashLoop {
        model: String,
        crashes: usize,
    },
    Unloaded {
        model: String,
        idle: bool,
    },
    /// A model was added at run time ([`Supervisor::add_model`]).
    Added {
        model: String,
    },
    /// A model was removed at run time ([`Supervisor::remove_model`]).
    Removed {
        model: String,
    },
}

/// A hold on the node's accelerators ([`Supervisor::reserve`]): while it lasts the supervisor
/// starts no engine, so nothing is placed on the node. For a job that needs the GPU to itself,
/// such as a training run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Reservation {
    /// Who holds it: the application's name for the job (for example a run id). Releasing takes
    /// the same name.
    pub holder: String,
    /// When it was taken, in Unix milliseconds.
    pub since_ms: i64,
}

impl Reservation {
    pub fn new(holder: impl Into<String>) -> Self {
        let since_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64);
        Self { holder: holder.into(), since_ms }
    }
}

/// One model's row in the node's capability record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelStatus {
    pub id: String,
    pub engine: EngineKind,
    pub state: ModelState,
    pub in_flight: usize,
    pub swap_group: Option<String>,
    pub embedding: bool,
    pub pid: Option<u32>,
    /// The last measured load time.
    pub last_load_secs: Option<f32>,
    /// Model file size, for load-time estimates before one was measured.
    pub size_bytes: Option<u64>,
    /// A stock server this node only forwards to.
    pub wrapped: bool,
    /// Crashes in the current window.
    pub crashes: usize,
    /// The configured context window, if set.
    #[serde(default)]
    pub context: Option<u32>,
    /// The configured concurrent sequences, if set.
    #[serde(default)]
    pub slots: Option<u32>,
    /// What the model takes ([`ModelLaunch::capabilities`]); `None` in records from nodes before
    /// 0.1.3, where only `embedding` is known.
    #[serde(default)]
    pub capabilities: Option<Capabilities>,
    /// For a LoRA adapter's row: the base model it runs on (it shares the base's engine, state
    /// and capabilities). `None` for models.
    #[serde(default)]
    pub base: Option<String>,
    /// The context per sequence the engine reported when the model last loaded (llama-server's
    /// `/props` `n_ctx` per slot, vLLM's `max_model_len`): what one request can use, which with
    /// several slots is less than the configured [`context`](Self::context). `None` until measured,
    /// when the engine doesn't say, and in records from nodes before 0.4.9.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_tokens: Option<u32>,
}

impl ModelStatus {
    /// Its media and request limits ([`Capabilities::limits`]); unknown when the record has no
    /// capabilities (nodes before 0.1.3) or the node knows none.
    pub fn limits(&self) -> weirdnet_agent::MediaLimits {
        self.capabilities.map(|c| c.limits).unwrap_or_default()
    }

    /// `entry` with this row's inputs, limits and context window filled in from the record:
    /// the measured context per sequence ([`context_tokens`](Self::context_tokens)), else the
    /// configured `--ctx-size` when set; `entry`'s own values stay where the record has none.
    pub fn apply_to(&self, entry: weirdnet_agent::remote::catalog::CatalogModel) -> weirdnet_agent::remote::catalog::CatalogModel {
        let entry = match self.capabilities {
            Some(c) => c.apply_to(entry),
            None => entry,
        };
        match self.context_tokens.or(self.context) {
            Some(ctx) => entry.with_context(ctx),
            None => entry,
        }
    }

    /// Its catalog/1 purpose ([`Capabilities::purpose`]): embedding, transcription
    /// ([`Capabilities::transcribe`]) or chat.
    pub fn purpose(&self) -> weirdnet_agent::remote::catalog::Purpose {
        self.capabilities.unwrap_or_default().purpose(self.embedding)
    }

    /// A catalog/1 entry for this row in `state`: its id, [`purpose`](Self::purpose), and what
    /// [`apply_to`](Self::apply_to) fills in.
    pub fn catalog_model(&self, state: weirdnet_agent::remote::catalog::ModelState) -> weirdnet_agent::remote::catalog::CatalogModel {
        self.apply_to(weirdnet_agent::remote::catalog::CatalogModel::new(self.id.clone(), self.purpose(), state))
    }
}

/// When an adapter added with [`Supervisor::add_adapter`] or [`Supervisor::add_adapter_live`]
/// can serve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum AdapterChange {
    /// Now, or whenever the base is next ready: the base wasn't running, or its engine has the
    /// adapter loaded already (llama.cpp: the same file, loaded at start; vLLM: loaded at start
    /// or at run time).
    Live,
    /// After the base's next load: its engine is running without the adapter (llama-server
    /// loads adapter files only at start). Loading the adapter ([`Supervisor::load`]) restarts
    /// the base.
    NextLoad,
}

/// Where a pool id goes: its base model's slot, and the adapter if it is one.
struct Route {
    base: String,
    adapter: Option<LoraAdapter>,
}

struct Running {
    base: String,
    /// The adapters the engine has loaded: llama.cpp's in `--lora` order (they stay loaded for
    /// the engine's life); vLLM's by name, including those loaded at run time.
    adapters: Vec<LoraAdapter>,
    /// vLLM started with LoRA on and run-time updates allowed ([`ModelLaunch::runtime_lora`]).
    runtime_lora: bool,
    /// The projector the engine started with.
    mmproj: Option<String>,
    api_key: Option<String>,
    pid: Option<u32>,
    stop: Option<oneshot::Sender<()>>,
    exited: watch::Receiver<Option<String>>,
    /// Keeps a wrapped server's never-firing exit channel open.
    _exit_tx: Option<watch::Sender<Option<String>>>,
    /// The key-checking proxy in front of an MLX-LM server; stops when the engine record goes.
    _proxy: Option<crate::mlx::ProxyHandle>,
}

struct Slot {
    launch: ModelLaunch,
    state: ModelState,
    running: Option<Running>,
    in_flight: Arc<AtomicUsize>,
    /// Milliseconds since the supervisor's epoch when the last request finished.
    last_used: Arc<AtomicU64>,
    /// Bumped on every start, so a late exit notice from an old process is ignored.
    generation: u64,
    load_secs: Option<f32>,
    crashes: Vec<Instant>,
    crash_loop: bool,
    restart_at: Option<Instant>,
    /// What the last load measured (or the store kept from an earlier one).
    measured: Option<LimitsRecord>,
}

impl Running {
    /// Where the engine has `a` loaded: llama.cpp's `--lora` index (any id with the same file);
    /// vLLM's module with the same name and path.
    fn index(&self, engine: EngineKind, a: &LoraAdapter) -> Option<usize> {
        match engine {
            EngineKind::Llamacpp => self.adapters.iter().position(|x| x.path == a.path),
            _ => self.adapters.iter().position(|x| x.id == a.id && x.path == a.path),
        }
    }
}

impl Slot {
    /// Whether adapter `a` can serve on the running engine (or the base isn't running).
    fn serves(&self, a: &LoraAdapter) -> bool {
        self.running.as_ref().is_none_or(|r| r.index(self.launch.engine, a).is_some())
    }

    fn new(launch: ModelLaunch) -> Self {
        Self {
            launch,
            state: ModelState::Unloaded,
            running: None,
            in_flight: Arc::new(AtomicUsize::new(0)),
            last_used: Arc::new(AtomicU64::new(0)),
            generation: 0,
            load_secs: None,
            crashes: Vec::new(),
            crash_loop: false,
            restart_at: None,
            measured: None,
        }
    }

    /// What the model takes, with the operator's limits over the measured ones
    /// ([`MediaLimits::or`](weirdnet_agent::MediaLimits::or)).
    fn capabilities(&self) -> Capabilities {
        let mut c = self.launch.capabilities();
        if let Some(r) = &self.measured {
            c.limits = c.limits.or(r.limits.to_media());
        }
        c
    }
}

struct Inner {
    engines: HashMap<EngineKind, EngineConfig>,
    opts: SupervisorOptions,
    http: reqwest::Client,
    slots: Mutex<HashMap<String, Slot>>,
    /// Serializes load and unload, so swaps never interleave.
    ops: tokio::sync::Mutex<()>,
    events: broadcast::Sender<SupervisorEvent>,
    epoch: Instant,
    /// The node's reservation, if any.
    reserved: watch::Sender<Option<Reservation>>,
    /// Where measured limits are kept ([`Supervisor::set_limits_store`]).
    limits_store: Mutex<Option<ModelStore>>,
}

/// The per-node supervisor. Cheap to clone.
#[derive(Clone)]
pub struct Supervisor {
    inner: Arc<Inner>,
}

impl Supervisor {
    pub fn new(engines: Vec<EngineConfig>, models: Vec<ModelLaunch>, opts: SupervisorOptions) -> Self {
        let slots = models.into_iter().map(|launch| (launch.id.clone(), Slot::new(launch))).collect();
        let http =
            reqwest::Client::builder().connect_timeout(Duration::from_secs(5)).no_proxy().build().expect("an HTTP client without TLS always builds");
        let (events, _) = broadcast::channel(256);
        Self {
            inner: Arc::new(Inner {
                engines: engines.into_iter().map(|e| (e.kind, e)).collect(),
                opts,
                http,
                slots: Mutex::new(slots),
                ops: tokio::sync::Mutex::new(()),
                events,
                epoch: Instant::now(),
                reserved: watch::Sender::new(None),
                limits_store: Mutex::new(None),
            }),
        }
    }

    fn slots(&self) -> std::sync::MutexGuard<'_, HashMap<String, Slot>> {
        self.inner.slots.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn emit(&self, ev: SupervisorEvent) {
        let _ = self.inner.events.send(ev);
    }

    /// The slot serving `id`: a model's own, or the base of an adapter.
    fn route(&self, id: &str) -> Result<Route, RemoteError> {
        let slots = self.slots();
        if slots.contains_key(id) {
            return Ok(Route { base: id.to_string(), adapter: None });
        }
        slots
            .values()
            .find_map(|s| s.launch.adapters.iter().find(|a| a.id == id).map(|a| Route { base: s.launch.id.clone(), adapter: Some(a.clone()) }))
            .ok_or_else(|| unknown_model(id))
    }

    /// Whether `id` names a model or an adapter on this node.
    fn taken(slots: &HashMap<String, Slot>, id: &str) -> bool {
        slots.contains_key(id) || slots.values().any(|s| s.launch.adapters.iter().any(|a| a.id == id))
    }

    /// Adds LoRA adapter `adapter` to model `base`, after [`ModelLaunch::check_adapters`]. Its id
    /// must be free on the node. Returns [`AdapterChange::Live`] when the base isn't running or
    /// its engine already has the adapter loaded (on llama.cpp: the same file, for example one
    /// removed and added back), else [`AdapterChange::NextLoad`]. It never touches the engine:
    /// [`Supervisor::add_adapter_live`] also loads it into a running vLLM engine. Emits
    /// [`SupervisorEvent::Added`] for the adapter.
    pub fn add_adapter(&self, base: &str, adapter: LoraAdapter) -> Result<AdapterChange, RemoteError> {
        let mut launch = self.slots().get(base).map(|s| s.launch.clone()).ok_or_else(|| unknown_model(base))?;
        launch.adapters.push(adapter.clone());
        launch.check_adapters().map_err(bad_request)?;
        let running = {
            let mut slots = self.slots();
            if Self::taken(&slots, &adapter.id) {
                return Err(bad_request(format!("{} is already configured on this node", adapter.id)));
            }
            let slot = slots.get_mut(base).ok_or_else(|| unknown_model(base))?;
            slot.launch.adapters.push(adapter.clone());
            !slot.serves(&adapter)
        };
        self.emit(SupervisorEvent::Added { model: adapter.id });
        Ok(if running { AdapterChange::NextLoad } else { AdapterChange::Live })
    }

    /// [`Supervisor::add_adapter`], then, when the base runs on vLLM with LoRA on
    /// ([`ModelLaunch::runtime_lora`]), loads the adapter into the running engine
    /// (`/v1/load_lora_adapter`) so it serves at once, without restarting the base. If the engine
    /// refuses it, the adapter is taken off the base again and the engine's error returned.
    /// On llama.cpp an adapter file the engine didn't start with still waits for the next load
    /// ([`AdapterChange::NextLoad`]).
    pub async fn add_adapter_live(&self, base: &str, adapter: LoraAdapter) -> Result<AdapterChange, RemoteError> {
        let _op = self.inner.ops.lock().await;
        let change = self.add_adapter(base, adapter.clone())?;
        if change == AdapterChange::Live {
            return Ok(change);
        }
        let Some((url, key)) = self.runtime_engine(base) else { return Ok(change) };
        let body = json!({"lora_name": adapter.id, "lora_path": adapter.path});
        match self.post(format!("{url}/v1/load_lora_adapter"), key.as_deref(), &body).await {
            Ok(_) => {
                if let Some(r) = self.slots().get_mut(base).and_then(|s| s.running.as_mut()) {
                    r.adapters.push(adapter);
                }
                Ok(AdapterChange::Live)
            }
            Err(e) => {
                let _ = self.remove_adapter(base, &adapter.id);
                Err(e)
            }
        }
    }

    /// [`Supervisor::remove_adapter`], then, on a running vLLM engine with run-time LoRA, unloads
    /// it from the engine too (`/v1/unload_lora_adapter`), freeing its memory. llama-server keeps
    /// the file loaded (unused) until the base's next load; adding it back is then live.
    pub async fn remove_adapter_live(&self, base: &str, id: &str) -> Result<LoraAdapter, RemoteError> {
        let _op = self.inner.ops.lock().await;
        let removed = self.remove_adapter(base, id)?;
        let Some((url, key)) = self.runtime_engine(base) else { return Ok(removed) };
        let loaded = self.slots().get(base).and_then(|s| s.running.as_ref()).is_some_and(|r| r.adapters.iter().any(|x| x.id == id));
        if loaded {
            match self.post(format!("{url}/v1/unload_lora_adapter"), key.as_deref(), &json!({"lora_name": id})).await {
                Ok(_) => {
                    if let Some(r) = self.slots().get_mut(base).and_then(|s| s.running.as_mut()) {
                        r.adapters.retain(|x| x.id != id);
                    }
                }
                Err(e) => tracing::warn!(model = %base, adapter = %id, "engine kept the adapter loaded: {e}"),
            }
        }
        Ok(removed)
    }

    /// Changes how strongly adapter `id` applies, from the next request on (`None`: 1.0). Live:
    /// llama.cpp takes the scale per request, so the base keeps running. Requests may still set
    /// their own ([`Request::adapter_scale`]). vLLM applies adapters at full strength and refuses.
    pub fn set_adapter_scale(&self, id: &str, scale: Option<f32>) -> Result<(), RemoteError> {
        if let Some(s) = scale {
            check_scale(s)?;
        }
        let route = self.route(id)?;
        if route.adapter.is_none() {
            return Err(bad_request(format!("{id} is not an adapter")));
        }
        let mut slots = self.slots();
        let slot = slots.get_mut(&route.base).ok_or_else(|| unknown_model(id))?;
        if slot.launch.engine != EngineKind::Llamacpp {
            return Err(bad_request(format!("{id} runs on {}, which applies adapters at full strength", slot.launch.engine.name())));
        }
        if let Some(a) = slot.launch.adapters.iter_mut().find(|a| a.id == id) {
            a.scale = scale;
        }
        Ok(())
    }

    /// The URL and key of `base`'s engine when it is ready and takes adapters at run time.
    fn runtime_engine(&self, base: &str) -> Option<(String, Option<String>)> {
        let slots = self.slots();
        let s = slots.get(base)?;
        let r = s.running.as_ref().filter(|r| r.runtime_lora && s.state == ModelState::Ready)?;
        Some((r.base.clone(), r.api_key.clone()))
    }

    /// Stops serving adapter `id` on `base` (requests for it fail from now on; the engine keeps
    /// it loaded until the base's next load, see [`Supervisor::remove_adapter_live`]). Emits
    /// [`SupervisorEvent::Removed`].
    pub fn remove_adapter(&self, base: &str, id: &str) -> Result<LoraAdapter, RemoteError> {
        let removed = {
            let mut slots = self.slots();
            let slot = slots.get_mut(base).ok_or_else(|| unknown_model(base))?;
            let at = slot.launch.adapters.iter().position(|a| a.id == id).ok_or_else(|| bad_request(format!("{base} has no adapter {id}")))?;
            slot.launch.adapters.remove(at)
        };
        self.emit(SupervisorEvent::Removed { model: id.into() });
        Ok(removed)
    }

    /// Sets (or with `None` clears) llama.cpp model `id`'s multimodal projector
    /// ([`ModelLaunch::mmproj`]), from the next load on: its capabilities change at once, and
    /// [`AdapterChange::NextLoad`] says the running engine started with another projector
    /// (llama-server reads it only at start; loading the model restarts it).
    pub async fn set_projector_live(&self, id: &str, mmproj: Option<String>) -> Result<AdapterChange, RemoteError> {
        let _op = self.inner.ops.lock().await;
        let change = {
            let mut slots = self.slots();
            let slot = slots.get_mut(id).ok_or_else(|| unknown_model(id))?;
            if slot.launch.engine != EngineKind::Llamacpp || slot.launch.wrap.is_some() || slot.launch.embedding {
                return Err(bad_request(format!("{id}: only a llama.cpp chat model this node starts takes a projector")));
            }
            if let Some(p) = &mmproj
                && p.contains(',')
            {
                return Err(bad_request(format!("{id}: llama.cpp can't take a path with a comma ({p})")));
            }
            if slot.launch.mmproj != mmproj {
                // What the old projector measured no longer holds; the next load measures again.
                slot.measured = None;
            }
            slot.launch.mmproj = mmproj.clone();
            match &slot.running {
                Some(r) if r.mmproj != mmproj => AdapterChange::NextLoad,
                _ => AdapterChange::Live,
            }
        };
        Ok(change)
    }

    /// Adds a model the node may load from now on, without a restart (for example one just
    /// downloaded). It starts unloaded. Refused when the id is taken; [`Supervisor::remove_model`]
    /// first to replace one. Emits [`SupervisorEvent::Added`].
    pub fn add_model(&self, launch: ModelLaunch) -> Result<(), RemoteError> {
        if launch.id.trim().is_empty() {
            return Err(bad_request("a model needs an id"));
        }
        if let Some(w) = &launch.wrap {
            w.check().map_err(bad_request)?;
        }
        launch.check_adapters().map_err(bad_request)?;
        let id = launch.id.clone();
        let ids = launch.served_ids();
        {
            let mut slots = self.slots();
            if let Some(t) = ids.iter().find(|i| Self::taken(&slots, i)) {
                return Err(bad_request(format!("{t} is already configured on this node")));
            }
            slots.insert(id.clone(), Slot::new(launch));
        }
        self.seed_limits(&id);
        for model in ids {
            self.emit(SupervisorEvent::Added { model });
        }
        Ok(())
    }

    /// Unloads `id` (draining requests under way) and forgets it, with its adapters. Emits
    /// [`SupervisorEvent::Removed`] for each. Returns its launch settings. An adapter's id is
    /// refused: use [`Supervisor::remove_adapter`].
    pub async fn remove_model(&self, id: &str) -> Result<ModelLaunch, RemoteError> {
        let _op = self.inner.ops.lock().await;
        if let Route { base, adapter: Some(_) } = self.route(id)? {
            return Err(bad_request(format!("{id} is an adapter on {base}; remove it with remove_adapter")));
        }
        self.stop(id, false).await;
        let slot = self.slots().remove(id).ok_or_else(|| unknown_model(id))?;
        for model in slot.launch.served_ids() {
            self.emit(SupervisorEvent::Removed { model });
        }
        Ok(slot.launch)
    }

    /// The configured models' launch settings, sorted by id.
    pub fn launches(&self) -> Vec<ModelLaunch> {
        let mut out: Vec<ModelLaunch> = self.slots().values().map(|s| s.launch.clone()).collect();
        out.sort_by(|a, b| a.id.cmp(&b.id));
        out
    }

    /// Engine events from now on.
    pub fn subscribe(&self) -> broadcast::Receiver<SupervisorEvent> {
        self.inner.events.subscribe()
    }

    fn now_ms(&self) -> u64 {
        self.inner.epoch.elapsed().as_millis() as u64
    }

    /// One row per model, and one per adapter (with [`ModelStatus::base`] set, the base's engine,
    /// capabilities and state; `Unloaded` while the base runs without it).
    pub fn status(&self) -> Vec<ModelStatus> {
        let mut out: Vec<ModelStatus> = self
            .slots()
            .values()
            .flat_map(|s| {
                let row = Self::row(s);
                let adapters: Vec<ModelStatus> = s
                    .launch
                    .adapters
                    .iter()
                    .map(|a| {
                        let mut r = row.clone();
                        if !s.serves(a) {
                            r.state = ModelState::Unloaded;
                            r.pid = None;
                        }
                        r.id = a.id.clone();
                        r.base = Some(s.launch.id.clone());
                        r.size_bytes = std::fs::metadata(&a.path).ok().filter(|m| m.is_file()).map(|m| m.len());
                        r
                    })
                    .collect();
                std::iter::once(row).chain(adapters)
            })
            .collect();
        out.sort_by(|a, b| a.id.cmp(&b.id));
        out
    }

    fn row(s: &Slot) -> ModelStatus {
        ModelStatus {
            id: s.launch.id.clone(),
            engine: s.launch.engine,
            state: s.state.clone(),
            in_flight: s.in_flight.load(Ordering::SeqCst),
            swap_group: s.launch.swap_group.clone(),
            embedding: s.launch.embedding,
            pid: s.running.as_ref().and_then(|r| r.pid),
            last_load_secs: s.load_secs,
            size_bytes: if s.launch.wrap.is_some() { None } else { std::fs::metadata(&s.launch.source).ok().map(|m| m.len()) },
            wrapped: s.launch.wrap.is_some(),
            crashes: s.crashes.len(),
            context: s.launch.context,
            slots: s.launch.slots,
            capabilities: Some(s.capabilities()),
            base: None,
            context_tokens: s.measured.as_ref().and_then(|r| r.limits.context_tokens),
        }
    }

    /// Requests under way across all models.
    pub fn in_flight(&self) -> usize {
        self.slots().values().map(|s| s.in_flight.load(Ordering::SeqCst)).sum()
    }

    /// `id`'s state; an adapter has its base's (`Unloaded` while the base runs without it).
    pub fn state(&self, id: &str) -> Option<ModelState> {
        let route = self.route(id).ok()?;
        let slots = self.slots();
        let s = slots.get(&route.base)?;
        match &route.adapter {
            Some(a) if !s.serves(a) => Some(ModelState::Unloaded),
            _ => Some(s.state.clone()),
        }
    }

    /// What `id` takes ([`ModelLaunch::capabilities`]), with its limits: the operator's
    /// ([`ModelLaunch::limits`]) over those measured at load ([`Supervisor::measured_limits`]); an
    /// adapter takes what its base does.
    pub fn capabilities(&self, id: &str) -> Option<Capabilities> {
        let route = self.route(id).ok()?;
        self.slots().get(&route.base).map(Slot::capabilities)
    }

    /// What `id` (or an adapter's base) measured when it last loaded, as `weirdnet-model`'s
    /// [`LimitsRecord`]: for a llama.cpp model whose source is a GGUF file, what its header and
    /// its projector's say ([`weirdnet_model::limits::measure`]; llama.cpp `--image-max-tokens`,
    /// when passed, over the header's per-image cap for size-dependent projectors), and for every
    /// engine that reports it the context per sequence it runs with (llama-server `/props`, vLLM
    /// `/v1/models`). Every figure that can't be read is absent, never defaulted. `None` before
    /// the first load (unless the [store](Supervisor::set_limits_store) kept a record) or when
    /// nothing could be measured. The capabilities carry these under the operator's
    /// ([`Supervisor::capabilities`]); the context goes into [`ModelStatus::context_tokens`].
    pub fn measured_limits(&self, id: &str) -> Option<LimitsRecord> {
        let route = self.route(id).ok()?;
        self.slots().get(&route.base).and_then(|s| s.measured.clone())
    }

    /// Keeps measured limits in `store` (`weirdnet-model`'s [`ModelStore::record_limits`],
    /// next to the model's files) for llama.cpp models served from it, and reads them back now
    /// for models not measured yet, so an unloaded model reports its image and audio figures
    /// after a restart. A record read back has no context (that belongs to the run that measured
    /// it), and one taken with another projector is ignored. `None` stops keeping them.
    /// Blocking: reads the store.
    pub fn set_limits_store(&self, store: Option<ModelStore>) {
        *self.inner.limits_store.lock().unwrap_or_else(|e| e.into_inner()) = store;
        let ids: Vec<String> = self.slots().iter().filter(|(_, s)| s.measured.is_none()).map(|(id, _)| id.clone()).collect();
        for id in ids {
            self.seed_limits(&id);
        }
    }

    fn limits_store(&self) -> Option<ModelStore> {
        self.inner.limits_store.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn engine_args(&self, engine: EngineKind) -> Vec<String> {
        self.inner.engines.get(&engine).map(|e| e.extra_args.clone()).unwrap_or_default()
    }

    /// `id`'s record from the store, when it has none yet. Blocking.
    fn seed_limits(&self, id: &str) {
        let Some(store) = self.limits_store() else { return };
        let Some(launch) = self.slots().get(id).filter(|s| s.measured.is_none()).map(|s| s.launch.clone()) else { return };
        let Some(record) = crate::measure::kept(&store, &launch, &self.engine_args(launch.engine)) else { return };
        if let Some(s) = self.slots().get_mut(id)
            && s.measured.is_none()
            && s.launch == launch
        {
            s.measured = Some(record);
        }
    }

    /// The context per sequence the engine at `base` reports for `launch`: llama-server's
    /// `/props` (`default_generation_settings.n_ctx`, per slot), vLLM's `/v1/models`
    /// (`max_model_len`). `None` when it doesn't answer or doesn't say.
    async fn served_context(&self, launch: &ModelLaunch, base: &str, key: Option<&str>) -> Option<u32> {
        let get = |path: &str| {
            let mut req = self.inner.http.get(format!("{base}{path}")).timeout(Duration::from_secs(5));
            if let Some(k) = key {
                req = req.bearer_auth(k);
            }
            req
        };
        let read = |v: Option<u64>| v.filter(|n| *n > 0).and_then(|n| u32::try_from(n).ok());
        match launch.engine {
            EngineKind::Llamacpp => {
                let v: Value = get("/props").send().await.ok().filter(|r| r.status().is_success())?.json().await.ok()?;
                read(v["default_generation_settings"]["n_ctx"].as_u64())
            }
            EngineKind::Vllm => {
                let v: Value = get("/v1/models").send().await.ok().filter(|r| r.status().is_success())?.json().await.ok()?;
                let row = v["data"].as_array()?.iter().find(|m| m["id"].as_str() == Some(launch.id.as_str()))?;
                read(row["max_model_len"].as_u64())
            }
            _ => None,
        }
    }

    /// Measures `launch`, just loaded at `base`, and keeps the record in the store when set.
    async fn measure_loaded(&self, launch: &ModelLaunch, base: &str, key: Option<&str>) -> Option<LimitsRecord> {
        let context = self.served_context(launch, base, key).await;
        let (launch, args, store) = (launch.clone(), self.engine_args(launch.engine), self.limits_store());
        tokio::task::spawn_blocking(move || {
            let record = crate::measure::measure(&launch, &args, context)?;
            if let Some(store) = store {
                crate::measure::keep(&store, &launch, &record);
            }
            Some(record)
        })
        .await
        .ok()
        .flatten()
    }

    /// Starts `id` and waits until it is ready (see [`Supervisor::load_with`]).
    pub async fn load(&self, id: &str) -> Result<(), RemoteError> {
        self.load_with(id, false).await
    }

    /// Starts `id` and waits until its engine is ready. Loading a model in a swap group unloads the
    /// group's other models first. Already ready: returns at once. A model in a crash loop or
    /// waiting for its restart refuses unless `force` (an operator's explicit load), which also
    /// clears the crash history.
    ///
    /// An adapter loads its base. When the base is ready without it (added after the base
    /// started), the base is restarted.
    ///
    /// While the node is reserved ([`Supervisor::reserve`]) every load is refused, `force` too.
    pub async fn load_with(&self, id: &str, force: bool) -> Result<(), RemoteError> {
        let route = self.route(id)?;
        self.check_unreserved()?;
        let wanted = route.adapter;
        let id = route.base.as_str();
        // A request during a crash backoff waits it out (at most `backoff_max`) rather than
        // failing; the scheduled restart and this load serialize on `ops`.
        if !force {
            let wait = {
                let slots = self.slots();
                let slot = slots.get(id).ok_or_else(|| unknown_model(id))?;
                slot.restart_at.and_then(|at| at.checked_duration_since(Instant::now()))
            };
            if let Some(d) = wait {
                tokio::time::sleep(d.min(self.inner.opts.backoff_max)).await;
            }
        }
        let _op = self.inner.ops.lock().await;
        // Checked again under `ops`: a reservation taken while this load waited wins.
        self.check_unreserved()?;
        let (launch, group_mates) = {
            let mut slots = self.slots();
            let slot = slots.get_mut(id).ok_or_else(|| unknown_model(id))?;
            let has_wanted = wanted.as_ref().is_none_or(|w| slot.running.is_some() && slot.serves(w));
            let projector_current = slot.launch.wrap.is_some() || slot.running.as_ref().is_none_or(|r| r.mmproj == slot.launch.mmproj);
            if slot.state == ModelState::Ready && has_wanted && projector_current {
                return Ok(());
            }
            if force {
                slot.crashes.clear();
                slot.crash_loop = false;
                slot.restart_at = None;
            } else if slot.crash_loop {
                return Err(unavailable(format!("{id} is in a crash loop ({} crashes); fix it, then load it again with force", slot.crashes.len())));
            }
            let launch = slot.launch.clone();
            let mates: Vec<String> = match &launch.swap_group {
                Some(g) => slots
                    .values()
                    .filter(|s| s.launch.id != id && s.launch.swap_group.as_ref() == Some(g) && s.running.is_some())
                    .map(|s| s.launch.id.clone())
                    .collect(),
                None => Vec::new(),
            };
            (launch, mates)
        };
        for mate in group_mates {
            tracing::info!(model = %mate, for_model = %id, "swap group: unloading");
            self.stop(&mate, false).await;
        }
        // A failed engine may still have a process record; clear it.
        self.stop(id, false).await;
        let started = Instant::now();
        self.emit(SupervisorEvent::Starting { model: id.into() });
        let result = match &launch.wrap {
            Some(_) => self.attach(&launch).await,
            None => self.start(&launch).await,
        };
        match &result {
            Ok(()) => {
                let secs = started.elapsed().as_secs_f32();
                if let Some(s) = self.slots().get_mut(id) {
                    s.load_secs = Some(secs);
                    s.restart_at = None;
                    s.last_used.store(self.now_ms(), Ordering::SeqCst);
                }
                self.emit(SupervisorEvent::Ready { model: id.into(), load_secs: secs });
            }
            Err(e) => self.emit(SupervisorEvent::LoadFailed { model: id.into(), why: e.to_string() }),
        }
        result
    }

    /// Drains and stops `id`. A wrapped server is only detached (the node doesn't own it).
    /// An adapter's id is refused: unload its base.
    pub async fn unload(&self, id: &str) -> Result<(), RemoteError> {
        let _op = self.inner.ops.lock().await;
        if let Route { base, adapter: Some(_) } = self.route(id)? {
            return Err(bad_request(format!("{id} is an adapter on {base}; unload {base} instead")));
        }
        self.stop(id, false).await;
        Ok(())
    }

    /// Reserves the node for `holder`: from now on no engine starts (loads, loads on demand,
    /// crash restarts and swaps are refused as unavailable), and every running engine is drained
    /// (waiting up to `drain_timeout` for requests under way) and stopped. Returns once they are
    /// stopped. Taking it again under the same holder is a no-op; another holder's reservation is
    /// refused as unavailable. Held in memory until [`Supervisor::release`]: an application that
    /// must keep it across a restart records it and reserves again at start.
    pub async fn reserve(&self, holder: &str) -> Result<Reservation, RemoteError> {
        let holder = holder.trim();
        if holder.is_empty() {
            return Err(bad_request("a reservation needs a holder"));
        }
        let mut taken = None;
        let mut busy = None;
        self.inner.reserved.send_if_modified(|r| match r {
            Some(cur) if cur.holder == holder => {
                taken = Some(cur.clone());
                false
            }
            Some(cur) => {
                busy = Some(cur.holder.clone());
                false
            }
            None => {
                let new = Reservation::new(holder);
                taken = Some(new.clone());
                *r = Some(new);
                true
            }
        });
        if let Some(other) = busy {
            return Err(unavailable(format!("this node is reserved by {other}")));
        }
        let _op = self.inner.ops.lock().await;
        let ids: Vec<String> = self.slots().values().filter(|s| s.running.is_some()).map(|s| s.launch.id.clone()).collect();
        for id in ids {
            tracing::info!(model = %id, holder, "reserved: unloading");
            self.stop(&id, false).await;
        }
        Ok(taken.expect("set above"))
    }

    /// Ends `holder`'s reservation; models load again on the next request or load. `false` when
    /// the node was not reserved; another holder's reservation is refused as a bad request.
    pub fn release(&self, holder: &str) -> Result<bool, RemoteError> {
        let holder = holder.trim();
        let mut other = None;
        let released = self.inner.reserved.send_if_modified(|r| match r {
            Some(cur) if cur.holder == holder => {
                *r = None;
                true
            }
            Some(cur) => {
                other = Some(cur.holder.clone());
                false
            }
            None => false,
        });
        match other {
            Some(o) => Err(bad_request(format!("this node is reserved by {o}, not {holder}"))),
            None => Ok(released),
        }
    }

    /// The node's reservation, if any.
    pub fn reservation(&self) -> Option<Reservation> {
        self.inner.reserved.borrow().clone()
    }

    /// Changes to the reservation, from now on.
    pub fn reservation_changes(&self) -> watch::Receiver<Option<Reservation>> {
        self.inner.reserved.subscribe()
    }

    fn check_unreserved(&self) -> Result<(), RemoteError> {
        match &*self.inner.reserved.borrow() {
            Some(r) => Err(unavailable(format!("this node is reserved by {}: it starts no model until released", r.holder))),
            None => Ok(()),
        }
    }

    /// Stops every engine (daemon shutdown).
    pub async fn shutdown(&self) {
        let _op = self.inner.ops.lock().await;
        let ids: Vec<String> = self.slots().keys().cloned().collect();
        for id in ids {
            self.stop(&id, false).await;
        }
    }

    /// Unloads ready models idle past their `idle_ttl`. Returns what it unloaded. Run it every few
    /// seconds ([`Supervisor::spawn_reaper`]).
    pub async fn reap_idle(&self) -> Vec<String> {
        let now = self.now_ms();
        let idle: Vec<String> = self
            .slots()
            .values()
            .filter(|s| s.state == ModelState::Ready && s.in_flight.load(Ordering::SeqCst) == 0)
            .filter(|s| s.launch.idle_ttl.is_some_and(|ttl| now.saturating_sub(s.last_used.load(Ordering::SeqCst)) >= ttl.as_millis() as u64))
            .map(|s| s.launch.id.clone())
            .collect();
        let mut done = Vec::new();
        for id in idle {
            // Never wait behind a load or swap; try again next round.
            let Ok(_op) = self.inner.ops.try_lock() else { break };
            let still_idle = self.slots().get(&id).is_some_and(|s| s.state == ModelState::Ready && s.in_flight.load(Ordering::SeqCst) == 0);
            if still_idle {
                tracing::info!(model = %id, "idle: unloading");
                self.stop(&id, true).await;
                done.push(id);
            }
        }
        done
    }

    /// Runs [`Supervisor::reap_idle`] every `every` until the returned handle is aborted.
    pub fn spawn_reaper(&self, every: Duration) -> tokio::task::JoinHandle<()> {
        let me = self.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(every);
            loop {
                tick.tick().await;
                me.reap_idle().await;
            }
        })
    }

    /// A wrapped stock server: probe it and mark it ready.
    async fn attach(&self, launch: &ModelLaunch) -> Result<(), RemoteError> {
        let wrap = launch.wrap.as_ref().expect("attach is for wrapped models");
        wrap.check().map_err(unavailable)?;
        let base = wrap.base();
        let ok = self.inner.http.get(format!("{base}/health")).timeout(Duration::from_secs(5)).send().await.is_ok_and(|r| r.status().is_success());
        // A stock mlx_lm.server still needs the proxy for `default_model` and text-only checks.
        let (base, api_key, proxy) = if ok && launch.engine == EngineKind::Mlx {
            let l = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await.map_err(|e| unavailable(format!("mlx proxy: {e}")))?;
            let url = format!("http://{}", l.local_addr().map_err(|e| unavailable(format!("mlx proxy: {e}")))?);
            let key = random_hex(24);
            (url, Some(key.clone()), Some(crate::mlx::spawn_proxy(l, base, launch.id.clone(), key)))
        } else {
            (base, wrap.api_key.clone(), None)
        };
        let measured = if ok { self.measure_loaded(launch, &wrap.base(), wrap.api_key.as_deref()).await } else { None };
        let (tx, rx) = watch::channel(None);
        let mut slots = self.slots();
        let slot = slots.get_mut(&launch.id).expect("checked by load");
        slot.generation += 1;
        if measured.is_some() {
            slot.measured = measured;
        }
        if !ok {
            slot.state = ModelState::Failed(format!("wrapped server {base} is not answering /health"));
            return Err(unavailable(format!("{}: wrapped server {base} is not answering /health", launch.id)));
        }
        slot.state = ModelState::Ready;
        slot.running = Some(Running {
            base,
            adapters: Vec::new(),
            runtime_lora: false,
            mmproj: None,
            api_key,
            pid: None,
            stop: None,
            exited: rx,
            _exit_tx: Some(tx),
            _proxy: proxy,
        });
        Ok(())
    }

    async fn start(&self, launch: &ModelLaunch) -> Result<(), RemoteError> {
        let engine = self
            .inner
            .engines
            .get(&launch.engine)
            .ok_or_else(|| unavailable(format!("engine {} is not configured on this node", launch.engine.name())))?
            .clone();
        if let Err(why) = launch.check_adapters() {
            if let Some(s) = self.slots().get_mut(&launch.id) {
                s.state = ModelState::Failed(why.clone());
            }
            return Err(bad_request(why));
        }
        if launch.engine == EngineKind::Mlx && launch.embedding {
            let why = "MLX-LM serves no embeddings; run embedding models on llama.cpp or vLLM".to_string();
            if let Some(s) = self.slots().get_mut(&launch.id) {
                s.state = ModelState::Failed(why.clone());
            }
            return Err(bad_request(format!("{}: {why}", launch.id)));
        }
        // The engine is told a free port, which is free when picked but not held (it can't be
        // handed over): another process can bind it before the engine does. Then the engine fails
        // to bind and exits, or another server answers on the port, and either is caught below and
        // the launch is tried again on a new port.
        let mut tries = 0;
        loop {
            tries += 1;
            match self.start_once(launch, &engine).await {
                Ok(()) => return Ok(()),
                Err(Start::PortTaken(e)) if tries < PORT_TRIES => {
                    tracing::warn!(model = %launch.id, "{e}; trying another port");
                }
                Err(Start::PortTaken(e) | Start::Failed(e)) => return Err(e),
            }
        }
    }

    /// One launch of `launch` on a newly picked port, up to ready.
    async fn start_once(&self, launch: &ModelLaunch, engine: &EngineConfig) -> Result<(), Start> {
        let mlx = launch.engine == EngineKind::Mlx;
        let no_port = |e: std::io::Error| Start::Failed(unavailable(format!("no free loopback port: {e}")));
        // MLX-LM has no API key: it gets a private port, and the advertised one is the proxy's,
        // bound now so nothing else can take it.
        let (port, proxy_listener) = if mlx {
            let l = std::net::TcpListener::bind(("127.0.0.1", 0)).map_err(no_port)?;
            l.set_nonblocking(true).map_err(no_port)?;
            (l.local_addr().map_err(no_port)?.port(), Some(l))
        } else {
            (free_port().map_err(no_port)?, None)
        };
        let engine_port = if mlx { free_port().map_err(no_port)? } else { port };
        let api_key = random_hex(24);
        let log_from = self.log_len(&launch.id);
        let mut cmd = engine.command(launch, engine_port, &api_key);
        cmd.stdin(Stdio::null());
        match self.log_file(&launch.id) {
            Some((out, err)) => {
                cmd.stdout(out).stderr(err);
            }
            None => {
                cmd.stdout(Stdio::null()).stderr(Stdio::null());
            }
        }
        #[cfg(unix)]
        {
            cmd.process_group(0);
            #[cfg(target_os = "linux")]
            // SAFETY: prctl is async-signal-safe and touches no memory of the parent.
            unsafe {
                cmd.pre_exec(|| {
                    libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
                    Ok(())
                });
            }
        }
        let mut child = cmd.spawn().map_err(|e| Start::Failed(unavailable(format!("could not start {}: {e}", engine.program.display()))))?;
        let pid = child.id();
        let proxy = match proxy_listener.map(tokio::net::TcpListener::from_std) {
            Some(Ok(l)) => Some(crate::mlx::spawn_proxy(l, format!("http://127.0.0.1:{engine_port}"), launch.id.clone(), api_key.clone())),
            Some(Err(e)) => {
                let _ = terminate(&mut child, pid, self.inner.opts.stop_grace).await;
                return Err(Start::Failed(unavailable(format!("mlx proxy: {e}"))));
            }
            None => None,
        };
        let (exit_tx, exit_rx) = watch::channel(None);
        let (stop_tx, stop_rx) = oneshot::channel::<()>();
        let generation = {
            let mut slots = self.slots();
            let slot = slots.get_mut(&launch.id).expect("checked by load");
            slot.generation += 1;
            slot.state = ModelState::Loading;
            slot.running = Some(Running {
                base: format!("http://127.0.0.1:{port}"),
                adapters: launch.adapters.clone(),
                runtime_lora: launch.runtime_lora(),
                mmproj: launch.mmproj.clone(),
                api_key: Some(api_key.clone()),
                pid,
                stop: Some(stop_tx),
                exited: exit_rx.clone(),
                _exit_tx: None,
                _proxy: proxy,
            });
            slot.generation
        };
        tracing::info!(model = %launch.id, pid, port, engine = launch.engine.name(), "engine starting");

        // The monitor owns the child: it reaps it and does the stopping.
        let grace = self.inner.opts.stop_grace;
        let me = self.clone();
        let id = launch.id.clone();
        let port_taken = Arc::new(AtomicBool::new(false));
        let taken = port_taken.clone();
        tokio::spawn(async move {
            let status = tokio::select! {
                s = child.wait() => s,
                _ = stop_rx => terminate(&mut child, pid, grace).await,
            };
            let desc = match status {
                Ok(s) => format!("engine exited ({s})"),
                Err(e) => format!("engine wait failed: {e}"),
            };
            // An engine that dies while loading because its port was taken is not a crash: the
            // launch is tried again on another port.
            let lost_port = me.is_loading(&id, generation) && me.lost_port(&id, engine_port, log_from);
            taken.store(lost_port, Ordering::SeqCst);
            // Record the exit (and count a crash) before anyone waiting on it wakes up.
            me.exited(&id, generation, desc.clone(), lost_port);
            let _ = exit_tx.send(Some(desc));
        });

        let started = Instant::now();
        let health = format!("http://127.0.0.1:{port}/health");
        let mut exited = exit_rx;
        loop {
            let gone = exited.borrow().clone();
            if let Some(desc) = gone {
                if port_taken.load(Ordering::SeqCst) {
                    return Err(Start::PortTaken(self.failed(
                        &launch.id,
                        generation,
                        format!("{desc} while loading: port {engine_port} was taken by another process"),
                    )));
                }
                return Err(Start::Failed(self.failed(&launch.id, generation, format!("{desc} while loading{}", self.log_tail(&launch.id)))));
            }
            if let Ok(resp) = self.inner.http.get(&health).timeout(Duration::from_secs(2)).send().await
                && resp.status().is_success()
            {
                match self.identify(launch, port, &api_key).await {
                    Some(true) => break,
                    // Healthy, but not ours: another server holds the port and ours can't bind it.
                    Some(false) => {
                        self.stop(&launch.id, false).await;
                        return Err(Start::PortTaken(self.failed(
                            &launch.id,
                            generation,
                            format!("another server answers on port {port} (not this launch's key or model)"),
                        )));
                    }
                    None => {}
                }
            }
            if started.elapsed() > engine.ready_timeout {
                self.stop(&launch.id, false).await;
                return Err(Start::Failed(self.failed(
                    &launch.id,
                    generation,
                    format!("not ready after {:?}{}", engine.ready_timeout, self.log_tail(&launch.id)),
                )));
            }
            let _ = tokio::time::timeout(Duration::from_millis(200), exited.changed()).await;
        }
        let measured = self.measure_loaded(launch, &format!("http://127.0.0.1:{port}"), Some(&api_key)).await;
        let mut slots = self.slots();
        let slot = slots.get_mut(&launch.id).expect("slot exists");
        if slot.generation == generation && slot.state == ModelState::Loading {
            if measured.is_some() {
                slot.measured = measured;
            }
            slot.state = ModelState::Ready;
            tracing::info!(model = %launch.id, secs = started.elapsed().as_secs_f32(), "engine ready");
            Ok(())
        } else {
            Err(Start::Failed(unavailable(format!("{} stopped while loading", launch.id))))
        }
    }

    /// Whether the server answering on `port` is this launch's engine: it takes the launch's key
    /// on `/v1/models` (llama-server and vLLM check it there) and lists the launch's id. `None`
    /// while that can't be told yet (no answer, or a server error). MLX-LM has no key and its port
    /// is the proxy's, bound before the engine starts, so it is always ours.
    async fn identify(&self, launch: &ModelLaunch, port: u16, key: &str) -> Option<bool> {
        if launch.engine == EngineKind::Mlx {
            return Some(true);
        }
        let resp =
            self.inner.http.get(format!("http://127.0.0.1:{port}/v1/models")).bearer_auth(key).timeout(Duration::from_secs(5)).send().await.ok()?;
        let status = resp.status();
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return Some(false);
        }
        if !status.is_success() {
            return None;
        }
        let v: Value = resp.json().await.ok()?;
        Some(lists_model(&v, &launch.id))
    }

    fn is_loading(&self, id: &str, generation: u64) -> bool {
        self.slots().get(id).is_some_and(|s| s.generation == generation && s.state == ModelState::Loading)
    }

    /// Whether an engine that just exited lost its port to another process: something else holds
    /// `port` now, or its log since the launch (from byte `log_from`) says the address was in use.
    fn lost_port(&self, id: &str, port: u16, log_from: u64) -> bool {
        if std::net::TcpListener::bind(("127.0.0.1", port)).is_err() {
            return true;
        }
        let Some(dir) = &self.inner.opts.log_dir else { return false };
        let Ok(bytes) = std::fs::read(dir.join(format!("{id}.log"))) else { return false };
        let new = bytes.get(log_from as usize..).unwrap_or_default();
        says_address_in_use(&String::from_utf8_lossy(new))
    }

    fn log_len(&self, id: &str) -> u64 {
        let Some(dir) = &self.inner.opts.log_dir else { return 0 };
        std::fs::metadata(dir.join(format!("{id}.log"))).map(|m| m.len()).unwrap_or(0)
    }

    /// The monitor saw the engine exit. A stop sets `Unloaded` before signalling, so anything else
    /// is a crash: count it, and schedule a restart with backoff unless it is a crash loop.
    /// `lost_port`: it died while loading because its port was taken; the launch is retried, so
    /// it is not counted as a crash.
    fn exited(&self, id: &str, generation: u64, desc: String, lost_port: bool) {
        let opts = &self.inner.opts;
        let (event, restart_in) = {
            let mut slots = self.slots();
            let Some(slot) = slots.get_mut(id).filter(|s| s.generation == generation) else { return };
            slot.running = None;
            if slot.state == ModelState::Unloaded {
                return;
            }
            if lost_port {
                slot.state = ModelState::Failed(format!("{desc}: its port was taken"));
                return;
            }
            let was_serving = slot.state == ModelState::Ready;
            let now = Instant::now();
            slot.crashes.retain(|t| now.duration_since(*t) < opts.crash_window);
            slot.crashes.push(now);
            let n = slot.crashes.len();
            tracing::warn!(model = %id, crashes = n, "{desc}");
            if n >= opts.max_crashes {
                slot.crash_loop = true;
                slot.state = ModelState::Failed(format!("crash loop: {n} crashes in {}s ({desc})", opts.crash_window.as_secs()));
                (SupervisorEvent::CrashLoop { model: id.into(), crashes: n }, None)
            } else {
                slot.state = ModelState::Failed(desc.clone());
                // A crash while loading is reported to whoever is loading; only a crash while
                // serving restarts on its own.
                let delay = (opts.restart && was_serving).then(|| backoff(opts.backoff_base, opts.backoff_max, n));
                slot.restart_at = delay.map(|d| now + d);
                (SupervisorEvent::Crashed { model: id.into(), why: desc, crashes: n, restart_in_secs: delay.map(|d| d.as_secs_f32()) }, delay)
            }
        };
        self.emit(event);
        if let Some(delay) = restart_in {
            let me = self.clone();
            let id = id.to_string();
            tokio::spawn(async move {
                tokio::time::sleep(delay).await;
                let due = me.slots().get(&id).is_some_and(|s| {
                    matches!(s.state, ModelState::Failed(_)) && !s.crash_loop && s.restart_at.is_some_and(|at| at <= Instant::now())
                });
                if due {
                    tracing::info!(model = %id, "restarting after crash");
                    if let Some(s) = me.slots().get_mut(&id) {
                        s.restart_at = None;
                    }
                    if let Err(e) = me.load(&id).await {
                        tracing::warn!(model = %id, "restart failed: {e}");
                    }
                }
            });
        }
    }

    fn failed(&self, id: &str, generation: u64, why: String) -> RemoteError {
        if let Some(slot) = self.slots().get_mut(id).filter(|s| s.generation == generation)
            && !slot.crash_loop
        {
            slot.state = ModelState::Failed(why.clone());
        }
        unavailable(format!("{id}: {why}"))
    }

    /// Drains, marks unloaded and stops the engine. Caller holds `ops`.
    async fn stop(&self, id: &str, idle: bool) {
        let in_flight = {
            let mut slots = self.slots();
            let Some(slot) = slots.get_mut(id) else { return };
            if slot.running.is_none() {
                if !matches!(slot.state, ModelState::Failed(_)) {
                    slot.state = ModelState::Unloaded;
                }
                return;
            }
            slot.state = ModelState::Draining;
            slot.in_flight.clone()
        };
        let deadline = Instant::now() + self.inner.opts.drain_timeout;
        while in_flight.load(Ordering::SeqCst) > 0 && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let (stop, mut exited) = {
            let mut slots = self.slots();
            let Some(slot) = slots.get_mut(id) else { return };
            slot.state = ModelState::Unloaded;
            match slot.running.as_mut() {
                Some(r) if r.stop.is_some() => (r.stop.take(), r.exited.clone()),
                Some(_) => {
                    // Wrapped: detach only.
                    slot.running = None;
                    drop(slots);
                    self.emit(SupervisorEvent::Unloaded { model: id.into(), idle });
                    return;
                }
                None => return,
            }
        };
        if let Some(stop) = stop {
            let _ = stop.send(());
        }
        let wait = self.inner.opts.stop_grace + Duration::from_secs(5);
        let _ = tokio::time::timeout(wait, exited.wait_for(Option::is_some)).await;
        tracing::info!(model = %id, "engine stopped");
        self.emit(SupervisorEvent::Unloaded { model: id.into(), idle });
    }

    /// The engine, request target, URL, key and in-flight guard for pool id `id` (a model, or an
    /// adapter its ready base's engine has loaded). `scale` overrides the adapter's own.
    fn routed(&self, id: &str, scale: Option<f32>) -> Result<(EngineKind, Target, String, Option<String>, InFlight), RemoteError> {
        let route = self.route(id)?;
        if let Some(s) = scale {
            check_scale(s)?;
            if route.adapter.is_none() {
                return Err(bad_request(format!("{id} is not an adapter; an adapter scale doesn't apply")));
            }
        }
        let (engine, target) = {
            let slots = self.slots();
            let slot = slots.get(&route.base).ok_or_else(|| unknown_model(id))?;
            let engine = slot.launch.engine;
            let target = match &route.adapter {
                None => Target::Base { alias: route.base.clone(), has_adapters: slot.running.as_ref().is_some_and(|r| !r.adapters.is_empty()) },
                Some(a) => {
                    if scale.is_some() && engine != EngineKind::Llamacpp {
                        return Err(bad_request(format!("{id} runs on {}, which applies adapters at full strength", engine.name())));
                    }
                    let index = slot
                        .running
                        .as_ref()
                        .and_then(|r| r.index(engine, a))
                        .ok_or_else(|| unavailable(format!("{id} is not loaded; load it (this restarts {})", route.base)))?;
                    Target::Adapter { base: route.base.clone(), index, id: a.id.clone(), scale: scale.unwrap_or(a.effective_scale()) }
                }
            };
            (engine, target)
        };
        let (base, key, guard) = self.target(&route.base)?;
        Ok((engine, target, base, key, guard))
    }

    /// Where requests for `id` go, with the in-flight guard taken.
    fn target(&self, id: &str) -> Result<(String, Option<String>, InFlight), RemoteError> {
        let slots = self.slots();
        let slot = slots.get(id).ok_or_else(|| unknown_model(id))?;
        match (&slot.state, &slot.running) {
            (ModelState::Ready, Some(r)) => {
                Ok((r.base.clone(), r.api_key.clone(), InFlight::new(slot.in_flight.clone(), slot.last_used.clone(), self.inner.epoch)))
            }
            (state, _) => Err(unavailable(format!("{id} is not ready ({state:?})"))),
        }
    }

    async fn post(&self, url: String, key: Option<&str>, body: &Value) -> Result<reqwest::Response, RemoteError> {
        self.send(self.inner.http.post(url).json(body), key).await
    }

    async fn send(&self, mut req: reqwest::RequestBuilder, key: Option<&str>) -> Result<reqwest::Response, RemoteError> {
        if let Some(k) = key {
            req = req.bearer_auth(k);
        }
        let resp = req.send().await.map_err(|e| engine_error(format!("request failed: {e}")))?;
        let status = resp.status();
        if status.as_u16() == 503 || status.as_u16() == 429 {
            return Err(RemoteError::Busy);
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(match status.as_u16() {
                400 | 422 => bad_request(trim(&body)),
                _ => engine_error(format!("{status}: {}", trim(&body))),
            });
        }
        Ok(resp)
    }

    /// Transcribes `request.audio` with `id`, which must be ready and transcribe here
    /// ([`Capabilities::transcribe`]; otherwise `Unsupported`, so a pool tries elsewhere). Returns
    /// the transcript and the engine's token counts when it gives them.
    ///
    /// - llama.cpp (a model with an audio projector, such as Voxtral): a greedy
    ///   `/v1/chat/completions` request with the clip as a base64 `input_audio` part and an
    ///   instruction to answer with the transcript only, naming the language and the prompt when
    ///   given. No segments (the model can't time its text); the duration is read from a PCM WAV
    ///   clip's header. llama-server takes WAV and MP3.
    /// - vLLM (Whisper and other transcription models): `/v1/audio/transcriptions`, with the
    ///   language and prompt, `verbose_json` (segments, language, duration) when timestamps are
    ///   asked for.
    pub async fn transcribe(&self, id: &str, request: &TranscribeRequest) -> Result<(Transcript, Option<Usage>), RemoteError> {
        let engine = self.engine_of(id)?;
        if !self.capabilities(id).is_some_and(|c| c.transcribe) {
            return Err(RemoteError::Unsupported { message: format!("model `{id}` does not transcribe on this node") });
        }
        request.check()?;
        match engine {
            EngineKind::Llamacpp => {
                let mut stream = self.chat(id, &crate::transcribe::chat_request(request)).await?;
                let mut text = String::new();
                while let Some(chunk) = stream.next().await {
                    match chunk? {
                        Chunk::Text(t) => text.push_str(&t),
                        Chunk::Done { usage, .. } => {
                            let mut t = Transcript::new(text.trim());
                            if let Some(lang) = request.language.as_deref().map(str::trim).filter(|l| !l.is_empty()) {
                                t = t.with_language(lang);
                            }
                            if let Some(ms) = crate::transcribe::wav_duration_ms(&request.audio) {
                                t = t.with_duration_ms(ms);
                            }
                            return Ok((t, usage));
                        }
                        _ => {}
                    }
                }
                Err(engine_error("engine stream ended early"))
            }
            EngineKind::Vllm => {
                let (base, key, _guard) = self.target(id)?;
                let format = request.format.trim().to_ascii_lowercase();
                let file = reqwest::multipart::Part::bytes(request.audio.clone()).file_name(format!("audio.{format}"));
                let mut form = reqwest::multipart::Form::new()
                    .part("file", file)
                    .text("model", id.to_string())
                    .text("temperature", "0")
                    .text("response_format", if request.timestamps { "verbose_json" } else { "json" });
                if request.timestamps {
                    form = form.text("timestamp_granularities[]", "segment");
                }
                if let Some(l) = request.language.as_deref().map(str::trim).filter(|l| !l.is_empty()) {
                    form = form.text("language", l.to_string());
                }
                if let Some(p) = request.prompt.as_deref().filter(|p| !p.trim().is_empty()) {
                    form = form.text("prompt", p.to_string());
                }
                let resp = self.send(self.inner.http.post(format!("{base}/v1/audio/transcriptions")).multipart(form), key.as_deref()).await?;
                let v: Value = resp.json().await.map_err(|e| engine_error(format!("bad transcription response: {e}")))?;
                let t = crate::transcribe::from_openai(&v).ok_or_else(|| engine_error("transcription response has no text"))?;
                Ok((t, crate::transcribe::usage(&v)))
            }
            other => Err(RemoteError::Unsupported { message: format!("{id} runs on {}, which serves no transcription", other.name()) }),
        }
    }

    /// Streams a chat answer from `id`, which must be ready. Dropping the stream aborts the engine
    /// request. For an adapter on llama.cpp, [`Request::adapter_scale`] overrides its scale.
    pub async fn chat(&self, id: &str, request: &Request) -> Result<ChunkStream, RemoteError> {
        let (engine, target, base, key, guard) = self.routed(id, request.adapter_scale)?;
        let body = openai::chat_body_for(request, &target, engine)?;
        let resp = self.post(format!("{base}/v1/chat/completions"), key.as_deref(), &body).await?;
        Ok(stream_chunks(resp, guard))
    }

    /// Streams a raw text completion of `prompt` from `id`, which must be ready. Response
    /// formats and tools don't apply.
    pub async fn complete(&self, id: &str, prompt: &str, sampling: &Sampling, seed: Option<u64>) -> Result<ChunkStream, RemoteError> {
        self.complete_scaled(id, prompt, sampling, seed, None).await
    }

    /// [`Supervisor::complete`] with an adapter scale for this request (llama.cpp adapters only;
    /// `None` uses the adapter's own).
    pub async fn complete_scaled(
        &self,
        id: &str,
        prompt: &str,
        sampling: &Sampling,
        seed: Option<u64>,
        adapter_scale: Option<f32>,
    ) -> Result<ChunkStream, RemoteError> {
        let (engine, target, base, key, guard) = self.routed(id, adapter_scale)?;
        let body = openai::completion_body_for(prompt, sampling, seed, &target, engine);
        let resp = self.post(format!("{base}/v1/completions"), key.as_deref(), &body).await?;
        Ok(stream_chunks(resp, guard))
    }

    fn engine_of(&self, id: &str) -> Result<EngineKind, RemoteError> {
        if let Route { base, adapter: Some(_) } = self.route(id)? {
            return Err(bad_request(format!("{id} is an adapter on {base}; adapters serve chat and completions only")));
        }
        self.slots().get(id).map(|s| s.launch.engine).ok_or_else(|| unknown_model(id))
    }

    /// Embeds `input` with `id` (an embedding model). Returns the vectors in input order and the
    /// prompt tokens used.
    pub async fn embed(&self, id: &str, input: &[String]) -> Result<(Vec<Vec<f32>>, u64), RemoteError> {
        if self.engine_of(id)? == EngineKind::Mlx {
            return Err(bad_request(format!("{id} runs on MLX-LM, which serves no embeddings")));
        }
        let (base, key, _guard) = self.target(id)?;
        let resp =
            self.post(format!("{base}/v1/embeddings"), key.as_deref(), &json!({"model": id, "input": input, "encoding_format": "float"})).await?;
        let v: Value = resp.json().await.map_err(|e| engine_error(format!("bad embeddings response: {e}")))?;
        let mut rows: Vec<(u64, Vec<f32>)> = v["data"]
            .as_array()
            .ok_or_else(|| engine_error("embeddings response has no data"))?
            .iter()
            .enumerate()
            .map(|(i, d)| {
                (
                    d["index"].as_u64().unwrap_or(i as u64),
                    d["embedding"].as_array().map(|e| e.iter().filter_map(Value::as_f64).map(|x| x as f32).collect()).unwrap_or_default(),
                )
            })
            .collect();
        rows.sort_by_key(|(i, _)| *i);
        if rows.len() != input.len() {
            return Err(engine_error(format!("asked for {} embeddings, got {}", input.len(), rows.len())));
        }
        Ok((rows.into_iter().map(|(_, e)| e).collect(), v["usage"]["prompt_tokens"].as_u64().unwrap_or(0)))
    }

    fn log_file(&self, id: &str) -> Option<(std::fs::File, std::fs::File)> {
        let dir = self.inner.opts.log_dir.as_ref()?;
        std::fs::create_dir_all(dir).ok()?;
        let f = std::fs::OpenOptions::new().create(true).append(true).open(dir.join(format!("{id}.log"))).ok()?;
        Some((f.try_clone().ok()?, f))
    }

    fn log_tail(&self, id: &str) -> String {
        let Some(dir) = &self.inner.opts.log_dir else { return String::new() };
        let Ok(text) = std::fs::read_to_string(dir.join(format!("{id}.log"))) else { return String::new() };
        let lines: Vec<&str> = text.lines().rev().take(8).collect();
        let tail: Vec<&str> = lines.into_iter().rev().collect();
        format!("; log {}:\n{}", dir.join(format!("{id}.log")).display(), tail.join("\n"))
    }
}

/// Adapter scales must be finite and within ±[`MAX_ADAPTER_SCALE`].
fn check_scale(s: f32) -> Result<(), RemoteError> {
    if s.is_finite() && s.abs() <= MAX_ADAPTER_SCALE {
        Ok(())
    } else {
        Err(bad_request(format!("adapter scale {s} must be within ±{MAX_ADAPTER_SCALE}")))
    }
}

/// The largest adapter scale a request or [`Supervisor::set_adapter_scale`] may set.
pub const MAX_ADAPTER_SCALE: f32 = 16.0;

/// `base * 2^(n-1)`, at most `max`.
fn backoff(base: Duration, max: Duration, n: usize) -> Duration {
    let factor = 1u32.checked_shl(n.saturating_sub(1).min(16) as u32).unwrap_or(u32::MAX);
    base.saturating_mul(factor).min(max)
}

fn stream_chunks(resp: reqwest::Response, guard: InFlight) -> ChunkStream {
    let (tx, rx) = tokio::sync::mpsc::channel(64);
    tokio::spawn(async move {
        let _guard = guard;
        let mut bytes = resp.bytes_stream();
        let mut parser = SseParser::default();
        let mut decoder = ChunkDecoder::default();
        loop {
            let payloads = match bytes.next().await {
                Some(Ok(chunk)) => parser.feed(&chunk),
                Some(Err(e)) => {
                    let _ = tx.send(Err(engine_error(format!("stream broke: {e}")))).await;
                    return;
                }
                None => {
                    let rest: Vec<String> = parser.finish().into_iter().collect();
                    for p in rest {
                        if !send_all(&tx, decoder.decode(&p)).await {
                            return;
                        }
                    }
                    let last = if decoder.complete() { decoder.finish().map(Ok) } else { Some(Err(engine_error("engine stream ended early"))) };
                    if let Some(last) = last {
                        let _ = tx.send(last).await;
                    }
                    return;
                }
            };
            for p in payloads {
                let done = p.trim() == "[DONE]";
                if !send_all(&tx, decoder.decode(&p)).await || done {
                    return;
                }
            }
        }
    });
    Box::pin(ReceiverStream::new(rx))
}

/// Sends decoded frames; false when the stream should end (receiver gone, or an error was sent).
async fn send_all(tx: &tokio::sync::mpsc::Sender<Result<Chunk, RemoteError>>, evs: Result<Vec<Chunk>, RemoteError>) -> bool {
    match evs {
        Ok(evs) => {
            for ev in evs {
                if tx.send(Ok(ev)).await.is_err() {
                    return false;
                }
            }
            true
        }
        Err(e) => {
            let _ = tx.send(Err(e)).await;
            false
        }
    }
}

/// Counts a request in flight; on drop, records when the model was last used.
struct InFlight {
    n: Arc<AtomicUsize>,
    last_used: Arc<AtomicU64>,
    epoch: Instant,
}

impl InFlight {
    fn new(n: Arc<AtomicUsize>, last_used: Arc<AtomicU64>, epoch: Instant) -> Self {
        n.fetch_add(1, Ordering::SeqCst);
        Self { n, last_used, epoch }
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        self.last_used.store(self.epoch.elapsed().as_millis() as u64, Ordering::SeqCst);
        self.n.fetch_sub(1, Ordering::SeqCst);
    }
}

async fn terminate(child: &mut tokio::process::Child, pid: Option<u32>, grace: Duration) -> std::io::Result<std::process::ExitStatus> {
    #[cfg(unix)]
    if let Some(pid) = pid.and_then(|p| i32::try_from(p).ok()) {
        // SAFETY: plain syscall; the negative pid targets the engine's own process group.
        unsafe {
            libc::kill(-pid, libc::SIGTERM);
        }
        if let Ok(status) = tokio::time::timeout(grace, child.wait()).await {
            return status;
        }
        // SAFETY: as above.
        unsafe {
            libc::kill(-pid, libc::SIGKILL);
        }
        return child.wait().await;
    }
    let _ = (pid, grace);
    child.start_kill()?;
    child.wait().await
}

/// How many ports a launch tries when the one it was given turns out to be taken.
const PORT_TRIES: usize = 3;

/// How one launch attempt failed.
enum Start {
    /// Another process took the port it was given; worth another try on a new port.
    PortTaken(RemoteError),
    Failed(RemoteError),
}

/// `/v1/models` lists `id` (as a model id, or a llama-server alias).
fn lists_model(v: &Value, id: &str) -> bool {
    v["data"].as_array().is_some_and(|d| {
        d.iter().any(|m| m["id"].as_str() == Some(id) || m["aliases"].as_array().is_some_and(|a| a.iter().any(|x| x.as_str() == Some(id))))
    })
}

/// An engine's log says it could not bind its address (llama-server, uvicorn/vLLM, MLX-LM).
fn says_address_in_use(log: &str) -> bool {
    let l = log.to_ascii_lowercase();
    ["address already in use", "couldn't bind", "could not bind", "errno 98", "errno 48", "eaddrinuse", "os error 98", "os error 48", "error 10048"]
        .iter()
        .any(|p| l.contains(p))
}

fn free_port() -> std::io::Result<u16> {
    Ok(std::net::TcpListener::bind(("127.0.0.1", 0))?.local_addr()?.port())
}

/// Random hex from the OS RNG.
pub(crate) fn random_hex(bytes: usize) -> String {
    let mut buf = vec![0u8; bytes];
    getrandom::fill(&mut buf).expect("the OS RNG is available");
    buf.iter().map(|b| format!("{b:02x}")).collect()
}

fn trim(s: &str) -> String {
    let s = s.trim();
    if s.len() > 500 { format!("{}...", &s[..s.floor_char_boundary(500)]) } else { s.to_string() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_status_limits_propagate_and_old_rows_read() {
        use weirdnet_agent::remote::catalog::{CatalogModel, ModelState as CatalogState, Purpose};
        use weirdnet_agent::{MediaLimits, Modalities};
        // A row as a 0.4.7 node sends it: no limits anywhere.
        let old = r#"{"id":"g","engine":"llamacpp","state":{"state":"ready"},"in_flight":0,"swap_group":null,"embedding":false,"pid":null,
            "last_load_secs":null,"size_bytes":null,"wrapped":false,"crashes":0,"context":8192,"slots":null,
            "capabilities":{"chat":true,"embeddings":false,"vision":true,"audio":false},"base":null}"#;
        let row: ModelStatus = serde_json::from_str(old).unwrap();
        assert!(row.limits().is_unknown());
        let e = row.apply_to(CatalogModel::new("g", Purpose::Chat, CatalogState::Warm));
        assert_eq!((e.inputs, e.context), (Modalities::VISION, Some(8192)));
        assert!(e.limits.is_unknown());
        assert!(!serde_json::to_string(&row).unwrap().contains("limits"));

        let limits = MediaLimits::new().with_max_images(3).with_image_tokens(576).with_max_request_bytes(8 << 20);
        let launch = ModelLaunch::new("g", EngineKind::Llamacpp, "/m/g.gguf").with_mmproj("/p/mm.gguf").with_limits(limits);
        let mut row = row;
        row.capabilities = Some(launch.capabilities());
        let back: ModelStatus = serde_json::from_str(&serde_json::to_string(&row).unwrap()).unwrap();
        assert_eq!(back.limits(), limits);
        let e = back.apply_to(CatalogModel::new("g", Purpose::Chat, CatalogState::Warm));
        assert_eq!((e.limits, e.inputs), (limits, Modalities::new(true, true)));
        let none = ModelStatus { capabilities: None, ..back };
        assert!(none.limits().is_unknown());

        // The measured context per sequence wins over the configured one; old rows have none.
        assert_eq!(row.context_tokens, None);
        let measured = ModelStatus { context_tokens: Some(4096), ..none.clone() };
        assert_eq!(measured.apply_to(CatalogModel::new("g", Purpose::Chat, CatalogState::Warm)).context, Some(4096));
        let back: ModelStatus = serde_json::from_str(&serde_json::to_string(&measured).unwrap()).unwrap();
        assert_eq!(back.context_tokens, Some(4096));
        // Purpose: from the embedding flag and the capabilities (old rows: chat or embedding).
        assert_eq!(none.purpose(), Purpose::Chat);
        assert_eq!(ModelStatus { embedding: true, ..none.clone() }.purpose(), Purpose::Embedding);
        let t = ModelStatus { capabilities: Some(Capabilities::text().with_audio(true).with_transcribe(true)), ..none };
        assert_eq!(t.purpose(), Purpose::Transcribe);
        let e = t.catalog_model(CatalogState::Loadable);
        assert_eq!((e.id.as_str(), e.purpose, e.state, e.inputs), ("g", Purpose::Transcribe, CatalogState::Loadable, Modalities::new(false, true)));
    }

    #[test]
    fn the_engine_is_recognised_by_its_model_list() {
        assert!(lists_model(&json!({"data": [{"id": "qwen"}]}), "qwen"));
        assert!(lists_model(&json!({"data": [{"id": "x.gguf", "aliases": ["qwen"]}]}), "qwen"));
        assert!(!lists_model(&json!({"data": [{"id": "other"}]}), "qwen"));
        assert!(!lists_model(&json!({"status": "ok"}), "qwen"));
        assert!(says_address_in_use("srv  start: couldn't bind HTTP server socket, hostname: 127.0.0.1, port: 4100"));
        assert!(says_address_in_use("ERROR:    [Errno 98] error while attempting to bind on address ('127.0.0.1', 8000): address already in use"));
        assert!(!says_address_in_use("loading model\nCUDA error: out of memory"));
    }

    #[test]
    fn backoff_doubles_and_caps() {
        let b = |n| backoff(Duration::from_secs(1), Duration::from_secs(60), n);
        assert_eq!(b(1), Duration::from_secs(1));
        assert_eq!(b(2), Duration::from_secs(2));
        assert_eq!(b(4), Duration::from_secs(8));
        assert_eq!(b(10), Duration::from_secs(60));
        assert_eq!(b(100), Duration::from_secs(60));
    }
}
