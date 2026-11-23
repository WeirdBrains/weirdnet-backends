//! `/weirdnet-backends/models/1`: the owner manages a model node's model files over
//! `weirdnet-net`: search the Hugging Face Hub, see which files fit the node, pull one (split
//! GGUFs too) and serve it without a restart, list and remove what is on disk. LoRA adapters
//! too: [`Pull::adapter_for`] pulls one for a served base model and adds it to that base;
//! [`ModelsCommand::ServeAdapter`], [`ModelsCommand::UnserveAdapter`] and
//! [`ModelsCommand::AdapterScale`] add a stored one to a base, take one off and change its scale
//! while the base keeps running where the engine allows (see [`Supervisor::add_adapter_live`]).
//!
//! Only the owner may use it: [`ModelsService`] takes an [`Authorize`] and runs nothing before
//! it admits the caller. Pass `weirdnet_agent::serve::RosterAuth` for the owner's linked devices
//! (with `require_roles` to narrow it to devices holding a role the owner granted).
//!
//! The Hugging Face token never travels: the node uses its own ([`ModelStore::from_env`]), and
//! nothing in this protocol carries one.
//!
//! Each stream carries one [`ModelsRequest`] (CBOR, with the caller's certificate) and its
//! answers ([`ModelsReply`]): one answer, or for [`ModelsCommand::Pull`] a stream of
//! [`ModelsReply::Progress`] then [`ModelsReply::Pulled`]. Closing the stream cancels a pull;
//! the partial file is kept, so pulling again resumes it.

use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use weirdnet_agent::remote::{Authorize, Caller, Channel, Dialer, RemoteError, to_cbor};
use weirdnet_agent::serve::{channel, stream_limits};
use weirdnet_model::fit::{self, Fit, FitOptions};
use weirdnet_model::gguf::GgufInfo;
use weirdnet_model::hf::{AdapterFormat, RepoAdapter, RepoModel, Search, SearchHit};
use weirdnet_model::hw::Budget;
use weirdnet_model::{Consent, ModelStore, Progress, StoredAdapter, StoredModel, download_adapter, download_parts};
use weirdnet_net::node::{Message, Node};

use crate::engine::{EngineKind, LoraAdapter, ModelLaunch};
use crate::error::bad_request;
use crate::supervisor::{AdapterChange, ModelState, Supervisor};

/// The protocol id.
pub const MODELS_PROTOCOL: &str = "/weirdnet-backends/models/1";
/// Progress messages go out at most this often.
const PROGRESS_EVERY: Duration = Duration::from_millis(250);

/// What to pull.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Pull {
    /// The Hub repo, `owner/name`.
    pub repo: String,
    /// A model of the repo ([`RepoModel::id`], such as `Qwen3-8B-Q4_K_M`, or one of its file
    /// names). `None` picks the largest that fits the node ([`fit::choose`]).
    pub model: Option<String>,
    /// Context and concurrent requests, for the fit and for serving.
    pub context: Option<u32>,
    pub slots: Option<u32>,
    /// Add it to the supervisor once downloaded (as the model id, on the service's engine).
    pub serve: bool,
    /// Also load it (implies `serve`).
    pub load: bool,
    /// Pull a LoRA adapter of the repo (by [`Pull::model`], or its only one) for this served base
    /// model (a supervisor id) instead of a model. The adapter's format must suit the base's
    /// engine, and its architecture and base model must match the base's where both are known.
    /// With `serve` it is added to the base ([`Supervisor::add_adapter`]).
    #[serde(default)]
    pub adapter_for: Option<String>,
}

impl Pull {
    pub fn new(repo: impl Into<String>) -> Self {
        Self { repo: repo.into(), ..Default::default() }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn with_context(mut self, context: u32) -> Self {
        self.context = Some(context);
        self
    }

    pub fn with_slots(mut self, slots: u32) -> Self {
        self.slots = Some(slots);
        self
    }

    /// Pull an adapter for served base model `base` (see [`Pull::adapter_for`]).
    pub fn with_adapter_for(mut self, base: impl Into<String>) -> Self {
        self.adapter_for = Some(base.into());
        self
    }

    /// Serve it after the download, and load it too when `load`.
    pub fn serving(mut self, load: bool) -> Self {
        self.serve = true;
        self.load = load;
        self
    }
}

/// A models/1 request. Nodes before 0.4.1 refuse the adapter commands.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ModelsCommand {
    /// Hub repos (GGUF only), most downloaded first.
    Search {
        query: String,
        limit: u32,
    },
    /// A repo's models measured against the node's budget, largest first.
    Files {
        repo: String,
        context: Option<u32>,
        slots: Option<u32>,
    },
    Pull(Pull),
    /// The models on the node's disk.
    List,
    /// Stop serving (if served) and delete a stored model by id.
    Remove {
        id: String,
    },
    /// Serve stored adapter `id` ([`StoredAdapter::id`]) on served base model `base`, under the
    /// same id, live where the engine allows ([`Supervisor::add_adapter_live`]). `scale` as in
    /// [`LoraAdapter::scale`].
    ServeAdapter {
        id: String,
        base: String,
        #[serde(default)]
        scale: Option<f32>,
    },
    /// Stop serving adapter `id` (its served id), keeping its files
    /// ([`Supervisor::remove_adapter_live`]).
    UnserveAdapter {
        id: String,
    },
    /// Change served adapter `id`'s scale ([`Supervisor::set_adapter_scale`]).
    AdapterScale {
        id: String,
        scale: Option<f32>,
    },
}

// The service refuses scales that aren't finite, so equality is total for the commands it runs.
impl Eq for ModelsCommand {}

/// What a client sends: its certificate and one command.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelsRequest {
    /// The caller's CBOR `NodeCert`, for the authorize hook.
    #[serde(default, with = "serde_bytes")]
    pub cert: Option<Vec<u8>>,
    pub command: ModelsCommand,
}

/// A repo's models against this node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RepoFits {
    pub repo: String,
    /// The header of the repo's smallest model, when it could be read (the KV figures then come
    /// from it, else they are estimates).
    pub info: Option<GgufInfo>,
    /// The node's budget, when it has one (without one nothing `fits`).
    pub budget: Option<Budget>,
    pub context: u32,
    pub slots: u32,
    /// Largest first.
    pub fits: Vec<Fit>,
    /// The largest that fits.
    pub best: Option<String>,
    /// The repo's LoRA adapters (never in `fits`).
    #[serde(default)]
    pub adapters: Vec<RepoAdapter>,
}

/// A finished pull.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Pulled {
    pub id: String,
    pub repo: String,
    /// Paths relative to the store, first part first.
    pub files: Vec<String>,
    pub size: u64,
    /// Added to the supervisor (now or before).
    pub served: bool,
    pub loaded: bool,
    /// For an adapter: the base model it was pulled for.
    #[serde(default)]
    pub adapter_for: Option<String>,
    /// For an adapter added to its base now: when it serves.
    #[serde(default)]
    pub change: Option<AdapterChange>,
}

/// A stored model or LoRA adapter and, when the supervisor serves it, its state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct StoredEntry {
    /// A model on disk (`None` for an adapter).
    #[serde(default)]
    pub model: Option<StoredModel>,
    /// An adapter on disk (`None` for a model).
    #[serde(default)]
    pub adapter: Option<StoredAdapter>,
    /// The supervisor's id for it, when served.
    pub served_as: Option<String>,
    pub state: Option<ModelState>,
    /// For a served adapter: the base model it runs on.
    #[serde(default)]
    pub base: Option<String>,
}

impl StoredEntry {
    /// The stored model's or adapter's id.
    pub fn id(&self) -> &str {
        match (&self.model, &self.adapter) {
            (Some(m), _) => &m.id,
            (_, Some(a)) => &a.id,
            _ => "",
        }
    }

    /// Bytes on disk.
    pub fn size(&self) -> u64 {
        self.model.as_ref().map(|m| m.size).or(self.adapter.as_ref().map(|a| a.size)).unwrap_or(0)
    }
}

/// What a models/1 node answers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ModelsReply {
    Hits(Vec<SearchHit>),
    Fits(Box<RepoFits>),
    Progress {
        done: u64,
        total: u64,
    },
    Pulled(Pulled),
    List(Vec<StoredEntry>),
    Removed {
        id: String,
        freed: u64,
    },
    /// [`ModelsCommand::ServeAdapter`] done: when it serves.
    AdapterServed {
        id: String,
        base: String,
        change: AdapterChange,
    },
    /// [`ModelsCommand::UnserveAdapter`] done.
    AdapterUnserved {
        id: String,
        base: String,
    },
    /// [`ModelsCommand::AdapterScale`] done.
    AdapterScaled {
        id: String,
        scale: Option<f32>,
    },
    Failed(RemoteError),
}

fn model_error(e: weirdnet_model::Error) -> RemoteError {
    match e {
        weirdnet_model::Error::BadRepo(_) | weirdnet_model::Error::Consent => bad_request(e.to_string()),
        weirdnet_model::Error::Http(_) => RemoteError::unavailable(e.to_string()),
        e => RemoteError::Failed { message: e.to_string() },
    }
}

async fn blocking<T: Send + 'static>(f: impl FnOnce() -> weirdnet_model::Result<T> + Send + 'static) -> Result<T, RemoteError> {
    tokio::task::spawn_blocking(f).await.map_err(|e| RemoteError::Failed { message: format!("worker: {e}") })?.map_err(model_error)
}

/// A model node's models/1 side. Transport-neutral ([`ModelsService::serve`] takes a channel);
/// [`ModelsServer`] runs it on a `weirdnet-net` node. Its methods also work in-process (a local
/// CLI on the node itself).
pub struct ModelsService {
    store: ModelStore,
    supervisor: Supervisor,
    auth: Arc<dyn Authorize>,
    budget: Option<Budget>,
    engine: EngineKind,
    pulling: Mutex<HashSet<String>>,
}

impl ModelsService {
    /// `store` is where files go (and its Hub and token); new models are served by `supervisor`;
    /// `auth` admits the owner only.
    pub fn new(store: ModelStore, supervisor: Supervisor, auth: Arc<dyn Authorize>) -> Self {
        Self { store, supervisor, auth, budget: None, engine: EngineKind::Llamacpp, pulling: Mutex::default() }
    }

    /// The memory budget fits are measured against (`weirdnet_model::hw::budget`).
    pub fn with_budget(mut self, budget: Budget) -> Self {
        self.budget = Some(budget);
        self
    }

    /// The engine pulled models are served on (default llama.cpp, the GGUF engine).
    pub fn with_engine(mut self, engine: EngineKind) -> Self {
        self.engine = engine;
        self
    }

    pub fn store(&self) -> &ModelStore {
        &self.store
    }

    /// Hub repos matching `query` (at most `limit`, 100 at most).
    pub async fn search(&self, query: &str, limit: u32) -> Result<Vec<SearchHit>, RemoteError> {
        let hub = self.store.hub_client();
        let search = Search::new(query).with_limit(limit.clamp(1, 100) as usize);
        blocking(move || hub.search(&search)).await
    }

    async fn repo(&self, repo: &str) -> Result<(Vec<RepoModel>, Option<GgufInfo>), RemoteError> {
        let hub = self.store.hub_client();
        let repo = repo.to_string();
        blocking(move || {
            let models = hub.models(&repo)?;
            // Every quantization shares the architecture; read the smallest one's header.
            let info = models.iter().find(|m| m.complete).and_then(|m| hub.gguf_info(&repo, &m.first().name).ok());
            Ok((models, info))
        })
        .await
    }

    fn options(context: Option<u32>, slots: Option<u32>, info: Option<&GgufInfo>) -> FitOptions {
        let mut opts = FitOptions::default();
        if let Some(c) = context {
            opts = opts.with_context(c);
        }
        if let Some(s) = slots {
            opts = opts.with_slots(s);
        }
        if let Some(i) = info {
            opts = opts.with_info(i);
        }
        opts
    }

    /// `repo`'s models measured against the node's budget.
    pub async fn files(&self, repo: &str, context: Option<u32>, slots: Option<u32>) -> Result<RepoFits, RemoteError> {
        let (models, info) = self.repo(repo).await?;
        let opts = Self::options(context, slots, info.as_ref());
        let budget = self.budget.unwrap_or(Budget { pool_bytes: 0, reserve_bytes: 0, usable_bytes: 0 });
        let fits = fit::rank(&models, &budget, &opts);
        let best = fits.iter().find(|f| f.fits).map(|f| f.model.id.clone());
        let adapters = self.adapters(repo).await.unwrap_or_default();
        Ok(RepoFits { repo: repo.into(), info, budget: self.budget, context: opts.context, slots: opts.slots, fits, best, adapters })
    }

    async fn adapters(&self, repo: &str) -> Result<Vec<RepoAdapter>, RemoteError> {
        let hub = self.store.hub_client();
        let repo = repo.to_string();
        blocking(move || hub.adapters(&repo)).await
    }

    /// Whether `adapter` suits served model `base`: the format for its engine (GGUF LoRA on
    /// llama.cpp, PEFT on vLLM), and the architecture and base model against the base GGUF's
    /// header where both sides are known.
    fn check_adapter_for(&self, base: &str, adapter: &RepoAdapter) -> Result<(), RemoteError> {
        let launch =
            self.supervisor.launches().into_iter().find(|l| l.id == base).ok_or_else(|| bad_request(format!("{base} is not served on this node")))?;
        match (launch.engine, adapter.format) {
            (EngineKind::Llamacpp, AdapterFormat::GgufLora) | (EngineKind::Vllm, AdapterFormat::Peft) => {}
            (engine, format) => return Err(bad_request(format!("{} is {format:?}, which {} can't serve", adapter.id, engine.name()))),
        }
        if launch.engine != EngineKind::Llamacpp {
            return Ok(());
        }
        let header = std::fs::File::open(&launch.source)
            .ok()
            .and_then(|f| weirdnet_model::gguf::read_header(std::io::BufReader::new(f)).ok())
            .ok_or_else(|| bad_request(format!("{base}: can't read the header of {}", launch.source)))?;
        if let Some(arch) = &adapter.architecture
            && *arch != header.architecture
        {
            return Err(bad_request(format!("{} is for {arch} models; {base} is {}", adapter.id, header.architecture)));
        }
        let known: Vec<&String> = header.base_model.iter().chain(header.repo.iter()).collect();
        if let Some(want) = &adapter.base_model
            && !known.is_empty()
            && !known.iter().any(|k| k.eq_ignore_ascii_case(want))
        {
            return Err(bad_request(format!("{} is for {want}; {base} comes from {}", adapter.id, known[0])));
        }
        Ok(())
    }

    async fn pull_adapter(&self, pull: &Pull, base: &str, progress: impl FnMut(Progress) -> bool + Send + 'static) -> Result<Pulled, RemoteError> {
        if !self.supervisor.launches().iter().any(|l| l.id == base) {
            return Err(bad_request(format!("{base} is not served on this node")));
        }
        let adapters = self.adapters(&pull.repo).await?;
        let adapter = match &pull.model {
            Some(want) => {
                adapters.iter().find(|a| a.id == *want || a.files.iter().any(|f| f.name == *want || f.name.rsplit('/').next() == Some(want.as_str())))
            }
            None if adapters.len() == 1 => adapters.first(),
            None if adapters.is_empty() => None,
            None => {
                let ids: Vec<&str> = adapters.iter().map(|a| a.id.as_str()).collect();
                return Err(bad_request(format!("{} has adapters {}; name one", pull.repo, ids.join(", "))));
            }
        }
        .cloned()
        .ok_or_else(|| bad_request(format!("{} has no adapter {}", pull.repo, pull.model.as_deref().unwrap_or(""))))?;
        self.check_adapter_for(base, &adapter)?;
        let key = format!("{}/{}", pull.repo, adapter.id);
        if !self.pulling.lock().unwrap_or_else(|e| e.into_inner()).insert(key.clone()) {
            return Err(RemoteError::Busy);
        }
        let (store, repo, a) = (self.store.clone(), pull.repo.clone(), adapter.clone());
        let mut progress = progress;
        let done = blocking(move || download_adapter(&store, &repo, &a, &Consent::given_adapter(&a), &mut progress)).await;
        self.pulling.lock().unwrap_or_else(|e| e.into_inner()).remove(&key);
        let path = done?;
        let dir = format!("adapters/{}", adapter.id);
        let files = adapter.files.iter().map(|f| format!("{dir}/{}", f.name.rsplit('/').next().unwrap_or(&f.name))).collect();
        let (mut served, mut loaded, mut change) = (false, false, None);
        if pull.serve || pull.load {
            served = true;
            let path = path.display().to_string();
            let already = self.supervisor.launches().iter().any(|l| l.adapters.iter().any(|x| x.path == path));
            if !already {
                change = Some(self.supervisor.add_adapter_live(base, LoraAdapter::new(adapter.id.clone(), path)).await?);
            }
            if pull.load {
                self.supervisor.load(&adapter.id).await?;
                loaded = true;
            }
        }
        Ok(Pulled { id: adapter.id, repo: pull.repo.clone(), files, size: adapter.size, served, loaded, adapter_for: Some(base.into()), change })
    }

    /// Downloads `pull` (resuming a partial file) and serves it if asked. `progress` returns
    /// `false` to cancel. One pull per model at a time.
    pub async fn pull(&self, pull: &Pull, progress: impl FnMut(Progress) -> bool + Send + 'static) -> Result<Pulled, RemoteError> {
        if let Some(base) = &pull.adapter_for {
            return self.pull_adapter(pull, base, progress).await;
        }
        let (models, info) = self.repo(&pull.repo).await?;
        let model = match &pull.model {
            Some(want) => models
                .iter()
                .find(|m| m.id == *want || m.parts.iter().any(|p| p.name == *want || p.name.rsplit('/').next() == Some(want.as_str())))
                .cloned()
                .ok_or_else(|| bad_request(format!("{} has no model {want}", pull.repo)))?,
            None => {
                let budget = self.budget.ok_or_else(|| bad_request("this node has no memory budget; name the model to pull"))?;
                fit::choose(&models, &budget, &Self::options(pull.context, pull.slots, info.as_ref()))
                    .map(|f| f.model)
                    .ok_or_else(|| bad_request(format!("nothing in {} fits this node", pull.repo)))?
            }
        };
        if !model.complete {
            return Err(bad_request(format!("{} is missing parts in {}", model.id, pull.repo)));
        }
        let specs = model.specs(&pull.repo).map_err(model_error)?;
        let key = format!("{}/{}", pull.repo, model.id);
        if !self.pulling.lock().unwrap_or_else(|e| e.into_inner()).insert(key.clone()) {
            return Err(RemoteError::Busy);
        }
        let store = self.store.clone();
        let parts = specs.clone();
        let mut progress = progress;
        let done = blocking(move || download_parts(&store, &parts, &Consent::given(&parts[0]), &mut progress)).await;
        self.pulling.lock().unwrap_or_else(|e| e.into_inner()).remove(&key);
        let path = done?;
        let files: Vec<String> = specs.iter().map(|s| s.file.clone()).collect();
        let (mut served, mut loaded) = (false, false);
        if pull.serve || pull.load {
            served = true;
            if self.served_as(&path).is_none() {
                let mut launch = ModelLaunch::new(model.id.clone(), self.engine, path.display().to_string());
                launch.context = pull.context;
                launch.slots = pull.slots;
                self.supervisor.add_model(launch)?;
            }
            if pull.load {
                let id = self.served_as(&path).unwrap_or_else(|| model.id.clone());
                self.supervisor.load(&id).await?;
                loaded = true;
            }
        }
        Ok(Pulled { id: model.id, repo: pull.repo.clone(), files, size: model.size, served, loaded, adapter_for: None, change: None })
    }

    /// The supervisor id serving the file at `path`, if any.
    fn served_as(&self, path: &Path) -> Option<String> {
        let path = path.display().to_string();
        self.supervisor.launches().into_iter().find(|l| l.wrap.is_none() && l.source == path).map(|l| l.id)
    }

    /// The models and then the adapters on disk, with the supervisor's state for the served ones.
    pub fn list(&self) -> Vec<StoredEntry> {
        let models = self.store.list().into_iter().map(|model| {
            let served_as = self.served_as(&self.store.dir().join(&model.files[0]));
            let state = served_as.as_deref().and_then(|id| self.supervisor.state(id));
            StoredEntry { model: Some(model), adapter: None, served_as, state, base: None }
        });
        let launches = self.supervisor.launches();
        let adapters = self.store.adapters().into_iter().map(|a| {
            let path = a.path.display().to_string();
            let served = launches.iter().find_map(|l| l.adapters.iter().find(|x| x.path == path).map(|x| (x.id.clone(), l.id.clone())));
            let (served_as, base) = served.unzip();
            let state = served_as.as_deref().and_then(|id| self.supervisor.state(id));
            StoredEntry { model: None, adapter: Some(a), served_as, state, base }
        });
        models.chain(adapters).collect()
    }

    /// Stops serving `id` (a stored model's id, or the supervisor id serving it) and deletes its
    /// files. Returns the bytes freed.
    pub async fn remove(&self, id: &str) -> Result<u64, RemoteError> {
        let entries = self.list();
        let matches: Vec<&StoredEntry> = entries.iter().filter(|e| e.id() == id || e.served_as.as_deref() == Some(id)).collect();
        let entry = match matches.as_slice() {
            [one] => *one,
            [] => return Err(bad_request(format!("no stored model {id}"))),
            _ => return Err(bad_request(format!("{id} names {} stored models; remove by the served id", matches.len()))),
        };
        if self.pulling.lock().unwrap_or_else(|e| e.into_inner()).iter().any(|k| k.rsplit('/').next() == Some(entry.id())) {
            return Err(RemoteError::Busy);
        }
        let store = self.store.clone();
        if let Some(adapter) = entry.adapter.clone() {
            if let (Some(served), Some(base)) = (&entry.served_as, &entry.base) {
                self.supervisor.remove_adapter_live(base, served).await?;
            }
            return blocking(move || store.remove_adapter(&adapter)).await;
        }
        let model = entry.model.clone().ok_or_else(|| bad_request(format!("no stored model {id}")))?;
        if let Some(served) = &entry.served_as {
            self.supervisor.remove_model(served).await?;
        }
        blocking(move || store.remove_local(&model)).await
    }

    /// Serves stored adapter `id` on served base model `base` (see [`ModelsCommand::ServeAdapter`]).
    pub async fn serve_adapter(&self, id: &str, base: &str, scale: Option<f32>) -> Result<AdapterChange, RemoteError> {
        let stored = self.store.adapters().into_iter().find(|a| a.id == id).ok_or_else(|| bad_request(format!("no stored adapter {id}")))?;
        let path = stored.path.display().to_string();
        if let Some(l) = self.supervisor.launches().iter().find(|l| l.adapters.iter().any(|x| x.path == path)) {
            return Err(bad_request(format!("{id} is already served on {}", l.id)));
        }
        let mut adapter = LoraAdapter::new(id, path);
        adapter.scale = scale;
        self.supervisor.add_adapter_live(base, adapter).await
    }

    /// Stops serving adapter `id` and keeps its files. Returns the base it was on.
    pub async fn unserve_adapter(&self, id: &str) -> Result<String, RemoteError> {
        let base = self
            .supervisor
            .launches()
            .into_iter()
            .find(|l| l.adapters.iter().any(|a| a.id == id))
            .map(|l| l.id)
            .ok_or_else(|| bad_request(format!("no served adapter {id}")))?;
        self.supervisor.remove_adapter_live(&base, id).await?;
        Ok(base)
    }

    /// Changes served adapter `id`'s scale (see [`ModelsCommand::AdapterScale`]).
    pub fn set_adapter_scale(&self, id: &str, scale: Option<f32>) -> Result<(), RemoteError> {
        self.supervisor.set_adapter_scale(id, scale)
    }

    async fn run(&self, command: ModelsCommand) -> Result<ModelsReply, RemoteError> {
        Ok(match command {
            ModelsCommand::Search { query, limit } => ModelsReply::Hits(self.search(&query, limit).await?),
            ModelsCommand::Files { repo, context, slots } => ModelsReply::Fits(Box::new(self.files(&repo, context, slots).await?)),
            ModelsCommand::List => ModelsReply::List(self.list()),
            ModelsCommand::Remove { id } => {
                let freed = self.remove(&id).await?;
                ModelsReply::Removed { id, freed }
            }
            ModelsCommand::ServeAdapter { id, base, scale } => {
                let change = self.serve_adapter(&id, &base, scale).await?;
                ModelsReply::AdapterServed { id, base, change }
            }
            ModelsCommand::UnserveAdapter { id } => {
                let base = self.unserve_adapter(&id).await?;
                ModelsReply::AdapterUnserved { id, base }
            }
            ModelsCommand::AdapterScale { id, scale } => {
                self.set_adapter_scale(&id, scale)?;
                ModelsReply::AdapterScaled { id, scale }
            }
            ModelsCommand::Pull(_) => unreachable!("handled by serve"),
        })
    }

    /// Serves one models/1 stream. `caller` is the transport-authenticated peer.
    pub async fn serve(&self, mut ch: Channel, caller: Caller) -> Result<(), RemoteError> {
        let Some(msg) = ch.recv_cbor::<ModelsRequest>().await? else { return Ok(()) };
        let caller = match (caller.cert_bytes.is_some(), msg.cert) {
            (false, Some(cert)) => caller.with_cert(cert),
            (_, _) => caller,
        };
        if let Err(e) = self.auth.authorize(&caller) {
            return ch.send_cbor(&ModelsReply::Failed(e)).await;
        }
        let pull = match msg.command {
            ModelsCommand::Pull(p) => p,
            other => {
                let answer = self.run(other).await.unwrap_or_else(ModelsReply::Failed);
                return ch.send_cbor(&answer).await;
            }
        };
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Progress>();
        let stop = cancel.clone();
        let mut last: Option<Instant> = None;
        let report = move |p: Progress| {
            let due = p.done == p.total || last.is_none_or(|t| t.elapsed() >= PROGRESS_EVERY);
            if due {
                last = Some(Instant::now());
                let _ = tx.send(p);
            }
            !stop.load(Ordering::SeqCst)
        };
        let Channel { tx: out, rx: inbound } = &mut ch;
        let work = self.pull(&pull, report);
        tokio::pin!(work);
        let mut closed = false;
        let result = loop {
            tokio::select! {
                r = &mut work => break r,
                Some(p) = rx.recv() => {
                    if !closed {
                        out.send(to_cbor(&ModelsReply::Progress { done: p.done, total: p.total })?).await.ok();
                    }
                }
                r = inbound.recv(), if !closed => {
                    if matches!(r, Ok(None) | Err(_)) {
                        closed = true;
                        cancel.store(true, Ordering::SeqCst);
                    }
                }
            }
        };
        if closed {
            return Ok(());
        }
        while let Ok(p) = rx.try_recv() {
            out.send(to_cbor(&ModelsReply::Progress { done: p.done, total: p.total })?).await?;
        }
        let answer = result.map(ModelsReply::Pulled).unwrap_or_else(ModelsReply::Failed);
        out.send(to_cbor(&answer)?).await
    }
}

/// Serves a [`ModelsService`] on a `weirdnet-net` node. Dropping it stops serving.
pub struct ModelsServer {
    task: tokio::task::JoinHandle<()>,
}

impl ModelsServer {
    /// Starts accepting [`MODELS_PROTOCOL`] streams. Call from a Tokio runtime.
    pub fn start<Q: Message, R: Message>(node: &Arc<Node<Q, R>>, service: Arc<ModelsService>) -> Result<ModelsServer, RemoteError> {
        let mut incoming = node.accept_streams(MODELS_PROTOCOL, stream_limits()).map_err(|e| RemoteError::Failed { message: e.to_string() })?;
        let task = tokio::spawn(async move {
            while let Some(stream) = futures::StreamExt::next(&mut incoming).await {
                let caller = Caller::new(stream.peer().to_string());
                let svc = service.clone();
                tokio::spawn(async move {
                    let _ = svc.serve(channel(stream), caller).await;
                });
            }
        });
        Ok(ModelsServer { task })
    }
}

impl Drop for ModelsServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// The owner's side of models/1, over a [`Dialer`] (`weirdnet_agent::pool::NetDialer` on
/// Weirdnet).
pub struct ModelsClient {
    dialer: Arc<dyn Dialer>,
    cert: Option<Vec<u8>>,
}

impl ModelsClient {
    pub fn new(dialer: Arc<dyn Dialer>) -> Self {
        Self { dialer, cert: None }
    }

    /// This device's CBOR `NodeCert` (`weirdnet_agent::pool::encode_cert`), presented to the node.
    pub fn with_cert(mut self, cert: Option<Vec<u8>>) -> Self {
        self.cert = cert;
        self
    }

    async fn open(&self, node: &str, command: ModelsCommand) -> Result<Channel, RemoteError> {
        let mut ch = self.dialer.open(node, MODELS_PROTOCOL).await?;
        ch.send_cbor(&ModelsRequest { cert: self.cert.clone(), command }).await?;
        Ok(ch)
    }

    async fn call(&self, node: &str, command: ModelsCommand) -> Result<ModelsReply, RemoteError> {
        let mut ch = self.open(node, command).await?;
        match ch.recv_cbor::<ModelsReply>().await? {
            Some(ModelsReply::Failed(e)) => Err(e),
            Some(m) => Ok(m),
            None => Err(RemoteError::Closed),
        }
    }

    fn unexpected(what: &str, got: ModelsReply) -> RemoteError {
        RemoteError::protocol(format!("expected {what}, got {got:?}"))
    }

    pub async fn search(&self, node: &str, query: &str, limit: u32) -> Result<Vec<SearchHit>, RemoteError> {
        match self.call(node, ModelsCommand::Search { query: query.into(), limit }).await? {
            ModelsReply::Hits(h) => Ok(h),
            other => Err(Self::unexpected("hits", other)),
        }
    }

    pub async fn files(&self, node: &str, repo: &str, context: Option<u32>, slots: Option<u32>) -> Result<RepoFits, RemoteError> {
        match self.call(node, ModelsCommand::Files { repo: repo.into(), context, slots }).await? {
            ModelsReply::Fits(f) => Ok(*f),
            other => Err(Self::unexpected("fits", other)),
        }
    }

    /// Pulls on `node`, calling `progress` as it goes. Dropping the future cancels the pull
    /// (the node keeps the partial file for a later resume).
    pub async fn pull(&self, node: &str, pull: Pull, mut progress: impl FnMut(Progress) + Send) -> Result<Pulled, RemoteError> {
        let mut ch = self.open(node, ModelsCommand::Pull(pull)).await?;
        loop {
            match ch.recv_cbor::<ModelsReply>().await? {
                Some(ModelsReply::Progress { done, total }) => progress(Progress { done, total }),
                Some(ModelsReply::Pulled(p)) => return Ok(p),
                Some(ModelsReply::Failed(e)) => return Err(e),
                Some(other) => return Err(Self::unexpected("progress", other)),
                None => return Err(RemoteError::Closed),
            }
        }
    }

    pub async fn list(&self, node: &str) -> Result<Vec<StoredEntry>, RemoteError> {
        match self.call(node, ModelsCommand::List).await? {
            ModelsReply::List(l) => Ok(l),
            other => Err(Self::unexpected("a list", other)),
        }
    }

    /// Serves stored adapter `id` on `base` on `node`; returns when it serves.
    pub async fn serve_adapter(&self, node: &str, id: &str, base: &str, scale: Option<f32>) -> Result<AdapterChange, RemoteError> {
        match self.call(node, ModelsCommand::ServeAdapter { id: id.into(), base: base.into(), scale }).await? {
            ModelsReply::AdapterServed { change, .. } => Ok(change),
            other => Err(Self::unexpected("adapter served", other)),
        }
    }

    /// Stops serving adapter `id` on `node`, keeping its files; returns the base it was on.
    pub async fn unserve_adapter(&self, node: &str, id: &str) -> Result<String, RemoteError> {
        match self.call(node, ModelsCommand::UnserveAdapter { id: id.into() }).await? {
            ModelsReply::AdapterUnserved { base, .. } => Ok(base),
            other => Err(Self::unexpected("adapter unserved", other)),
        }
    }

    /// Changes served adapter `id`'s scale on `node` (`None`: 1.0).
    pub async fn set_adapter_scale(&self, node: &str, id: &str, scale: Option<f32>) -> Result<(), RemoteError> {
        match self.call(node, ModelsCommand::AdapterScale { id: id.into(), scale }).await? {
            ModelsReply::AdapterScaled { .. } => Ok(()),
            other => Err(Self::unexpected("adapter scaled", other)),
        }
    }

    /// Removes a stored model; returns the bytes freed.
    pub async fn remove(&self, node: &str, id: &str) -> Result<u64, RemoteError> {
        match self.call(node, ModelsCommand::Remove { id: id.into() }).await? {
            ModelsReply::Removed { freed, .. } => Ok(freed),
            other => Err(Self::unexpected("removed", other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, BufReader, Write};

    use weirdnet_agent::remote::MemoryDialer;

    use super::*;
    use crate::supervisor::SupervisorOptions;

    /// A GGUF header (llama, one block, one KV head of size 1) padded to `len` bytes.
    fn gguf(len: usize) -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend(3u32.to_le_bytes());
        out.extend(0u64.to_le_bytes());
        let kv: [(&str, Option<u32>); 5] = [
            ("general.architecture", None),
            ("llama.block_count", Some(1)),
            ("llama.attention.head_count", Some(1)),
            ("llama.attention.head_count_kv", Some(1)),
            ("llama.attention.key_length", Some(1)),
        ];
        out.extend((kv.len() as u64).to_le_bytes());
        for (k, v) in kv {
            out.extend((k.len() as u64).to_le_bytes());
            out.extend(k.as_bytes());
            match v {
                None => {
                    out.extend(8u32.to_le_bytes());
                    out.extend(5u64.to_le_bytes());
                    out.extend(b"llama");
                }
                Some(n) => {
                    out.extend(4u32.to_le_bytes());
                    out.extend(n.to_le_bytes());
                }
            }
        }
        out.resize(len, 0);
        out
    }

    /// A LoRA adapter GGUF header for `arch`, padded to `len` bytes.
    fn lora(arch: &str, len: usize) -> Vec<u8> {
        let mut out = crate::engine::tests::lora_gguf(arch);
        out.resize(len, 0);
        out
    }

    /// The Hub: a search, repo `o/n` with a single Q4_0 file and a two-part Q8_0, and the files.
    fn hub() -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let info = r#"{"id":"o/n","siblings":[
            {"rfilename":"m-Q4_0.gguf","size":4000},
            {"rfilename":"Q8_0/m-Q8_0-00001-of-00002.gguf","size":3000},
            {"rfilename":"Q8_0/m-Q8_0-00002-of-00002.gguf","size":3000}]}"#;
        let search = r#"[{"id":"o/n","downloads":9,"likes":1}]"#;
        let tune = r#"{"id":"o/tune","tags":["gguf","base_model:adapter:Org/Base"],"siblings":[
            {"rfilename":"tune-LoRA-F16.gguf","size":600},{"rfilename":"qwen-LoRA-F16.gguf","size":600}]}"#;
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut first = String::new();
                reader.read_line(&mut first).unwrap();
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line.trim().is_empty() {
                        break;
                    }
                }
                let path = first.split_whitespace().nth(1).unwrap_or("/").split('?').next().unwrap_or("").to_string();
                let body: Option<Vec<u8>> = match path.as_str() {
                    "/api/models" => Some(search.into()),
                    "/api/models/o/n" => Some(info.into()),
                    "/o/n/resolve/main/m-Q4_0.gguf" => Some(gguf(4000)),
                    "/o/n/resolve/main/Q8_0/m-Q8_0-00001-of-00002.gguf" => Some(gguf(3000)),
                    "/o/n/resolve/main/Q8_0/m-Q8_0-00002-of-00002.gguf" => Some(vec![7; 3000]),
                    "/api/models/o/tune" => Some(tune.into()),
                    "/o/tune/resolve/main/tune-LoRA-F16.gguf" => Some(lora("llama", 600)),
                    "/o/tune/resolve/main/qwen-LoRA-F16.gguf" => Some(lora("qwen3", 600)),
                    _ => None,
                };
                let _ = match body {
                    Some(b) => {
                        let _ = stream.write_all(format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", b.len()).as_bytes());
                        stream.write_all(&b)
                    }
                    None => stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"),
                };
            }
        });
        url
    }

    fn dialer(svc: Arc<ModelsService>, peer: &'static str) -> Arc<dyn Dialer> {
        let d = MemoryDialer::new();
        d.route_only("model-node", &[MODELS_PROTOCOL], move |ch, _| {
            let svc = svc.clone();
            tokio::spawn(async move {
                let _ = svc.serve(ch, Caller::new(peer)).await;
            });
        });
        Arc::new(d)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn the_owner_searches_pulls_serves_lists_and_removes() {
        let tmp = tempfile::tempdir().unwrap();
        let store = ModelStore::new(tmp.path()).with_hub(hub());
        let sup = Supervisor::new(vec![], vec![], SupervisorOptions::default());
        let auth: Arc<dyn Authorize> = Arc::new(|c: &Caller| if c.peer == "owner" { Ok(()) } else { Err(RemoteError::refused("owner only")) });
        // Q4_0 (4000 bytes plus 32 KiB of KV, plus 10%) fits; the 6000-byte Q8_0 doesn't.
        let budget = Budget { pool_bytes: 41_000, reserve_bytes: 0, usable_bytes: 41_000 };
        let svc = Arc::new(ModelsService::new(store, sup.clone(), auth).with_budget(budget));
        let owner = ModelsClient::new(dialer(svc.clone(), "owner"));
        let stranger = ModelsClient::new(dialer(svc.clone(), "stranger"));

        assert!(matches!(stranger.list("model-node").await, Err(RemoteError::Refused { .. })));
        assert!(matches!(stranger.pull("model-node", Pull::new("o/n"), |_| {}).await, Err(RemoteError::Refused { .. })));
        assert!(!tmp.path().join("m-Q4_0.gguf").exists(), "nothing ran for the stranger");

        let hits = owner.search("model-node", "m", 5).await.unwrap();
        assert_eq!(hits.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(), ["o/n"]);

        let fits = owner.files("model-node", "o/n", None, None).await.unwrap();
        assert_eq!(fits.info.as_ref().map(|i| i.kv_bytes_per_token()), Some(4));
        assert_eq!(fits.best.as_deref(), Some("m-Q4_0"));
        assert_eq!(fits.fits.iter().map(|f| (f.model.id.as_str(), f.fits)).collect::<Vec<_>>(), [("m-Q8_0", false), ("m-Q4_0", true)]);

        let mut seen = Vec::new();
        let pulled = owner.pull("model-node", Pull::new("o/n").serving(false), |p| seen.push(p)).await.unwrap();
        assert_eq!((pulled.id.as_str(), pulled.size, pulled.served, pulled.loaded), ("m-Q4_0", 4000, true, false));
        assert_eq!(seen.last().map(|p| (p.done, p.total)), Some((4000, 4000)));
        assert_eq!(sup.state("m-Q4_0"), Some(ModelState::Unloaded), "served without a restart");
        let launch = sup.launches().into_iter().find(|l| l.id == "m-Q4_0").unwrap();
        assert_eq!(launch.source, tmp.path().join("m-Q4_0.gguf").display().to_string());

        let split = owner.pull("model-node", Pull::new("o/n").with_model("m-Q8_0"), |_| {}).await.unwrap();
        assert_eq!(split.files, ["Q8_0/m-Q8_0-00001-of-00002.gguf", "Q8_0/m-Q8_0-00002-of-00002.gguf"]);
        assert!(!split.served);
        let err = owner.pull("model-node", Pull::new("o/n").with_model("nope"), |_| {}).await.unwrap_err();
        assert!(matches!(err, RemoteError::BadRequest { .. }), "{err:?}");

        let list = owner.list("model-node").await.unwrap();
        assert_eq!(list.iter().map(|e| (e.id(), e.served_as.as_deref())).collect::<Vec<_>>(), [("m-Q4_0", Some("m-Q4_0")), ("m-Q8_0", None)]);
        assert_eq!(list[0].state, Some(ModelState::Unloaded));

        assert_eq!(owner.remove("model-node", "m-Q4_0").await.unwrap(), 4000);
        assert_eq!(sup.state("m-Q4_0"), None, "no longer served");
        assert_eq!(owner.remove("model-node", "m-Q8_0").await.unwrap(), 6000);
        assert!(owner.list("model-node").await.unwrap().is_empty());
        assert!(matches!(owner.remove("model-node", "m-Q8_0").await, Err(RemoteError::BadRequest { .. })));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn adapters_pull_for_a_served_base_list_and_remove() {
        let tmp = tempfile::tempdir().unwrap();
        let sup = Supervisor::new(vec![], vec![], SupervisorOptions::default());
        let auth: Arc<dyn Authorize> = Arc::new(|_: &Caller| Ok(()));
        let svc = ModelsService::new(ModelStore::new(tmp.path()).with_hub(hub()), sup.clone(), auth);
        let fits = svc.files("o/tune", None, None).await.unwrap();
        assert!(fits.fits.is_empty(), "adapters are never fit-ranked");
        assert_eq!(fits.adapters.iter().map(|a| a.id.as_str()).collect::<Vec<_>>(), ["qwen-LoRA-F16", "tune-LoRA-F16"]);

        let err = svc.pull(&Pull::new("o/tune").with_adapter_for("m-Q4_0"), |_| true).await.unwrap_err();
        assert!(err.to_string().contains("not served"), "{err}");
        svc.pull(&Pull::new("o/n").with_model("m-Q4_0").serving(false), |_| true).await.unwrap();
        let err = svc.pull(&Pull::new("o/tune").with_adapter_for("m-Q4_0"), |_| true).await.unwrap_err();
        assert!(err.to_string().contains("name one"), "{err}");
        let err = svc.pull(&Pull::new("o/tune").with_model("qwen-LoRA-F16").with_adapter_for("m-Q4_0"), |_| true).await.unwrap_err();
        assert!(err.to_string().contains("qwen3"), "architecture mismatch: {err}");

        let pulled = svc.pull(&Pull::new("o/tune").with_model("tune-LoRA-F16").with_adapter_for("m-Q4_0").serving(false), |_| true).await.unwrap();
        assert_eq!((pulled.adapter_for.as_deref(), pulled.change, pulled.served), (Some("m-Q4_0"), Some(AdapterChange::Live), true));
        assert_eq!(pulled.files, ["adapters/tune-LoRA-F16/tune-LoRA-F16.gguf"]);
        assert_eq!(sup.state("tune-LoRA-F16"), Some(ModelState::Unloaded));
        let base = sup.launches().into_iter().find(|l| l.id == "m-Q4_0").unwrap();
        assert_eq!(base.adapters.iter().map(|a| a.id.as_str()).collect::<Vec<_>>(), ["tune-LoRA-F16"]);

        let list = svc.list();
        assert_eq!(
            list.iter().map(|e| (e.id(), e.served_as.as_deref(), e.base.as_deref())).collect::<Vec<_>>(),
            [("m-Q4_0", Some("m-Q4_0"), None), ("tune-LoRA-F16", Some("tune-LoRA-F16"), Some("m-Q4_0"))]
        );
        assert_eq!(list[1].size(), 600);

        // Take it off and put it back with a scale, over models/1, keeping the files.
        let svc = Arc::new(svc);
        let owner = ModelsClient::new(dialer(svc.clone(), "owner"));
        assert_eq!(owner.unserve_adapter("model-node", "tune-LoRA-F16").await.unwrap(), "m-Q4_0");
        assert!(sup.state("tune-LoRA-F16").is_none());
        assert!(matches!(owner.unserve_adapter("model-node", "tune-LoRA-F16").await, Err(RemoteError::BadRequest { .. })));
        assert_eq!(svc.list()[1].served_as, None, "still on disk");
        assert!(matches!(owner.serve_adapter("model-node", "nope", "m-Q4_0", None).await, Err(RemoteError::BadRequest { .. })));
        assert_eq!(owner.serve_adapter("model-node", "tune-LoRA-F16", "m-Q4_0", Some(0.5)).await.unwrap(), AdapterChange::Live);
        assert!(matches!(owner.serve_adapter("model-node", "tune-LoRA-F16", "m-Q4_0", None).await, Err(RemoteError::BadRequest { .. })), "served");
        owner.set_adapter_scale("model-node", "tune-LoRA-F16", Some(0.25)).await.unwrap();
        let scale = |sup: &Supervisor| sup.launches()[0].adapters[0].scale;
        assert_eq!(scale(&sup), Some(0.25));
        assert!(matches!(owner.set_adapter_scale("model-node", "m-Q4_0", Some(0.25)).await, Err(RemoteError::BadRequest { .. })));
        assert!(owner.set_adapter_scale("model-node", "tune-LoRA-F16", Some(f32::INFINITY)).await.is_err());
        assert_eq!(scale(&sup), Some(0.25));

        assert_eq!(svc.remove("tune-LoRA-F16").await.unwrap(), 600);
        assert!(sup.state("tune-LoRA-F16").is_none(), "no longer served");
        assert_eq!(svc.list().len(), 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn without_a_budget_the_model_must_be_named() {
        let tmp = tempfile::tempdir().unwrap();
        let sup = Supervisor::new(vec![], vec![], SupervisorOptions::default());
        let auth: Arc<dyn Authorize> = Arc::new(|_: &Caller| Ok(()));
        let svc = ModelsService::new(ModelStore::new(tmp.path()).with_hub(hub()), sup, auth);
        let err = svc.pull(&Pull::new("o/n"), |_| true).await.unwrap_err();
        assert!(err.to_string().contains("no memory budget"), "{err}");
        assert!(svc.files("o/n", None, None).await.unwrap().best.is_none());
        let err = svc.pull(&Pull::new("o/n").with_model("m-Q4_0"), |_| false).await.unwrap_err();
        assert!(err.to_string().contains("cancelled"), "{err}");
        assert!(matches!(svc.pull(&Pull::new("bad"), |_| true).await, Err(RemoteError::BadRequest { .. })));
    }
}
