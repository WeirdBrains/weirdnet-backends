//! How each engine is launched and probed.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::process::Command;
use weirdnet_agent::remote::catalog::CatalogModel;
use weirdnet_agent::{MediaLimits, Modalities};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum EngineKind {
    /// llama.cpp `llama-server`.
    Llamacpp,
    /// `vllm serve` (Linux, NVIDIA sm_80+ or supported ROCm).
    Vllm,
    /// MLX-LM's `mlx_lm.server` (macOS on Apple silicon). Text only, no embeddings; the
    /// supervisor puts a key-checking proxy in front of it (see [`crate::mlx`]).
    Mlx,
}

impl EngineKind {
    pub fn name(self) -> &'static str {
        match self {
            EngineKind::Llamacpp => "llamacpp",
            EngineKind::Vllm => "vllm",
            EngineKind::Mlx => "mlx",
        }
    }

    /// The most this engine can serve. A model may serve less (see [`ModelLaunch::capabilities`]).
    pub fn capabilities(self) -> Capabilities {
        match self {
            EngineKind::Llamacpp | EngineKind::Vllm => {
                Capabilities { chat: true, embeddings: true, vision: true, audio: true, ..Capabilities::default() }
            }
            EngineKind::Mlx => Capabilities::text(),
        }
    }
}

/// What a model takes, so routing can keep requests off models that can't serve them: images
/// or audio off text-only models, embeddings off chat-only engines such as MLX-LM. With its
/// media and request [`limits`](Capabilities::limits) when known, so clients can size images,
/// audio and whole requests (catalog/1 [`CatalogModel::limits`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Capabilities {
    /// Chat and raw text completions.
    pub chat: bool,
    /// `/v1/embeddings`.
    pub embeddings: bool,
    /// Image content parts.
    pub vision: bool,
    /// Audio content parts.
    pub audio: bool,
    /// Media and request limits (images per request, image token cost and edge, audio length and
    /// token rate, request bytes), each absent when unknown, never defaulted: measured when the
    /// model loads or set by the operator ([`ModelLaunch::limits`]). Left out of records when none
    /// is known; absent in records from nodes before 0.4.8.
    #[serde(default, skip_serializing_if = "MediaLimits::is_unknown")]
    pub limits: MediaLimits,
}

impl Capabilities {
    /// Text chat and completions only.
    pub fn text() -> Self {
        Self { chat: true, ..Self::default() }
    }

    /// Embeddings only.
    pub fn embeddings() -> Self {
        Self { embeddings: true, ..Self::default() }
    }

    pub fn with_vision(mut self, on: bool) -> Self {
        self.vision = on;
        self
    }

    pub fn with_audio(mut self, on: bool) -> Self {
        self.audio = on;
        self
    }

    /// See [`Capabilities::limits`].
    pub fn with_limits(mut self, limits: MediaLimits) -> Self {
        self.limits = limits;
        self
    }

    /// A catalog/1 entry with these inputs and limits: `entry` with [`CatalogModel::inputs`]
    /// and [`CatalogModel::limits`] set from these (the rest of `entry` is kept).
    pub fn apply_to(&self, entry: CatalogModel) -> CatalogModel {
        entry.with_inputs(self.inputs()).with_limits(self.limits)
    }

    /// The inputs beyond text, as `weirdnet-agent` [`Modalities`] (what catalog entries carry).
    pub fn inputs(&self) -> Modalities {
        Modalities::new(self.vision, self.audio)
    }

    /// What a request needing `needed` would lack here (`"images"`, `"audio"`, ...), if anything.
    pub fn lacking(&self, needed: Modalities) -> Option<&'static str> {
        self.inputs().lacking(needed)
    }

    /// What both allow, with the stricter limits ([`MediaLimits::tightest`]: a limit only one
    /// side knows is kept).
    pub fn intersect(self, other: Self) -> Self {
        Self {
            chat: self.chat && other.chat,
            embeddings: self.embeddings && other.embeddings,
            vision: self.vision && other.vision,
            audio: self.audio && other.audio,
            limits: self.limits.tightest(other.limits),
        }
    }
}

/// One engine binary and how long it may take to come up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineConfig {
    pub kind: EngineKind,
    pub program: PathBuf,
    /// Arguments added to every launch of this engine.
    pub extra_args: Vec<String>,
    pub ready_timeout: Duration,
}

impl EngineConfig {
    pub fn llamacpp(program: impl Into<PathBuf>) -> Self {
        Self { kind: EngineKind::Llamacpp, program: program.into(), extra_args: Vec::new(), ready_timeout: Duration::from_secs(300) }
    }

    pub fn vllm(program: impl Into<PathBuf>) -> Self {
        Self { kind: EngineKind::Vllm, program: program.into(), extra_args: Vec::new(), ready_timeout: Duration::from_secs(1200) }
    }

    /// `program` is `mlx_lm.server`, such as [`crate::mlx::server_in`] of a venv made by
    /// [`crate::mlx::install`].
    pub fn mlx(program: impl Into<PathBuf>) -> Self {
        Self { kind: EngineKind::Mlx, program: program.into(), extra_args: Vec::new(), ready_timeout: Duration::from_secs(1200) }
    }

    /// The command for `model` on `127.0.0.1:port`. The API key goes in the environment, not the
    /// command line, so other local users can't read it from the process list.
    ///
    /// MLX-LM takes no API key: the supervisor starts it on a private port and checks the key in
    /// a proxy on the advertised one, so `api_key` is not passed to it.
    pub fn command(&self, model: &ModelLaunch, port: u16, api_key: &str) -> Command {
        let mut cmd = Command::new(&self.program);
        match self.kind {
            EngineKind::Llamacpp => {
                cmd.args(["--model", &model.source, "--host", "127.0.0.1", "--port", &port.to_string(), "--alias", &model.id, "--no-webui"]);
                if let Some(ctx) = model.context {
                    cmd.args(["--ctx-size", &ctx.to_string()]);
                }
                if let Some(slots) = model.slots {
                    cmd.args(["--parallel", &slots.to_string()]);
                }
                if let Some(ngl) = model.gpu_layers {
                    cmd.args(["--n-gpu-layers", &ngl.to_string()]);
                }
                if model.embedding {
                    cmd.arg("--embedding");
                }
                if let Some(mmproj) = &model.mmproj {
                    cmd.args(["--mmproj", mmproj]);
                }
                // Adapters load at scale 0; each request names its own (`"lora"` in the body).
                // `--lora` rather than `--lora-scaled FNAME:SCALE`, which splits on `:` and so
                // breaks on Windows paths; the scale is sent per request anyway.
                for a in &model.adapters {
                    cmd.args(["--lora", &a.path]);
                }
                if !model.adapters.is_empty() {
                    cmd.arg("--lora-init-without-apply");
                }
                cmd.env("LLAMA_API_KEY", api_key);
            }
            EngineKind::Vllm => {
                cmd.args(["serve", &model.source, "--host", "127.0.0.1", "--port", &port.to_string(), "--served-model-name", &model.id]);
                if let Some(ctx) = model.context {
                    cmd.args(["--max-model-len", &ctx.to_string()]);
                }
                if let Some(slots) = model.slots {
                    cmd.args(["--max-num-seqs", &slots.to_string()]);
                }
                if model.runtime_lora() {
                    cmd.arg("--enable-lora");
                    if !model.adapters.is_empty() {
                        // One `--lora-modules` with every module: vLLM keeps only the last flag.
                        cmd.arg("--lora-modules");
                        for a in &model.adapters {
                            cmd.arg(serde_json::json!({"name": a.id, "path": a.path, "base_model_name": model.id}).to_string());
                        }
                    }
                    let max = (model.adapters.len() as u32).max(model.lora_slots.unwrap_or(0)).max(1);
                    cmd.args(["--max-loras", &max.to_string()]);
                    // Lets the supervisor add and remove adapters on the running engine
                    // (`/v1/load_lora_adapter`); the endpoints sit behind the API key on loopback.
                    cmd.env("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "True");
                }
                cmd.env("VLLM_API_KEY", api_key);
            }
            EngineKind::Mlx => {
                // mlx_lm.server stops at 512 tokens unless told otherwise; the others run to the
                // context window.
                let max = model.context.unwrap_or(crate::mlx::DEFAULT_MAX_TOKENS).to_string();
                cmd.args(["--model", &model.source, "--host", "127.0.0.1", "--port", &port.to_string(), "--max-tokens", &max]);
            }
        }
        cmd.args(&self.extra_args).args(&model.extra_args);
        cmd
    }
}

/// A LoRA adapter served on top of a [`ModelLaunch`]'s base model, under its own pool id.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct LoraAdapter {
    /// Pool id; requests for it go to the base model's engine with the adapter applied.
    pub id: String,
    /// A LoRA GGUF for llama.cpp; a PEFT directory (or Hub id) for vLLM.
    pub path: String,
    /// How strongly it applies (llama.cpp per request; default 1.0). vLLM ignores it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<f32>,
}

// `check_adapters` refuses a scale that isn't finite, so equality is total for served adapters.
impl Eq for LoraAdapter {}

impl LoraAdapter {
    pub fn new(id: impl Into<String>, path: impl Into<String>) -> Self {
        Self { id: id.into(), path: path.into(), scale: None }
    }

    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = Some(scale);
        self
    }

    /// The scale requests use: [`LoraAdapter::scale`], else 1.0.
    pub fn effective_scale(&self) -> f32 {
        self.scale.unwrap_or(1.0)
    }
}

/// A model this node may be told to load.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelLaunch {
    /// Pool id; also the engine's served model name.
    pub id: String,
    pub engine: EngineKind,
    /// A GGUF path for llama.cpp; a path or Hub id for vLLM; an MLX-format model directory (or
    /// Hub id) for MLX-LM.
    pub source: String,
    pub context: Option<u32>,
    /// Concurrent requests (llama.cpp slots, vLLM max sequences).
    pub slots: Option<u32>,
    /// llama.cpp `--n-gpu-layers`; `None` keeps the engine's default (all that fit).
    pub gpu_layers: Option<i32>,
    /// Models in the same group are never resident together.
    pub swap_group: Option<String>,
    pub embedding: bool,
    pub extra_args: Vec<String>,
    /// Unload after this long without requests; `None` keeps it loaded.
    #[serde(default, with = "opt_secs")]
    pub idle_ttl: Option<Duration>,
    /// Wrap a stock server already running on loopback (such as `vllm serve` or `llama-server`
    /// started by hand) instead of starting an engine: the node only probes, advertises and forwards.
    #[serde(default)]
    pub wrap: Option<Wrap>,
    /// LoRA adapters served on this base (llama.cpp and vLLM only; see
    /// [`ModelLaunch::check_adapters`]). Each is its own pool id.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub adapters: Vec<LoraAdapter>,
    /// vLLM only: start the engine with LoRA on (`--enable-lora`, `--max-loras` at least this)
    /// even without adapters, so adapters can be added and removed while it runs
    /// ([`crate::Supervisor::add_adapter_live`]). A vLLM base started with adapters can take more
    /// at run time without it. llama.cpp ignores it: llama-server loads adapter files only at start.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora_slots: Option<u32>,
    /// llama.cpp only: a multimodal projector GGUF (`--mmproj`), so the model takes images
    /// (and audio, when the projector has it). See [`crate::Supervisor::set_projector_live`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mmproj: Option<String>,
    /// Media and request limits the operator sets for this model on this hardware (such as
    /// `image_tokens` or `max_images` a node can't measure); they become
    /// [`Capabilities::limits`]. Unknown by default, and never defaulted in code.
    #[serde(default, skip_serializing_if = "MediaLimits::is_unknown")]
    pub limits: MediaLimits,
}

/// A stock OpenAI-compatible server the node forwards to but doesn't own.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Wrap {
    /// Base URL without `/v1`, such as `http://127.0.0.1:8000`. Must be loopback.
    pub url: String,
    /// The server's API key, if it wants one (read from the environment or a file by the caller).
    #[serde(default, skip_serializing)]
    pub api_key: Option<String>,
}

impl std::fmt::Debug for Wrap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Wrap").field("url", &self.url).field("api_key", &self.api_key.as_ref().map(|_| "<redacted>")).finish()
    }
}

impl Wrap {
    /// Only loopback URLs: the wrapped server is on this machine, reached without TLS.
    pub fn check(&self) -> Result<(), String> {
        let rest = self.url.strip_prefix("http://").ok_or_else(|| format!("wrap url {} must start with http:// (loopback only)", self.url))?;
        let host = match rest.strip_prefix('[') {
            Some(v6) => v6.split(']').next().unwrap_or(""),
            None => rest.split(['/', ':']).next().unwrap_or(""),
        };
        if matches!(host, "127.0.0.1" | "localhost" | "::1") { Ok(()) } else { Err(format!("wrap url {} is not loopback", self.url)) }
    }

    pub fn base(&self) -> String {
        self.url.trim_end_matches('/').trim_end_matches("/v1").to_string()
    }
}

mod opt_secs {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(d: &Option<Duration>, s: S) -> Result<S::Ok, S::Error> {
        match d {
            Some(d) => s.serialize_some(&d.as_secs()),
            None => s.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Duration>, D::Error> {
        Ok(Option::<u64>::deserialize(d)?.map(Duration::from_secs))
    }
}

impl ModelLaunch {
    pub fn new(id: impl Into<String>, engine: EngineKind, source: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            engine,
            source: source.into(),
            context: None,
            slots: None,
            gpu_layers: None,
            swap_group: None,
            embedding: false,
            extra_args: Vec::new(),
            idle_ttl: None,
            wrap: None,
            adapters: Vec::new(),
            lora_slots: None,
            mmproj: None,
            limits: MediaLimits::UNKNOWN,
        }
    }

    /// See [`ModelLaunch::limits`].
    pub fn with_limits(mut self, limits: MediaLimits) -> Self {
        self.limits = limits;
        self
    }

    /// See [`ModelLaunch::mmproj`].
    pub fn with_mmproj(mut self, path: impl Into<String>) -> Self {
        self.mmproj = Some(path.into());
        self
    }

    /// See [`ModelLaunch::lora_slots`].
    pub fn with_lora_slots(mut self, slots: u32) -> Self {
        self.lora_slots = Some(slots);
        self
    }

    /// Whether the engine starts able to take adapters at run time: vLLM with adapters or
    /// [`ModelLaunch::lora_slots`].
    pub fn runtime_lora(&self) -> bool {
        self.engine == EngineKind::Vllm && self.wrap.is_none() && (!self.adapters.is_empty() || self.lora_slots.is_some_and(|n| n > 0))
    }

    pub fn with_adapters(mut self, adapters: Vec<LoraAdapter>) -> Self {
        self.adapters = adapters;
        self
    }

    /// The pool ids this launch serves: the base's, then each adapter's.
    pub fn served_ids(&self) -> Vec<String> {
        std::iter::once(self.id.clone()).chain(self.adapters.iter().map(|a| a.id.clone())).collect()
    }

    /// Whether the adapters can be served: llama.cpp or vLLM (not MLX-LM, not a wrapped server,
    /// not an embedding model), unique non-empty ids other than the base's, finite scales, and
    /// paths the engine takes. On llama.cpp each adapter must be a LoRA GGUF (`general.type` =
    /// `adapter`, `adapter.type` = `lora`) with the base GGUF's architecture; both headers are
    /// read. On vLLM a local adapter must be a PEFT LoRA directory; anything else must look like a
    /// Hub id.
    pub fn check_adapters(&self) -> Result<(), String> {
        if self.adapters.is_empty() {
            return Ok(());
        }
        let id = &self.id;
        if self.wrap.is_some() {
            return Err(format!("{id}: a wrapped server can't take adapters from this node"));
        }
        if self.embedding {
            return Err(format!("{id}: embedding models take no adapters"));
        }
        if !matches!(self.engine, EngineKind::Llamacpp | EngineKind::Vllm) {
            return Err(format!("{id}: {} serves no adapters here; use llama.cpp or vLLM", self.engine.name()));
        }
        let mut seen = std::collections::HashSet::new();
        for a in &self.adapters {
            if a.id.trim().is_empty() {
                return Err(format!("{id}: an adapter needs an id"));
            }
            if a.id == *id {
                return Err(format!("{id}: adapter id {} is the base model's", a.id));
            }
            if !seen.insert(a.id.as_str()) {
                return Err(format!("{id}: adapter id {} is used twice", a.id));
            }
            if a.scale.is_some_and(|s| !s.is_finite()) {
                return Err(format!("{}: the scale must be a finite number", a.id));
            }
        }
        match self.engine {
            EngineKind::Llamacpp => {
                let header = |path: &str| -> Result<weirdnet_model::gguf::GgufHeader, String> {
                    let f = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
                    weirdnet_model::gguf::read_header(std::io::BufReader::new(f)).map_err(|e| format!("{path}: {e}"))
                };
                let base = header(&self.source)?;
                if base.kind.is_adapter() {
                    return Err(format!("{id}: {} is an adapter, not a base model", self.source));
                }
                for a in &self.adapters {
                    if a.path.contains(',') {
                        return Err(format!("{}: llama.cpp can't take a path with a comma ({})", a.id, a.path));
                    }
                    let h = header(&a.path)?;
                    if !h.kind.is_lora() {
                        return Err(format!("{}: {} is not a LoRA adapter GGUF", a.id, a.path));
                    }
                    if h.architecture != base.architecture {
                        return Err(format!("{}: adapter architecture {} doesn't match the base's {}", a.id, h.architecture, base.architecture));
                    }
                }
            }
            _ => {
                for a in &self.adapters {
                    let p = Path::new(&a.path);
                    if p.is_dir() {
                        let config: serde_json::Value = std::fs::read(p.join("adapter_config.json"))
                            .ok()
                            .and_then(|b| serde_json::from_slice(&b).ok())
                            .ok_or_else(|| format!("{}: {} has no readable adapter_config.json", a.id, a.path))?;
                        if !config["peft_type"].as_str().unwrap_or("LORA").eq_ignore_ascii_case("lora") {
                            return Err(format!("{}: {} is not a LoRA adapter", a.id, a.path));
                        }
                    } else if p.exists() {
                        return Err(format!("{}: vLLM takes a PEFT directory, not the file {}", a.id, a.path));
                    } else if weirdnet_model::hf::check_repo(&a.path).is_err() {
                        return Err(format!("{}: {} is neither a directory nor a Hub id", a.id, a.path));
                    }
                }
            }
        }
        Ok(())
    }

    /// What this model takes. Embedding models only embed (and can't run on MLX-LM at all). A
    /// llama.cpp model takes images and audio when it has a multimodal projector
    /// ([`ModelLaunch::mmproj`], or `--mmproj` in `extra_args`); the projector decides which, and
    /// llama-server refuses what it lacks. vLLM and MLX-LM models are text only here. The
    /// [`limits`](ModelLaunch::limits) are carried over as set.
    pub fn capabilities(&self) -> Capabilities {
        let own = if self.embedding {
            Capabilities::embeddings()
        } else {
            let media = self.engine == EngineKind::Llamacpp && self.has_projector();
            Capabilities::text().with_vision(media).with_audio(media)
        };
        own.with_limits(self.limits).intersect(self.engine.capabilities())
    }

    fn has_projector(&self) -> bool {
        self.mmproj.is_some() || self.extra_args.iter().any(|a| matches!(a.split('=').next(), Some("--mmproj" | "-mm" | "--mmproj-url" | "-mmu")))
    }
}

/// The first executable called `name` on `PATH`.
pub fn find_in_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|d| d.join(name)).find(|p| is_executable(p))
}

fn is_executable(p: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(p).is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
    }
    #[cfg(not(unix))]
    {
        p.is_file()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn args(cmd: &Command) -> Vec<String> {
        cmd.as_std().get_args().map(|a| a.to_string_lossy().into_owned()).collect()
    }

    #[test]
    fn llama_args() {
        let mut m = ModelLaunch::new("qwen3-8b", EngineKind::Llamacpp, "/m/q.gguf");
        m.context = Some(8192);
        m.slots = Some(2);
        let cmd = EngineConfig::llamacpp("/bin/llama-server").command(&m, 4321, "k");
        let a = args(&cmd);
        assert!(a.windows(2).any(|w| w == ["--host", "127.0.0.1"]));
        assert!(a.windows(2).any(|w| w == ["--port", "4321"]));
        assert!(a.windows(2).any(|w| w == ["--alias", "qwen3-8b"]));
        assert!(a.windows(2).any(|w| w == ["--parallel", "2"]));
        assert!(!a.iter().any(|x| x == "k"), "the key stays out of argv");
    }

    #[test]
    fn a_projector_goes_on_the_command_and_into_capabilities() {
        let m = ModelLaunch::new("v", EngineKind::Llamacpp, "/m/v.gguf");
        assert!(!m.capabilities().vision);
        let m = m.with_mmproj("/m/mmproj.gguf");
        let a = args(&EngineConfig::llamacpp("/bin/llama-server").command(&m, 4321, "k"));
        assert!(a.windows(2).any(|w| w == ["--mmproj", "/m/mmproj.gguf"]));
        assert!(m.capabilities().vision);
        let back: ModelLaunch = serde_json::from_value(serde_json::to_value(&m).unwrap()).unwrap();
        assert_eq!(back.mmproj.as_deref(), Some("/m/mmproj.gguf"));
    }

    /// A GGUF header with string, u32 and f32 values (enough for adapter checks).
    pub(crate) fn gguf(kv: &[(&str, serde_json::Value)]) -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend(3u32.to_le_bytes());
        out.extend(0u64.to_le_bytes());
        out.extend((kv.len() as u64).to_le_bytes());
        let s = |out: &mut Vec<u8>, s: &str| {
            out.extend((s.len() as u64).to_le_bytes());
            out.extend(s.as_bytes());
        };
        for (k, v) in kv {
            s(&mut out, k);
            match v {
                serde_json::Value::String(t) => {
                    out.extend(8u32.to_le_bytes());
                    s(&mut out, t);
                }
                serde_json::Value::Number(n) if n.is_u64() => {
                    out.extend(4u32.to_le_bytes());
                    out.extend((n.as_u64().unwrap() as u32).to_le_bytes());
                }
                serde_json::Value::Number(n) => {
                    out.extend(6u32.to_le_bytes());
                    out.extend((n.as_f64().unwrap() as f32).to_le_bytes());
                }
                _ => unreachable!(),
            }
        }
        out
    }

    pub(crate) fn base_gguf(arch: &str) -> Vec<u8> {
        gguf(&[("general.architecture", arch.into()), (format!("{arch}.block_count").as_str(), 1.into())])
    }

    pub(crate) fn lora_gguf(arch: &str) -> Vec<u8> {
        gguf(&[
            ("general.architecture", arch.into()),
            ("general.type", "adapter".into()),
            ("adapter.type", "lora".into()),
            ("adapter.lora.alpha", 16.5.into()),
        ])
    }

    #[test]
    fn llama_adapter_args_and_checks() {
        let tmp = tempfile::tempdir().unwrap();
        let w = |name: &str, b: Vec<u8>| {
            let p = tmp.path().join(name);
            std::fs::write(&p, b).unwrap();
            p.display().to_string()
        };
        let base = w("base.gguf", base_gguf("qwen3"));
        let (a, b) = (w("a.gguf", lora_gguf("qwen3")), w("b.gguf", lora_gguf("qwen3")));
        let m = ModelLaunch::new("qwen", EngineKind::Llamacpp, &base)
            .with_adapters(vec![LoraAdapter::new("tune-a", &a).with_scale(0.5), LoraAdapter::new("tune-b", &b)]);
        assert_eq!(m.served_ids(), ["qwen", "tune-a", "tune-b"]);
        m.check_adapters().unwrap();
        let argv = args(&EngineConfig::llamacpp("llama-server").command(&m, 1, "k"));
        let loras: Vec<&String> = argv.windows(2).filter(|w| w[0] == "--lora").map(|w| &w[1]).collect();
        assert_eq!(loras, [&a, &b], "one --lora per adapter, in order");
        assert!(argv.iter().any(|x| x == "--lora-init-without-apply"));
        assert!(
            !args(&EngineConfig::llamacpp("l").command(&ModelLaunch::new("x", EngineKind::Llamacpp, &base), 1, "k"))
                .iter()
                .any(|x| x.starts_with("--lora"))
        );

        let bad =
            |adapters: Vec<LoraAdapter>| ModelLaunch::new("qwen", EngineKind::Llamacpp, &base).with_adapters(adapters).check_adapters().unwrap_err();
        assert!(bad(vec![LoraAdapter::new("qwen", &a)]).contains("base model's"));
        assert!(bad(vec![LoraAdapter::new("t", &a), LoraAdapter::new("t", &b)]).contains("twice"));
        assert!(bad(vec![LoraAdapter::new("t", &a).with_scale(f32::NAN)]).contains("finite"));
        assert!(bad(vec![LoraAdapter::new("t", &base)]).contains("not a LoRA"));
        assert!(bad(vec![LoraAdapter::new("t", w("llama.gguf", lora_gguf("llama")))]).contains("architecture"));
        assert!(bad(vec![LoraAdapter::new("t", tmp.path().join("missing.gguf").display().to_string())]).contains("missing.gguf"));
        assert!(bad(vec![LoraAdapter::new("t", w("a,b.gguf", lora_gguf("qwen3")))]).contains("comma"));
        let mut emb = m.clone();
        emb.embedding = true;
        assert!(emb.check_adapters().unwrap_err().contains("embedding"));
        let mut wrapped = m.clone();
        wrapped.wrap = Some(Wrap { url: "http://127.0.0.1:1".into(), api_key: None });
        assert!(wrapped.check_adapters().unwrap_err().contains("wrapped"));
        let mlx = ModelLaunch::new("m", EngineKind::Mlx, &base).with_adapters(vec![LoraAdapter::new("t", &a)]);
        assert!(mlx.check_adapters().unwrap_err().contains("mlx"));
        let json = serde_json::to_string(&ModelLaunch::new("x", EngineKind::Llamacpp, "/m")).unwrap();
        assert!(!json.contains("adapters"), "no adapters, same JSON as before");
    }

    #[test]
    fn vllm_adapter_args_and_checks() {
        let tmp = tempfile::tempdir().unwrap();
        let peft = tmp.path().join("tune");
        std::fs::create_dir(&peft).unwrap();
        std::fs::write(peft.join("adapter_config.json"), br#"{"peft_type":"LORA","r":8}"#).unwrap();
        let peft = peft.display().to_string();
        let m = ModelLaunch::new("big", EngineKind::Vllm, "Qwen/Qwen3-8B")
            .with_adapters(vec![LoraAdapter::new("tune", &peft), LoraAdapter::new("hub", "owner/some-lora")]);
        m.check_adapters().unwrap();
        let a = args(&EngineConfig::vllm("vllm").command(&m, 9000, "k"));
        assert_eq!(a.iter().filter(|x| *x == "--lora-modules").count(), 1, "vLLM keeps only the last --lora-modules");
        let at = a.iter().position(|x| x == "--lora-modules").unwrap();
        let mods: Vec<serde_json::Value> = a[at + 1..at + 3].iter().map(|j| serde_json::from_str(j).unwrap()).collect();
        assert_eq!(mods[0], serde_json::json!({"name": "tune", "path": peft, "base_model_name": "big"}));
        assert_eq!(mods[1]["name"], "hub");
        assert!(a.iter().any(|x| x == "--enable-lora"));
        assert!(a.windows(2).any(|w| w == ["--max-loras", "2"]));
        let cmd = EngineConfig::vllm("vllm").command(&m, 9000, "k");
        let env: Vec<_> = cmd.as_std().get_envs().map(|(k, v)| (k.to_owned(), v.map(|v| v.to_owned()))).collect();
        assert!(env.iter().any(|(k, v)| k == "VLLM_ALLOW_RUNTIME_LORA_UPDATING" && v.as_deref() == Some("True".as_ref())));
        // LoRA on without adapters at start, for adapters added while it runs.
        let empty = ModelLaunch::new("big", EngineKind::Vllm, "Qwen/Qwen3-8B").with_lora_slots(4);
        assert!(empty.runtime_lora());
        let a = args(&EngineConfig::vllm("vllm").command(&empty, 9000, "k"));
        assert!(a.iter().any(|x| x == "--enable-lora") && !a.iter().any(|x| x == "--lora-modules"));
        assert!(a.windows(2).any(|w| w == ["--max-loras", "4"]));
        let plain = args(&EngineConfig::vllm("vllm").command(&ModelLaunch::new("big", EngineKind::Vllm, "Q/Q"), 9000, "k"));
        assert!(!plain.iter().any(|x| x == "--enable-lora"));
        assert!(!ModelLaunch::new("q", EngineKind::Llamacpp, "/q.gguf").with_lora_slots(2).runtime_lora(), "llama.cpp ignores it");
        let file = tmp.path().join("x.gguf");
        std::fs::write(&file, lora_gguf("qwen3")).unwrap();
        let bad = |path: String| {
            ModelLaunch::new("big", EngineKind::Vllm, "Q/Q").with_adapters(vec![LoraAdapter::new("t", path)]).check_adapters().unwrap_err()
        };
        assert!(bad(file.display().to_string()).contains("PEFT directory"));
        assert!(bad("/no/such/dir".into()).contains("Hub id"));
    }

    #[test]
    fn wrap_is_loopback_only() {
        let w = |u: &str| Wrap { url: u.into(), api_key: None }.check();
        assert!(w("http://127.0.0.1:8000").is_ok());
        assert!(w("http://localhost:8000/").is_ok());
        assert!(w("http://[::1]:8000").is_ok());
        assert!(w("https://127.0.0.1:8000").is_err());
        assert!(w("http://10.0.0.2:8000").is_err());
        assert!(w("http://127.0.0.1.evil:8000").is_err());
        assert_eq!(Wrap { url: "http://127.0.0.1:8000/v1/".into(), api_key: None }.base(), "http://127.0.0.1:8000");
    }

    #[test]
    fn vllm_args() {
        let m = ModelLaunch::new("big", EngineKind::Vllm, "Qwen/Qwen3-8B");
        let a = args(&EngineConfig::vllm("vllm").command(&m, 9000, "k"));
        assert_eq!(&a[..2], ["serve", "Qwen/Qwen3-8B"]);
        assert!(a.windows(2).any(|w| w == ["--served-model-name", "big"]));
    }

    #[test]
    fn mlx_args() {
        let mut m = ModelLaunch::new("qwen3-8b", EngineKind::Mlx, "/m/mlx/Qwen3-8B-4bit");
        m.context = Some(4096);
        let cmd = EngineConfig::mlx("/v/bin/mlx_lm.server").command(&m, 9, "k");
        assert_eq!(args(&cmd), ["--model", "/m/mlx/Qwen3-8B-4bit", "--host", "127.0.0.1", "--port", "9", "--max-tokens", "4096"]);
        assert!(cmd.as_std().get_envs().all(|(_, v)| v != Some(std::ffi::OsStr::new("k"))), "MLX-LM gets no key");
        m.context = None;
        let a = args(&EngineConfig::mlx("mlx_lm.server").command(&m, 9, "k"));
        assert_eq!(a.last().map(String::as_str), Some("32768"));
        assert_eq!(serde_json::to_string(&EngineKind::Mlx).unwrap(), "\"mlx\"");
    }

    #[test]
    fn capabilities_as_modalities() {
        let media = Capabilities::text().with_vision(true).with_audio(true);
        assert_eq!(media.inputs(), Modalities::new(true, true));
        assert_eq!(Capabilities::text().lacking(Modalities::VISION), Some("images"));
        assert_eq!(Capabilities::text().with_vision(true).lacking(Modalities::new(true, true)), Some("audio"));
        assert_eq!(media.lacking(Modalities::AUDIO), None);
        assert_eq!(
            ModelLaunch::new("x", EngineKind::Mlx, "/m/x").capabilities().lacking(Modalities::VISION),
            Some("images"),
            "MLX never takes images"
        );
    }

    #[test]
    fn capabilities() {
        let mut m = ModelLaunch::new("x", EngineKind::Mlx, "/m/x");
        assert_eq!(m.capabilities(), Capabilities::text());
        m.extra_args = vec!["--mmproj".into(), "/p".into()];
        assert!(!m.capabilities().vision, "MLX-LM never takes images");
        m.embedding = true;
        assert_eq!(m.capabilities(), Capabilities::default(), "MLX-LM serves no embeddings");

        let mut l = ModelLaunch::new("g", EngineKind::Llamacpp, "/m/g.gguf");
        assert_eq!(l.capabilities(), Capabilities::text());
        l.extra_args = vec!["--mmproj=/p/mm.gguf".into()];
        let c = l.capabilities();
        assert!(c.chat && c.vision && c.audio && !c.embeddings);
        l.embedding = true;
        assert_eq!(l.capabilities(), Capabilities::embeddings());

        let v = ModelLaunch::new("v", EngineKind::Vllm, "Qwen/Qwen3-8B");
        assert_eq!(v.capabilities(), Capabilities::text());
        assert!(!EngineKind::Mlx.capabilities().embeddings && EngineKind::Llamacpp.capabilities().vision);
    }

    fn limits() -> MediaLimits {
        MediaLimits::new().with_max_images(4).with_image_tokens(256).with_image_edge(896).with_audio_max_seconds(30).with_max_request_bytes(16 << 20)
    }

    #[test]
    fn limits_reach_capabilities_and_catalog_entries() {
        assert!(Capabilities::text().limits.is_unknown());
        let l = ModelLaunch::new("g", EngineKind::Llamacpp, "/m/g.gguf").with_mmproj("/p/mm.gguf").with_limits(limits());
        let c = l.capabilities();
        assert_eq!(c.limits, limits(), "the operator's limits survive the engine's (which knows none)");
        assert!(c.vision && c.audio);
        let both = c.intersect(Capabilities::text().with_vision(true).with_limits(MediaLimits::new().with_max_images(2)));
        assert_eq!((both.limits.max_images, both.limits.image_tokens), (Some(2), Some(256)), "stricter wins, one-sided kept");
        use weirdnet_agent::remote::catalog::{ModelState as CatalogState, Purpose};
        let entry = c.apply_to(CatalogModel::new("g", Purpose::Chat, CatalogState::Warm).with_replicas(1));
        assert_eq!((entry.inputs, entry.limits, entry.replicas), (Modalities::new(true, true), limits(), 1));
        // Through the wire and back.
        let wire: CatalogModel = serde_json::from_str(&serde_json::to_string(&entry).unwrap()).unwrap();
        assert_eq!(wire.limits, limits());
    }

    #[test]
    fn records_without_limits_read_as_before() {
        let old = r#"{"chat":true,"embeddings":false,"vision":true,"audio":false}"#;
        let c: Capabilities = serde_json::from_str(old).unwrap();
        assert_eq!(c, Capabilities::text().with_vision(true));
        assert!(c.limits.is_unknown());
        assert_eq!(serde_json::to_string(&c).unwrap(), old, "no limits: the record looks as before");
        let with: Capabilities = serde_json::from_str(&serde_json::to_string(&c.with_limits(limits())).unwrap()).unwrap();
        assert_eq!(with.limits, limits());
        let cbor = weirdnet_agent::remote::to_cbor(&c.with_limits(limits())).unwrap();
        assert_eq!(weirdnet_agent::remote::from_cbor::<Capabilities>(&cbor).unwrap(), c.with_limits(limits()));
        assert_eq!(weirdnet_agent::remote::from_cbor::<Capabilities>(&weirdnet_agent::remote::to_cbor(&c).unwrap()).unwrap(), c);
        let launch = ModelLaunch::new("g", EngineKind::Llamacpp, "/m/g.gguf");
        let json = serde_json::to_string(&launch).unwrap();
        assert!(!json.contains("limits"));
        let back: ModelLaunch = serde_json::from_str(&json).unwrap();
        assert_eq!(back, launch);
        let with = launch.with_limits(limits());
        assert_eq!(serde_json::from_str::<ModelLaunch>(&serde_json::to_string(&with).unwrap()).unwrap(), with);
    }
}
