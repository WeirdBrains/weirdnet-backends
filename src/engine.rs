//! How each engine is launched and probed.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::process::Command;
use weirdnet_agent::Modalities;

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
            EngineKind::Llamacpp | EngineKind::Vllm => Capabilities { chat: true, embeddings: true, vision: true, audio: true },
            EngineKind::Mlx => Capabilities::text(),
        }
    }
}

/// What a model takes, so routing can keep requests off models that can't serve them: images
/// or audio off text-only models, embeddings off chat-only engines such as MLX-LM.
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

    /// The inputs beyond text, as `weirdnet-agent` [`Modalities`] (what catalog entries carry).
    pub fn inputs(&self) -> Modalities {
        Modalities::new(self.vision, self.audio)
    }

    /// What a request needing `needed` would lack here (`"images"`, `"audio"`, ...), if anything.
    pub fn lacking(&self, needed: Modalities) -> Option<&'static str> {
        self.inputs().lacking(needed)
    }

    /// What both allow.
    pub fn intersect(self, other: Self) -> Self {
        Self {
            chat: self.chat && other.chat,
            embeddings: self.embeddings && other.embeddings,
            vision: self.vision && other.vision,
            audio: self.audio && other.audio,
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

/// A model this node may be told to load.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
        }
    }

    /// What this model takes. Embedding models only embed (and can't run on MLX-LM at all). A
    /// llama.cpp model takes images and audio when it has a multimodal projector (`--mmproj` in
    /// `extra_args`); the projector decides which, and llama-server refuses what it lacks. vLLM
    /// and MLX-LM models are text only here.
    pub fn capabilities(&self) -> Capabilities {
        let own = if self.embedding {
            Capabilities::embeddings()
        } else {
            let media = self.engine == EngineKind::Llamacpp && self.has_projector();
            Capabilities::text().with_vision(media).with_audio(media)
        };
        own.intersect(self.engine.capabilities())
    }

    fn has_projector(&self) -> bool {
        self.extra_args.iter().any(|a| matches!(a.split('=').next(), Some("--mmproj" | "-mm" | "--mmproj-url" | "-mmu")))
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
mod tests {
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
}
