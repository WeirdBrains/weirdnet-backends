//! The Weirdnet model-node wrapper.
//!
//! A model node runs models and takes requests, like a llama.cpp or vLLM server (it can be one:
//! a stock server it only wraps). This crate starts and stops the engines and serves them; it
//! decides nothing beyond the node: what to load is the caller's decision.
//!
//! - [`engine`]: how llama.cpp `llama-server`, `vllm serve`, vLLM-Omni (`vllm serve --omni`) and
//!   MLX-LM's `mlx_lm.server` are launched and probed, each model's [`Capabilities`] (chat,
//!   embeddings, images, audio, video, transcription, speech), and
//!   [`Wrap`] for a stock OpenAI-compatible server on loopback, and [`LoraAdapter`]s served
//!   next to a llama.cpp or vLLM base model.
//! - [`mlx`]: installing a pinned `mlx-lm` into a venv, and the proxy the supervisor puts in front
//!   of `mlx_lm.server` (API key check, readiness after a one-token completion, `default_model`
//!   rewrite, text only).
//! - [`supervisor`]: the local supervisor: load (random loopback port, per-process API key in
//!   the environment, readiness probe), drain and unload, swap groups, idle unload, crash
//!   restarts with backoff and crash loops, a [`Reservation`] that holds the node's GPU for one
//!   job (nothing is placed there until it is released), [`SupervisorEvent`]s and per-model logs; chat,
//!   completion and embedding forwarding; adapters routed to their base ([`AdapterChange`]).
//! - [`sse`]: OpenAI server-sent events decoded into [`Chunk`]s, and [`ToolCalls`] to join
//!   streamed tool calls.
//! - [`openai`]: engine request bodies from `weirdnet-agent` infer/2 requests.
//! - [`host`]: [`ModelHost`], a `weirdnet-agent` `InferHost` serving raw infer/2 from the
//!   supervisor's models, transcription (infer/2 `Transcribe`) and speech out (infer/2 `Speak`,
//!   llama-server's and vLLM-Omni's `/v1/audio/speech`) included.
//! - [`fence`]: placement-lease fencing ([`Fence`], [`FenceGuard`], [`FenceRefusal`]): a model node
//!   refuses node/1 `Load` and `Unload` from a lease holder with an older term than it has seen.
//! - `node` (feature): `/weirdnet-backends/node/1` over `weirdnet-net`: signed capability
//!   records, `Load`, `Unload`, `Drain`, `Reserve`/`Release`, `GroupUp`/`GroupDown` and health.
//! - `models` (feature `node`): `/weirdnet-backends/models/1`, for the owner only: search the
//!   Hugging Face Hub, see which files fit the node, pull (split GGUFs too) and serve without a
//!   restart, list and remove; LoRA adapters for a served base too. Uploads of model files made
//!   elsewhere (resumable, checked, with provenance), copies between nodes and a disk report,
//!   with writes under a [`models::WritePolicy`].
//!
//! Platforms: the supervisor runs on Unix and Windows. On Unix an engine gets its own process
//! group, stops with SIGTERM then SIGKILL after a grace period, and on Linux dies with its
//! parent; on Windows the engine process is killed outright and processes it started are not.
//! vLLM and vLLM-Omni run on Linux only; MLX-LM on macOS on Apple silicon only.

pub mod engine;
pub mod error;
pub mod fence;
pub mod host;
#[cfg(feature = "node")]
pub mod link;
mod measure;
pub mod mlx;
#[cfg(feature = "node")]
pub mod models;
#[cfg(feature = "node")]
pub mod node;
pub mod openai;
mod speak;
pub mod sse;
pub mod supervisor;
mod transcribe;

pub use engine::{Capabilities, EngineConfig, EngineKind, LoraAdapter, ModelLaunch, Wrap, find_in_path};
pub use fence::{Fence, FenceGuard, FenceRefusal};
pub use host::{EngineModel, ModelHost, RAW_INFER_PROTOCOL, RawInferDialer, Speed};
pub use openai::Target;
pub use sse::{Chunk, ChunkDecoder, SseParser, ToolCalls};
pub use supervisor::{
    AdapterChange, ChunkStream, MAX_ADAPTER_SCALE, ModelState, ModelStatus, Reservation, Supervisor, SupervisorEvent, SupervisorOptions,
};
pub use weirdnet_agent::speech::SpeechProfile;

/// What `messages` need beyond text (images, audio, video).
pub(crate) fn model_needs(messages: &[weirdnet_agent::Message]) -> weirdnet_agent::Modalities {
    weirdnet_agent::model::needs(messages)
}
