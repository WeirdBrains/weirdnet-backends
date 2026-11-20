//! The Weirdnet model-node wrapper.
//!
//! A model node runs models and takes requests, like a llama.cpp or vLLM server (it can be one:
//! a stock server it only wraps). This crate starts and stops the engines and serves them; it
//! decides nothing beyond the node: what to load is the caller's decision.
//!
//! - [`engine`]: how llama.cpp `llama-server`, `vllm serve` and MLX-LM's `mlx_lm.server` are
//!   launched and probed, each model's [`Capabilities`] (chat, embeddings, images, audio), and
//!   [`Wrap`] for a stock OpenAI-compatible server on loopback.
//! - [`mlx`]: installing a pinned `mlx-lm` into a venv, and the proxy the supervisor puts in front
//!   of `mlx_lm.server` (API key check, readiness after a one-token completion, `default_model`
//!   rewrite, text only).
//! - [`supervisor`]: the local supervisor: load (random loopback port, per-process API key in
//!   the environment, readiness probe), drain and unload, swap groups, idle unload, crash
//!   restarts with backoff and crash loops, [`SupervisorEvent`]s and per-model logs; chat,
//!   completion and embedding forwarding.
//! - [`sse`]: OpenAI server-sent events decoded into [`Chunk`]s, and [`ToolCalls`] to join
//!   streamed tool calls.
//! - [`openai`]: engine request bodies from `weirdnet-agent` infer/2 requests.
//! - [`host`]: [`ModelHost`], a `weirdnet-agent` `InferHost` serving raw infer/2 from the
//!   supervisor's models.
//! - `node` (feature): `/weirdnet-backends/node/1` over `weirdnet-net`: signed capability
//!   records, `Load`, `Unload`, `Drain`, `GroupUp`/`GroupDown` and health.
//!
//! Platforms: the supervisor runs on Unix and Windows. On Unix an engine gets its own process
//! group, stops with SIGTERM then SIGKILL after a grace period, and on Linux dies with its
//! parent; on Windows the engine process is killed outright and processes it started are not.
//! vLLM runs on Linux only; MLX-LM on macOS on Apple silicon only.

pub mod engine;
pub mod error;
pub mod host;
#[cfg(feature = "node")]
pub mod link;
pub mod mlx;
#[cfg(feature = "node")]
pub mod node;
pub mod openai;
pub mod sse;
pub mod supervisor;

pub use engine::{Capabilities, EngineConfig, EngineKind, ModelLaunch, Wrap, find_in_path};
pub use host::{EngineModel, ModelHost, RAW_INFER_PROTOCOL, RawInferDialer, Speed};
pub use sse::{Chunk, ChunkDecoder, SseParser, ToolCalls};
pub use supervisor::{ChunkStream, ModelState, ModelStatus, Supervisor, SupervisorEvent, SupervisorOptions};
