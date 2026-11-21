# weirdnet-backends

The Weirdnet model-node wrapper. A model node runs models and takes requests, like a llama.cpp,
vLLM or MLX-LM server (it can even be one: a stock server it only wraps). This crate starts and stops
the engines and serves them to inference nodes; what to load is always the caller's decision.

- `EngineConfig` / `ModelLaunch`: how `llama-server`, `vllm serve` and MLX-LM's
  `mlx_lm.server` are launched (loopback only, random port, per-process API key in the
  environment, or checked by a proxy for MLX-LM) and probed. `Wrap` attaches to a stock
  OpenAI-compatible server on loopback without owning it.
- `Capabilities`: what each model takes (chat, embeddings, images, audio), in `ModelStatus`
  and so in capability records, so routing keeps image, audio and embedding requests off
  models that can't serve them. `ModelLaunch::capabilities` derives it: embedding models only
  embed; llama.cpp models with a multimodal projector (`--mmproj`) take images and audio;
  vLLM and MLX-LM models are text only.
- `Supervisor`: load and unload with draining, **swap groups** (loading one model of a group
  unloads the others), **idle unload**, crash restarts with exponential backoff and crash-loop
  detection, `SupervisorEvent`s, per-model logs; chat, completion and embedding forwarding.
  `add_model` and `remove_model` change the served models at run time, without a restart.
- `ChunkDecoder` / `ToolCalls`: OpenAI server-sent events decoded into `Chunk`s (text,
  reasoning, tool-call fragments, finish reason and usage).
- `ModelHost`: a [`weirdnet-agent`](https://crates.io/crates/weirdnet-agent) 0.5 `InferHost`
  serving **raw infer/2** from the supervisor's models. Cancel aborts the engine request;
  assistant prefixes continue (resume); draining refuses new requests.
- Images and audio: `weirdnet-agent` 0.5 content parts (`image_url` with a `data:` URL,
  `input_audio`) reach llama.cpp unchanged, in the OpenAI shape. `ModelHost` refuses them for a
  model whose `Capabilities` lack them (`Unsupported`, so a pool tries another node) and refuses
  remote image URLs (`BadRequest`). `Capabilities::inputs()` gives the `Modalities` an inference
  node puts in its catalog entry (`CatalogModel::with_inputs`).

```rust
use std::sync::Arc;
use weirdnet_backends::{EngineConfig, EngineKind, ModelHost, ModelLaunch, Supervisor, SupervisorOptions};

let chat = ModelLaunch::new("qwen3-8b", EngineKind::Llamacpp, "/models/qwen3-8b-q4_k_m.gguf");
let sup = Supervisor::new(vec![EngineConfig::llamacpp("llama-server")], vec![chat], SupervisorOptions::default());
sup.load("qwen3-8b").await?;
let host = Arc::new(ModelHost::new(sup.clone(), auth)); // serve with weirdnet_agent::remote::infer::v2::serve
```

## MLX-LM (Apple silicon)

`EngineKind::Mlx` runs MLX-format models with `mlx_lm.server` on macOS on Apple silicon:

```rust
use weirdnet_backends::{EngineConfig, EngineKind, ModelLaunch, mlx};

let server = mlx::install(&engines_dir.join("mlx"))?; // pinned mlx-lm in a venv (uv, else python3)
let mut qwen = ModelLaunch::new("qwen3-8b", EngineKind::Mlx, "/models/mlx/Qwen3-8B-4bit");
qwen.context = Some(32768); // becomes --max-tokens
let engines = vec![EngineConfig::mlx(server), EngineConfig::llamacpp("llama-server")];
```

- Launch: `mlx_lm.server --model <dir> --host 127.0.0.1 --port <p> --max-tokens <context>`
  (32768 when no context is set; MLX-LM's own default is 512).
- MLX-LM has no API key, so it runs on a private loopback port and the model's port is a small
  proxy in the node that checks the per-process key as a bearer token, rewrites each request's
  `model` to MLX-LM's `default_model`, lists the pool id in `/v1/models` and streams responses
  through.
- A wrapped stock `mlx_lm.server` (`ModelLaunch::wrap` with `EngineKind::Mlx`) gets the same
  proxy.
- Ready only after a one-token completion succeeds (MLX-LM's `/health` is 200 before the
  weights are loaded).
- Text only: image and audio content parts get 400, and so does `/v1/embeddings`;
  `Supervisor::embed` refuses MLX models, and an MLX model with `embedding` set fails to load.
  Run embedding and vision models on llama.cpp.
- `mlx::install` puts `mlx-lm==0.32.2` (`mlx::MLX_LM_VERSION`) into a venv made by `uv`
  (Python 3.12) when it is on `PATH`, else by `python3 -m venv`; `mlx::install_commands` gives
  the commands for apps that run them themselves.

## The `node` feature: `/weirdnet-backends/node/1`

How inference nodes drive model nodes over [`weirdnet-net`](https://crates.io/crates/weirdnet-net).

- **Capability records**, pushed to every subscriber: hardware and memory budget
  ([`weirdnet-model`](https://crates.io/crates/weirdnet-model) `hw`), engines, every
  configured model (state, revision, quantization, context and slot limits, load-time
  estimate, measured prefill/decode speed and time to first token), load (queued, active,
  slots, in flight), warm prefix hashes, swap groups, draining, whether the node takes
  group jobs and its private links (`NodeService::set_links`, `link::local_links`: address,
  class, interface, nominal and declared speed; public and CGNAT addresses are never
  advertised). Each is signed by the node key, chained to the user by its `NodeCert`, carries
  an increasing sequence number and expires after 90 s; it is refreshed every 15 s and on
  every change. `Verifier` checks the signature, the signer, the expiry and the user's
  certificate, and with `Verifier::with_roster` that the node is still linked (a revoked
  node's record is refused even with its old certificate); `Subscription` also rejects sequence numbers that go backwards.
- **Commands**: `Load` (with `force` for a crash loop), `Unload`, `Drain` (stop admissions and
  wait for requests under way, or resume), `GroupUp`/`GroupDown`, `Health`. Cross-machine swap
  groups (a Ray worker, an `rpc-server` on a private link) are started by the application's
  `GroupWorkers`; without one a node answers `Unsupported` and advertises `group_jobs: false`.
- `NodeServer::start` serves node/1 and raw inference on `/weirdnet-backends/infer/1`
  (`RAW_INFER_PROTOCOL`, the infer/2 framing) and, for 0.1.0 inference nodes, on
  `/weirdnet-agent/infer/2`; `NodeServer::start_with(.., NodeServerOptions::default().with_legacy_infer2(false))`
  leaves infer/2 free for a node that is also an inference node. Inference nodes wrap their
  model-node dialer in `RawInferDialer`, which tries the new id and falls back to infer/2. It
  announces `weirdnet-backends-node` with the protocols served; `NodeService::withdraw` ends subscriptions when the
  node leaves. `NodeClient` is the inference node's side, over any `weirdnet-agent` `Dialer`.
- Who may drive a model node is an `Authorize` hook; `weirdnet_agent::serve::RosterAuth`
  admits the user's linked devices.

## Platforms

| | Linux | macOS | Windows |
|---|---|---|---|
| llama.cpp | yes | yes | yes |
| vLLM | yes | no (vLLM is Linux-only) | no |
| MLX-LM | no | Apple silicon only | no |
| Graceful stop (SIGTERM, then SIGKILL after a grace period) | yes | yes | no: killed outright |
| Engine in its own process group (children stopped too) | yes | yes | no |
| Engine dies with the node process | yes (`PR_SET_PDEATHSIG`) | no | no |
| `node` feature | yes | yes | yes |

The Unix-only pieces are behind `cfg(unix)`; the library builds everywhere.

## Tests

The adapter tests run against a mock engine (the test binary re-executes itself as a fake
`llama-server`/`vllm serve`/`mlx_lm.server`), never a real one. The MLX proxy has unit tests
against a fake MLX-LM server in `src/mlx.rs`; no test runs a real `mlx_lm.server`. `tests/node.rs` runs node/1 and raw infer/2
between in-process `weirdnet-net` nodes on the memory transport.
