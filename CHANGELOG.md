# Changelog

## [Unreleased]

## [0.2.0] - 2026-11-20

### Added

- Image and audio content parts (`weirdnet-agent` 0.5 `Content`) reach llama.cpp unchanged; `ModelHost` refuses them for a model without those capabilities (`Unsupported`) and refuses remote image URLs (`BadRequest`).
- `Capabilities::inputs` and `Capabilities::lacking`.

### Changed

- Breaking: needs `weirdnet-agent` 0.5 (content parts).

## [0.1.3] - 2026-11-20

### Added

- MLX-LM engine (`EngineKind::Mlx`, `EngineConfig::mlx`) for Apple silicon: `mlx_lm.server` behind a proxy that checks the node's API key, rewrites `model` to `default_model`, refuses image, audio and embedding requests, and reports ready only after a one-token completion. Wrapped MLX-LM servers get the same proxy.
- `mlx` module: `install` puts a pinned `mlx-lm` (0.32.2) into a venv with `uv` or `python3`; `install_commands`, `server_in`, `supported`.
- `Capabilities` (chat, embeddings, vision, audio): `EngineKind::capabilities`, `ModelLaunch::capabilities`, `Supervisor::capabilities` and `ModelStatus::capabilities` in capability records.

### Changed

- `Supervisor::embed` refuses MLX models with a bad request.

## [0.1.2] - 2026-11-20

### Added

- Link info in capability records: `Capability::links` (`LinkInfo`, `LinkMeasure`, `AddrClass`), set with `NodeService::set_links`; `link::local_links` lists this machine's private interfaces with the NIC's nominal speed. Only loopback, link-local and private addresses are advertised. Old verifiers ignore the field.
- `Verifier::with_roster`: also refuses records from nodes that are unknown to, or revoked in, the `weirdnet-sync` roster.

### Changed

- Needs weirdnet-agent 0.4.4 and accepts weirdnet-sync 0.5 up to 0.7 (`>=0.5, <0.8`), so apps on sync 0.6 or 0.7 share one roster type.

## [0.1.1] - 2026-11-19

### Added

- `ModelHost` serves raw prompts (`weirdnet-agent` 0.4.1 `Input::Prompt`) through the engine's `/v1/completions`; a resume continues the prompt plus the assistant prefix. Response formats and tools don't apply to raw prompts.
- `ModelHost` embeddings report the engine's prompt tokens (`EmbedBackend::embed_counted`).
- `RAW_INFER_PROTOCOL` (`/weirdnet-backends/infer/1`): raw inference in the infer/2 framing under its own id, so one node can be an inference node and a model node. `NodeServer` always serves it and lists it in the model-node announcement.
- `NodeServerOptions` and `NodeServer::start_with`: `legacy_infer2` (default on) keeps serving raw inference on `/weirdnet-agent/infer/2` for 0.1.0 inference nodes; turn it off on a node that is also an inference node.
- `RawInferDialer`: wraps the dialer of an inference node's model-node `Pool`; infer/2 opens try `/weirdnet-backends/infer/1` first and fall back to infer/2 on nodes that don't accept it (remembered per node).

### Changed

- Depends on `weirdnet-agent` `"0.4.1"` and `weirdnet-model` `"0.4"` (caret requirements, not exact pins).

## [0.1.0] - 2026-11-19

### Added

- Engine adapters for llama.cpp `llama-server` and `vllm serve` (`EngineConfig`, `ModelLaunch`), and `Wrap` for a stock OpenAI-compatible server on loopback.
- `Supervisor`: loads on a random loopback port with a per-process API key and a readiness probe; drain and unload; swap groups; idle unload (`reap_idle`, `spawn_reaper`); crash restarts with backoff and crash loops; `SupervisorEvent`s and per-model logs; chat, completion and embedding forwarding.
- SSE decoding into `Chunk`s (text, reasoning, tool-call fragments, finish and usage) and `ToolCalls` to join streamed calls.
- OpenAI request bodies from `weirdnet-agent` 0.4 infer/2 requests, including assistant-prefix continuation.
- `ModelHost`: a `weirdnet-agent` `InferHost` serving raw infer/2 (and embeddings) from the supervisor's models, with draining and measured speed.
- `node` feature: `/weirdnet-backends/node/1` with signed capability records (node key, user certificate, sequence, 90 s expiry, 15 s refresh), `Load`, `Unload`, `Drain`, `GroupUp`/`GroupDown` through the application's `GroupWorkers`, health, `NodeServer` (node/1 plus raw infer/2 on a `weirdnet-net` node) and `NodeClient`.
