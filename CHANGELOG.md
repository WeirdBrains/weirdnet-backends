# Changelog

## [Unreleased]

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
